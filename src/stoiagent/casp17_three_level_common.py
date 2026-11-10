#!/usr/bin/env python3
"""
Common utilities for the three-level CASP17 stoichiometry experiment.

Level 0:
    original target description/source + full subunit sequences only.

Level 1:
    same initial input + callable research actions:
        RCSB.search
        RCSB.searchSequence
        RCSB.getPDB
        UniProt.search
        UniProt.get
        PubMed.search
        PubMed.readAbstract

Level 2:
    full StoiAgent (templates/shared-template evidence + the same research
    actions + deterministic stoichiometry validation).

Model loading, CASP17 sequence parsing, Qwen3-4B literature summarization, and
stoichiometry parsing come from agent_common/utils, shared with stage1_agent and D.
"""

import copy
import json
import os
import re
from collections import Counter, defaultdict
from urllib.parse import quote

import requests

from . import agent_common
from .agent_common import (
    compute_generation_budget,
    filter_llm_reply,
    has_candidate_header,
    parse_candidate_stoichiometries,
    reseed,
    resolve_casp17_sequence_target,
    validate_stoichiometry,
)
from .utils import generate_chain_ids, sort_subunits, normalize_stoich, simplest_stoich, strip_quotes as _strip_quotes
# D's parse_confidences differs from stage1's (it does not accept 'support = <x>'); kept as-is.
from .D_core_three_level import parse_confidences


# ---------------------------------------------------------------------
# Per-target historical evidence cutoff (calendar-date precision)
# ---------------------------------------------------------------------
def resolve_target_information_cutoff(
    target_id,
    *,
    target_release_dates=None,
    target_release_date=None,
    default_cutoff="2026-04-26",
):
    """Resolve the LAST FULL DAY before a CASP target became public.

    When target_release_dates is supplied it must be a CSV containing
    target_id,release_date (YYYY-MM-DD). All targets must be present; missing,
    ambiguous, invalid, or duplicated entries raise instead of silently
    reverting to a different evidence policy.

    When only a date is known, allowing its release day would be unsafe:
    databases generally do not expose reliable timestamps for every field.
    Therefore the cutoff is strictly release_date - 1 calendar day.

    Returns: (effective_cutoff_iso_or_None, target_release_date_or_None, source)
    """
    import csv
    from datetime import date, timedelta
    from pathlib import Path

    tid = str(target_id or "").strip().upper()
    if not tid:
        raise ValueError("target_id must be nonempty")
    if target_release_dates and target_release_date:
        raise ValueError("Use --target-release-dates OR --target-release-date, not both")

    raw_release = None
    provenance = "static information cutoff"
    if target_release_date is not None:
        raw_release = str(target_release_date).strip()
        provenance = "explicit target release date"
    elif target_release_dates:
        path = Path(target_release_dates)
        mapping = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"target_id", "release_date"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Release-date CSV {path} must have headers target_id,release_date"
                )
            for row_num, row in enumerate(reader, start=2):
                identifier = str(row.get("target_id") or "").strip().upper()
                value = str(row.get("release_date") or "").strip()
                if not identifier or not value:
                    raise ValueError(f"Blank target ID or release date at {path}:{row_num}")
                if identifier in mapping:
                    raise ValueError(f"Duplicate target ID {identifier!r} in {path}")
                try:
                    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                        raise ValueError("expected YYYY-MM-DD")
                    date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid release date for {identifier} at {path}:{row_num}: {value!r}"
                    ) from exc
                mapping[identifier] = value
        if tid not in mapping:
            raise ValueError(
                f"Missing release date for target {tid} in {path}; "
                "refusing to silently use another cutoff"
            )
        raw_release = mapping[tid]
        provenance = f"release-date CSV: {path}"

    if raw_release is not None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_release):
            raise ValueError(f"Release date for {tid} must be YYYY-MM-DD: {raw_release!r}")
        try:
            release = date.fromisoformat(raw_release)
            cutoff = release - timedelta(days=1)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid target release date for {tid}: {raw_release!r}") from exc
        return cutoff.isoformat(), release.isoformat(), provenance

    static = str(default_cutoff or "").strip()
    if not static:
        # Backward-compatible mode, only when the caller explicitly disables
        # the global cutoff AND provides no per-target release-date source.
        return None, None, "cutoff disabled (no release-date source)"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", static):
        raise ValueError(f"Static information cutoff must be YYYY-MM-DD: {static!r}")
    date.fromisoformat(static)
    return static, None, provenance


# ---------------------------------------------------------------------
# Target-information helpers independent of template evidence
# ---------------------------------------------------------------------

def _canonical_subunit_ids(n):
    ids = generate_chain_ids(max(1, n))
    return ids[:n]


def build_target_info_without_templates(target_id, catalog):
    """
    Build target information directly from the official CASP17 sequence file,
    without looking at the StoiBench template JSON contents.

    CASP sequence records are mapped, in released order, to A/B/C/... .
    The target ID is retained internally but is never emitted by
    format_blinded_target_info().
    """
    source_target_id = resolve_casp17_sequence_target(target_id, catalog)
    records = list(catalog[source_target_id])
    if not records:
        raise RuntimeError(f"No CASP17 sequence records for {target_id}")

    subunit_ids = _canonical_subunit_ids(len(records))
    mapped = {su: dict(rec) for su, rec in zip(subunit_ids, records)}

    description = next(
        (r.get("description", "") for r in records if str(r.get("description", "")).strip()),
        "",
    )
    organism = next(
        (r.get("organism", "") for r in records if str(r.get("organism", "")).strip()),
        "",
    )

    return {
        "target_id": str(target_id).upper(),
        "sequence_source_target_id": source_target_id,
        "description": description,
        "organism": organism,
        "subunits": mapped,
    }


def format_blinded_target_info(target_info):
    """LLM-visible original target information; CASP target IDs are omitted."""
    lines = ["Original target information:"]
    if target_info.get("description"):
        lines.append(f"- Target/protein description: {target_info['description']}")
    if target_info.get("organism"):
        lines.append(f"- Organism/source: {target_info['organism']}")

    lines.append("- Distinct protein subunits:")
    for su in sort_subunits(target_info["subunits"].keys()):
        rec = target_info["subunits"][su]
        src_n = rec.get("source_subunit_number")
        if src_n is not None:
            src = f"released sequence record/subunit {src_n}"
        else:
            src = f"released sequence record {rec.get('occurrence_index')}"
        lines.append(
            f"  - Subunit {su}: {src}; length={rec.get('sequence_length')} residues"
        )
        lines.append(f"    Sequence: {rec.get('sequence', '')}")
    return "\n".join(lines)


def subunit_sequences(target_info):
    return {
        su: rec["sequence"]
        for su, rec in target_info.get("subunits", {}).items()
    }


def subunit_lengths(target_info):
    return {
        su: int(rec["sequence_length"])
        for su, rec in target_info.get("subunits", {}).items()
    }


# ---------------------------------------------------------------------
# Level-0 / Level-1 prompts
# ---------------------------------------------------------------------

def minimum_baseline_candidate(target_info):
    """Same simplest-complex baseline concept used by the full D/StoiAgent."""
    subunits = sort_subunits(target_info["subunits"].keys())
    baseline = simplest_stoich(subunits)
    if len(subunits) == 1 and baseline == f"{subunits[0]}1":
        baseline = f"{subunits[0]}2"
    return baseline


def _candidate_output_policy(target_info):
    baseline = minimum_baseline_candidate(target_info)
    return (
        "Stage-1 output policy:\n"
        "- Propose at least 5 COMPLETE and UNIQUE stoichiometry candidates.\n"
        f"- Include the simple baseline candidate {baseline} so Stage 2 can compare it.\n"
        "- Every candidate must contain every target subunit exactly once in canonical "
        "subunit order, with a positive integer copy number.\n"
        "- For a single-subunit complex, X1 is invalid.\n"
        "- Give each candidate a support score from 0 to 1 based ONLY on the evidence "
        "available in this Stage-1 condition.\n"
        "- The Final Answer should be your best Stage-1 candidate, but downstream "
        "Stage 2 will re-rank the candidate pool using AlphaFold3 ranking scores.\n\n"
        "Required final format:\n"
        "Thought: <1-3 short sentences>\n"
        "Candidates:\n"
        "1) <complete stoichiometry>  support=<0..1>  support_evidence=<short phrase>\n"
        "2) <complete stoichiometry>  support=<0..1>  support_evidence=<short phrase>\n"
        "3) ...\n4) ...\n5) ...\n"
        "Final Answer: <one of the candidates>\n"
        "Support Score: <support score of the Final Answer>"
    )


def build_input_only_messages(target_info):
    subunits = sort_subunits(target_info["subunits"].keys())
    system = {
        "role": "system",
        "content": (
            "You predict the biological stoichiometry (copy number of each distinct "
            "protein subunit) of a protein complex.\n"
            f"The distinct protein subunits are: {', '.join(subunits)}.\n"
            "Use ONLY the target description/source and amino-acid sequences supplied "
            "by the user. Do not assume access to databases, templates, literature, "
            "or external tools.\n\n" + _candidate_output_policy(target_info)
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"{format_blinded_target_info(target_info)}\n\n"
            "Generate the Stage-1 candidate stoichiometries for downstream AlphaFold3 re-ranking."
        ),
    }
    return [system, user]


def build_tool_agent_messages(target_info, budgets):
    subunits = sort_subunits(target_info["subunits"].keys())
    system = {
        "role": "system",
        "content": (
            "You are a research agent that predicts biological protein-complex stoichiometry.\n"
            f"The distinct protein subunits are: {', '.join(subunits)}.\n"
            "Start only from the released target description/source and sequences. "
            "You may request external evidence using the Actions below. Python will "
            "execute an Action and return an Observation; never invent Observation lines.\n\n"
            "Available Actions:\n"
            'Action: RCSB.search("text query")\n'
            'Action: RCSB.searchSequence("A")\n'
            'Action: RCSB.getPDB("1ABC")\n'
            'Action: UniProt.search("text query")\n'
            'Action: UniProt.get("P12345")\n'
            'Action: PubMed.search("text query")\n'
            'Action: PubMed.readAbstract("PMID")\n\n'
            "Use tools selectively. Search first when you do not already know an "
            "accession/PDB/PMID. Integrate retrieved biological-assembly, subunit, "
            "oligomerization, and literature evidence with the sequence/description.\n"
            f"Tool budgets: PDB searches={budgets['pdb_search']}, "
            f"PDB reads={budgets['pdb_read']}, UniProt searches={budgets['uniprot_search']}, "
            f"UniProt reads={budgets['uniprot_read']}, PubMed searches={budgets['pubmed_search']}, "
            f"abstract reads={budgets['abstract_read']}.\n"
            "When research is sufficient, stop calling tools and produce the Stage-1 candidate pool.\n\n"
            + _candidate_output_policy(target_info)
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"{format_blinded_target_info(target_info)}\n\n"
            "Research only as needed, then generate the Stage-1 candidate stoichiometries "
            "for downstream AlphaFold3 re-ranking."
        ),
    }
    return [system, user]


# ---------------------------------------------------------------------
# Research tools
# ---------------------------------------------------------------------

ACTION_RE = re.compile(
    r'Action(?:\s*\[[^\]]+\])?\s*:\s*'
    r'(?P<tool>'
    r'RCSB\.searchSequence|RCSB\.search|RCSB\.getPDB|'
    r'UniProt\.search|UniProt\.get|'
    r'PubMed\.search|PubMed\.readAbstract'
    r')\s*\(\s*(?P<q>"[^"]*"|\'[^\']*\'|[^)]*?)\s*\)',
    flags=re.IGNORECASE,
)




def parse_actions(reply):
    actions = []
    for m in ACTION_RE.finditer(reply or ""):
        actions.append((m.group(0), m.group("tool"), _strip_quotes(m.group("q"))))
    return actions


def _safe_json_response(resp):
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(
            f"Expected JSON from {resp.url}; HTTP {resp.status_code}; "
            f"body={resp.text[:500]!r}"
        ) from e


class ResearchToolExecutor:
    """External research with fail-closed cutoff checks.
    UniProt cutoff mode excludes CURRENT entries with any audit update after
    the cutoff. It does not reconstruct archived UniProtKB entries or historical
    search rankings. Pre-cutoff PDB records may have later metadata revisions.
    """
    def __init__(
        self,
        dc,
        target_info,
        *,
        information_cutoff=None,
        budgets=None,
        timeout=45,
    ):
        self.dc = dc
        self.target_info = target_info
        self.sequences = subunit_sequences(target_info)
        self.information_cutoff = information_cutoff
        self.timeout = int(timeout)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "StoiAgent-CASP17/1.0"})

        default = {
            "pdb_search": 4,
            "pdb_read": 6,
            "uniprot_search": 4,
            "uniprot_read": 6,
            "pubmed_search": 3,
            "abstract_read": 8,
        }
        if budgets:
            default.update({k: int(v) for k, v in budgets.items()})
        self.budgets = default
        self.used = Counter()
        self.seen = set()

    def _budget(self, key):
        if self.used[key] >= self.budgets[key]:
            return False, (
                f"Tool budget exhausted for {key}: "
                f"{self.used[key]}/{self.budgets[key]}."
            )
        self.used[key] += 1
        return True, None

    def _cutoff_date(self):
        if not self.information_cutoff:
            return None
        return str(self.information_cutoff)[:10]

    # --------------------------- RCSB ---------------------------

    @staticmethod
    def _pdb_date(value):
        """Parse complete ISO/ISO-timestamp dates; reject missing/partial values."""
        from datetime import date

        if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:$|T)", value):
            return None
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None

    def _pdb_release_revision_dates(self, pdb_id):
        """Read original release AND latest revision dates from RCSB's core API.

        The same entry request used for the previous release-date filter exposes
        rcsb_accession_info.revision_date. When revision history is present, use
        its maximum date as a cross-check (never a way to weaken the cutoff).
        """
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        data = _safe_json_response(self.session.get(url, timeout=self.timeout))
        info = data.get("rcsb_accession_info") or {}
        release = self._pdb_date(info.get("initial_release_date"))
        if release is None:
            raise ValueError("missing/invalid rcsb_accession_info.initial_release_date")

        raw_revision = info.get("revision_date")
        latest = self._pdb_date(raw_revision)
        if raw_revision not in (None, "") and latest is None:
            raise ValueError(f"invalid rcsb_accession_info.revision_date: {raw_revision!r}")

        history = data.get("pdbx_audit_revision_history") or []
        if not isinstance(history, list):
            raise ValueError("pdbx_audit_revision_history is not a list")
        for record in history:
            if not isinstance(record, dict):
                raise ValueError("invalid revision-history record")
            raw_date = record.get("revision_date")
            parsed = self._pdb_date(raw_date)
            if parsed is None:
                raise ValueError(f"missing/invalid history revision_date: {raw_date!r}")
            latest = max(latest, parsed) if latest is not None else parsed

        # Fail closed rather than silently assuming that an unverified structure
        # has never been updated. Histories can supply a date if the summary
        # field is absent, but at least one verifiable revision date is required.
        if latest is None:
            raise ValueError("no verifiable latest revision date in entry metadata")
        if latest < release:
            raise ValueError("latest revision date predates initial release date")
        return release, latest

    def _pdb_allowed(self, pdb_id):
        """Option A: exclude any PDB first released OR revised after cutoff.

        Return (allowed, metadata) so search results and direct retrievals use
        the exact same policy and show which date caused a rejection.
        """
        cutoff_text = self._cutoff_date()
        meta = {"release": "NA", "latest_revision": "NA", "reason": ""}
        if not cutoff_text:
            return True, meta

        cutoff = self._pdb_date(cutoff_text)
        if cutoff is None:
            meta["reason"] = f"invalid information cutoff: {cutoff_text!r}"
            return False, meta
        try:
            release, latest_revision = self._pdb_release_revision_dates(pdb_id)
            meta.update({"release": release.isoformat(),
                         "latest_revision": latest_revision.isoformat()})
        except Exception as e:
            meta["reason"] = f"could not verify PDB release/revision dates: {e}"
            return False, meta

        if release > cutoff:
            meta["reason"] = f"initial release {release} is after cutoff {cutoff}"
            return False, meta
        if latest_revision > cutoff:
            meta["reason"] = f"latest revision {latest_revision} is after cutoff {cutoff}"
            return False, meta
        return True, meta

    def rcsb_search(self, query, rows=8):
        query = str(query).strip()
        if not query:
            return "RCSB.search requires a non-empty text query."

        ok, msg = self._budget("pdb_search")
        if not ok:
            return msg

        payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": str(query)},
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {"start": 0, "rows": int(rows)},
                "results_content_type": ["experimental"],
            },
        }
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 204:
                return "RCSB search returned no hits."
            data = _safe_json_response(resp)
        except Exception as e:
            return f"RCSB text search failed: {e}"

        ids = []
        for item in data.get("result_set", []):
            pdb = str(item.get("identifier", "")).split("_")[0].upper()
            if not re.fullmatch(r"[A-Z0-9]{4}", pdb):
                continue
            allowed, pdb_meta = self._pdb_allowed(pdb)
            if allowed:
                ids.append((pdb, pdb_meta, item.get("score")))
            if len(ids) >= rows:
                break

        if not ids:
            return "RCSB search returned no entries passing the configured cutoff."
        return "RCSB search hits:\n" + "\n".join(
            f"- {pdb}; release={meta['release']}; latest_revision={meta['latest_revision']}; score={score}"
            for pdb, meta, score in ids
        )

    def rcsb_search_sequence(self, subunit_or_sequence, rows=8):
        """
        Search RCSB by sequence.

        The action argument may be either:
          - a target subunit label, e.g. "C"; or
          - a raw amino-acid sequence.

        Argument validation happens BEFORE charging the pdb_search budget so a
        malformed tool call cannot consume the budget and block a corrected call.
        """
        raw = str(subunit_or_sequence).strip()
        su = raw.upper()

        if su in self.sequences:
            seq = self.sequences[su]
            source_label = f"target subunit {su}"
        else:
            seq = re.sub(r"\s+", "", raw).upper()
            aa = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
            if len(seq) < 20 or any(ch not in aa for ch in seq):
                return (
                    "RCSB.searchSequence expects either a target subunit label "
                    f"({', '.join(sorted(self.sequences))}) or a raw amino-acid "
                    "sequence of at least 20 residues."
                )
            source_label = f"raw sequence ({len(seq)} aa)"

        ok, msg = self._budget("pdb_search")
        if not ok:
            return msg

        payload = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 1e-3,
                    "identity_cutoff": 0.30,
                    "target": "pdb_protein_sequence",
                    "value": seq,
                },
            },
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": 0, "rows": max(int(rows) * 3, 20)},
                "results_content_type": ["experimental"],
            },
        }
        url = "https://search.rcsb.org/rcsbsearch/v2/query"
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 204:
                return f"RCSB sequence search for {source_label} returned no hits."
            data = _safe_json_response(resp)
        except Exception as e:
            return f"RCSB sequence search failed for {source_label}: {e}"

        hits = []
        seen = set()
        for item in data.get("result_set", []):
            ident = str(item.get("identifier", ""))
            pdb = ident.split("_")[0].upper()
            if pdb in seen or not re.fullmatch(r"[A-Z0-9]{4}", pdb):
                continue
            allowed, pdb_meta = self._pdb_allowed(pdb)
            if not allowed:
                continue
            seen.add(pdb)
            hits.append((pdb, ident, pdb_meta, item.get("score")))
            if len(hits) >= rows:
                break

        if not hits:
            return (
                f"RCSB sequence search for {source_label} returned no entries "
                "passing the configured cutoff."
            )
        return (
            f"RCSB sequence-search hits for {source_label}:\n"
            + "\n".join(
                f"- PDB={pdb}; entity={ident}; release={meta['release']}; latest_revision={meta['latest_revision']}; score={score}"
                for pdb, ident, meta, score in hits
            )
        )

    def rcsb_get(self, pdb_id):
        ok, msg = self._budget("pdb_read")
        if not ok:
            return msg
        pdb = str(pdb_id).strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{4}", pdb):
            return f"Invalid PDB ID: {pdb_id!r}"
        allowed, pdb_meta = self._pdb_allowed(pdb)
        if not allowed:
            return f"Blocked PDB {pdb}: {pdb_meta['reason']}."
        try:
            obs = self.dc.summarize_rcsb(pdb)
        except Exception as e:
            return f"RCSB.getPDB failed for {pdb}: {e}"
        prefix = (
            f"PDB {pdb}; initial_release_date={pdb_meta['release']}; "
            f"latest_revision_date={pdb_meta['latest_revision']}.\n"
        )
        return prefix + str(obs)

    # ------------------------- UniProt --------------------------

    @staticmethod
    def _protein_name(entry):
        pd = entry.get("proteinDescription", {}) or {}
        rec = pd.get("recommendedName", {}) or {}
        full = rec.get("fullName", {}) or {}
        value = full.get("value")
        if value:
            return value
        subs = pd.get("submissionNames", []) or []
        if subs:
            return (subs[0].get("fullName", {}) or {}).get("value", "")
        return ""


    @staticmethod
    def _uniprot_date(value):
        """Accept only complete ISO dates; incomplete history is not evidence of age."""
        from datetime import date

        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    def _uniprot_allowed(self, entry):
        """Conservative time freeze: only current entries unchanged since cutoff.

        This is intentionally stricter than recovering a historical UniProt
        record. Entries changed after the cutoff are omitted even if their
        stoichiometry annotation was already public before that date.
        """
        cutoff_text = self._cutoff_date()
        if not cutoff_text:
            return True, "No UniProt cutoff configured"

        cutoff = self._uniprot_date(cutoff_text)
        if cutoff is None:
            return False, f"Invalid information cutoff {cutoff_text!r}"

        audit = entry.get("entryAudit") or {}
        fields = (
            "firstPublicDate",
            "lastAnnotationUpdateDate",
            "lastSequenceUpdateDate",
        )
        dates = {}
        for field in fields:
            raw = audit.get(field)
            parsed = self._uniprot_date(raw)
            if parsed is None:
                return False, f"Missing/invalid UniProt entryAudit.{field}: {raw!r}"
            dates[field] = parsed

        if dates["firstPublicDate"] > cutoff:
            return False, f"UniProt entry first public on {dates['firstPublicDate']}"
        if dates["lastAnnotationUpdateDate"] > cutoff:
            return False, f"UniProt annotation updated on {dates['lastAnnotationUpdateDate']}"
        if dates["lastSequenceUpdateDate"] > cutoff:
            return False, f"UniProt sequence updated on {dates['lastSequenceUpdateDate']}"

        return True, (
            f"first_public={dates['firstPublicDate']}; "
            f"last_annotation_update={dates['lastAnnotationUpdateDate']}; "
            f"last_sequence_update={dates['lastSequenceUpdateDate']}"
        )

    @staticmethod
    def _uniprot_observation(entry, audit_note):
        """Build an observation ONLY from the validated UniProt JSON payload.

        Do not call DiscoveryCache.get_uniprot_observation under cutoff mode:
        that method can return unrelated/unverified live or cached annotations.
        """
        acc = entry.get("primaryAccession", "")
        uid = entry.get("uniProtkbId", "")
        org = (entry.get("organism") or {}).get("scientificName", "")
        seq = entry.get("sequence") or {}
        lines = [
            f"UniProt accession={acc}; entry={uid}",
            f"Protein={ResearchToolExecutor._protein_name(entry)}",
            f"Organism={org}; length={seq.get('length', 'NA')}",
            f"Historical eligibility: {audit_note}",
        ]
        gene_names = []
        for gene in entry.get("genes") or []:
            for name in [gene.get("geneName"), *(gene.get("synonyms") or [])]:
                if isinstance(name, dict) and name.get("value"):
                    gene_names.append(str(name["value"]))
        if gene_names:
            lines.append("Genes: " + ", ".join(dict.fromkeys(gene_names[:25])))

        for comment in entry.get("comments") or []:
            kind = str(comment.get("commentType", ""))
            if kind not in {"SUBUNIT", "FUNCTION", "SIMILARITY", "SUBCELLULAR LOCATION"}:
                continue
            texts = [str(t.get("value", "")).strip()
                     for t in comment.get("texts") or [] if isinstance(t, dict)]
            texts = [t for t in texts if t]
            if texts:
                lines.append(f"{kind}: " + " | ".join(texts)[:2500])

        # Historical record is unchanged since cutoff, so these cross-reference
        # links are also acceptable as record content (not necessarily as live
        # linked pages). Avoid dereferencing links without separate date checks.
        pdb_ids = []
        for xref in entry.get("uniProtKBCrossReferences") or []:
            if xref.get("database") == "PDB" and xref.get("id"):
                pdb_ids.append(str(xref["id"]))
        if pdb_ids:
            lines.append("PDB cross-references (IDs only): " + ", ".join(dict.fromkeys(pdb_ids[:30])))
        return "\n".join(lines)

    def uniprot_search(self, query, size=8):
        ok, msg = self._budget("uniprot_search")
        if not ok:
            return msg

        size = max(1, int(size))
        cutoff_enabled = bool(self._cutoff_date())
        # Retrieve more than the display limit, since strict date filtering
        # can exclude many otherwise relevant search hits.
        fetch_size = min(max(size * 5, size), 500) if cutoff_enabled else size
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {"query": str(query), "format": "json", "size": fetch_size}

        try:
            data = _safe_json_response(
                self.session.get(url, params=params, timeout=self.timeout)
            )
        except Exception as e:
            return f"UniProt search failed: {e}"

        lines = []
        excluded = 0
        for entry in data.get("results", []):
            allowed, reason = self._uniprot_allowed(entry)
            if not allowed:
                excluded += 1
                continue
            acc = entry.get("primaryAccession", "")
            uid = entry.get("uniProtkbId", "")
            name = self._protein_name(entry)
            org = (entry.get("organism") or {}).get("scientificName", "")
            length = (entry.get("sequence") or {}).get("length", "")
            lines.append(
                f"- accession={acc}; entry={uid}; protein={name}; "
                f"organism={org}; length={length}"
                + (f"; {reason}" if cutoff_enabled else "")
            )
            if len(lines) >= size:
                break

        if not lines:
            return (
                "UniProt search returned no entries passing the configured cutoff."
                if cutoff_enabled else "UniProt search returned no hits."
            )
        result = "UniProt search hits:\n" + "\n".join(lines)
        if cutoff_enabled:
            result += (
                f"\nCutoff={self._cutoff_date()}; filtered {excluded} "
                "retrieved entries with missing, invalid, or post-cutoff audit dates. "
                "Current search ranking is not a historical UniProt search snapshot."
            )
        else:
            result += "\nNOTE: UniProt data are current; no historical cutoff applied."
        return result

    def uniprot_get(self, accession):
        ok, msg = self._budget("uniprot_read")
        if not ok:
            return msg

        acc = str(accession).strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", acc):
            return f"Invalid UniProt accession: {accession!r}"

        # Preserve original behavior if historical filtering is explicitly
        # disabled. Under cutoff mode, NEVER output an unverified cache value.
        if not self._cutoff_date():
            try:
                obs = self.dc.get_uniprot_observation(acc)
                return str(obs) + "\nNOTE: UniProt cutoff is disabled; current/cache annotations."
            except Exception as e:
                return f"UniProt.get failed for {acc}: {e}"

        try:
            url = f"https://rest.uniprot.org/uniprotkb/{quote(acc, safe='')}.json"
            entry = _safe_json_response(self.session.get(url, timeout=self.timeout))
        except Exception as e:
            return f"UniProt.get failed for {acc}: {e}"

        allowed, reason = self._uniprot_allowed(entry)
        if not allowed:
            return f"Blocked UniProt {acc}: {reason}; cutoff={self._cutoff_date()}"

        return self._uniprot_observation(entry, reason)

    # -------------------------- PubMed --------------------------

    def _pubmed_date_clause(self):
        """Search prefilter only; independent XML checks still gate each PMID."""
        cutoff = self._cutoff_date()
        if not cutoff:
            return ""
        y, m, d = cutoff.split("-")
        bound = f"{y}/{m}/{d}"
        return (
            f' AND ("1900/01/01"[Date - Publication] : "{bound}"[Date - Publication])'
            f' AND ("1900/01/01"[Date - Create] : "{bound}"[Date - Create])'
        )

    @staticmethod
    def _pubmed_xml_date(node, *, allow_partial=False):
        """Return (earliest, latest) possible publication dates or None.

        A month/year publication date is allowed only if its ENTIRE range is
        before the cutoff. PubMed processing dates must be complete.
        """
        from datetime import date
        import calendar

        if node is None:
            return None
        year_text = node.findtext("Year")
        month_text = node.findtext("Month")
        day_text = node.findtext("Day")
        if year_text is None or not re.fullmatch(r"\d{4}", year_text.strip()):
            return None
        year = int(year_text)
        try:
            date(year, 1, 1)
        except ValueError:
            return None

        if not month_text or not month_text.strip():
            if not allow_partial or day_text:
                return None
            return date(year, 1, 1), date(year, 12, 31)

        month_str = month_text.strip()
        if month_str.isdigit():
            month = int(month_str)
        else:
            month_names = {
                name.lower(): i
                for i in range(1, 13)
                for name in (calendar.month_name[i], calendar.month_abbr[i])
            }
            month = month_names.get(month_str.lower())

        if month is None or not 1 <= month <= 12:
            return None
        if not day_text or not day_text.strip():
            if not allow_partial:
                return None
            return date(year, month, 1), date(
                year, month, calendar.monthrange(year, month)[1]
            )
        try:
            d = date(year, month, int(day_text.strip()))
            return d, d
        except (ValueError, TypeError):
            return None

    def _pubmed_fetch_records(self, pmids):
        """Fetch structured PubMed XML, batching PMIDs to avoid N+1 API calls."""
        import xml.etree.ElementTree as ET

        ids = [str(x).strip() for x in pmids]
        if not ids or any(not re.fullmatch(r"\d{5,10}", x) for x in ids):
            raise ValueError("Invalid PMID list for PubMed XML verification")
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        resp = self.session.get(
            url,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        records = {}
        for record in root.findall(".//PubmedArticle"):
            pmid = (record.findtext("./MedlineCitation/PMID") or "").strip()
            if pmid and pmid in ids:
                if pmid in records:
                    raise ValueError(f"Duplicate PMID {pmid} in PubMed XML response")
                records[pmid] = record
        # A missing record remains missing; the validator will reject it.
        return records

    def _pubmed_record_allowed(self, pmid, record):
        """Strict time-frozen policy for publication, creation, and revisions.

        Only publication events (not received/accepted/submitted/indexed dates)
        establish public availability. The *latest possible* date of an
        imprecisely dated publication must still precede the cutoff.
        """
        from datetime import date

        cutoff_text = self._cutoff_date()
        if not cutoff_text:
            return True, "No cutoff configured"
        try:
            cutoff = date.fromisoformat(cutoff_text)
        except ValueError:
            return False, f"Invalid information cutoff: {cutoff_text!r}"
        if record is None:
            return False, f"PMID {pmid} not present in PubMed XML response"
        citation = record.find("./MedlineCitation")
        if citation is None:
            return False, "Missing MedlineCitation"
        xml_pmid = (citation.findtext("PMID") or "").strip()
        if xml_pmid != str(pmid):
            return False, f"Mismatched PMID in XML: {xml_pmid!r}"

        dates = {}
        for tag in ("DateCreated", "DateRevised"):
            parsed = self._pubmed_xml_date(citation.find(tag))
            if parsed is None:
                return False, f"Missing/incomplete PubMed {tag}"
            dates[tag] = parsed[0]
            if parsed[0] > cutoff:
                return False, f"PubMed {tag}={parsed[0]} after cutoff={cutoff}"
        if dates["DateRevised"] < dates["DateCreated"]:
            return False, "PubMed DateRevised precedes DateCreated"

        publication_evidence = []
        article = citation.find("Article")
        if article is not None:
            for node in article.findall("./ArticleDate"):
                if node.get("DateType", "Electronic").lower() == "electronic":
                    parsed = self._pubmed_xml_date(node, allow_partial=False)
                    if parsed:
                        publication_evidence.append((parsed, "ArticleDate:Electronic"))
            journal_pubdate = article.find("./Journal/JournalIssue/PubDate")
            if journal_pubdate is not None:
                parsed = self._pubmed_xml_date(journal_pubdate, allow_partial=True)
                if parsed:
                    publication_evidence.append((parsed, "JournalIssue:PubDate"))

        history = record.find("./PubmedData/History")
        if history is not None:
            allowed_status = {"epublish", "aheadofprint", "ppublish", "ecollection"}
            for node in history.findall("PubMedPubDate"):
                status = (node.get("PubStatus") or "").lower()
                if status not in allowed_status:
                    continue
                parsed = self._pubmed_xml_date(node, allow_partial=False)
                if parsed:
                    publication_evidence.append((parsed, f"History:{status}"))

        # Only dates that definitely occurred before the cutoff can establish
        # historical availability. No receipt/acceptance or index date counts.
        eligible = [
            (bounds[1], label) for bounds, label in publication_evidence
            if bounds[1] <= cutoff
        ]
        if not eligible:
            return False, "No verified pre-cutoff public publication date in PubMed XML"
        first_public, source = min(eligible)
        return True, (
            f"publication_by={first_public} ({source}); "
            f"PubMed DateCreated={dates['DateCreated']}; "
            f"DateRevised={dates['DateRevised']}"
        )

    def _pubmed_allowed(self, pmid, *, record=None):
        """Verify an arbitrary PMID, even one not discovered by PubMed.search."""
        if not self._cutoff_date():
            return True, "No cutoff configured"
        if record is None:
            try:
                record = self._pubmed_fetch_records([pmid]).get(str(pmid))
            except Exception as e:
                return False, f"Could not verify PubMed XML dates: {e}"
        return self._pubmed_record_allowed(pmid, record)

    def pubmed_search(self, query, size=8):
        query = str(query).strip()
        if not query:
            return "PubMed.search requires a non-empty query."
        ok, msg = self._budget("pubmed_search")
        if not ok:
            return msg

        term = f"({query})" + self._pubmed_date_clause()
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        try:
            sr = _safe_json_response(
                self.session.get(
                    f"{base}/esearch.fcgi",
                    params={
                        "db": "pubmed", "term": term, "retmode": "json",
                        "retmax": int(size), "sort": "relevance",
                    },
                    timeout=self.timeout,
                )
            )
            ids = [str(x) for x in sr.get("esearchresult", {}).get("idlist", [])]
            if not ids:
                return "PubMed search returned no hits."
            sm = _safe_json_response(
                self.session.get(
                    f"{base}/esummary.fcgi",
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                    timeout=self.timeout,
                )
            )
            # Batch XML retrieval: independently check each returned record
            # BEFORE exposing its title, source or dates to the benchmark LLM.
            records = self._pubmed_fetch_records(ids) if self._cutoff_date() else {}
        except Exception as e:
            return f"PubMed search failed: {e}"

        lines = []
        result = sm.get("result", {})
        for pmid in ids:
            verification = ""
            if self._cutoff_date():
                allowed, verification = self._pubmed_record_allowed(
                    pmid, records.get(pmid)
                )
                if not allowed:
                    continue
            item = result.get(pmid, {}) or {}
            title = str(item.get("title", "")).replace("\n", " ").strip()
            source = item.get("source", "")
            if self._cutoff_date():
                lines.append(
                    f"- PMID={pmid}; {verification}; source={source}; title={title}"
                )
            else:
                pubdate = item.get("pubdate", "")
                lines.append(
                    f"- PMID={pmid}; date={pubdate}; source={source}; title={title}"
                )
        if not lines:
            return "PubMed search returned no entries passing the configured cutoff."
        return "PubMed search hits:\n" + "\n".join(lines)

    @staticmethod
    def _extract_pmid(arg):
        s = str(arg).strip()
        m = re.search(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/)?(\d{5,10})", s)
        return m.group(1) if m else None

    def pubmed_read(self, arg):
        ok, msg = self._budget("abstract_read")
        if not ok:
            return msg
        pmid = self._extract_pmid(arg)
        if not pmid:
            return (
                "PubMed.readAbstract expects a PMID or a pubmed.ncbi.nlm.nih.gov "
                f"URL; received {arg!r}."
            )
        allowed, verification = self._pubmed_allowed(pmid)
        if not allowed:
            return (
                f"Blocked PMID {pmid}: {verification}; "
                f"information cutoff={self._cutoff_date()}."
            )
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        try:
            summary = agent_common.get_or_generate_literature_summary(url, self.dc)
        except Exception as e:
            return (
                f"PubMed abstract summary failed for PMID {pmid} with fixed "
                f"{agent_common.SUMMARY_MODEL_KEY}: {e}"
            )
        return f"PMID={pmid}; {verification}.\n{summary}"

    # ------------------------- dispatcher ------------------------

    def execute(self, tool, arg):
        name = str(tool).lower()
        signature = (name, str(arg).strip())
        if signature in self.seen:
            return "This exact Action was already executed; use the previous Observation."
        self.seen.add(signature)

        if name == "rcsb.search":
            return self.rcsb_search(arg)
        if name == "rcsb.searchsequence":
            return self.rcsb_search_sequence(arg)
        if name == "rcsb.getpdb":
            return self.rcsb_get(arg)
        if name == "uniprot.search":
            return self.uniprot_search(arg)
        if name == "uniprot.get":
            return self.uniprot_get(arg)
        if name == "pubmed.search":
            return self.pubmed_search(arg)
        if name == "pubmed.readabstract":
            return self.pubmed_read(arg)
        return f"Unsupported Action: {tool}"

    def usage_dict(self):
        return {
            k: {
                "used": int(self.used[k]),
                "budget": int(self.budgets[k]),
            }
            for k in self.budgets
        }


# ---------------------------------------------------------------------
# Generic input-only/tool-enabled agent loop
# ---------------------------------------------------------------------

SUPPORT_RE = re.compile(
    r"Support\s*Score\s*:\s*([01](?:\.\d+)?|0?\.\d+)",
    flags=re.IGNORECASE,
)


def parse_final_answer(reply):
    m = re.search(r"Final\s+Answer\s*:\s*([A-Z]{1,2}\d+(?:[A-Z]{1,2}\d+)*)", reply or "", re.I)
    if not m:
        return None
    try:
        return normalize_stoich(m.group(1))
    except Exception:
        return None


def parse_support(reply):
    m = SUPPORT_RE.search(reply or "")
    if not m:
        return None
    try:
        x = float(m.group(1))
        return max(0.0, min(1.0, x))
    except Exception:
        return None


def _append_message(messages, role, content):
    if role == "user" and messages and messages[-1].get("role") == "user":
        messages[-1]["content"] += "\n\n" + str(content)
    else:
        messages.append({"role": role, "content": str(content)})


def run_simple_agent(
    llm,
    messages,
    target_info,
    *,
    temperature,
    seed,
    tool_executor=None,
    max_turns=12,
    min_candidates=5,
    finalization_turns=4,
):
    """Run one Stage-1 sample and return a validated candidate pool.

    Candidate repair is incremental: valid candidates from an imperfect list are
    retained, and the model is asked only to replace missing/invalid candidates.
    """
    reseed(seed)
    subunits = sort_subunits(target_info["subunits"].keys())
    lengths = subunit_lengths(target_info)
    baseline = minimum_baseline_candidate(target_info)
    seen_replies = set()
    trace = []
    final = None
    support = None
    final_reply = None
    final_candidates = []
    final_conf = {}

    # Persist validated candidates across repair turns.
    accepted = {}  # stoich -> support

    max_turns = int(max_turns)
    finalization_turns = max(1, min(int(finalization_turns), max_turns))
    research_last_turn = max(0, max_turns - finalization_turns)
    finalization_notice_added = False

    def accepted_text():
        if not accepted:
            return "(none yet)"
        return ", ".join(f"{s} (support={accepted[s]:.3f})" for s in accepted)

    def merge_valid_candidates(candidates, confidences):
        """Accept good rows, return row-specific problems for only the bad rows."""
        problems = []
        newly_accepted = []
        seen_this_reply = set()

        for i, stoich in enumerate(candidates, start=1):
            if stoich in seen_this_reply:
                problems.append(f"{i}) {stoich}: duplicate within this response")
                continue
            seen_this_reply.add(stoich)

            valid, msg = validate_stoichiometry(stoich, subunits)
            if not valid:
                problems.append(f"{i}) {stoich}: {msg}")
                continue
            if stoich not in confidences:
                problems.append(f"{i}) {stoich}: missing explicit support=<0..1>")
                continue

            score = float(confidences[stoich])
            accepted[stoich] = score
            newly_accepted.append(stoich)

        return newly_accepted, problems

    for turn in range(1, max_turns + 1):
        in_finalization = turn > research_last_turn
        if in_finalization and not finalization_notice_added:
            _append_message(
                messages,
                "user",
                "RESEARCH PHASE CLOSED. Do not request any more Actions. Using the "
                "evidence already collected, finish the Stage-1 candidate pool. "
                "Previously accepted valid candidates remain accepted; do not discard "
                "them just because a different candidate needs repair."
            )
            finalization_notice_added = True

        max_new = compute_generation_budget(messages, llm)
        raw = llm.generate(
            messages,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
        )
        reply = filter_llm_reply(raw)
        final_reply = reply

        if reply in seen_replies:
            _append_message(
                messages,
                "user",
                "Do not repeat the previous response. Keep previously accepted candidates "
                "and provide only the missing/replacement candidate(s), or give the completed "
                "candidate pool and Final Answer."
            )
            trace.append({
                "turn": turn,
                "reply": reply,
                "actions": [],
                "observations": [],
                "validation_error": "Repeated response",
                "phase": "finalization" if in_finalization else "research",
                "accepted_candidates": dict(accepted),
                "messages": copy.deepcopy(messages),
            })
            continue
        seen_replies.add(reply)

        actions = parse_actions(reply) if tool_executor is not None else []
        if tool_executor is None:
            accidental = parse_actions(reply)
            if accidental:
                _append_message(messages, "assistant", reply)
                _append_message(
                    messages,
                    "user",
                    "External Actions are not available in this condition. Use only the "
                    "supplied target information. Previously accepted candidates remain "
                    f"accepted: {accepted_text()}. Finish the candidate pool."
                )
                trace.append({
                    "turn": turn, "reply": reply, "actions": [], "observations": [],
                    "phase": "finalization", "accepted_candidates": dict(accepted),
                    "messages": copy.deepcopy(messages)
                })
                continue

        candidate_final = parse_final_answer(reply)
        candidates = parse_candidate_stoichiometries(reply)
        confidences = parse_confidences(reply)

        if actions:
            _append_message(messages, "assistant", reply)
            if in_finalization:
                rejection = (
                    "Research is closed for this sample. Do NOT request additional Actions. "
                    f"Already accepted candidates: {accepted_text()}. Finish or repair the "
                    "candidate pool now."
                )
                _append_message(messages, "user", rejection)
                trace.append({
                    "turn": turn, "reply": reply,
                    "actions": [x[0] for x in actions], "observations": [],
                    "validation_error": rejection, "phase": "finalization",
                    "accepted_candidates": dict(accepted),
                    "messages": copy.deepcopy(messages),
                })
                continue

            observations = []
            for full_action, tool, arg in actions:
                obs = tool_executor.execute(tool, arg)
                observations.append({"action": full_action, "tool": tool, "arg": arg, "observation": obs})
            _append_message(messages, "user", "\n\n".join(
                f"{x['action']}\nObservation: {x['observation']}" for x in observations
            ))
            if turn == research_last_turn:
                _append_message(
                    messages,
                    "user",
                    "Research turns are now exhausted. On the next turn do not request "
                    "Actions; finish the Stage-1 candidate pool."
                )
                finalization_notice_added = True
            trace.append({
                "turn": turn, "reply": reply,
                "actions": [x[0] for x in actions], "observations": observations,
                "phase": "research", "accepted_candidates": dict(accepted),
                "messages": copy.deepcopy(messages),
            })
            continue

        row_problems = []
        if candidates:
            _, row_problems = merge_valid_candidates(candidates, confidences)

        # A malformed candidate section with no parseable rows still needs repair.
        if has_candidate_header(reply) and not candidates:
            row_problems.append(
                "Candidates section could not be parsed; use numbered rows like "
                "1) A1B1C1  support=0.80  support_evidence=..."
            )

        missing_baseline = baseline not in accepted
        need_n = max(0, int(min_candidates) - len(accepted))

        final_problem = None
        if candidate_final:
            valid, reason = validate_stoichiometry(candidate_final, subunits)
            if not valid:
                final_problem = f"Final Answer {candidate_final} is invalid: {reason}"
            elif candidate_final not in accepted:
                final_problem = (
                    f"Final Answer {candidate_final} is not currently in the validated candidate pool."
                )

        # If there are enough accepted candidates, ignore bad extras instead of
        # forcing a rewrite of the entire list.  Only unresolved requirements matter.
        unresolved = []
        if need_n > 0:
            unresolved.append(f"need {need_n} more valid unique candidate(s)")
        if missing_baseline:
            unresolved.append(f"must include baseline {baseline}")
        if final_problem:
            unresolved.append(final_problem)

        if unresolved:
            _append_message(messages, "assistant", reply)
            bad_text = (" Rejected rows: " + " | ".join(row_problems)) if row_problems else ""
            repair = (
                f"Accepted and retained candidates: {accepted_text()}. "
                + "; ".join(unresolved)
                + ". Do NOT replace or discard the accepted candidates. "
                  "Provide only the needed replacement/additional candidate rows with "
                  "support scores (or restate the full list if you prefer)."
                + bad_text
            )
            _append_message(messages, "user", repair)
            trace.append({
                "turn": turn, "reply": reply, "actions": [], "observations": [],
                "validation_error": repair,
                "phase": "finalization" if in_finalization else "research",
                "accepted_candidates": dict(accepted),
                "rejected_candidate_problems": row_problems,
                "messages": copy.deepcopy(messages),
            })
            continue

        # Requirements are satisfied. Bad extra rows are simply ignored.
        if candidate_final is None:
            # Ask for only the final choice; keep the accepted pool intact.
            _append_message(messages, "assistant", reply)
            _append_message(
                messages,
                "user",
                f"The validated candidate pool is complete: {accepted_text()}. "
                "Choose one of these as Final Answer and give its Support Score."
            )
            trace.append({
                "turn": turn, "reply": reply, "actions": [], "observations": [],
                "phase": "finalization" if in_finalization else "research",
                "accepted_candidates": dict(accepted),
                "rejected_candidate_problems": row_problems,
                "messages": copy.deepcopy(messages),
            })
            continue

        final = candidate_final
        support = accepted.get(final, parse_support(reply))
        final_candidates = list(accepted.keys())
        final_conf = dict(accepted)
        _append_message(messages, "assistant", reply)
        trace.append({
            "turn": turn, "reply": reply, "actions": [], "observations": [],
            "phase": "finalization" if in_finalization else "research",
            "accepted_candidates": dict(accepted),
            "rejected_candidate_problems": row_problems,
            "messages": copy.deepcopy(messages),
        })
        break

    return {
        "final_answer": final,
        "support": support,
        "candidates": final_candidates,
        "candidate_confidences": final_conf,
        "last_reply": final_reply,
        "messages": messages,
        "trace": trace,
        "tool_usage": tool_executor.usage_dict() if tool_executor else {},
    }


def aggregate_samples(samples):
    """
    Rank candidates by mean confidence over ALL stochastic samples.

    If a candidate is absent from a sample, that sample contributes 0.0.
    Therefore:

        global_avg = sum(observed confidences) / N_total_samples
                   = conditional_avg * n_observed / N_total_samples

    This prevents a one-off high-confidence candidate from outranking a
    consistently proposed candidate.
    """
    conf_lists = defaultdict(list)
    final_counts = Counter()
    n_total = len(samples)

    for sample in samples:
        if sample.get("final_answer"):
            final_counts[sample["final_answer"]] += 1
        for stoich, score in (sample.get("candidate_confidences") or {}).items():
            try:
                conf_lists[stoich].append(float(score))
            except Exception:
                pass

    conditional_avg = {
        stoich: sum(vals) / len(vals)
        for stoich, vals in conf_lists.items()
        if vals
    }
    global_avg = {
        stoich: (sum(vals) / n_total if n_total else 0.0)
        for stoich, vals in conf_lists.items()
        if vals
    }
    observed_n = {stoich: len(vals) for stoich, vals in conf_lists.items()}

    ranked = sorted(
        global_avg,
        key=lambda stoich: (
            -global_avg[stoich],
            -observed_n[stoich],
            -conditional_avg[stoich],
            stoich,
        ),
    )

    return {
        "n_samples": n_total,
        "n_samples_with_candidates": sum(bool(s.get("candidates")) for s in samples),
        "final_answer_counts": dict(final_counts),
        # Keep this key for downstream compatibility. It is now the GLOBAL mean.
        "avg_candidate_support": global_avg,
        "global_avg_candidate_support": global_avg,
        "conditional_avg_candidate_support": conditional_avg,
        "candidate_observation_counts": observed_n,
        "ranked": ranked,
        "top1": ranked[0] if ranked else None,
        "top5": ranked[:5],
    }


def save_simple_results(outdir, base_name, level_name, model_key, temperature, samples):
    """Write diagnostics, D-style traces, and D-compatible phase1 candidate files for AF3 Stage 2."""
    os.makedirs(outdir, exist_ok=True)
    agg = aggregate_samples(samples)
    payload = {"level": level_name, "model": model_key, "temperature": float(temperature), "aggregate": agg, "samples": samples}
    json_path = os.path.join(outdir, f"{base_name}_{level_name}_samples.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(outdir, f"{base_name}_{level_name}_summary.out")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Level: {level_name}\nModel: {model_key}\nTemperature: {temperature}\n")
        f.write(f"Samples with valid candidate sets: {agg['n_samples_with_candidates']}/{agg['n_samples']}\n")
        f.write("Ranking score: sum(candidate confidence across runs) / total runs; absent candidate = 0\n")
        f.write(f"Stage-1 top candidate: {agg['top1'] or 'NA'}\n\n")
        for i, s in enumerate(agg["ranked"], 1):
            f.write(
                f"{i}. {s:<12} "
                f"avg_conf={agg['avg_candidate_support'][s]:.3f} "
                f"n={agg['candidate_observation_counts'][s]}/{agg['n_samples']} "
                f"conditional_avg={agg['conditional_avg_candidate_support'][s]:.3f}\n"
            )

    # Save a full D-style cumulative trace for every stochastic Stage-1 sample.
    # Each turn repeats the cumulative conversation exactly as D does.
    for idx, sample in enumerate(samples, 1):
        sample_id = int(sample.get("sample_id", idx))
        trace_path = os.path.join(outdir, f"{base_name}_sample{sample_id:02d}_phase1.trace")
        with open(trace_path, "w", encoding="utf-8") as f:
            trace = sample.get("trace") or []
            if not trace:
                f.write("(No trace recorded.)\n")
                continue
            for step in trace:
                f.write(f"\n--- Turn {step.get('turn', '?')} ---\n")
                step_messages = step.get("messages") or []
                if step_messages:
                    for m in step_messages:
                        f.write(f"{str(m.get('role', 'unknown')).upper()}: {m.get('content', '')}\n\n")
                else:
                    # Backward-compatible fallback for old cached/simple traces.
                    if step.get("reply") is not None:
                        f.write(f"ASSISTANT: {step.get('reply', '')}\n\n")
                    for obs in step.get("observations") or []:
                        f.write(f"ACTION: {obs.get('action', '')}\n")
                        f.write(f"OBSERVATION: {obs.get('observation', '')}\n\n")
                    if step.get("validation_error"):
                        f.write(f"VALIDATION_ERROR: {step['validation_error']}\n\n")

        # Also write a compact non-cumulative trace for debugging.  The original
        # .trace above is retained unchanged for D compatibility.
        compact_path = os.path.join(
            outdir, f"{base_name}_sample{sample_id:02d}_phase1.compact.trace"
        )
        with open(compact_path, "w", encoding="utf-8") as cf:
            for step in sample.get("trace") or []:
                cf.write(
                    f"\n--- Turn {step.get('turn', '?')} "
                    f"[{step.get('phase', 'unknown')}] ---\n"
                )
                if step.get("reply") is not None:
                    cf.write(f"ASSISTANT: {step.get('reply', '')}\n\n")
                for obs in step.get("observations") or []:
                    cf.write(f"ACTION: {obs.get('action', '')}\n")
                    cf.write(f"OBSERVATION: {obs.get('observation', '')}\n\n")
                if step.get("validation_error"):
                    cf.write(f"VALIDATION_ERROR: {step['validation_error']}\n\n")

    # Existing downstream Stage 2 consumes this D-style Phase-1 format.
    k_label = 15 if len(samples) >= 15 else len(samples)
    top5_path = os.path.join(outdir, f"{base_name}_phase1_top5_k{k_label}.out")
    with open(top5_path, "w", encoding="utf-8") as f:
        f.write(f"=== Top-5 Predictions for {base_name} (phase1, first {k_label} samples) ===\n\n")
        if not agg["ranked"]:
            f.write("RESULT: EMPTY_AFTER_FILTER\nTop-5: (none)\n")
        else:
            for i, s in enumerate(agg["ranked"][:5], 1):
                f.write(f"{i}. {s:<12} avg_conf={agg['avg_candidate_support'][s]:.3f}\n")

    topall_path = os.path.join(outdir, f"{base_name}_phase1_topall_k{k_label}.out")
    with open(topall_path, "w", encoding="utf-8") as f:
        f.write(f"=== All Predictions for {base_name} (phase1, first {k_label} samples) ===\n\n")
        if not agg["ranked"]:
            f.write("RESULT: EMPTY_AFTER_FILTER\nCandidates: (none)\n")
        else:
            for i, s in enumerate(agg["ranked"], 1):
                f.write(f"{i}. {s:<12} avg_conf={agg['avg_candidate_support'][s]:.3f}\n")
    return json_path, summary_path, agg
