#!/usr/bin/env python3
"""StoiAgent Stage 1: shared templates with a compact subunit-template fallback.

Run from this repository with a prepared target JSON and CASP17 sequence file.
"""
import re
import copy
import os
import argparse
import json
import gzip
from collections import Counter

import numpy as np


from .discovery_cache import DiscoveryCache
from .utils import sort_subunits, normalize_stoich, simplest_stoich, strip_quotes as _strip_quotes
from . import agent_common
from .agent_common import (
    BENCHMARK_MODEL_KEYS,
    HFLLM,
    MODEL_MAP,
    SEEDS_30,
    STOICH_RE,
    SUMMARY_MODEL_KEY,
    compute_generation_budget,
    extract_casp17_target_id,
    filter_llm_reply,
    format_casp17_target_info_block,
    has_candidate_header,
    infer_subunit_lengths_from_json,
    load_casp17_sequence_catalog,
    parse_candidate_stoichiometries,
    reseed,
    resolve_casp17_sequence_target,
    sanitize_chat_messages,
    validate_stoichiometry,
)
from .research_tools import ResearchToolExecutor, resolve_target_information_cutoff


EXPERIMENT_VARIANT = 'S1_stoiagent_shared_template_compact_soft_fallback_search'
EXPERIMENT_DESCRIPTION = 'Stage-1 StoiAgent: shared-template-only evidence when available; otherwise fall back to the supplied v8 compact/soft StoiAgent prompts (significant-copy observations and up to three representative hits per subunit) for homo- and hetero-multimers; matched research tools and L1 output-validity rules.'
MAX_ABSTRACT_READS = 10

# Level-2 research configuration. A fresh executor is created per stochastic
# sample, so tool budgets reset for every sample.
L2_TARGET_INFO = None
L2_INFORMATION_CUTOFF = "2026-04-26"
L2_RESEARCH_BUDGETS = {
    "pdb_search": 4,
    "pdb_read": 6,
    "uniprot_search": 4,
    "uniprot_read": 6,
    "pubmed_search": 3,
    "abstract_read": 8,
}
# Prefer direct shared templates. If none jointly map >=2 distinct target
# subunits, use the supplied v8 compact/soft prompt instead. A homo-multimer has
# one distinct target subunit, so it uses the subunit fallback. All templates
# remain advisory. No Phase-2/reference stoichiometry information is exposed.


# =========================================================
# Action regex
# =========================================================
_ACTION_RE = re.compile(
    r'Action(?:\s*\[[^\]]+\])?\s*:\s*'
    r'(?P<tool>'
    r'RCSB\.searchSequence|RCSB\.search|RCSB\.getPDB|'
    r'UniProt\.search|UniProt\.get|'
    r'PubMed\.search|PubMed\.readAbstract'
    r')\s*'
    r'\(\s*(?P<q>"[^"]*"|\'[^\']*\'|[^)]*?)\s*\)',
    flags=re.IGNORECASE
)


def validate_and_clean(reply, seen=None):
    if seen is None:
        seen = set()

    final_answer = None
    m = re.search(r"Final Answer\s*:(.*)", reply, flags=re.IGNORECASE)
    after_colon = m.group(1).strip() if m else None
    if after_colon:
        stoich_match = re.search(r"\b([A-Z]{1,2}\d+([A-Z]{1,2}\d+)*)\b", after_colon)
        final_answer = stoich_match.group(1) if stoich_match else None

    extra_actions = []
    for m in _ACTION_RE.finditer(reply):
        tool = m.group("tool")
        arg = _strip_quotes(m.group("q")).strip()
        sig = f"{tool}|{arg}"
        if sig not in seen:
            seen.add(sig)
            full_action = m.group(0)
            extra_actions.append((full_action, tool, arg))

    return final_answer, extra_actions, seen


def parse_confidences(reply_text):
    confs = {}
    for line in reply_text.splitlines():
        m = re.search(
            r"([A-Z]{1,2}\d+(?:[A-Z]{1,2}\d+)*)\s+(?:fused_)?support\s*=\s*<?\s*([0-9]*\.?[0-9]+)\s*>?",
            line,
            re.I
        )
        if m:
            try:
                stoich = normalize_stoich(m.group(1))
            except ValueError:
                continue

            try:
                score = float(m.group(2))
            except Exception:
                continue

            score = max(0.0, min(1.0, score))
            confs[stoich] = score
    return confs


def parse_candidate_support_evidence(reply_text):
    """Return stoichiometry -> model-provided support_evidence text."""
    evidence = {}
    for line in reply_text.splitlines():
        line = line.strip()
        m = re.search(rf"^\s*\d+\s*[\.\)]\s*{STOICH_RE}\b(?P<rest>.*)$", line)
        if not m:
            continue
        try:
            stoich = normalize_stoich(m.group(1))
        except ValueError:
            continue
        rest = m.group("rest") or ""
        em = re.search(r"\bsupport_evidence\s*=\s*(.+?)\s*$", rest, flags=re.IGNORECASE)
        if em:
            reason = em.group(1).strip()
            if reason:
                evidence[stoich] = reason
    return evidence


def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _template_copy_count(hit):
    for key in ("copies", "template_chain_count", "chain_count"):
        value = hit.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _compact_template_hit(hit):
    """One compact direct-evidence representation; no duplicated derived priors."""
    bits = [f"PDB={hit.get('id', 'NA')}"]
    copy_n = _template_copy_count(hit)
    if copy_n is not None:
        bits.append(f"template_copies={copy_n}")
    cov = _safe_float(hit.get("coverage"))
    if cov is not None:
        bits.append(f"coverage={cov:.3f}")
    ev = _safe_float(hit.get("evalue"))
    if ev is not None:
        bits.append(f"evalue={ev:.2e}")
    sto = hit.get("template_complex_stoichiometry")
    if sto not in (None, "", "?", "N/A"):
        bits.append(f"template_assembly={sto}")
    methods = hit.get("experimental_methods")
    if methods:
        bits.append(f"method={methods}")
    resolution = hit.get("resolution")
    if resolution not in (None, "", "N/A"):
        bits.append(f"resolution={resolution}")
    title = hit.get("title")
    if title:
        title = re.sub(r"\s+", " ", str(title)).strip()
        bits.append(f"title={title[:160]}")
    citation = hit.get("citation") or hit.get("url")
    if citation and citation != "N/A":
        bits.append(f"citation={citation}")
    return "; ".join(bits)


def _copy_distribution(hits, evalue_threshold=1.0):
    counts = Counter()
    for hit in hits:
        ev = _safe_float(hit.get("evalue"), 1e99)
        if ev is None or ev > evalue_threshold:
            continue
        n = _template_copy_count(hit)
        if n is not None and n > 0:
            counts[n] += 1
    return counts


def _shared_covered_subunits(record):
    """Return unique target subunits jointly mapped by one shared-template record."""
    covered = record.get("covered_subunits", []) or []
    if not isinstance(covered, (list, tuple, set)):
        covered = []
    covered = [str(x) for x in covered if str(x).strip()]

    # Older JSON bundles may omit covered_subunits but retain per_subunit_hits.
    if not covered:
        per_sub = record.get("per_subunit_hits", {}) or {}
        if isinstance(per_sub, dict):
            covered = [str(x) for x in per_sub.keys() if str(x).strip()]

    return sort_subunits(list(dict.fromkeys(covered)))


def _compact_shared_template_lines(
    shared_templates_for_agent, shared_template_summary, max_items=5,
    subunit_ids=None,
):
    """Render only direct templates that jointly map at least two target subunits."""
    lines = []
    seen_pdb = set()
    target_subunits = set(subunit_ids) if subunit_ids is not None else None

    def covered_target_subunits(record):
        covered = _shared_covered_subunits(record)
        if target_subunits is not None:
            covered = [su for su in covered if su in target_subunits]
        return covered

    # Prefer the richer direct shared-template records when available.  Do not use
    # top-pattern/consensus restatements: the goal is one representation per source PDB.
    rich = []
    if shared_template_summary:
        rich = list(shared_template_summary.get("shared_templates", []) or [])

    for item in rich:
        covered = covered_target_subunits(item)
        if len(covered) < 2:
            continue
        pdb_id = str(item.get("id", "NA"))
        if pdb_id in seen_pdb:
            continue
        seen_pdb.add(pdb_id)

        bits = [f"PDB={pdb_id}", f"covers={covered}"]
        missing = item.get("missing_target_subunits", [])
        if missing:
            bits.append(f"missing={missing}")
        patterns = item.get("target_stoich_patterns_from_shared_template", [])
        if patterns:
            bits.append(f"observed_pattern={patterns}")
        cov = _safe_float(item.get("best_coverage"))
        if cov is not None:
            bits.append(f"best_coverage={cov:.3f}")
        ev = _safe_float(item.get("best_evalue"))
        if ev is not None:
            bits.append(f"best_evalue={ev:.2e}")
        stype = item.get("shared_template_type")
        if stype:
            bits.append(f"type={stype}")
        lines.append("- " + "; ".join(bits))
        if len(lines) >= max_items:
            return lines

    # Fill remaining slots from direct shared-template records.  Per-subunit copy
    # observations are allowed here only because they come from the SAME shared PDB;
    # isolated single-subunit templates are handled separately by the fallback.
    for rec in (shared_templates_for_agent or []):
        covered = covered_target_subunits(rec)
        if len(covered) < 2:
            continue
        pdb_id = str(rec.get("id", "NA"))
        if pdb_id in seen_pdb:
            continue
        seen_pdb.add(pdb_id)

        bits = [f"PDB={pdb_id}", f"covers={covered}"]
        cov = _safe_float(rec.get("best_coverage"))
        if cov is not None:
            bits.append(f"best_coverage={cov:.3f}")
        ev = _safe_float(rec.get("best_evalue"))
        if ev is not None:
            bits.append(f"best_evalue={ev:.2e}")

        copy_obs = []
        for su in covered:
            hit = (rec.get("per_subunit_hits") or {}).get(su)
            if not hit:
                continue
            n = _template_copy_count(hit)
            if n is not None:
                copy_obs.append(f"{su}:{n}")
        if copy_obs:
            bits.append("same_PDB_copy_observations=" + ",".join(copy_obs))

        lines.append("- " + "; ".join(bits))
        if len(lines) >= max_items:
            break

    return lines


def select_structural_template_evidence(
    templates, shared_templates_for_agent=None, shared_template_summary=None,
):
    """Use shared evidence if present; otherwise fall back for every subunit.

    Presence is determined from direct records in either shared input, never
    from summary counts or predicted consensus patterns. Partial shared records
    still count as shared, preserving the original shared-template behavior.
    """
    shared_lines = _compact_shared_template_lines(
        shared_templates_for_agent, shared_template_summary,
        max_items=5, subunit_ids=templates.keys(),
    )
    if shared_lines:
        return {"mode": "shared_templates", "lines": shared_lines}
    has_subunit_templates = any(
        isinstance(hit, dict)
        for hits in templates.values() for hit in (hits or [])
    )
    return {
        "mode": "subunit_fallback" if has_subunit_templates else "no_templates",
    }


def _build_shared_template_user_prompt_from_json(
    templates,
    shared_templates_for_agent,
    subunit_consensus_summary,
    shared_template_summary,
    constraint_summary,
    subunit_lengths,
    target_info=None,
    partial_scaffold=None,
    scaffold_completion_candidates=None,
    scaffold_copy_options=None,
    symmetry_info=None,
):
    """v9 Stage-1 prompt: L1 + shared/multi-subunit template evidence only."""
    subunit_ids = sort_subunits(list(templates.keys()))
    native_candidate = simplest_stoich(subunit_ids)
    if len(subunit_ids) == 1 and native_candidate == f"{subunit_ids[0]}1":
        native_candidate = f"{subunit_ids[0]}2"

    lines = ["Predict the biological stoichiometry of this protein complex."]

    target_block = format_casp17_target_info_block(target_info)
    if target_block:
        lines.extend(["", target_block])

    lines.extend([
        "",
        "1. Shared/multi-subunit structural-template evidence (ADVISORY, not constraints):",
        "- Only templates that jointly map at least two target subunits are shown here.",
        "- Each listed PDB is one structural evidence source; fields derived from the same PDB are not independent evidence.",
        "- A mapped biological assembly from a homologous/shared template may support a relative copy-number pattern, but it is not a hard constraint on the target.",
        "- Do not assume that a partial shared-template pattern determines the copy number of uncovered target subunits.",
        "- Target-specific external database/literature evidence may confirm or contradict shared-template evidence.",
    ])

    shared_lines = _compact_shared_template_lines(
        shared_templates_for_agent,
        shared_template_summary,
        max_items=5,
        subunit_ids=subunit_ids,
    )
    if shared_lines:
        lines.extend(shared_lines)
    else:
        lines.append("- no shared template jointly mapping at least two target subunits")

    lines.extend([
        "",
        "2. Optional external research (same tool interface and budgets as S1_external_search):",
        "- Use external research selectively when it can resolve biological assembly, oligomerization, conflicting evidence, or target-specific identity.",
        "- Available Actions:",
        '  Action: RCSB.search("text query")',
        '  Action: RCSB.searchSequence("A")',
        '  Action: RCSB.getPDB("1ABC")',
        '  Action: UniProt.search("text query")',
        '  Action: UniProt.get("P12345")',
        '  Action: PubMed.search("text query")',
        '  Action: PubMed.readAbstract("PMID")',
        "- Do not invent Observation lines.",
        "",
        "Stage-1 output policy (matched to L0/L1):",
        "- Propose at least 5 COMPLETE and UNIQUE stoichiometry candidates.",
        f"- Include the simple baseline candidate {native_candidate} so Stage 2 can compare the same baseline across variants.",
        "- Every candidate must contain every target subunit exactly once in canonical subunit order, with a positive integer copy number.",
        "- For a single-subunit complex, X1 is invalid.",
        "- Give each candidate a support score from 0 to 1 based on the evidence available in this condition.",
        "- Shared-template patterns are suggestions only; include or rank them only when you judge them biologically plausible.",
        "- The Final Answer should be your best Stage-1 candidate; downstream Stage 2 will re-rank the candidate pool using AlphaFold3 ranking scores.",
        "",
        "Required final format:",
        "Thought: <1-3 short sentences>",
        "Candidates:",
        "1) <complete stoichiometry>  support=0.70  support_evidence=<short phrase>",
        "2) <complete stoichiometry>  support=0.25  support_evidence=<short phrase>",
        "IMPORTANT: support must be a bare numeric value from 0 to 1 with NO angle brackets.",
        "Correct: support=0.70    Incorrect: support=<0.70>",
        "3) ...",
        "4) ...",
        "5) ...",
        "Final Answer: <one of the candidates>",
        "Support Score: <support score of the Final Answer>",
    ])
    return "\n".join(lines)


def _build_compact_soft_user_prompt_from_json(
    templates,
    shared_templates_for_agent,
    subunit_consensus_summary,
    shared_template_summary,
    constraint_summary,
    subunit_lengths,
    target_info=None,
    partial_scaffold=None,
    scaffold_completion_candidates=None,
    scaffold_copy_options=None,
    symmetry_info=None,
):
    """
    Compact StoiAgent Stage-1 prompt.

    Design goal: L1 + structural-template evidence, not a second rule engine.
    Derived consensus/scaffold/symmetry candidates are intentionally NOT exposed
    as required anchors. Template observations are advisory evidence only.
    """
    subunit_ids = sort_subunits(list(templates.keys()))
    native_candidate = simplest_stoich(subunit_ids)
    if len(subunit_ids) == 1 and native_candidate == f"{subunit_ids[0]}1":
        native_candidate = f"{subunit_ids[0]}2"

    lines = ["Predict the biological stoichiometry of this protein complex."]

    target_block = format_casp17_target_info_block(target_info)
    if target_block:
        lines.extend(["", target_block])

    lines.extend([
        "",
        "1. Compact structural-template evidence (ADVISORY, not constraints):",
        "- Template evidence is homolog/structure evidence and may not transfer exactly to the target biological assembly.",
        "- Do NOT combine copy numbers from unrelated per-subunit templates into a full-complex stoichiometry automatically.",
        "- Repeated summaries or multiple records from the same PDB are not independent evidence.",
        "- A shared template covering multiple target subunits is usually more directly informative than an isolated subunit homolog, but neither is mandatory.",
        "- Target-specific external database/literature evidence, when retrieved, may confirm or contradict template evidence.",
    ])

    # One concise summary per subunit.  The count summary and representative
    # hits are explicitly marked as the same underlying evidence so the model
    # should not double-count them.
    for su in subunit_ids:
        hits = list(templates.get(su, []) or [])
        dist = _copy_distribution(hits)
        if dist:
            dist_text = ", ".join(f"copy{n}:{count}" for n, count in sorted(dist.items()))
        else:
            dist_text = "none"
        lines.append(f"- Subunit {su}: significant-template copy observations={dist_text} (summary of the same template hits below; not independent evidence).")

        # Keep at most three representatives, preferring the existing ranking.
        reps = []
        for hit in hits:
            ev = _safe_float(hit.get("evalue"), 1e99)
            if ev is not None and ev <= 1.0:
                reps.append(hit)
            if len(reps) >= 3:
                break
        if len(reps) < 3:
            for hit in hits:
                if hit not in reps:
                    reps.append(hit)
                if len(reps) >= 3:
                    break
        if reps:
            for hit in reps:
                lines.append("  - " + _compact_template_hit(hit))
        else:
            lines.append("  - no template evidence")

    lines.extend(["", "2. Shared/multi-subunit template evidence (ADVISORY):"])
    shared_lines = _compact_shared_template_lines(
        shared_templates_for_agent,
        shared_template_summary,
        max_items=5,
    )
    if shared_lines:
        lines.extend(shared_lines)
    else:
        lines.append("- no shared multi-subunit template evidence")

    lines.extend([
        "",
        "3. Optional external research (same tool interface and budgets as S1_external_search):",
        "- Use external research selectively when it can resolve biological assembly, oligomerization, conflicting evidence, or target-specific identity.",
        "- Available Actions:",
        '  Action: RCSB.search("text query")',
        '  Action: RCSB.searchSequence("A")',
        '  Action: RCSB.getPDB("1ABC")',
        '  Action: UniProt.search("text query")',
        '  Action: UniProt.get("P12345")',
        '  Action: PubMed.search("text query")',
        '  Action: PubMed.readAbstract("PMID")',
        "- Do not invent Observation lines.",
        "",
        "Stage-1 output policy (matched to L0/L1):",
        "- Propose at least 5 COMPLETE and UNIQUE stoichiometry candidates.",
        f"- Include the simple baseline candidate {native_candidate} so Stage 2 can compare the same baseline across variants.",
        "- Every candidate must contain every target subunit exactly once in canonical subunit order, with a positive integer copy number.",
        "- For a single-subunit complex, X1 is invalid.",
        "- Give each candidate a support score from 0 to 1 based on the evidence available in this condition.",
        "- Template-derived patterns, scaffold completions, and symmetry patterns are suggestions only; include them only if you judge them biologically plausible.",
        "- The Final Answer should be your best Stage-1 candidate; downstream Stage 2 will re-rank the candidate pool using AlphaFold3 ranking scores.",
        "",
        "Required final format:",
        "Thought: <1-3 short sentences>",
        "Candidates:",
        "1) <complete stoichiometry>  support=<0..1>  support_evidence=<short phrase>",
        "2) <complete stoichiometry>  support=<0..1>  support_evidence=<short phrase>",
        "3) ...",
        "4) ...",
        "5) ...",
        "Final Answer: <one of the candidates>",
        "Support Score: <support score of the Final Answer>",
    ])
    return "\n".join(lines)


def _build_shared_template_system_message_from_json(templates):
    subunit_ids = sort_subunits(list(templates.keys()))
    return {
        "role": "system",
        "content": (
            "You are an AI research agent that predicts protein-complex stoichiometry.\n"
            f"Target subunits: {', '.join(subunit_ids)}.\n"
            "Reason from the original target sequence/description, advisory shared/multi-subunit structural-template evidence, "
            "and selectively requested RCSB/UniProt/PubMed evidence.\n"
            "No isolated per-subunit template-copy evidence is provided in this condition. "
            "Shared-template evidence is not a hard constraint and must not override stronger target-specific evidence merely because it is present.\n"
            "Do not double-count multiple fields from the same PDB as independent evidence.\n"
            "Follow the common Stage-1 output format and never invent Observation lines."
        )
    }


def _build_compact_soft_system_message_from_json(templates):
    subunit_ids = sort_subunits(list(templates.keys()))
    return {
        "role": "system",
        "content": (
            "You are an AI research agent that predicts protein-complex stoichiometry.\n"
            f"Target subunits: {', '.join(subunit_ids)}.\n"
            "Reason from the original target sequence/description, compact advisory structural-template evidence, "
            "and selectively requested RCSB/UniProt/PubMed evidence.\n"
            "Template evidence is not a hard constraint and must not override stronger target-specific evidence merely because it is present.\n"
            "Do not double-count derived summaries and their source template records as independent evidence.\n"
            "Follow the common Stage-1 output format and never invent Observation lines."
        )
    }


def build_phase1_user_prompt_from_json(
    templates,
    shared_templates_for_agent,
    subunit_consensus_summary,
    shared_template_summary,
    constraint_summary,
    subunit_lengths,
    target_info=None,
    partial_scaffold=None,
    scaffold_completion_candidates=None,
    scaffold_copy_options=None,
    symmetry_info=None,
    template_evidence=None,
):
    """Use the shared-only prompt or the supplied v8 compact/soft fallback."""
    if template_evidence is None:
        template_evidence = select_structural_template_evidence(
            templates, shared_templates_for_agent, shared_template_summary,
        )
    use_shared = template_evidence["mode"] == "shared_templates"
    builder = (
        _build_shared_template_user_prompt_from_json if use_shared
        else _build_compact_soft_user_prompt_from_json
    )
    # No qualifying shared record exists on the fallback path. Do not pass
    # single-subunit or unrelated records to v8's permissive shared formatter.
    return builder(
        templates=templates,
        shared_templates_for_agent=shared_templates_for_agent if use_shared else [],
        subunit_consensus_summary=subunit_consensus_summary,
        shared_template_summary=shared_template_summary if use_shared else {},
        constraint_summary=constraint_summary,
        subunit_lengths=subunit_lengths,
        target_info=target_info,
        partial_scaffold=partial_scaffold,
        scaffold_completion_candidates=scaffold_completion_candidates,
        scaffold_copy_options=scaffold_copy_options,
        symmetry_info=symmetry_info,
    )


def build_system_message_from_json(templates, template_evidence=None):
    """Pair each selected user prompt with its original system prompt."""
    if template_evidence is not None and template_evidence["mode"] == "shared_templates":
        return _build_shared_template_system_message_from_json(templates)
    return _build_compact_soft_system_message_from_json(templates)


# =========================================================
# Agent loop
# =========================================================
def run_agent(
    llm,
    messages,
    templates,
    shared_templates_for_agent,
    dc,
    read_template=True,
    max_turns=12,
    finalization_turns=4,
    allowed_final_answers=None,
    required_candidates=None,
    partial_scaffold=None,
    scaffold_completion_candidates=None,
    symmetry_preserving_candidates=None,
    min_completion_candidates=0,
    min_total_candidates=0,
):
    subunit_ids = sort_subunits(list(templates.keys()))
    trace = []
    seen_actions = set()
    final_answer = None
    accepted = {}  # validated stoich -> support; retained across repair turns
    accepted_evidence = {}  # validated stoich -> original model support_evidence

    single_subunit_symbol = subunit_ids[0] if len(subunit_ids) == 1 else None
    role = "user"
    scaffold_completion_candidates = scaffold_completion_candidates or []
    symmetry_preserving_candidates = symmetry_preserving_candidates or []
    required_candidates = required_candidates or []

    turn_count = 1
    research_tools = None
    if L2_TARGET_INFO is not None:
        research_tools = ResearchToolExecutor(
            dc,
            L2_TARGET_INFO,
            information_cutoff=L2_INFORMATION_CUTOFF,
            budgets=L2_RESEARCH_BUDGETS,
        )

    max_turns = int(max_turns)
    finalization_turns = max(1, min(int(finalization_turns), max_turns))
    research_last_turn = max(0, max_turns - finalization_turns)
    finalization_notice_added = False

    def append_safe_message(msg_list, role_, content):
        if role_ == "system" and msg_list:
            role_ = "user"
        if role_ == "user" and msg_list and msg_list[-1]["role"] == "user":
            msg_list[-1]["content"] += f"\n\n{content}"
        else:
            msg_list.append({"role": role_, "content": content})

    def accepted_text():
        if not accepted:
            return "(none yet)"
        return ", ".join(f"{s} (support={accepted[s]:.3f})" for s in accepted)

    def merge_candidates(candidates, confidences, support_evidence):
        problems = []
        seen_local = set()
        for i, stoich in enumerate(candidates, start=1):
            if stoich in seen_local:
                problems.append(f"{i}) {stoich}: duplicate within this response")
                continue
            seen_local.add(stoich)
            valid, msg = validate_stoichiometry(stoich, subunit_ids)
            if not valid:
                problems.append(f"{i}) {stoich}: {msg}")
                continue
            if stoich not in confidences:
                problems.append(f"{i}) {stoich}: missing explicit numeric support, e.g. support=0.70")
                continue
            accepted[stoich] = float(confidences[stoich])
            reason = str(support_evidence.get(stoich, "")).strip()
            if reason:
                accepted_evidence[stoich] = reason
            elif stoich not in accepted_evidence:
                accepted_evidence[stoich] = "No support_evidence provided in model output."
        return problems

    for turn in range(turn_count, max_turns + 1):
        in_finalization = turn > research_last_turn
        if in_finalization and not finalization_notice_added:
            append_safe_message(
                messages, role,
                "RESEARCH PHASE CLOSED. Do not request additional Actions. Finish or "
                "repair the Stage-1 candidate pool. Valid candidates already accepted "
                "from earlier attempts remain accepted; replace only invalid/missing rows."
            )
            finalization_notice_added = True

        max_new = compute_generation_budget(messages, llm)
        raw_reply = llm.generate(messages, max_new_tokens=max_new)
        reply = filter_llm_reply(raw_reply)

        tmp_seen_actions = set(seen_actions)
        parsed_final, extra_actions, tmp_seen_actions = validate_and_clean(reply, tmp_seen_actions)
        if parsed_final:
            try:
                parsed_final = normalize_stoich(parsed_final)
            except ValueError:
                parsed_final = None

        candidates = parse_candidate_stoichiometries(reply)
        confidences = parse_confidences(reply)
        support_evidence = parse_candidate_support_evidence(reply)

        # Handle research actions first. During reserved finalization turns they
        # are blocked so repair attempts cannot be consumed by more searching.
        if extra_actions:
            append_safe_message(messages, "assistant", reply)
            if in_finalization:
                msg = (
                    "Research is closed. Do not request more Actions. "
                    f"Already accepted candidates: {accepted_text()}. Finish/repair the candidate pool."
                )
                append_safe_message(messages, role, msg)
                trace.append({
                    "turn": turn, "phase": "finalization", "validation_error": msg,
                    "accepted_candidates": dict(accepted), "messages": copy.deepcopy(messages)
                })
                continue

            seen_actions = tmp_seen_actions
            reply_lines = []
            for full_action, tool, arg in extra_actions:
                if research_tools is None:
                    obs = "Research tools are unavailable because target information was not initialized."
                elif arg == "N/A" or len(arg) == 0:
                    obs = "Action argument is empty."
                else:
                    obs = research_tools.execute(tool, arg)
                reply_lines.append(f"{full_action}\nObservation: {obs}")
            if reply_lines:
                append_safe_message(messages, "user", "\n\n".join(reply_lines))
            if turn == research_last_turn:
                append_safe_message(
                    messages, role,
                    "Research turns are now exhausted. On the next turn do not request Actions; "
                    "finish the Stage-1 candidate pool."
                )
                finalization_notice_added = True
            print(f"[research] tool usage: {research_tools.usage_dict() if research_tools else {}}")
            trace.append({
                "turn": turn, "phase": "research", "accepted_candidates": dict(accepted),
                "messages": copy.deepcopy(messages)
            })
            continue

        row_problems = []
        if candidates:
            row_problems = merge_candidates(candidates, confidences, support_evidence)
        elif has_candidate_header(reply):
            row_problems.append(
                "Candidates section could not be parsed; use numbered candidate rows with numeric support, e.g. support=0.70 (no angle brackets)."
            )

        # Validate requirements against the cumulative accepted pool, not only
        # the latest response. This allows one-row replacement after one bad row.
        unresolved = []
        if min_total_candidates > 0 and len(accepted) < min_total_candidates:
            unresolved.append(
                f"need {min_total_candidates - len(accepted)} more valid unique candidate(s)"
            )
        missing_required = [c for c in required_candidates if c not in accepted]
        if missing_required:
            unresolved.append(f"missing required baseline candidate(s) {missing_required}")

        if scaffold_completion_candidates and min_completion_candidates > 0:
            present = [c for c in accepted if c in set(scaffold_completion_candidates)]
            if len(present) < min_completion_candidates:
                unresolved.append(
                    f"need {min_completion_candidates - len(present)} more scaffold-completion candidate(s); "
                    f"examples: {', '.join(scaffold_completion_candidates[:6])}"
                )

        if symmetry_preserving_candidates:
            present_sym = [c for c in accepted if c in set(symmetry_preserving_candidates)]
            if not present_sym:
                unresolved.append(
                    "need at least one symmetry-preserving completion; examples: "
                    + ", ".join(symmetry_preserving_candidates[:4])
                )

        if parsed_final:
            valid, reason = validate_stoichiometry(parsed_final, subunit_ids)
            if not valid:
                unresolved.append(f"Final Answer {parsed_final} is invalid: {reason}")
            elif parsed_final not in accepted:
                unresolved.append(f"Final Answer {parsed_final} is not in the validated candidate pool")
            elif allowed_final_answers is not None and parsed_final not in allowed_final_answers:
                unresolved.append(
                    "Final Answer must be one of the AF3-allowed stoichiometries: "
                    + ", ".join(sorted(allowed_final_answers))
                )

        if unresolved:
            append_safe_message(messages, "assistant", reply)
            bad = (" Rejected rows only: " + " | ".join(row_problems)) if row_problems else ""
            repair = (
                f"Accepted and retained candidates: {accepted_text()}. "
                + "; ".join(unresolved)
                + ". Keep the accepted candidates. Replace/add only the missing or invalid row(s) "
                  "with numbered candidate lines and support scores."
                + bad
            )
            append_safe_message(messages, role, repair)
            trace.append({
                "turn": turn,
                "phase": "finalization" if in_finalization else "research",
                "validation_error": repair,
                "accepted_candidates": dict(accepted),
                "rejected_candidate_problems": row_problems,
                "messages": copy.deepcopy(messages),
            })
            final_answer = None
            continue

        # If all constraints are satisfied, invalid extra rows are ignored rather
        # than causing the entire valid list to be discarded.
        if parsed_final is None:
            append_safe_message(messages, "assistant", reply)
            append_safe_message(
                messages, role,
                f"The validated candidate pool is complete: {accepted_text()}. "
                "Choose one accepted candidate as Final Answer and give its Support Score."
            )
            trace.append({
                "turn": turn, "phase": "finalization" if in_finalization else "research",
                "accepted_candidates": dict(accepted),
                "rejected_candidate_problems": row_problems,
                "messages": copy.deepcopy(messages),
            })
            continue

        final_answer = parsed_final

        # Canonical merged final response so downstream Stage 2 sees the entire
        # repaired candidate pool even when the last model turn supplied only one replacement row.
        lines = ["Candidates:"]
        for i, (sto, score) in enumerate(accepted.items(), 1):
            reason = accepted_evidence.get(sto, "No support_evidence provided in model output.")
            lines.append(
                f"{i}) {sto}  support={score:.6f}  support_evidence={reason}"
            )
        lines.append(f"Final Answer: {final_answer}")
        lines.append(f"Support Score: {accepted[final_answer]:.6f}")
        merged_reply = "\n".join(lines)
        append_safe_message(messages, "assistant", merged_reply)
        trace.append({
            "turn": turn,
            "phase": "finalization" if in_finalization else "research",
            "accepted_candidates": dict(accepted),
            "rejected_candidate_problems": row_problems,
            "messages": copy.deepcopy(messages),
        })
        break

    if not final_answer:
        # Do not silently fabricate a replacement candidate list. Preserve any
        # validated partial pool for diagnostics and mark the sample incomplete.
        append_safe_message(
            messages, "assistant",
            "Final Answer: INCOMPLETE_CANDIDATE_REPAIR"
        )
        trace.append({
            "turn": max_turns + 1,
            "phase": "incomplete",
            "accepted_candidates": dict(accepted),
            "messages": copy.deepcopy(messages),
        })

    print(f"[research] final tool usage: {research_tools.usage_dict() if research_tools else {}}")
    return messages, trace, final_answer


# =========================================================
# Top-k runner
# =========================================================
def run_agent_topk(
    llm,
    messages,
    templates,
    shared_templates_for_agent,
    dc,
    temperature,
    sample_times,
    sample_seeds,
    outdir,
    base_name,
    subunit_lengths,
    *,
    phase1_cache_dir: str = None,
    force_rerun_phase1: bool = False,
    shared_template_summary: dict = None,
    required_baseline_candidates=None,
    partial_scaffold=None,
    scaffold_completion_candidates=None,
    symmetry_preserving_candidates=None,
    min_completion_candidates=0,
    min_total_candidates=0,
    max_turns_p1=12,
    finalization_turns_p1=4,
):
    all_samples = []

    for i, seed in enumerate(sample_seeds[:sample_times], 1):
        sample_messages = copy.deepcopy(messages)

        print(f"\n🧠 Running sample {i}/{sample_times} (seed={seed})")

        combined_p1_glob = run_agent_single(
            llm, sample_messages, templates, shared_templates_for_agent, dc,
            temperature=temperature,
            outdir=outdir,
            base_name=base_name,
            subunit_lengths=subunit_lengths,
            seed=seed,
            sample_id=i,
            phase1_cache_dir=phase1_cache_dir,
            force_rerun_phase1=force_rerun_phase1,
            shared_template_summary=shared_template_summary,
            required_baseline_candidates=required_baseline_candidates,
            partial_scaffold=partial_scaffold,
            scaffold_completion_candidates=scaffold_completion_candidates,
            symmetry_preserving_candidates=symmetry_preserving_candidates,
            min_completion_candidates=min_completion_candidates,
            min_total_candidates=min_total_candidates,
            max_turns_p1=max_turns_p1,
            finalization_turns_p1=finalization_turns_p1,
        )

        all_samples.append({
            "sample_id": i,
            "phase1": {
                "stoich": list(combined_p1_glob.keys()),
                "conf": list(combined_p1_glob.values())
            },
        })

    return all_samples


def run_agent_single(
    llm,
    messages,
    templates,
    shared_templates_for_agent,
    dc,
    *,
    temperature: float,
    outdir: str = None,
    base_name: str = "run",
    subunit_lengths: dict = None,
    seed: int = 1234,
    sample_id: int = 1,
    phase1_cache_dir: str = None,
    force_rerun_phase1: bool = False,
    max_turns_p1: int = 12,
    finalization_turns_p1: int = 4,
    shared_template_summary: dict = None,
    required_baseline_candidates=None,
    partial_scaffold=None,
    scaffold_completion_candidates=None,
    symmetry_preserving_candidates=None,
    min_completion_candidates=0,
    min_total_candidates=0,
):
    def save_json_gz(path: str, obj: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    def load_json_gz(path: str) -> dict:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    def phase1_cache_path(cache_dir: str, base_name_: str, sample_id_: int) -> str:
        return os.path.join(cache_dir, EXPERIMENT_VARIANT, base_name_, f"sample{sample_id_:02d}.phase1.json.gz")

    if subunit_lengths is None:
        subunit_lengths = {}

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    if phase1_cache_dir is None:
        phase1_cache_dir = os.path.join(outdir or ".", "phase1_cache")

    cache_path = phase1_cache_path(phase1_cache_dir, base_name, sample_id)

    reseed(seed)

    orig_generate = llm.generate

    def stochastic_generate(msgs_, max_new_tokens=1024):
        return orig_generate(
            msgs_,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    llm.generate = stochastic_generate

    p1_conf_dict = {}
    meta = {"phase1_final": None, "phase1_loaded_from_cache": False}

    msgs_after_p1 = None
    last_reply_p1 = None
    answer1 = None
    phase1_candidates = []

    valid_subunits = set(sort_subunits(list(templates.keys())))
    template_evidence_mode = select_structural_template_evidence(
        templates, shared_templates_for_agent, shared_template_summary,
    )["mode"]
    single_subunit_symbol = None
    if len(valid_subunits) == 1:
        single_subunit_symbol = list(valid_subunits)[0]

    try:
        can_load_cache = (not force_rerun_phase1) and os.path.exists(cache_path)
        if can_load_cache:
            cached = load_json_gz(cache_path)
            msgs_after_p1 = sanitize_chat_messages(cached["msgs_after_p1"])
            last_reply_p1 = cached["last_reply_p1"]
            answer1 = cached["answer1"]
            p1_conf_dict = {k: float(v) for k, v in cached.get("p1_conf_dict", {}).items()}
            phase1_candidates = cached.get("phase1_candidates", [])
            meta["phase1_final"] = answer1
            meta["phase1_loaded_from_cache"] = True
        else:
            meta["phase1_loaded_from_cache"] = False
            messages_p1 = copy.deepcopy(messages)

            # The base prompt already contains the selected shared or fallback
            # complete compact template evidence.  Do not append duplicate
            # shared-template summaries, consensus anchors, scaffold priors, or
            # other derived observations here.
            messages_p1 = sanitize_chat_messages(messages_p1)

            msgs1, trace1, answer1 = run_agent(
                llm, messages_p1, templates, shared_templates_for_agent, dc,
                read_template=False,
                max_turns=max_turns_p1,
                finalization_turns=finalization_turns_p1,
                required_candidates=required_baseline_candidates,
                partial_scaffold=None,
                scaffold_completion_candidates=None,
                symmetry_preserving_candidates=None,
                min_completion_candidates=0,
                min_total_candidates=5,
            )

            meta["phase1_final"] = answer1
            msgs_after_p1 = msgs1

            last_reply_p1 = trace1[-1]["messages"][-1]["content"]
            raw_p1 = parse_confidences(last_reply_p1)
            phase1_candidates = parse_candidate_stoichiometries(last_reply_p1)

            p1_conf_dict = {}
            for s, c in raw_p1.items():
                if set(re.findall(r"[A-Z]{1,2}", s)).issubset(valid_subunits) and not (
                    len(valid_subunits) == 1 and s.strip() == f"{single_subunit_symbol}1"
                ):
                    p1_conf_dict[s] = float(c)

            save_json_gz(cache_path, {
                "experiment_variant": EXPERIMENT_VARIANT,
                "template_evidence_mode": template_evidence_mode,
                "model_key": getattr(llm, "model_key", None),
                "temperature": float(temperature),
                "seed": int(seed),
                "sample_id": int(sample_id),
                "answer1": answer1,
                "last_reply_p1": last_reply_p1,
                "p1_conf_dict": p1_conf_dict,
                "phase1_candidates": phase1_candidates,
                "msgs_after_p1": msgs_after_p1,
            })
            print(f"💾 Saved Phase 1 cache → {cache_path}")

            if outdir is not None:
                tr1_path = os.path.join(outdir, f"{base_name}_sample{sample_id:02d}_phase1.trace")
                with open(tr1_path, "w", encoding="utf-8") as f:
                    for step in trace1:
                        f.write(f"\n--- Turn {step['turn']} ---\n")
                        for m in step["messages"]:
                            f.write(f"{m['role'].upper()}: {m['content']}\n\n")

    finally:
        llm.generate = orig_generate

    return p1_conf_dict


# def parse_casp17_sequence_catalog(text: str) -> dict:
#     """Parse CASP sequence records with automatic target ID matching."""
#     catalog = defaultdict(list)

#     current_header = None
#     seq_parts = []
#     occurrence_by_target = defaultdict(int)

#     def flush():
#         nonlocal current_header, seq_parts
#         if current_header is None:
#             return

#         raw = current_header[1:] if current_header.startswith(">") else current_header

#         target_match = re.match(
#             r"\s*([HT]\d{4})(?:o(?=\b))?",
#             raw,
#             flags=re.IGNORECASE,
#         )

#         if not target_match:
#             current_header = None
#             seq_parts = []
#             return

#         original_target_id = target_match.group(1).upper()

#         # Automatically normalize CASP16 naming conventions
#         target_id = original_target_id

#         # H1xxx -> H0xxx (CASP16)
#         if re.fullmatch(r"H1[0-2]\d{2}", target_id):
#             target_id = "H0" + target_id[2:]

#         # T0xxx -> T0xxxo (CASP16)
#         elif re.fullmatch(r"T0\d{3}", target_id):
#             target_id += "o"

#         occurrence_by_target[target_id] += 1

#         rec = _parse_casp17_header(
#             current_header,
#             occurrence_index=occurrence_by_target[target_id],
#         )

#         sequence = re.sub(r"\s+", "", "".join(seq_parts)).upper()
#         if not sequence:
#             raise ValueError(
#                 f"Empty sequence for CASP header: {current_header}"
#             )

#         rec["sequence"] = sequence
#         rec["sequence_length"] = len(sequence)
#         rec["length_matches_header"] = (
#             len(sequence) == rec["declared_length"]
#         )

#         catalog[target_id].append(rec)

#         current_header = None
#         seq_parts = []

#     for raw_line in text.splitlines():
#         line = raw_line.strip()
#         if not line:
#             continue

#         if line.startswith(">"):
#             flush()
#             current_header = line
#             seq_parts = []
#         else:
#             if current_header is None:
#                 raise ValueError(
#                     f"Sequence line encountered before a CASP header: {line[:60]}"
#                 )
#             seq_parts.append(line)

#     flush()
#     return dict(catalog)


# def resolve_casp17_sequence_target(target_id: str, catalog: dict) -> str:
#     """
#     Resolve the sequence-release record for a benchmark target.

#     Phase-2 CASP IDs used by StoiBench (e.g. H2346/T2342) may not have a
#     separate sequence entry because their original Phase-1 sequence entry is
#     H1346/T1342. If exact lookup fails, try that deterministic 2xxx -> 1xxx
#     alias.
#     """
#     target_id = str(target_id).upper()
#     if target_id in catalog:
#         return target_id

#     m = re.fullmatch(r"([HT])2(\d{3})", target_id)
#     if m:
#         phase1_id = f"{m.group(1)}1{m.group(2)}"
#         if phase1_id in catalog:
#             print(
#                 f"[CASP17] sequence alias: {target_id} -> {phase1_id} "
#                 "(using original Phase-1 sequence release)"
#             )
#             return phase1_id

#     raise KeyError(
#         f"No CASP17 sequence information found for target {target_id}. "
#         "Exact lookup and the 2xxx -> 1xxx Phase-1 alias both failed."
#     )


def build_casp17_target_info(
    target_id: str,
    subunit_ids,
    catalog: dict,
) -> dict:
    """
    Map the ordered CASP17 sequence records onto StoiBench canonical subunits.

    The mapping is intentionally by source-file record order rather than by
    the literal CASP 'subunit N' integer. This handles records such as a target
    whose only protein sequence is labeled 'subunit 2' because subunit 1 is a
    non-protein component; the sole protein sequence should still map to
    StoiBench protein subunit A.
    """
    subunit_ids = sort_subunits(list(subunit_ids))
    source_target_id = resolve_casp17_sequence_target(target_id, catalog)
    records = list(catalog[source_target_id])

    if len(records) != len(subunit_ids):
        details = [
            (
                r.get("source_subunit_number"),
                r.get("declared_length"),
                r.get("description"),
            )
            for r in records
        ]
        raise RuntimeError(
            "CASP17 sequence-record count does not match the StoiBench "
            f"distinct-subunit count for {target_id}: "
            f"CASP records={len(records)}, StoiBench subunits={len(subunit_ids)} "
            f"{subunit_ids}. CASP records={details}. "
            "Refusing to guess a sequence-to-subunit mapping."
        )

    mapped = {}
    for su, rec in zip(subunit_ids, records):
        mapped[su] = dict(rec)

    # Most records for a multi-subunit target repeat the same target-level
    # description/source. Retain the first non-empty value.
    description = next(
        (r.get("description", "") for r in records if r.get("description", "").strip()),
        "",
    )
    organism = next(
        (r.get("organism", "") for r in records if r.get("organism", "").strip()),
        "",
    )

    return {
        "target_id": str(target_id).upper(),
        "sequence_source_target_id": source_target_id,
        "description": description,
        "organism": organism,
        "subunits": mapped,
    }


def load_json_bundle_from_file(file_path: str, casp17_sequence_catalog=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    templates = data["templates"]
    raw_shared_templates_for_agent = data.get("shared_templates_for_agent", [])
    shared_templates_for_agent = [
        rec for rec in raw_shared_templates_for_agent
        if len(_shared_covered_subunits(rec)) >= 2
    ]
    subunit_consensus_summary = data.get("subunit_consensus_summary", {})
    shared_template_summary = data.get("shared_template_summary", {})
    constraint_summary = data.get("constraint_summary", {})

    # C inferred approximate sequence lengths from aligned_cols/coverage.
    # D uses the exact original CASP17 sequence release whenever available.
    inferred_subunit_lengths = infer_subunit_lengths_from_json(templates)

    target_id = extract_casp17_target_id(file_path)
    if casp17_sequence_catalog is None:
        raise RuntimeError(
            "D requires the CASP17 sequence catalog. "
            "Load it with load_casp17_sequence_catalog() and pass it to "
            "load_json_bundle_from_file()."
        )

    target_info = build_casp17_target_info(
        target_id=target_id,
        subunit_ids=templates.keys(),
        catalog=casp17_sequence_catalog,
    )

    exact_subunit_lengths = {
        su: int(rec["sequence_length"])
        for su, rec in target_info["subunits"].items()
    }
    subunit_lengths = dict(inferred_subunit_lengths)
    subunit_lengths.update(exact_subunit_lengths)

    print(
        f"[CASP17] {target_id}: description={target_info.get('description')!r}; "
        f"organism/source={target_info.get('organism')!r}"
    )
    print(
        "[CASP17] exact subunit lengths: "
        + ", ".join(
            f"{su}={subunit_lengths[su]}"
            for su in sort_subunits(subunit_lengths.keys())
        )
    )

    native_baseline = simplest_stoich(sort_subunits(list(templates.keys())))
    if len(templates) == 1 and native_baseline == f"{sort_subunits(list(templates.keys()))[0]}1":
        native_baseline = f"{sort_subunits(list(templates.keys()))[0]}2"
    # Match L0/L1 exactly: only the simple baseline is required.
    # Template consensus/scaffold/symmetry candidates are advisory, never mandatory.
    required_baseline_candidates = [native_baseline]

    # Select direct shared evidence when present, otherwise compact subunit
    # evidence. Do not inject consensus, scaffold, or symmetry candidate anchors;
    # the only mandatory candidate remains the matched L0/L1 simple baseline.
    template_evidence = select_structural_template_evidence(
        templates, shared_templates_for_agent, shared_template_summary,
    )
    print(f"[template evidence] mode={template_evidence['mode']}")
    partial_scaffold = None
    scaffold_completion_candidates = []
    scaffold_copy_options = {}
    symmetry_info = {}

    min_total_candidates = 5
    min_completion_candidates = 0

    sys_msg = build_system_message_from_json(templates, template_evidence=template_evidence)
    user_msg = {
        "role": "user",
        "content": build_phase1_user_prompt_from_json(
            templates=templates,
            shared_templates_for_agent=shared_templates_for_agent,
            subunit_consensus_summary=subunit_consensus_summary,
            shared_template_summary=shared_template_summary,
            constraint_summary=constraint_summary,
            subunit_lengths=subunit_lengths,
            target_info=target_info,
            partial_scaffold=partial_scaffold,
            scaffold_completion_candidates=scaffold_completion_candidates,
            scaffold_copy_options=scaffold_copy_options,
            symmetry_info=symmetry_info,
            template_evidence=template_evidence,
        )
    }

    return {
        "messages": [sys_msg, user_msg],
        "templates": templates,
        "shared_templates_for_agent": shared_templates_for_agent,
        "template_evidence_mode": template_evidence["mode"],
        "subunit_lengths": subunit_lengths,
        "target_info": target_info,
        "casp17_target_id": target_id,
        "casp17_sequence_source_target_id": target_info.get("sequence_source_target_id"),
        "shared_template_summary": shared_template_summary,
        "subunit_consensus_summary": subunit_consensus_summary,
        "constraint_summary": constraint_summary,
        "required_baseline_candidates": required_baseline_candidates,
        "partial_scaffold": partial_scaffold,
        "scaffold_completion_candidates": scaffold_completion_candidates,
        "scaffold_copy_options": scaffold_copy_options,
        "symmetry_info": symmetry_info,
        "symmetry_preserving_candidates": symmetry_info.get("symmetry_preserving_candidates", []),
        "min_completion_candidates": min_completion_candidates,
        "min_total_candidates": min_total_candidates,
    }


# =========================================================
# Main
# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_DESCRIPTION)
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default="Qwen3.6-27B", choices=MODEL_MAP.keys())
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max_abstract_reads", type=int, default=8, help="Maximum PubMed abstract reads per stochastic sample.")
    parser.add_argument("--max_turns", type=int, default=12, help="Maximum Stage-1 turns per stochastic sample (matched to S1_external_search).")
    parser.add_argument("--finalization_turns", type=int, default=4, help="Last N Stage-1 turns reserved for candidate generation/repair (matched to S1_external_search).")
    parser.add_argument("--max_pdb_searches", type=int, default=4)
    parser.add_argument("--max_pdb_reads", type=int, default=6)
    parser.add_argument("--max_uniprot_searches", type=int, default=4)
    parser.add_argument("--max_uniprot_reads", type=int, default=6)
    parser.add_argument("--max_pubmed_searches", type=int, default=3)
    parser.add_argument(
        "--information_cutoff",
        default="2026-04-26",
        help=(
            "Conservative cutoff for RCSB release dates and PubMed publication dates. "
            "Default: 2026-04-26. Use '' to disable."
        ),
    )
    parser.add_argument(
        "--casp17_seq_file",
        default="casp17.HT1.seq.txt",
        help=(
            "Local cache/path for the original CASP17 HT1 sequence release. "
            "If missing, D downloads it once from the official Prediction Center URL. "
            "Default: casp17.HT1.seq.txt"
        ),
    )
    parser.add_argument(
        "--target-release-dates",
        default=None,
        help="CSV with target_id,release_date (YYYY-MM-DD); each target uses the previous day as cutoff.",
    )
    parser.add_argument(
        "--target-release-date",
        default=None,
        help="Release date YYYY-MM-DD of this single target; cutoff is previous day.",
    )

    parser.add_argument("--phase1_cache_dir", default=None)
    parser.add_argument("--force_rerun_phase1", action="store_true")
    parser.add_argument(
        "--summary_cache_dir",
        default=os.path.join("article_cache", SUMMARY_MODEL_KEY),
        help=("Persistent cache for summaries generated by the fixed Qwen3-4B summary agent. "
              "A cached summary is reused; if a requested summary is missing, Qwen3-4B generates "
              "and caches it on demand. Default: article_cache/Qwen3-4B"),
    )
    args = parser.parse_args()

    MAX_ABSTRACT_READS = max(0, int(args.max_abstract_reads))
    os.makedirs(args.outdir, exist_ok=True)

    # One fixed Qwen3-4B summary agent is used for every benchmark backbone.
    # Summaries are lazy: cache hits are reused; cache misses are generated by
    # Qwen3-4B only when that literature item is actually requested.
    FROZEN_SUMMARY_CACHE_DIR = os.path.abspath(args.summary_cache_dir)
    agent_common.FROZEN_SUMMARY_CACHE_DIR = FROZEN_SUMMARY_CACHE_DIR  # read by the shared summary cache
    os.makedirs(FROZEN_SUMMARY_CACHE_DIR, exist_ok=True)

    print(f"Literature summary agent: {SUMMARY_MODEL_KEY} (Qwen/Qwen3-4B-Instruct-2507)")
    print("Pre-CASP17 benchmark panel: " + ", ".join(BENCHMARK_MODEL_KEYS))
    print(f"Summary cache: {FROZEN_SUMMARY_CACHE_DIR}")

    # Load/cache the original CASP17 target descriptions and sequences once.
    casp17_sequence_catalog = load_casp17_sequence_catalog(args.casp17_seq_file)

    dc = DiscoveryCache(
        rcsb_cache="pdb_json_cache",
        uniprot_cache="uniprot_cache_test",
        pubmed_cache="pubmed_cache_test",
        article_cache=FROZEN_SUMMARY_CACHE_DIR
    )

    infile = args.json_file
    jsonfile = os.path.basename(infile)
    print(f"\n[ablation] {EXPERIMENT_VARIANT}: {EXPERIMENT_DESCRIPTION}")
    print(f"\n\n=== Processing {jsonfile} ===")

    bundle = load_json_bundle_from_file(
        infile,
        casp17_sequence_catalog=casp17_sequence_catalog,
    )

    L2_TARGET_INFO = bundle["target_info"]
    L2_INFORMATION_CUTOFF, TARGET_RELEASE_DATE, CUTOFF_SOURCE = resolve_target_information_cutoff(
        bundle["casp17_target_id"],
        target_release_dates=args.target_release_dates,
        target_release_date=args.target_release_date,
        default_cutoff=args.information_cutoff,
    )
    print(f"[temporal evidence] target={bundle['casp17_target_id']} "
          f"release={TARGET_RELEASE_DATE or 'NA'} "
          f"cutoff={L2_INFORMATION_CUTOFF or 'disabled'} source={CUTOFF_SOURCE}")
    L2_RESEARCH_BUDGETS = {
        "pdb_search": int(args.max_pdb_searches),
        "pdb_read": int(args.max_pdb_reads),
        "uniprot_search": int(args.max_uniprot_searches),
        "uniprot_read": int(args.max_uniprot_reads),
        "pubmed_search": int(args.max_pubmed_searches),
        "abstract_read": int(args.max_abstract_reads),
    }


    print(f"Information cutoff (PDB/PubMed): {L2_INFORMATION_CUTOFF or 'disabled'}")
    print(f"Matched research budgets: {L2_RESEARCH_BUDGETS}")
    print(
        "WARNING: UniProt REST/cache observations are current unless a historically "
        "frozen UniProt cache is supplied."
    )

    messages = bundle["messages"]
    templates = bundle["templates"]
    shared_templates_for_agent = bundle["shared_templates_for_agent"]

    # Load the benchmark backbone; external research is requested on demand.
    # This variant does not auto-read literature associated with templates. PubMed
    # remains available only through the same explicit research actions as L1.
    llm = HFLLM(args.model)

    subunit_lengths = bundle["subunit_lengths"]
    shared_template_summary = bundle["shared_template_summary"]
    required_baseline_candidates = bundle["required_baseline_candidates"]
    partial_scaffold = bundle.get("partial_scaffold")
    scaffold_completion_candidates = bundle.get("scaffold_completion_candidates", [])
    symmetry_preserving_candidates = bundle.get("symmetry_preserving_candidates", [])
    min_completion_candidates = bundle.get("min_completion_candidates", 0)
    min_total_candidates = bundle.get("min_total_candidates", 0)

    base_name = jsonfile.replace(".json", "")

    all_samples = run_agent_topk(
        llm,
        messages,
        templates,
        shared_templates_for_agent,
        dc,
        temperature=args.temperature,
        sample_times=15,
        sample_seeds=SEEDS_30,
        outdir=args.outdir,
        base_name=base_name,
        subunit_lengths=subunit_lengths,
        phase1_cache_dir=args.phase1_cache_dir,
        force_rerun_phase1=args.force_rerun_phase1,
        shared_template_summary=shared_template_summary,
        required_baseline_candidates=required_baseline_candidates,
        partial_scaffold=partial_scaffold,
        scaffold_completion_candidates=scaffold_completion_candidates,
        symmetry_preserving_candidates=symmetry_preserving_candidates,
        min_completion_candidates=min_completion_candidates,
        min_total_candidates=min_total_candidates,
        max_turns_p1=args.max_turns,
        finalization_turns_p1=args.finalization_turns,
    )

    for k in [15]: # [5, 10, 15]:
        subset = all_samples[:k]

        all_stoichs = set()
        for s in subset:
            phase_data = s["phase1"]
            all_stoichs.update(phase_data["stoich"])

        if not all_stoichs:
            outfile = os.path.join(args.outdir, f"{base_name}_phase1_top5_k{k}.out")
            with open(outfile, "w", encoding="utf-8") as f:
                f.write(f"=== Top-5 Predictions for {base_name} (phase1, first {k} samples) ===\n\n")
                f.write("RESULT: EMPTY_AFTER_FILTER\n")
                f.write("Reason: no stoichiometries survived post-filtering.\n")
                f.write("Top-5: (none)\n")
            print(f"⚠️ Empty-after-filter result written → {outfile}")

            outall = os.path.join(args.outdir, f"{base_name}_phase1_topall_k{k}.out")
            with open(outall, "w", encoding="utf-8") as f:
                f.write(f"=== All Predictions for {base_name} (phase1, first {k} samples) ===\n\n")
                f.write("RESULT: EMPTY_AFTER_FILTER\n")
                f.write("Reason: no stoichiometries survived post-filtering.\n")
                f.write("Candidates: (none)\n")

        # Global averaging across all k stochastic samples. If a candidate is
        # absent from one run, that run contributes confidence 0.0. This avoids
        # promoting one-off high-confidence candidates above consistently proposed
        # candidates.
        conf_dict = {stoich: [] for stoich in all_stoichs}
        observed_n = {stoich: 0 for stoich in all_stoichs}

        for s in subset:
            phase_data = s["phase1"]
            this_sample = dict(zip(phase_data["stoich"], phase_data["conf"]))
            for stoich in all_stoichs:
                if stoich in this_sample:
                    observed_n[stoich] += 1
                conf_dict[stoich].append(float(this_sample.get(stoich, 0.0)))

        avg_conf = {stoich: float(np.mean(vals)) for stoich, vals in conf_dict.items()}
        ranked = sorted(
            avg_conf.items(),
            key=lambda x: (-x[1], -observed_n[x[0]], x[0]),
        )

        outfile = os.path.join(args.outdir, f"{base_name}_phase1_top5_k{k}.out")
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f"=== Top-5 Predictions for {base_name} (phase1, first {k} samples) ===\n\n")
            for i, (s, c) in enumerate(ranked[:5], 1):
                f.write(f"{i}. {s:<12} avg_conf={c:.3f} n={observed_n[s]}/{k}\n")

        print(f"✅ Saved phase1 Top-5 (global average; absent=0) for first {k} samples → {outfile}")

        outall = os.path.join(args.outdir, f"{base_name}_phase1_topall_k{k}.out")
        with open(outall, "w", encoding="utf-8") as f:
            f.write(f"=== All Predictions for {base_name} (phase1, first {k} samples) ===\n\n")
            for i, (s, c) in enumerate(ranked, 1):
                f.write(f"{i}. {s:<12} avg_conf={c:.3f} n={observed_n[s]}/{k}\n")

        print(f"✅ Saved phase1 ALL candidates (global average) for first {k} samples → {outall}")
