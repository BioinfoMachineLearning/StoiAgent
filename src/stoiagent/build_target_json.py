#!/usr/bin/env python3
"""Build one Stage-1 target JSON from a multi-record target FASTA.

Jackhmmer, HHmake and HHsearch run automatically. HHR results are internal
intermediates; callers only provide the target FASTA and installed databases.
"""

import os
import re
import json
import math
import argparse
from collections import Counter

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner

from . import parsers
from .utils import sort_subunits, linkify_citation

# =========================================================
# Config
# =========================================================
MAX_TEMPLATE_COUNT = 30
MAX_TEMPLATE_SCAN = 500
MIN_SHARED_SUBUNITS = 2

# Subunit-consensus rules
SUBUNIT_SIG_EVALUE = 1.0
SUBUNIT_STRONG_MIN_FREQ = 0.60
SUBUNIT_ALLOWED_MIN_FREQ = 0.20

# Shared-template ranking rules
SHARED_STRONG_COV = 0.50
SHARED_STRONG_EVALUE = 1e-3

# =========================================================
# Chain / stoichiometry utilities
# =========================================================


def safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def safe_str(x, default=""):
    if x is None:
        return default
    try:
        return str(x)
    except Exception:
        return default


def safe_neg_log10_evalue(x, floor=1e-300):
    try:
        v = float(x)
        if v <= 0:
            v = floor
        return -math.log10(max(v, floor))
    except Exception:
        return 0.0

def normalize_by_max(value, max_value):
    if max_value <= 0:
        return 0.0
    return float(value) / float(max_value)

# =========================================================
# Positive-composition helpers
# =========================================================

def generate_positive_compositions(total, parts):
    """
    Generate all ordered positive integer compositions of `total`
    into `parts` positive parts.

    Example:
      total=3, parts=2 -> [[1,2], [2,1]]
    """
    if parts <= 0 or total < parts:
        return []
    if parts == 1:
        return [[total]]

    out = []
    for first in range(1, total - parts + 2):
        for rest in generate_positive_compositions(total - first, parts - 1):
            out.append([first] + rest)
    return out

# =========================================================
# Prompt helper
# =========================================================

def make_system_message(subunit_ids):
    return {
        "role": "system",
        "content": (
            "You are an AI agent that predicts protein stoichiometry.\n"
            f"This protein contains {len(subunit_ids)} distinct subunit(s): {', '.join(subunit_ids)}.\n"
            "You should use homolog templates and their primary citations (PubMed/DOI) to gather evidence.\n"
            "There are two evidence sources in the input:\n"
            "1) Per-subunit templates\n"
            "2) Shared templates observed across multiple subunits\n"
            "Treat the shared-template section as assembly-level evidence.\n"
            "Do not assume that every shared template must also rank highly within each subunit’s local top list.\n"
            "Important rule about templates:\n"
            "- A template is considered to exist in multiple subunits only if it is explicitly listed under more than one subunit in the user’s input.\n"
            "- Do not assume or infer that a template is present in multiple subunits unless it is literally shown under each subunit’s section or in the explicit shared-template section.\n"
            "- If the same PDB ID appears under multiple subunits, treat that as especially strong assembly-level evidence.\n"
            "- If multiple target subunits all align to the same repeated template chain, interpret the total template copy number as a total allocation across those target subunits rather than copying that number onto each target subunit.\n"
            "Workflow:\n"
            "- The environment will automatically generate initial Actions for all templates and fetch their PubMed/DOI abstracts as Observations.\n"
            "- Your main responsibility is to read the Observations and produce one line:\n"
            "Final Answer: <your stoichiometry prediction>\n"
            "When to do more research:\n"
            "- If Observations conflict, are missing copy numbers, or do not clearly imply a single stoichiometry, emit new Actions before giving the Final Answer.\n"
            "Allowed Actions:\n"
            "- Action: PubMed.readAbstract('<DOI_or_PubMed_URL>')\n"
            "- Action: PubMed.search('<Search query built from Template metadata: use only biological terms from the template metadata (e.g., protein names, UniProt IDs, keywords, or entities). Do NOT include raw subunit labels (e.g., A, B, C, D) in search queries. These are dataset identifiers, not biological terms. You can also use informative terms from Title, Keywords, or Entity (e.g., DegS serine protease PDZ stoichiometry)>')\n"
            "- Action: RCSB.getPDB('<PDB_ID>')\n"
            "- Action: UniProt.get('<UniProt_ID>')\n"
            "After each Action, wait for an Observation, then continue reasoning.\n"
            "Stop expanding once a single stoichiometry consistent with all subunits is supported.\n"
            "Important rules:\n"
            "- You may output 'Thought:' lines to explain your reasoning.\n"
            "- You may output 'Action: ...' lines to request tool calls.\n"
            "- You may output one 'Final Answer: ...' line when you are done.\n"
            "- You must NEVER include PubMed abstracts, summaries, or results yourself.\n"
            "- You must NEVER write lines starting with 'Observation:'.\n"
            "Observations will ONLY be provided by the environment after you request an Action.\n"
            "Important rules for Final Answer:\n"
            f"- The Final Answer must list the exact subunit IDs in this canonical order: {', '.join(subunit_ids)}\n"
            "- Always follow this order strictly and never reorder subunits (e.g., if subunits are A and B, use 'A1B2', not 'B2A1').\n"
            "- Do not use symbols like ':', '-', or words like 'dimer'/'trimer'.\n"
            "- Example valid answers: 'A1B1', 'A2', 'A2B2C4'.\n"
            "- Example invalid answers: 'B2A1', '1:1', 'heterodimer', '2 copies of A'.\n"
        )
    }

# =========================================================
# Sequence / entity helpers
# =========================================================

def compute_sequence_identity(seq1, seq2):
    if not seq1 or not seq2:
        return 0.0
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    return aligner.score(seq1, seq2) / min(len(seq1), len(seq2))

def extract_subunit_copies(stoich_string: str, subunit_id: str) -> str:
    if not stoich_string or stoich_string == "?":
        return "?"
    matches = re.findall(r"([A-Z]{1,2})(\d*)", stoich_string)
    for sym, count in matches:
        if sym == subunit_id:
            return f"{count if count else '1'}"
    return "?"

def map_template_to_entity(template_hit_sequence: str, pdb_data: dict) -> dict:
    best_match = {}
    best_identity = 0.0

    entities = pdb_data.get("entry", {}).get("polymer_entities", [])
    for e in entities:
        seq = e.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
        if not seq:
            continue

        identity = compute_sequence_identity(template_hit_sequence, seq)
        if identity > best_identity:
            best_identity = identity

            source_org = ""
            host_org = ""
            if "rcsb_entity_source_organism" in e and e["rcsb_entity_source_organism"]:
                source_org = e["rcsb_entity_source_organism"][0].get("ncbi_scientific_name", "")
            if "rcsb_entity_host_organism" in e and e["rcsb_entity_host_organism"]:
                host_org = e["rcsb_entity_host_organism"][0].get("ncbi_scientific_name", "")

            uniprot_ids = e.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids")
            if uniprot_ids and isinstance(uniprot_ids, list):
                uniprot = ",".join(uniprot_ids)
            else:
                uniprot = ""

            best_match = {
                "description": e.get("entity_poly", {}).get("pdbx_description", ""),
                "source_organism": source_org,
                "host_organism": host_org,
                "uniprot": uniprot
            }

    if best_identity >= 0.90:
        return best_match
    return {}

# =========================================================
# Template metadata builder
# =========================================================

def build_template_metadata(
    template_hit,
    su,
    subunit_len,
    template_fasta_dir,
    template_atom_dir,
    pdb_json_dir,
    template_true_stoich_dict
):
    template_id = template_hit.name[:4]
    template_json_file = os.path.join(pdb_json_dir, template_id + ".json")
    try:
        with open(template_json_file) as f:
            data = json.load(f)
    except Exception:
        data = {}

    pdb_data = data.get("data", {})

    struct_title = pdb_data.get("entry", {}).get("struct", {}).get("title", "N/A")
    struct_keywords = pdb_data.get("entry", {}).get("struct_keywords", {})
    keywords = struct_keywords.get("text", "") if struct_keywords else ""

    exp_methods = [m.get("method") for m in pdb_data.get("entry", {}).get("exptl", []) if m.get("method")]
    refine_list = pdb_data.get("entry", {}).get("refine")
    if isinstance(refine_list, list) and refine_list:
        resolution = refine_list[0].get("ls_d_res_high", "N/A")
    else:
        resolution = "N/A"

    citation_branch = pdb_data.get("entry", {}).get("citation", [])
    citation_doi = citation_branch[0].get("pdbx_database_id_DOI", "") if citation_branch else ""
    cite_url = linkify_citation(citation_doi) if citation_doi else "N/A"

    full_stoich = template_true_stoich_dict.get(template_id, "?")

    template_hit_atom_file = os.path.join(template_atom_dir, template_hit.name + ".atom.gz")
    try:
        template_hit_sequence = parsers.read_pdb_sequence(template_hit_atom_file)
    except Exception:
        template_hit_sequence = ""

    template_fasta_file = os.path.join(template_fasta_dir, template_id + "_initial.fasta")
    subunit_in_full_stoich = ""
    try:
        records = list(SeqIO.parse(template_fasta_file, "fasta"))
        if len(records) == 1:
            subunit_in_full_stoich = records[0].id
        else:
            for record in records:
                template_seq = str(record.seq)
                identity = compute_sequence_identity(template_hit_sequence, template_seq)
                if identity > 0.90:
                    subunit_in_full_stoich = record.id
                    break
    except FileNotFoundError:
        pass

    num_copies = extract_subunit_copies(full_stoich, subunit_in_full_stoich)

    aligned_cols = getattr(template_hit, "aligned_cols", 0)
    coverage = aligned_cols / subunit_len if subunit_len > 0 else 0.0
    coverage_label = (
        "high" if coverage >= 0.8 else
        "moderate" if coverage >= 0.5 else
        "low"
    )

    entity_match = {}
    if pdb_data and template_hit_sequence:
        entity_match = map_template_to_entity(template_hit_sequence, pdb_data)

    return {
        "id": template_id,
        "sum_probs": safe_float(getattr(template_hit, "prob", None), 0.0),
        "evalue": safe_float(getattr(template_hit, "e_value", None), 1e99),
        "pvalue": safe_float(getattr(template_hit, "p_value", None), 1e99),
        "aligned_cols": safe_int(aligned_cols, 0),
        "coverage": safe_float(coverage, 0.0),
        "coverage_label": coverage_label,

        "query_range": safe_str(getattr(template_hit, "query_range", ""), ""),
        "hit_range": safe_str(getattr(template_hit, "hit_range", ""), ""),
        "identity": safe_float(getattr(template_hit, "identity", 0.0), 0.0),

        "copies": safe_str(num_copies, "?"),
        "template_chain_count": safe_str(num_copies, "?"),
        "template_chain_id": safe_str(subunit_in_full_stoich, ""),
        "aligned_to_target_subunit": su,
        "template_complex_stoichiometry": safe_str(full_stoich, "?"),

        "url": safe_str(cite_url, "N/A"),
        "citation": safe_str(cite_url, "N/A"),
        "citation_doi": safe_str(citation_doi, ""),

        "title": safe_str("" if struct_title == "N/A" else struct_title, ""),
        "keywords": safe_str(keywords, ""),
        "experimental_methods": exp_methods,
        "resolution": resolution,

        "entity_description": safe_str(entity_match.get("description", ""), ""),
        "entity_source_organism": safe_str(entity_match.get("source_organism", ""), ""),
        "entity_host_organism": safe_str(entity_match.get("host_organism", ""), ""),
        "entity_uniprot": safe_str(entity_match.get("uniprot", ""), ""),

    }

# =========================================================
# Shared-template helpers
# =========================================================

def compute_shared_template_map_from_hits(template_hits, min_shared_subunits=MIN_SHARED_SUBUNITS):
    pdb_to_subunits = {}
    for su, hits in template_hits.items():
        seen_pdbs_this_subunit = set()
        for hit in hits:
            pdb_id = hit.name[:4]
            if pdb_id not in seen_pdbs_this_subunit:
                pdb_to_subunits.setdefault(pdb_id, set()).add(su)
                seen_pdbs_this_subunit.add(pdb_id)

    return {
        pdb_id: sorted(list(subs))
        for pdb_id, subs in pdb_to_subunits.items()
        if len(subs) >= min_shared_subunits
    }

def infer_shared_template_type(covered_subunits, all_subunits):
    covered_count = len(covered_subunits)
    total_count = len(all_subunits)
    major_threshold = max(2, math.ceil(total_count / 3))

    if covered_count == total_count:
        return "FULL_COMPLEX_SHARED"
    elif covered_count >= major_threshold:
        return "MAJOR_PARTIAL_SHARED"
    else:
        return "PARTIAL_SHARED"

def infer_target_patterns_from_shared_template(subunit_details, all_subunits):
    """
    Return one or more target-oriented patterns inferred from one shared template.

    Key rule for homomeric transfer:
    - If all covered target subunits map to the same template chain, and that
      template chain has total multiplicity N, then infer all positive integer
      allocations of N across those covered target subunits.
    """
    covered_subunits = [su for su in all_subunits if su in subunit_details]
    if not covered_subunits:
        return [], "partial"

    is_single_subunit_system = (len(all_subunits) == 1)

    chain_ids = []
    copy_values = []
    same_chain_info_complete = True
    for su in covered_subunits:
        chain_id = subunit_details[su].get("template_chain_id", "")
        copies = subunit_details[su].get("copies")
        if not chain_id or copies in (None, "", "?"):
            same_chain_info_complete = False
            break
        try:
            chain_ids.append(chain_id)
            copy_values.append(int(copies))
        except Exception:
            same_chain_info_complete = False
            break

    same_template_chain = (
        same_chain_info_complete and
        len(set(chain_ids)) == 1 and
        len(set(copy_values)) == 1
    )

    if same_template_chain:
        total_template_copies = copy_values[0]
        num_covered = len(covered_subunits)

        allocations = generate_positive_compositions(total_template_copies, num_covered)
        if not allocations:
            return [], "partial"

        candidate_patterns = []
        for alloc in allocations:
            parts = []
            valid = True
            for su, c in zip(covered_subunits, alloc):
                if is_single_subunit_system and c < 2:
                    c = 2
                if c <= 0:
                    valid = False
                    break
                parts.append(f"{su}{c}")
            if valid:
                candidate_patterns.append("".join(parts))

        candidate_patterns = sorted(set(candidate_patterns))
        if not candidate_patterns:
            return [], "partial"

        if len(candidate_patterns) == 1:
            return candidate_patterns, "consistent"
        return candidate_patterns, "ambiguous_composition"

    parts = []
    consistency = "consistent"
    for su in all_subunits:
        if su not in subunit_details:
            continue

        copies = subunit_details[su].get("copies")
        if copies is None or str(copies) in {"", "?"}:
            consistency = "partial"
            continue

        try:
            c = int(copies)
        except Exception:
            consistency = "partial"
            continue

        if is_single_subunit_system and c < 2:
            c = 2

        parts.append(f"{su}{c}")

    if not parts:
        return [], "partial"

    return ["".join(parts)], consistency

def rank_inferred_shared_patterns(shared_templates, all_subunits):
    pattern_groups = {}

    for item in shared_templates:
        patterns = item.get("target_stoich_patterns_from_shared_template", [])
        for pattern in patterns:
            if not pattern:
                continue

            if pattern not in pattern_groups:
                pattern_groups[pattern] = {
                    "pattern": pattern,
                    "supporting_template_ids": [],
                    "supporting_template_count": 0,
                    "max_covered_count": 0,
                    "mean_covered_count": 0.0,
                    "best_is_full_complex": False,
                    "best_is_major_partial_shared": False,
                    "mean_best_coverage": 0.0,
                    "best_coverage": 0.0,
                    "best_evalue": 1e99,
                    "copy_consistency_labels": [],
                }

            g = pattern_groups[pattern]
            g["supporting_template_ids"].append(item["id"])
            g["supporting_template_count"] += 1
            g["max_covered_count"] = max(g["max_covered_count"], int(item.get("covered_count", 0)))
            g["best_is_full_complex"] = g["best_is_full_complex"] or bool(item.get("is_full_complex_template", False))
            g["best_is_major_partial_shared"] = (
                g["best_is_major_partial_shared"] or
                (item.get("shared_template_type") == "MAJOR_PARTIAL_SHARED")
            )
            g["best_coverage"] = max(g["best_coverage"], float(item.get("best_coverage", 0.0)))
            g["best_evalue"] = min(g["best_evalue"], float(item.get("best_evalue", 1e99)))
            g["copy_consistency_labels"].append(item.get("copy_consistency", "partial"))

    if not pattern_groups:
        return []

    for pattern, g in pattern_groups.items():
        covered_counts = []
        coverages = []
        for item in shared_templates:
            if pattern not in item.get("target_stoich_patterns_from_shared_template", []):
                continue
            covered_counts.append(int(item.get("covered_count", 0)))
            coverages.append(float(item.get("best_coverage", 0.0)))

        g["mean_covered_count"] = float(sum(covered_counts) / len(covered_counts)) if covered_counts else 0.0
        g["mean_best_coverage"] = float(sum(coverages) / len(coverages)) if coverages else 0.0

        labels = g["copy_consistency_labels"]
        if all(x == "consistent" for x in labels):
            g["copy_consistency"] = "consistent"
        elif any(x == "ambiguous_composition" for x in labels):
            g["copy_consistency"] = "ambiguous_composition"
        elif any(x == "consistent" for x in labels):
            g["copy_consistency"] = "mixed"
        else:
            g["copy_consistency"] = "partial"

    total_subunits = max(1, len(all_subunits))
    max_support_count = max(g["supporting_template_count"] for g in pattern_groups.values())

    ranked = []
    for pattern, g in pattern_groups.items():
        covered_frac = normalize_by_max(g["max_covered_count"], total_subunits)
        support_frac = normalize_by_max(g["supporting_template_count"], max_support_count)
        full_complex_bonus = 1.0 if g["best_is_full_complex"] else 0.0
        major_partial_bonus = 1.0 if g["best_is_major_partial_shared"] else 0.0
        coverage_score = g["mean_best_coverage"]
        evalue_score = min(1.0, safe_neg_log10_evalue(g["best_evalue"]) / 50.0)

        support_score = (
            0.30 * covered_frac +
            0.20 * full_complex_bonus +
            0.10 * major_partial_bonus +
            0.20 * support_frac +
            0.10 * coverage_score +
            0.10 * evalue_score
        )

        ranked.append({
            "pattern": g["pattern"],
            "supporting_template_ids": sorted(g["supporting_template_ids"]),
            "supporting_template_count": int(g["supporting_template_count"]),
            "max_covered_count": int(g["max_covered_count"]),
            "mean_covered_count": float(round(g["mean_covered_count"], 4)),
            "best_is_full_complex": bool(g["best_is_full_complex"]),
            "best_is_major_partial_shared": bool(g["best_is_major_partial_shared"]),
            "mean_best_coverage": float(round(g["mean_best_coverage"], 4)),
            "best_coverage": float(round(g["best_coverage"], 4)),
            "best_evalue": float(g["best_evalue"]),
            "copy_consistency": g["copy_consistency"],
            "support_score": float(round(support_score, 6)),
        })

    ranked.sort(
        key=lambda x: (
            -x["max_covered_count"],
            -int(x["best_is_full_complex"]),
            -int(x["best_is_major_partial_shared"]),
            -x["supporting_template_count"],
            -x["mean_best_coverage"],
            x["best_evalue"],
            x["pattern"],
        )
    )

    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx

    return ranked

def build_shared_template_records(
    subunit_sequences,
    template_hits_full,
    shared_template_map,
    template_fasta_dir,
    template_atom_dir,
    template_pdb_json_dir,
    true_labels
):
    shared_records = []

    for pdb_id, covered_subunits in sorted(shared_template_map.items()):
        per_subunit_hits = {}

        for su in covered_subunits:
            hits = template_hits_full.get(su, [])
            matched_hits = [h for h in hits if h.name[:4] == pdb_id]
            if not matched_hits:
                continue

            best_hit = sorted(
                matched_hits,
                key=lambda h: (
                    float(getattr(h, "e_value", 1e99)) if getattr(h, "e_value", None) is not None else 1e99,
                    -float(getattr(h, "prob", 0.0))
                )
            )[0]

            subunit_len = len(subunit_sequences[su])
            meta = build_template_metadata(
                best_hit,
                su,
                subunit_len,
                template_fasta_dir,
                template_atom_dir,
                template_pdb_json_dir,
                true_labels
            )
            per_subunit_hits[su] = meta

        if len(per_subunit_hits) < 2:
            continue

        best_coverage = max(v["coverage"] for v in per_subunit_hits.values())
        best_evalue = min(v["evalue"] for v in per_subunit_hits.values())

        shared_records.append({
            "id": pdb_id,
            "covered_subunits": sort_subunits(list(per_subunit_hits.keys())),
            "covered_count": len(per_subunit_hits),
            "best_coverage": best_coverage,
            "best_evalue": best_evalue,
            "per_subunit_hits": {
                su: per_subunit_hits[su]
                for su in sort_subunits(per_subunit_hits.keys())
            }
        })

    shared_records.sort(
        key=lambda x: (
            -x["covered_count"],
            -x["best_coverage"],
            x["best_evalue"],
            x["id"]
        )
    )
    return shared_records

def compute_shared_template_summary_from_shared_records(shared_template_records, all_subunits):
    major_threshold = max(2, math.ceil(len(all_subunits) / 3))
    shared_templates = []
    pattern_counter = Counter()

    for rec in shared_template_records:
        pdb_id = rec["id"]
        covered_subunits = sort_subunits(rec["covered_subunits"])
        covered_count = len(covered_subunits)
        missing_target_subunits = [s for s in all_subunits if s not in covered_subunits]
        shared_type = infer_shared_template_type(covered_subunits, all_subunits)

        subunit_details = {}
        for su in covered_subunits:
            hit = rec["per_subunit_hits"][su]
            subunit_details[su] = {
                "coverage": float(hit["coverage"]),
                "evalue": float(hit["evalue"]),
                "copies": hit["copies"],
                "template_chain_id": hit.get("template_chain_id", ""),
                "rank": 1,
            }

        target_patterns, copy_consistency = infer_target_patterns_from_shared_template(
            subunit_details, all_subunits
        )
        for p in target_patterns:
            pattern_counter[p] += 1

        is_full_complex_template = (covered_count == len(all_subunits))
        is_major_partial_shared = (covered_count >= major_threshold and covered_count < len(all_subunits))

        shared_templates.append({
            "id": pdb_id,
            "covered_subunits": covered_subunits,
            "covered_count": covered_count,
            "major_partial_threshold": major_threshold,
            "is_full_complex_template": is_full_complex_template,
            "is_major_partial_shared": is_major_partial_shared,
            "missing_target_subunits": missing_target_subunits,
            "shared_template_type": shared_type,
            "best_coverage": float(rec["best_coverage"]),
            "best_evalue": float(rec["best_evalue"]),
            "target_stoich_patterns_from_shared_template": target_patterns,
            "copy_consistency": copy_consistency,
            "subunit_details": {
                su: {
                    "coverage": float(subunit_details[su]["coverage"]),
                    "evalue": float(subunit_details[su]["evalue"]),
                    "copies": subunit_details[su]["copies"],
                    "template_chain_id": subunit_details[su].get("template_chain_id", ""),
                    "rank": int(subunit_details[su]["rank"]),
                }
                for su in sort_subunits(subunit_details.keys())
            },
        })

    shared_templates.sort(
        key=lambda x: (
            -x["covered_count"],
            -int(x["is_full_complex_template"]),
            -int(x["is_major_partial_shared"]),
            -x["best_coverage"],
            x["best_evalue"],
            x["id"],
        )
    )

    strong_shared_templates = [
        x for x in shared_templates
        if (
            x["best_coverage"] >= SHARED_STRONG_COV
            and x["best_evalue"] <= SHARED_STRONG_EVALUE
            and x["covered_count"] >= major_threshold
        )
    ]

    top_shared_pattern = pattern_counter.most_common(1)[0][0] if pattern_counter else None
    inferred_shared_patterns = rank_inferred_shared_patterns(shared_templates, all_subunits)
    top_ranked_shared_pattern = inferred_shared_patterns[0]["pattern"] if inferred_shared_patterns else None

    return {
        "num_shared_templates": len(shared_templates),
        "num_strong_shared_templates": len(strong_shared_templates),
        "major_partial_threshold": major_threshold,
        "best_shared_pdb": shared_templates[0]["id"] if shared_templates else None,
        "best_shared_cover": shared_templates[0]["covered_count"] if shared_templates else 0,
        "shared_template_pattern_counts": dict(pattern_counter),
        "top_shared_template_pattern": top_shared_pattern,
        "top_ranked_shared_pattern": top_ranked_shared_pattern,
        "inferred_shared_patterns": inferred_shared_patterns,
        "shared_templates": shared_templates,
        "strong_shared_templates": strong_shared_templates,
    }

# =========================================================
# Common subunit stoichiometry summary
# =========================================================

def compute_subunit_consensus_summary_from_template_dict(template_dict):
    """
    For single-subunit systems, A1 is invalid everywhere.
    """
    all_subunits = sort_subunits(list(template_dict.keys()))
    per_subunit = {}
    combined_parts = []

    is_single_subunit_system = (len(all_subunits) == 1)

    for su in all_subunits:
        hits = template_dict.get(su, [])
        sig_counts = []
        all_counts = []

        for hit in hits:
            evalue = safe_float(hit.get("evalue", None), None)
            copies = hit.get("copies", None)
            c = safe_int(copies, None)
            if c is None:
                continue

            if is_single_subunit_system and c == 1:
                continue

            all_counts.append(c)
            if evalue is not None and evalue <= SUBUNIT_SIG_EVALUE:
                sig_counts.append(c)

        use_counts = sig_counts if sig_counts else all_counts

        if use_counts:
            raw_counter = Counter(use_counts)
            dominant_copy = raw_counter.most_common(1)[0][0]
            total = sum(raw_counter.values())
            freq_map = {str(k): v / total for k, v in sorted(raw_counter.items())}
        else:
            dominant_copy = 2 if is_single_subunit_system else 1
            freq_map = {str(dominant_copy): 1.0}

        if is_single_subunit_system and dominant_copy < 2:
            dominant_copy = 2

        if is_single_subunit_system:
            freq_map = {k: v for k, v in freq_map.items() if int(k) >= 2}
            if not freq_map:
                freq_map = {"2": 1.0}

        dominant_freq = max(freq_map.values()) if freq_map else 1.0

        if not use_counts:
            confidence_label = "no_evidence"
        elif dominant_freq >= SUBUNIT_STRONG_MIN_FREQ:
            confidence_label = "strong"
        elif dominant_freq >= 0.40:
            confidence_label = "moderate"
        else:
            confidence_label = "weak"

        all_observed_copies = sorted(int(k) for k in freq_map.keys())

        preferred_copies = [
            int(k) for k, v in freq_map.items()
            if v >= SUBUNIT_ALLOWED_MIN_FREQ
        ]
        if dominant_copy not in preferred_copies:
            preferred_copies = sorted(set(preferred_copies + [dominant_copy]))

        if is_single_subunit_system:
            all_observed_copies = [c for c in all_observed_copies if c >= 2]
            preferred_copies = [c for c in preferred_copies if c >= 2]
            if not all_observed_copies:
                all_observed_copies = [2]
            if not preferred_copies:
                preferred_copies = [2]

        per_subunit[su] = {
            "dominant_copy": int(dominant_copy),
            "dominant_frequency": float(round(dominant_freq, 4)),
            "all_observed_copies": all_observed_copies,
            "preferred_copies": preferred_copies,
            "copy_frequency_map": {k: float(round(v, 4)) for k, v in freq_map.items()},
            "confidence_label": confidence_label,
            "num_templates_used": int(len(use_counts)),
        }

        combined_parts.append(f"{su}{dominant_copy}")

    return {
        "common_subunit_stoichiometry_pattern": "".join(combined_parts),
        "per_subunit": per_subunit,
    }

# =========================================================
# Constraint summary
# =========================================================

def build_constraint_summary(template_dict, subunit_consensus_summary, shared_template_summary):
    all_subunits = sort_subunits(list(template_dict.keys()))
    major_threshold = shared_template_summary.get("major_partial_threshold", max(2, math.ceil(len(all_subunits) / 3)))

    common_constraints = []
    for su in all_subunits:
        info = subunit_consensus_summary["per_subunit"].get(su, {})
        common_constraints.append({
            "subunit": su,
            "dominant_copy": info.get("dominant_copy"),
            "all_observed_copies": info.get("all_observed_copies", []),
            "preferred_copies": info.get("preferred_copies", []),
            "confidence_label": info.get("confidence_label", "weak"),
            "constraint_type": "soft" if info.get("confidence_label", "weak") != "strong" else "strong_soft",
            "rule": (
                f"Subunit {su} most commonly appears with copy {info.get('dominant_copy')}; "
                f"preferred copies are {info.get('preferred_copies', [])}; "
                f"all observed copies are {info.get('all_observed_copies', [])}."
            )
        })

    inferred_patterns = shared_template_summary.get("inferred_shared_patterns", [])
    shared_constraints = []

    for item in inferred_patterns[:10]:
        pattern = item["pattern"]
        pairs = re.findall(r"([A-Z]{1,2})(\d+)", pattern)
        covered_subunits = [p[0] for p in pairs]
        copy_map = {su: int(c) for su, c in pairs}

        if item["max_covered_count"] == len(all_subunits):
            ctype = "strong"
        elif item["max_covered_count"] >= major_threshold:
            ctype = "semi_strong"
        else:
            ctype = "soft"

        shared_constraints.append({
            "pattern": pattern,
            "rank": item["rank"],
            "covered_subunits": covered_subunits,
            "copy_map": copy_map,
            "supporting_template_count": item["supporting_template_count"],
            "max_covered_count": item["max_covered_count"],
            "best_is_full_complex": item["best_is_full_complex"],
            "best_is_major_partial_shared": item.get("best_is_major_partial_shared", False),
            "support_score": item["support_score"],
            "copy_consistency": item.get("copy_consistency", "partial"),
            "constraint_type": ctype,
            "rule": (
                f"Shared-template pattern {pattern} is supported by "
                f"{item['supporting_template_count']} shared template(s) and its best template covers "
                f"{item['max_covered_count']} target subunit(s). "
                f"The MAJOR_PARTIAL_SHARED threshold is {major_threshold}."
            )
        })

    top_common_pattern = subunit_consensus_summary.get("common_subunit_stoichiometry_pattern")
    top_shared_pattern = shared_template_summary.get("top_ranked_shared_pattern")

    combined = {
        "top_common_subunit_pattern": top_common_pattern,
        "top_shared_template_pattern": top_shared_pattern,
        "agreement": (top_common_pattern == top_shared_pattern) if (top_common_pattern and top_shared_pattern) else False,
        "resolution_rule": (
            "If a high-ranked shared-template pattern covers at least the MAJOR_PARTIAL_SHARED threshold of target subunits, "
            "prefer it over the common per-subunit pattern. "
            "Use the common per-subunit pattern mainly as a fallback prior or to fill uncovered subunits."
        )
    }

    return {
        "common_subunit_stoichiometry_constraints": common_constraints,
        "shared_template_stoichiometry_constraints": shared_constraints,
        "combined_constraint_view": combined,
    }

# =========================================================
# Template collection (generated search results)
# =========================================================

def collect_template_hits_no_rescue(
    subunit_inputs: dict,
    release_dates,
    deposit_date_threshold=None,
    core_top_k: int = MAX_TEMPLATE_COUNT,
    scan_top_k: int = MAX_TEMPLATE_SCAN,
):
    """
    Collect template hits for one target from generated HHsearch results.

    Args:
        subunit_inputs:
            Dict of the form:
            {
                "A": {"hhr_file": "...", "fasta_file": "..."},
                "B": {"hhr_file": "...", "fasta_file": "..."},
                ...
            }

        release_dates:
            Dict mapping template PDB ID -> release date.

        deposit_date_threshold:
            If provided, exclude templates with release_date >= threshold.

        core_top_k:
            Number of top hits kept per subunit for the final output.

        scan_top_k:
            Number of hits scanned per subunit for shared-template detection.

    Returns:
        final_hits:
            Dict[subunit -> top core_top_k hits]

        template_hits_full:
            Dict[subunit -> top scan_top_k hits]
    """
    template_hits_full = {}

    subunit_names = sort_subunits(list(subunit_inputs.keys()))
    if not subunit_names:
        return {}, {}

    for subunit in subunit_names:
        info = subunit_inputs[subunit]
        hhr_file = info.get("hhr_file")

        with open(hhr_file, encoding="latin1") as f:
            hits = parsers.parse_hhr(f.read())

        hits = sorted(hits, key=lambda x: x.e_value)

        filtered_hits = []
        for hit in hits:
            pdbcode = hit.name[:4]
            if deposit_date_threshold:
                if pdbcode not in release_dates:
                    continue
                deposit_date = release_dates[pdbcode]
                if deposit_date >= deposit_date_threshold:
                    continue
            filtered_hits.append(hit)

        template_hits_full[subunit] = filtered_hits[:scan_top_k]
    if not template_hits_full:
        return {}, {}

    final_hits = {
        subunit: hits[:core_top_k]
        for subunit, hits in template_hits_full.items()
    }
    final_hits = {su: final_hits[su] for su in sort_subunits(final_hits.keys())}
    template_hits_full = {su: template_hits_full[su] for su in sort_subunits(template_hits_full.keys())}
    return final_hits, template_hits_full

# =========================================================
# User prompt
# =========================================================

def make_user_prompt(
    protein_id,
    subunit_sequences,
    template_hits,
    template_fasta_dir,
    template_atom_dir,
    pdb_json_dir,
    template_true_stoich_dict
) -> str:
    lines = [
        f"Predict stoichiometry for this protein ({protein_id}).",
        "There are two evidence sections below:",
        "1. Per-subunit templates",
        "2. Shared templates observed across multiple subunits",
        ""
    ]

    lines.append("Per-subunit templates:")

    for su in sort_subunits(template_hits.keys()):
        subunit_len = len(subunit_sequences[su])
        lines.append(f"- Subunit {su}, sequence length: {subunit_len}")

        for template_hit in template_hits[su]:
            meta = build_template_metadata(
                template_hit=template_hit,
                su=su,
                subunit_len=subunit_len,
                template_fasta_dir=template_fasta_dir,
                template_atom_dir=template_atom_dir,
                pdb_json_dir=pdb_json_dir,
                template_true_stoich_dict=template_true_stoich_dict
            )

            lines.append(
                f"Template: {meta['id']}, "
                f"sum_probs={meta['sum_probs']:.3f}, e-value={meta['evalue']}, "
                f"p-evalue={meta['pvalue']}, number_of_aligned_residues={meta['aligned_cols']}, "
                f"coverage={meta['coverage']:.2f} ({meta['coverage_label']} coverage), "
                f"template_chain_count={meta['template_chain_count']} "
                f"(template chain {meta['template_chain_id']} → aligns to subunit {su}), "
                f"template_complex_stoichiometry={meta['template_complex_stoichiometry'].replace(', ', '')} "
                f"(template-level only; chain letters are template-specific and not aligned to target labels), "
                f"citation={meta['citation']}"
            )

            if meta["title"]:
                lines.append(f"    Title: {meta['title']}")
            if meta["keywords"]:
                lines.append(f"    Keywords: {meta['keywords']}")
            if meta["experimental_methods"]:
                lines.append(
                    f"    Method: {', '.join(meta['experimental_methods'])}, Resolution: {meta['resolution']}Å"
                )
            if meta["entity_description"] or meta["entity_uniprot"]:
                lines.append(
                    f"    Entity: {meta['entity_description']} "
                    f"(Source Organism: {meta['entity_source_organism']}, "
                    f"Host Organism: {meta['entity_host_organism']}, "
                    f"UniProt:{meta['entity_uniprot']})"
                )

    return "\n".join(lines) + "\n"

def make_shared_templates_prompt(shared_template_records, shared_template_summary=None):
    lines = []
    if not shared_template_records:
        return ""

    lines.append("Shared templates observed across multiple target subunits:")

    summary_lookup = {}
    if shared_template_summary:
        for item in shared_template_summary.get("shared_templates", []):
            summary_lookup[item["id"]] = item

    for rec in shared_template_records:
        lines.append(
            f"- Shared Template: {rec['id']}, "
            f"covered_subunits={','.join(rec['covered_subunits'])}, "
            f"covered_count={rec['covered_count']}, "
            f"best_coverage={rec['best_coverage']:.2f}, "
            f"best_evalue={rec['best_evalue']}"
        )

        summary_item = summary_lookup.get(rec["id"])
        if summary_item:
            pats = summary_item.get("target_stoich_patterns_from_shared_template", [])
            if pats:
                lines.append(f"    Inferred target patterns from shared template: {', '.join(pats)}")
            lines.append(f"    Copy consistency: {summary_item.get('copy_consistency', 'partial')}")

        for su in sort_subunits(rec["per_subunit_hits"].keys()):
            hit = rec["per_subunit_hits"][su]
            lines.append(
                f"    Subunit {su}: "
                f"sum_probs={hit['sum_probs']:.3f}, "
                f"e-value={hit['evalue']}, "
                f"coverage={hit['coverage']:.2f} ({hit['coverage_label']} coverage), "
                f"template_chain_count={hit['template_chain_count']} "
                f"(template chain {hit['template_chain_id']} → aligns to subunit {su}), "
                f"template_complex_stoichiometry={hit['template_complex_stoichiometry']}, "
                f"citation={hit['citation']}"
            )
            if hit["title"]:
                lines.append(f"        Title: {hit['title']}")
            if hit["keywords"]:
                lines.append(f"        Keywords: {hit['keywords']}")
            if hit["experimental_methods"]:
                lines.append(
                    f"        Method: {', '.join(hit['experimental_methods'])}, Resolution: {hit['resolution']}Å"
                )
            if hit["entity_description"] or hit["entity_uniprot"]:
                lines.append(
                    f"        Entity: {hit['entity_description']} "
                    f"(Source Organism: {hit['entity_source_organism']}, "
                    f"Host Organism: {hit['entity_host_organism']}, "
                    f"UniProt:{hit['entity_uniprot']})"
                )

    return "\n".join(lines) + "\n"

# =========================================================
# JSON construction
# =========================================================

def build_protein_json(
    protein_id,
    subunit_inputs,
    true_labels,
    release_dates,
    template_fasta_dir,
    template_atom_dir,
    template_pdb_json_dir,
    max_template_count=MAX_TEMPLATE_COUNT,
    template_release_cutoff=None,
):
    template_hits, template_hits_full = collect_template_hits_no_rescue(
        subunit_inputs=subunit_inputs,
        release_dates=release_dates,
        deposit_date_threshold=template_release_cutoff,
        core_top_k=max_template_count,
    )

    subunit_sequences = {}
    for subunit in sort_subunits(subunit_inputs):
        fasta_file = subunit_inputs[subunit]["fasta_file"]
        records = list(SeqIO.parse(fasta_file, "fasta"))
        if not records:
            raise ValueError(f"No FASTA records found in {fasta_file}")
        subunit_sequences[subunit] = str(records[0].seq)

    shared_template_map_full = compute_shared_template_map_from_hits(
        template_hits_full,
        min_shared_subunits=MIN_SHARED_SUBUNITS
    )

    template_dict = {}
    for su in sort_subunits(subunit_inputs):
        entries = []
        subunit_len = len(subunit_sequences[su])

        for rank_idx, hit in enumerate(template_hits.get(su, []), start=1):
            meta = build_template_metadata(
                hit,
                su,
                subunit_len,
                template_fasta_dir,
                template_atom_dir,
                template_pdb_json_dir,
                true_labels
            )

            pdb_id_hit = meta["id"]
            shared_subunits = shared_template_map_full.get(pdb_id_hit, [])
            meta["rank"] = rank_idx
            meta["shared_across_subunits"] = shared_subunits
            meta["num_target_subunits_matched"] = len(shared_subunits)

            entries.append(meta)

        template_dict[su] = entries

    shared_template_records = build_shared_template_records(
        subunit_sequences=subunit_sequences,
        template_hits_full=template_hits_full,
        shared_template_map=shared_template_map_full,
        template_fasta_dir=template_fasta_dir,
        template_atom_dir=template_atom_dir,
        template_pdb_json_dir=template_pdb_json_dir,
        true_labels=true_labels
    )

    all_subunits = sort_subunits(list(template_dict.keys()))
    shared_template_summary = compute_shared_template_summary_from_shared_records(
        shared_template_records,
        all_subunits
    )
    subunit_consensus_summary = compute_subunit_consensus_summary_from_template_dict(template_dict)
    constraint_summary = build_constraint_summary(
        template_dict=template_dict,
        subunit_consensus_summary=subunit_consensus_summary,
        shared_template_summary=shared_template_summary,
    )

    user_prompt = make_user_prompt(
        protein_id,
        subunit_sequences,
        template_hits,
        template_fasta_dir,
        template_atom_dir,
        template_pdb_json_dir,
        true_labels
    )

    shared_prompt = make_shared_templates_prompt(shared_template_records, shared_template_summary)
    combined_user_prompt = user_prompt + ("\n" + shared_prompt if shared_prompt else "")

    base = {
        "protein_id": protein_id,
        "messages": [
            make_system_message(all_subunits),
            {"role": "user", "content": combined_user_prompt},
        ],
        "templates": template_dict,
        "shared_templates_for_agent": shared_template_records,
        "subunit_consensus_summary": subunit_consensus_summary,
        "shared_template_summary": shared_template_summary,
        "constraint_summary": constraint_summary,
    }

    return base

# =========================================================
# Main
# =========================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build one Stage-1 JSON from a target FASTA, generating template searches automatically."
    )
    parser.add_argument("--input_fasta", required=True,
                        help="Multi-record FASTA containing the target's distinct protein subunits.")
    parser.add_argument("--config", required=True,
                        help="JSON with search database, binary and template metadata paths.")
    parser.add_argument("--workdir", required=True,
                        help="Directory for generated search intermediates.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--protein_id", default=None,
                        help="Optional override for protein_id; otherwise output filename stem.")
    parser.add_argument("--max_template_count", type=int, default=5)
    parser.add_argument("--template_release_cutoff", default=None,
                        help="Optional YYYY-MM-DD cutoff; exclude templates released on or after this date.")

    args = parser.parse_args(argv)

    if args.max_template_count < 1:
        parser.error("--max_template_count must be at least 1")
    if args.template_release_cutoff and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.template_release_cutoff):
        parser.error("--template_release_cutoff must be YYYY-MM-DD")

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    def setting(*keys):
        for key in keys:
            if config.get(key):
                return config[key]
        parser.error(f"Missing config key: {' or '.join(keys)}")

    from .template_search import generate_template_search_inputs
    subunit_inputs = generate_template_search_inputs(
        args.input_fasta, args.workdir,
        uniref90_db=setting("uniref90_path"),
        hhdb_prefix=setting("hhdb_prefix"),
        jackhmmer_binary=config.get("jackhmmer_binary", "jackhmmer"),
        hhmake_binary=config.get("hhmake_binary", "hhmake"),
        hhsearch_binary=config.get("hhsearch_binary", "hhsearch"),
        cpu=int(config.get("search_cpu", 8)),
    )
    template_fasta_dir = setting("template_fasta_dir", "stoichiometry_pdb_fasta_folder")
    template_pdb_json_dir = setting("template_pdb_json_dir")
    template_atom_dir = setting("template_atom_dir")
    template_label_csv = setting("template_label_csv", "pdb_stoichiometry_database")

    training_label_df = pd.read_csv(template_label_csv)
    required_columns = {"pdb_id", "final_stoichiometry"}
    missing_columns = required_columns - set(training_label_df.columns)
    if missing_columns:
        parser.error(f"Template label CSV is missing columns: {', '.join(sorted(missing_columns))}")
    if args.template_release_cutoff and "release_date" not in training_label_df.columns:
        parser.error("--template_release_cutoff requires a release_date column in the template label CSV")
    true_labels = dict(zip(training_label_df["pdb_id"], training_label_df["final_stoichiometry"]))

    if "release_date" in training_label_df.columns:
        release_dates = dict(zip(training_label_df["pdb_id"], training_label_df["release_date"]))
    else:
        release_dates = {}

    protein_id = args.protein_id
    if protein_id is None:
        protein_id = os.path.splitext(os.path.basename(args.output_json))[0]

    result = build_protein_json(
        protein_id=protein_id,
        subunit_inputs=subunit_inputs,
        true_labels=true_labels,
        release_dates=release_dates,
        template_fasta_dir=template_fasta_dir,
        template_atom_dir=template_atom_dir,
        template_pdb_json_dir=template_pdb_json_dir,
        max_template_count=args.max_template_count,
        template_release_cutoff=args.template_release_cutoff,
    )

    out_parent = os.path.dirname(os.path.abspath(args.output_json))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {args.output_json}")

if __name__ == "__main__":
    main()
