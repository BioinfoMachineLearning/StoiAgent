#!/usr/bin/env python3
import re
import copy
import os
import argparse
import json
import gzip

import numpy as np


from .discovery_cache import DiscoveryCache
from .utils import (
    sort_subunits,
    normalize_stoich,
    parse_stoich,
    simplest_stoich,
    strip_quotes as _strip_quotes,
)
from . import agent_common
from .agent_common import (
    BENCHMARK_MODEL_KEYS,
    HFLLM,
    MODEL_MAP,
    SEEDS_30,
    SUMMARY_MODEL_KEY,
    build_shared_template_observation_from_summary,
    build_tb_prediction_block,
    classify_symmetry_completion_candidates,
    collect_unique_citations,
    compute_consensus_stoichiometry_by_subunit,
    compute_generation_budget,
    extract_casp17_target_id,
    filter_llm_reply,
    format_casp17_target_info_block,
    format_template_line,
    generate_partial_scaffold_completions,
    get_or_generate_literature_summary,
    get_required_baseline_candidates,
    has_candidate_header,
    infer_subunit_lengths_from_json,
    load_casp17_sequence_catalog,
    parse_candidate_stoichiometries,
    reseed,
    resolve_casp17_sequence_target,
    sanitize_chat_messages,
    validate_candidate_set,
)


EXPERIMENT_VARIANT = 'D_on_demand_simplified_rules_casp17_target_info'
EXPERIMENT_DESCRIPTION = 'C plus original CASP17 target description/source and full amino-acid sequence for each subunit, with CASP target identifiers hidden from the LLM.'
MAX_ABSTRACT_READS = 10
# Agent D adds only original CASP17 sequence-release information to C.
# It does not add Phase-2/reference stoichiometry information.


# =========================================================
# Action regex
# =========================================================
_ACTION_RE = re.compile(
    r'Action(?:\s*\[[^\]]+\])?\s*:\s*'
    r'(?P<tool>PubMed\.readAbstract|PubMed\.search|RCSB\.getPDB|UniProt\.get)\s*'
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
            r"([A-Z]{1,2}\d+(?:[A-Z]{1,2}\d+)*)\s+(?:fused_)?support\s*=\s*([0-9]*\.?[0-9]+)",
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
            # Keep the first row of a duplicated candidate, as the merge step
            # does; a later duplicate must not overwrite its support.
            confs.setdefault(stoich, score)
    return confs


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
):
    subunit_ids = sort_subunits(list(templates.keys()))
    lines = ["Predict stoichiometry for this protein complex."]
    lines.append(f"Subunits (canonical order): {subunit_ids}")
    if subunit_lengths:
        parts = [f"{su}={subunit_lengths[su]}" for su in subunit_ids if su in subunit_lengths]
        if parts:
            lines.append("Subunit sequence lengths: " + ", ".join(parts))

    # D differs from C only by adding the original CASP17 target metadata
    # and full amino-acid sequence for each distinct subunit.
    target_block = format_casp17_target_info_block(target_info)
    if target_block:
        lines.extend(["", target_block])

    # Same full structural evidence as C and the same simplified rule policy.
    lines.extend(["", "1. Templates for each subunit:"])
    for su in subunit_ids:
        lines.append(f"- Subunit {su}")
        hits = templates.get(su, [])
        if not hits:
            lines.append("  (No templates)")
            continue
        for idx, hit in enumerate(hits, start=1):
            lines.append("  - " + format_template_line(hit, su=su, rank=idx))

    lines.extend(["", "2. Templates covering multiple or all chains:"])
    if shared_templates_for_agent:
        for rec in shared_templates_for_agent:
            lines.append(
                f"- Shared template id={rec.get('id')}; "
                f"covered_subunits={rec.get('covered_subunits')}; "
                f"covered_count={rec.get('covered_count')}; "
                f"best_coverage={rec.get('best_coverage')}; "
                f"best_evalue={rec.get('best_evalue')}"
            )
            for su in sort_subunits(rec.get("per_subunit_hits", {}).keys()):
                lines.append("  - " + format_template_line(rec["per_subunit_hits"][su], su=su))
    else:
        lines.append("- No shared multi-subunit templates were detected.")

    # Preserve the same derived information, but avoid restating it as multiple rules.
    native_candidate = simplest_stoich(subunit_ids)
    if len(subunit_ids) == 1 and native_candidate == f"{subunit_ids[0]}1":
        native_candidate = f"{subunit_ids[0]}2"
    common_candidate = None
    if subunit_consensus_summary:
        common_candidate = subunit_consensus_summary.get("common_subunit_stoichiometry_pattern")

    lines.extend(["", "Derived candidate anchors:"])
    lines.append(f"- Native/simple baseline: {native_candidate}")
    if common_candidate:
        lines.append(f"- Per-subunit consensus: {common_candidate}")
    if partial_scaffold:
        lines.append(f"- Best shared-template partial scaffold: {partial_scaffold}")
    if scaffold_copy_options:
        for su in sort_subunits(scaffold_copy_options.keys()):
            lines.append(f"- Plausible copies for uncovered {su}: {scaffold_copy_options[su]}")
    if scaffold_completion_candidates:
        lines.append("- Precomputed full-complex scaffold completions: " + ", ".join(scaffold_completion_candidates[:6]))
    if symmetry_info:
        target_copy = symmetry_info.get("symmetry_target_copy")
        missing = symmetry_info.get("missing_subunits", [])
        preserving = symmetry_info.get("symmetry_preserving_candidates", [])
        if target_copy is not None and missing:
            lines.append(f"- Scaffold symmetry prior: repeated copy={target_copy}; uncovered={missing}")
        if preserving:
            lines.append("- Symmetry-preserving completions: " + ", ".join(preserving[:4]))

    lines.extend(["", "3. Literature analysis (ON-DEMAND):"])
    lines.append(
        "- Do not read every citation. Request an abstract only when it can materially resolve assembly/oligomerization, symmetry, a conflict, or full-versus-partial-complex uncertainty.\n"
        f"- At most {MAX_ABSTRACT_READS} unique template-associated abstracts may be read.\n"
        "- Use only citation URLs shown in the evidence: Action: PubMed.readAbstract(\"<citation URL>\")\n"
        "- PubMed.search is disabled. If no abstract is needed, continue with structural/TB evidence."
    )

    lines.extend(["", build_tb_prediction_block(
        subunit_consensus_summary=subunit_consensus_summary,
        shared_template_summary=shared_template_summary,
        constraint_summary=constraint_summary,
    )])

    # One non-redundant policy. The Python validator is unchanged from A/B.
    lines.extend([
        "",
        "Decision/output policy:",
        "- Weight shared multi-subunit templates more strongly than comparable isolated single-subunit templates.",
        "- Treat an incomplete shared pattern as a partial scaffold; infer missing copies from evidence rather than defaulting them to 1.",
        f"- Return at least 5 complete, unique candidates and include the baseline {native_candidate}" + (f" and consensus {common_candidate}." if common_candidate else "."),
        "- Include plausible scaffold completions and a symmetry-preserving completion when supported by the supplied derived evidence.",
        "- For a single-subunit target, X1 is invalid.",
        "- Assign fresh support scores after integrating structural evidence, any requested literature, and TB priors.",
        "",
        "Output:",
        "Thought: <1-3 short sentences>",
        "Candidates:",
        "1) <complete stoichiometry>  support=<0..1>  support_evidence=<short phrase>",
        "2) <complete stoichiometry>  support=<0..1>  support_evidence=<short phrase>",
        "Final Answer: <stoichiometry in canonical subunit order>",
        "Support Score: <score of top candidate>"
    ])
    return "\n".join(lines)


def build_system_message_from_json(templates):
    subunit_ids = sort_subunits(list(templates.keys()))
    return {
        "role": "system",
        "content": (
            "You are an AI agent that predicts protein-complex stoichiometry.\n"
            f"Target subunits: {', '.join(subunit_ids)}.\n"
            "Reason from per-subunit templates, shared/multi-subunit templates, selective literature, and TB priors.\n"
            "Follow the single decision/output policy in the user message; do not invent Observation lines.\n"
            "When literature is genuinely needed, request a listed citation with "
            "Action: PubMed.readAbstract(\"<citation URL>\")."
        )
    }


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
    max_turns=15,
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
    search_attempts = {}
    final_answer = None

    single_subunit_symbol = None
    if len(subunit_ids) == 1:
        single_subunit_symbol = subunit_ids[0]

    is_gemma = "gemma" in getattr(llm, "model_key", "").lower()

    # Runtime observations/corrections must not be system messages.
    # Qwen3.5/Qwen3.8 requires the system message to be at the beginning.
    role = "user"

    scaffold_completion_candidates = scaffold_completion_candidates or []
    symmetry_preserving_candidates = symmetry_preserving_candidates or []

    # B/C literature policy: do not preload any abstracts.
    # The LLM sees citation URLs in the structural evidence and may request only
    # the abstracts it considers decision-relevant.
    turn_count = 1
    citation_items = collect_unique_citations(templates, shared_templates_for_agent)
    allowed_citation_urls = {item["url"] for item in citation_items if item.get("url")}
    abstract_urls_read = set()
    abstract_reads_used = 0

    def append_safe_message(msg_list, role_, content):
        # Never introduce a new system message after the first turn.
        if role_ == "system" and msg_list:
            role_ = "user"

        if role_ == "user" and msg_list and msg_list[-1]["role"] == "user":
            msg_list[-1]["content"] += f"\n\n{content}"
        else:
            msg_list.append({"role": role_, "content": content})

    for turn in range(turn_count, max_turns + 1):
        max_new = compute_generation_budget(messages, llm)
        raw_reply = llm.generate(messages, max_new_tokens=max_new)
        reply = filter_llm_reply(raw_reply)

        tmp_seen_actions = set(seen_actions)
        final_answer, extra_actions, tmp_seen_actions = validate_and_clean(reply, tmp_seen_actions)
        if final_answer:
            try:
                final_answer = normalize_stoich(final_answer)
            except ValueError:
                final_answer = None

        candidates = parse_candidate_stoichiometries(reply)
        rejection_message = None

        if final_answer and extra_actions:
            rejection_message = (
                "Reminder: do not give the Final Answer until all requested Actions are resolved."
            )

        elif has_candidate_header(reply) and not candidates:
            rejection_message = (
                "Your Candidates section could not be parsed. "
                "Use numbered candidate lines such as:\n"
                "1) A2B2C2D2E2F1  support=0.98  support_evidence=...\n"
                "or:\n"
                "1. A2B2C2D2E2F1  support=0.98  support_evidence=...\n"
                "Each candidate must be a complete stoichiometry with all target subunits."
            )

        elif final_answer and not candidates and not extra_actions:
            rejection_message = (
                "Your final response must include a Candidates section with numbered candidate lines "
                "before the Final Answer."
            )

        elif candidates:
            ok, problems = validate_candidate_set(
                candidates,
                subunit_ids,
                require_unique=True
            )
            if not ok:
                rejection_message = (
                    "Your Candidates section is invalid.\n"
                    + "\n".join(f"- {p}" for p in problems)
                    + "\nPlease restate the Candidates section using only COMPLETE unique stoichiometries. "
                      "Partial scaffolds may be mentioned in Thought or support_evidence, but not as candidate lines."
                )
            elif min_total_candidates > 0 and len(candidates) < min_total_candidates:
                rejection_message = (
                    f"You must provide at least {min_total_candidates} ranked COMPLETE and UNIQUE candidates."
                )
            elif required_candidates:
                missing_required = [c for c in required_candidates if c not in candidates]
                if missing_required:
                    rejection_message = (
                        "Your candidate list is incomplete. "
                        "The following required baseline candidate(s) must be explicitly considered: "
                        f"{missing_required}. "
                        "Please restate the Candidates section including them, unless directly contradicted by strong evidence."
                    )
            elif scaffold_completion_candidates and min_completion_candidates > 0:
                present = [c for c in candidates if c in set(scaffold_completion_candidates)]
                if len(present) < min_completion_candidates:
                    rejection_message = (
                        f"The shared scaffold {partial_scaffold} is incomplete, so you must consider "
                        f"at least {min_completion_candidates} plausible full-complex completions. "
                        f"Examples for this target: {', '.join(scaffold_completion_candidates[:6])}."
                    )
            elif symmetry_preserving_candidates:
                present_sym = [c for c in candidates if c in set(symmetry_preserving_candidates)]
                if len(present_sym) < 1:
                    rejection_message = (
                        "You must include at least one symmetry-preserving completion because the partial scaffold "
                        f"suggests a dominant repeated symmetry. Examples: {', '.join(symmetry_preserving_candidates[:4])}."
                    )

        if rejection_message is None and final_answer:
            counts = parse_stoich(final_answer)
            missing = [s for s in subunit_ids if s not in counts]
            if missing:
                rejection_message = (
                    f"Reminder: the Final Answer must explicitly include all subunits {subunit_ids}. Missing: {missing}."
                )
            elif len(templates) == 1 and single_subunit_symbol and final_answer == f"{single_subunit_symbol}1":
                rejection_message = (
                    f"Reminder: {single_subunit_symbol}1 is invalid for a single-subunit system."
                )
            elif allowed_final_answers is not None and final_answer not in allowed_final_answers:
                rejection_message = (
                    "Correction required: your Final Answer must be one of the AF3-allowed stoichiometries only. "
                    f"Allowed: {', '.join(sorted(allowed_final_answers))}"
                )

        if rejection_message is not None:
            append_safe_message(messages, role, rejection_message)
            final_answer = None
            continue

        seen_actions = tmp_seen_actions
        append_safe_message(messages, "assistant", reply)
        trace.append({"turn": turn, "messages": copy.deepcopy(messages)})

        if extra_actions:
            pubmed_search_actions = []
            other_actions = []
            reply_lines = []

            for full_action, tool, arg in extra_actions:
                if tool.lower().startswith("pubmed.search"):
                    pubmed_search_actions.append((full_action, tool, arg))
                else:
                    other_actions.append((full_action, tool, arg))

            for full_action, tool, arg in other_actions:
                obs = "(No cached observation available)"
                if arg != "N/A" and len(arg) > 0:
                    if tool.lower().startswith("pubmed.readabstract"):
                        if arg not in allowed_citation_urls:
                            obs = (
                                "On-demand literature policy: this URL is not one of the template-associated "
                                "citations supplied in the evidence. Choose a listed citation URL."
                            )
                        elif arg in abstract_urls_read:
                            obs = "This abstract was already read earlier; use the previous Observation."
                        elif abstract_reads_used >= MAX_ABSTRACT_READS:
                            obs = (
                                f"On-demand abstract budget exhausted "
                                f"({abstract_reads_used}/{MAX_ABSTRACT_READS}). Continue with existing evidence."
                            )
                        else:
                            abstract_urls_read.add(arg)
                            abstract_reads_used += 1
                            print(
                                f"[literature] on-demand abstract "
                                f"{abstract_reads_used}/{MAX_ABSTRACT_READS}: {arg}"
                            )
                            try:
                                obs = get_or_generate_literature_summary(arg, dc)
                            except Exception as e:
                                obs = f"Literature summary generation failed with fixed {SUMMARY_MODEL_KEY}: {e}"
                    elif tool.lower().startswith("rcsb.getpdb"):
                        obs = dc.summarize_rcsb(arg)
                    elif tool.lower().startswith("uniprot.get"):
                        obs = dc.get_uniprot_observation(arg)

                reply_lines.append(f"{full_action}\nObservation: {obs}")

            # Keep literature retrieval controlled: no free-form PubMed search in B/C.
            for full_action, tool, arg in pubmed_search_actions:
                obs = (
                    "PubMed.search is disabled in this on-demand variant. "
                    "Use PubMed.readAbstract with a citation URL already shown in the template evidence."
                )
                reply_lines.append(f"{full_action}\nObservation: {obs}")

            if reply_lines:
                append_safe_message(messages, "user", "\n".join(reply_lines))

            trace.append({"turn": turn, "messages": copy.deepcopy(messages)})

        if final_answer:
            break

    if not final_answer:
        if allowed_final_answers:
            final_answer = sorted(allowed_final_answers)[0]
            append_safe_message(
                messages,
                "assistant",
                f"Final Answer (default AF3-allowed fallback): {final_answer}"
            )
            trace.append({"turn": max_turns + 1, "messages": copy.deepcopy(messages)})
            print(f"[literature] total on-demand abstracts read: {abstract_reads_used}/{MAX_ABSTRACT_READS}")
            return messages, trace, final_answer

        stoich_parts = []
        for subunit in subunit_ids:
            hits = templates[subunit]
            if hits:
                top1 = hits[0]
                copies = top1.get("copies")
                if copies and str(copies).isdigit():
                    stoich_parts.append(f"{subunit}{copies}")
                else:
                    stoich_parts.append(f"{subunit}1")
            else:
                stoich_parts.append(f"{subunit}1")

        final_answer = "".join(stoich_parts)
        if len(templates) == 1 and single_subunit_symbol and final_answer == f"{single_subunit_symbol}1":
            final_answer = f"{single_subunit_symbol}2"
        append_safe_message(messages, "assistant", f"Final Answer (default): {final_answer}")
        trace.append({"turn": max_turns + 1, "messages": copy.deepcopy(messages)})

    print(f"[literature] total on-demand abstracts read: {abstract_reads_used}/{MAX_ABSTRACT_READS}")
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
    max_turns_p1: int = 15,
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

            shared_template_obs = build_shared_template_observation_from_summary(shared_template_summary)
            if shared_template_obs:
                if messages_p1 and messages_p1[-1]["role"] == "user":
                    messages_p1[-1]["content"] += (
                        "\n\nAdditional shared-template observation:\n" + shared_template_obs
                    )
                else:
                    messages_p1.append({
                        "role": "user",
                        "content": "Additional shared-template observation:\n" + shared_template_obs
                    })

            if len(templates) >= 8:
                baseline = simplest_stoich(list(templates.keys()))
                consensus_msg = (
                    f"Observation: This complex contains {len(templates)} distinct subunits. "
                    f"Start from the simplest baseline stoichiometry {baseline}. "
                    "Increase multiplicity only when strongly supported by shared templates, literature, or strong TB evidence."
                )
            else:
                consensus_stoich, per_sub_counts = compute_consensus_stoichiometry_by_subunit(templates)
                baseline = simplest_stoich(list(templates.keys()))
                consensus_msg = (
                    "Observation: Based on significant per-subunit templates (E-value ≤ 1), the most common copy numbers are: "
                    + ", ".join(f"{su}={cnt}" for su, cnt in per_sub_counts.items())
                    + f". The per-subunit consensus pattern is {consensus_stoich}. "
                    + ("For single-subunit targets, copies=1 evidence is ignored and the minimum is 2. " if len(templates) == 1 else "")
                    + "This pattern must be considered as one TB candidate. "
                    + (f"The simplest stoichiometry {baseline} must also be considered. " if len(templates) >= 2 else "")
                    + "Shared multi-subunit templates should receive extra weight, but you must still refine the list using literature."
                )

            if messages_p1 and messages_p1[-1]["role"] == "user":
                messages_p1[-1]["content"] += (
                    "\n\nAdditional template-based observation:\n" + consensus_msg
                )
            else:
                messages_p1.append({
                    "role": "user",
                    "content": "Additional template-based observation:\n" + consensus_msg
                })

            # Defensive cleanup in case an older caller/cache supplied mid-chat
            # system messages. This does not alter the initial system instruction.
            messages_p1 = sanitize_chat_messages(messages_p1)

            msgs1, trace1, answer1 = run_agent(
                llm, messages_p1, templates, shared_templates_for_agent, dc,
                read_template=True,
                max_turns=max_turns_p1,
                required_candidates=required_baseline_candidates,
                partial_scaffold=partial_scaffold,
                scaffold_completion_candidates=scaffold_completion_candidates,
                min_completion_candidates=2 if scaffold_completion_candidates and len(scaffold_completion_candidates) >= 2 else 0,
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
    shared_templates_for_agent = data.get("shared_templates_for_agent", [])
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

    required_baseline_candidates = get_required_baseline_candidates(
        templates=templates,
        shared_template_summary=shared_template_summary,
        subunit_consensus_summary=subunit_consensus_summary,
    )

    partial_scaffold, scaffold_completion_candidates, scaffold_copy_options = generate_partial_scaffold_completions(
            templates=templates,
            shared_template_summary=shared_template_summary,
            subunit_consensus_summary=subunit_consensus_summary,
            max_copy=3,
            max_total_candidates=12,
        )

    symmetry_info = classify_symmetry_completion_candidates(
        partial_scaffold,
        scaffold_completion_candidates,
        sort_subunits(list(templates.keys())),
    )

    min_total_candidates = 5
    min_completion_candidates = min(2, len(scaffold_completion_candidates)) if scaffold_completion_candidates else 0

    sys_msg = build_system_message_from_json(templates)
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
        )
    }

    return {
        "messages": [sys_msg, user_msg],
        "templates": templates,
        "shared_templates_for_agent": shared_templates_for_agent,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default="Qwen3.6-27B", choices=MODEL_MAP.keys())
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max_abstract_reads", type=int, default=10, help="Maximum unique template-associated abstracts the agent may read on demand.")
    parser.add_argument(
        "--casp17_seq_file",
        default="casp17.HT1.seq.txt",
        help=(
            "Local cache/path for the original CASP17 HT1 sequence release. "
            "If missing, D downloads it once from the official Prediction Center URL. "
            "Default: casp17.HT1.seq.txt"
        ),
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
    messages = bundle["messages"]
    templates = bundle["templates"]
    shared_templates_for_agent = bundle["shared_templates_for_agent"]

    # Load the benchmark backbone first. B/C do not precompute literature.
    # If the benchmark LLM later requests a template-associated abstract:
    #   cached summary -> reuse it
    #   missing summary -> fixed Qwen3-4B generates and caches it on demand
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
