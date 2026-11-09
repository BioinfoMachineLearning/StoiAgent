#!/usr/bin/env python3
"""Stage-2 fixed-pool stoichiometry reranking, independently per LLM.

By default Stage-2 results now live next to Stage-1 under the same
raw/<TARGET>/stoichiometry/<TARGET>_subunit/Agent/<MODEL>/ hierarchy.
Two separate S2 variants are created for noaf3 and af3.  --outdir is
used for aggregate summary CSVs. Set --output-layout central for legacy layout.

Input:
  results/raw/<TARGET>/stoichiometry/<TARGET>_subunit/Agent/<MODEL>/<VARIANT>/
      *_phase1_topall_k15.out       (Stage-1 ranked list)
  results/raw/<TARGET>/stoichiometry/<TARGET>_subunit/AlphaFold3/<target_lower>/
      stoichiometry_results.csv     (may contain candidates outside Stage-1 Top-K)
  Local CASP17 sequence-release FASTA (same biological input used by Stage 1).

No candidate is created, deleted, or replaced. The model returns a Phase-1-style
numbered candidate ranking, a Final Answer, and free-form Support Evidence.
Python still writes JSON metadata for resumability; the LLM is NOT asked for JSON.
Modes 'noaf3' and 'af3' differ only in AF3 evidence supplied to the model.
This revision restores the score-only baseline. For a target containing one
protein type (A1/A2/A3/etc.), the only AF3 input is max_ranking_score.
Heteromeric targets retain the baseline max/average/num_models prompt.
Stage-1 support and biological information remain available to the same LLM.
After saving the rankings, the same model receives an explanation-only follow-up.
The initial ranking prompts and settings are unchanged. --explain-only adds
explanations to matching saved predictions without generating rankings.
AF3 rows outside each model's Stage-1 Top-K are intentionally never displayed.

Dependencies for --run: torch, transformers, accelerate; for Mistral:
  mistral-common; appropriate Transformers with multimodal and Mistral support.
--prepare-only and --evaluate-only use only the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "Qwen3.6-27B": ("Qwen/Qwen3.6-27B", "multimodal"),
    "Gemma-3-27B-it": ("google/gemma-3-27b-it", "multimodal"),
    "Mistral-Small-3.2-24B": ("mistralai/Mistral-Small-3.2-24B-Instruct-2506", "mistral3"),
    "OLMo-2-32B-Instruct": ("allenai/OLMo-2-0325-32B-Instruct", "causal"),
}
STOICH_RE = re.compile(r"^(?:[A-Z]\d+)+$")
CAND_LINE_RE = re.compile(
    r"^\s*(?P<rank>\d+)\s*[.):\-]\s*"
    r"(?P<sto>(?:[A-Za-z]\d+)+)"
    r"(?:\s+.*?\bavg_conf\s*=\s*(?P<conf>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?))?",
    re.I,
)
DEFAULT_AF3_TEMPLATE = (
    "{root}/{target}/stoichiometry/{target}_subunit/AlphaFold3/"
    "{target_lower}/stoichiometry_results.csv"
)
SYSTEM = """You are a scientific assistant reranking existing protein-complex stoichiometry hypotheses.
Your job is to ORDER the supplied candidate stoichiometries, NOT generate hypotheses.

HARD CONSTRAINTS:
1. Return EVERY supplied candidate exactly once, without additions, omissions, or changes.
2. Return a plain-text Candidates section: one numbered stoichiometry per line, from most to least plausible.
3. Return Final Answer: <stoichiometry> immediately after the candidate list.
4. Final Answer MUST equal the first-ranked candidate.
5. You may then write a Support Evidence section explaining the decision.
6. Do not return JSON, Markdown fences, new candidates, or additional numbered lists.
7. Do not browse or request external evidence. Do not assume access to evidence not shown.
8. Write the complete ranked list and Final Answer BEFORE any extended explanation.

SCIENTIFIC INTERPRETATION:
- Weigh the supplied target description, sequences, and Stage-1 support information.
- The Stage-1 confidence/support is a heuristic, NOT a calibrated probability.
- Do not automatically select the Stage-1 first candidate or the simplest stoichiometry.
- The only valid output is a permutation of the input candidates; consider all of them.
"""
AF3_RULES = """ALPHAFOLD3 EVIDENCE RULES:
- AF3 max_ranking_score and avg_ranking_score measure predicted-structure
  confidence, not the probability that a biological stoichiometry is correct.
- Compare both maximum and average scores. A high maximum with a much lower
  average may indicate that confidence varies among the generated models.
- Consider num_models when comparing maximum or average scores; scores based
  on different numbers of predictions are not necessarily directly comparable.
- Missing AF3 scores mean unavailable structural evidence, NOT a score of zero
  or negative biological evidence, including when a complex is too large to model.

EXPLICIT ASSEMBLY-SIZE INTERPRETATION RULES:
- Compare AF3 scores cautiously when candidates differ in total residue count,
  number of chains, or copy numbers (for example, A1B1 versus A2B2). A larger
  assembly introduces additional relative chain arrangements and potential
  interfaces, which can reduce global confidence even when the larger
  stoichiometry is biologically correct.
- A higher AF3 ranking score for a smaller complex is NOT, by itself, evidence
  that the smaller stoichiometry is biologically correct. Do not automatically
  prefer the smaller candidate merely because its AF3 score is higher.
- Do not automatically prefer a larger candidate merely because it has more
  copies, greater symmetry, or more apparently occupied binding sites.
- Look for target-specific biological evidence supporting or contradicting the
  additional subunit copies and interactions required by a larger candidate.
  If only global AF3 scores are supplied, do NOT invent interface-level scores,
  contacts, clashes, or structural observations that have not been provided.
- Do NOT apply any fixed correction, bonus, penalty, or numerical normalization
  of AF3 scores by sequence length, total residues, number of chains, or copies.
  Instead, assess size-related uncertainty when interpreting the raw scores.
- Neither the highest AF3 score nor the highest Stage-1 support must win.
  If the structural scores and biological evidence disagree, briefly explain
  the tradeoff and any uncertainty behind the final ordering.
"""
NO_AF3_RULES = """EVIDENCE RULES:
- Rank using the target description, sequences, and Stage-1 support information.
- Stage-1 rankings and support values are fallible and are not probabilities.
- Do not automatically favor the highest Stage-1 support or simplest stoichiometry.
- Do not assume knowledge of AlphaFold3 results or other unseen evidence.
"""


def norm_stoic(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).upper()


def read_truth(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Truth CSV not found: {path}")
    out: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        if not {"phase1_target", "stoichiometry"}.issubset(rd.fieldnames or []):
            raise ValueError("--pairs requires phase1_target,stoichiometry headers")
        for row in rd:
            t, s = row["phase1_target"].strip(), norm_stoic(row["stoichiometry"])
            if t and s:
                if t in out and out[t] != s:
                    raise ValueError(f"Conflicting truth for {t}")
                out[t] = s
    return out


def parse_stage1_ranked(path: Path) -> list[dict]:
    """Only numbered lines from a Phase-1 topall/Top-5 output are candidates."""
    rows, seen = [], set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = CAND_LINE_RE.match(line)
        if not m:
            continue
        sto = norm_stoic(m.group("sto"))
        if not STOICH_RE.fullmatch(sto) or sto in seen:
            continue
        seen.add(sto)
        c = m.group("conf")
        score = float(c) if c is not None else None
        rows.append({"stoic": sto, "stage1_rank": int(m.group("rank")), "stage1_support": score})
    rows.sort(key=lambda x: x["stage1_rank"])
    for i, row in enumerate(rows, 1):
        row["stage1_rank"] = i
    return rows


def find_stage1_file(root: Path, target: str, model: str, variant: str) -> Path | None:
    """Pick topall file from exactly one subunit directory; prefer largest k."""
    basedir = root / target / "stoichiometry"
    dirs = sorted(p for p in basedir.glob(f"*/Agent/{model}/{variant}") if p.is_dir())
    if len(dirs) > 1:
        raise ValueError(f"Ambiguous Stage-1 directories for {target}/{model}/{variant}: {dirs}")
    if not dirs:
        return None
    d = dirs[0]
    files = sorted(p for p in d.glob("*phase1_topall*.out") if p.is_file())
    if not files:
        files = sorted(p for p in d.glob("*phase1_top5*.out") if p.is_file())
    if not files:
        return None
    def key(p: Path):
        m = re.search(r"_k(\d+)", p.stem)
        return (int(m.group(1)) if m else -1, p.name)
    return max(files, key=key)


def read_af3_csv(path: Path) -> dict[str, dict]:
    """Index by 'stoic'; never inject other rows into candidate list."""
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        required = {"stoic", "max_ranking_score", "avg_ranking_score", "num_models"}
        if not required.issubset(rd.fieldnames or []):
            raise ValueError(f"AF3 CSV {path} lacks columns: {sorted(required - set(rd.fieldnames or []))}")
        for row in rd:
            sto = norm_stoic(row["stoic"])
            if not STOICH_RE.fullmatch(sto):
                warnings.warn(f"Ignoring invalid AF3 stoichiometry {sto!r} in {path}")
                continue
            if sto in out:
                raise ValueError(f"Duplicate AF3 stoichiometry {sto} in {path}")
            vals = {}
            for key in ["max_ranking_score", "avg_ranking_score"]:
                try:
                    val = float(row[key])
                    vals[key] = val if math.isfinite(val) else None
                except (TypeError, ValueError):
                    vals[key] = None
            try:
                cnt = int(float(row["num_models"]))
            except (ValueError, TypeError):
                cnt = None
            vals["num_models"] = cnt
            vals["af3_available"] = (
                all(vals[k] is not None for k in ["max_ranking_score", "avg_ranking_score"])
                and cnt is not None and cnt > 0
            )
            out[sto] = vals
    return out


def annotate_af3(candidates: list[dict], af3_index: dict[str, dict], csv_exists: bool) -> list[dict]:
    out = []
    for cand in candidates:
        r = dict(cand)
        rec = af3_index.get(cand["stoic"])
        if rec is not None and rec["af3_available"]:
            r.update(rec)
            r["af3_status"] = "scored"
        else:
            r["af3_available"] = False
            r["max_ranking_score"] = None
            r["avg_ranking_score"] = None
            r["num_models"] = None
            r["af3_status"] = (
                "csv_missing" if not csv_exists else
                "candidate_not_in_csv" if rec is None else "incomplete_scores"
            )
        out.append(r)
    return out


def parse_casp17_catalog(path: Path) -> dict[str, list[dict]]:
    """Parse the same CASP17 HT1 description/source/sequence records used in v9."""
    records: dict[str, list[dict]] = defaultdict(list)
    header = None
    parts: list[str] = []
    def flush():
        nonlocal header, parts
        if header is None:
            return
        m0 = re.match(r"^>?\s*([HT]\d{4})\b", header, re.I)
        if not m0:
            header, parts = None, []
            return
        h = header.lstrip(">").strip().rstrip(";|").strip()
        m = re.match(r"^(?P<target>[HT]\d{4})\s+(?P<body>.*?),\s*(?P<length>\d+)\s+residues\s*$", h, re.I)
        if not m:
            raise ValueError(f"Bad CASP17 sequence header: {header}")
        target = m.group("target")
        body = m.group("body").strip()
        body = re.sub(r",\s*subunit\s+\d+\s*$", "", body, flags=re.I)
        desc, org = (part.strip() for part in body.rsplit(",", 1)) if "," in body else (body, "")
        seq = re.sub(r"\s+", "", "".join(parts)).upper()
        if not seq:
            raise ValueError(f"Empty CASP17 sequence: {target}")
        if len(seq) != int(m.group("length")):
            raise ValueError(f"CASP17 length mismatch for {target}: {len(seq)} != {m.group('length')}")
        records[target].append({"description": desc, "organism": org, "sequence": seq})
        header, parts = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            flush()
            header, parts = line, []
        elif header is not None:
            parts.append(line)
        else:
            raise ValueError("FASTA sequence before header")
    flush()
    return dict(records)


def target_info_from_catalog(target: str, candidates: list[dict], catalog: dict) -> dict:
    source = target

    if source not in catalog:
        # Remove trailing 'o': T0217o -> T0217
        base = re.sub(r"^(T\d{4})o$", r"\1", target)

        if base in catalog:
            source = base
        else:
            # Map H0xxx/H2xxx -> H1xxx (also supports T targets)
            m = re.fullmatch(r"([HT])[02](\d{3})", base)
            if m and f"{m.group(1)}1{m.group(2)}" in catalog:
                source = f"{m.group(1)}1{m.group(2)}"
            else:
                raise ValueError(f"CASP17 sequence information missing for {target}")

    subunits = []
    for cand in candidates:
        for su in re.findall(r"([A-Z])\d+", cand["stoic"]):
            if su not in subunits:
                subunits.append(su)

    subunits.sort()
    recs = catalog[source]

    if len(subunits) != len(recs):
        raise ValueError(
            f"Subunit-count mismatch for {target}: candidate symbols {subunits}; "
            f"CASP17 sequence records {len(recs)}. Refusing to guess a mapping."
        )

    return {
        "description": next((r["description"] for r in recs if r["description"]), ""),
        "organism": next((r["organism"] for r in recs if r["organism"]), ""),
        "subunits": {su: r["sequence"] for su, r in zip(subunits, recs)},
    }



def load_candidate_evidence(path: Path | None) -> dict[tuple[str, str, str], str]:
    """Optional Stage-1 rationale CSV: target,model,stoichiometry,support_evidence."""
    if path is None:
        return {}
    d: dict[tuple[str, str, str], str] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rd = csv.DictReader(fh)
        req = {"target", "model", "stoichiometry", "support_evidence"}
        if not req.issubset(rd.fieldnames or []):
            raise ValueError(f"Evidence CSV needs {sorted(req)}")
        for row in rd:
            key = (row["target"].strip(), row["model"].strip(), norm_stoic(row["stoichiometry"]))
            reason = (row["support_evidence"] or "").strip()
            if key in d and d[key] != reason:
                raise ValueError(f"Conflicting evidence for {key}")
            d[key] = reason
    return d


def load_target_context(context_dir: Path | None, target: str) -> str:
    """Optional preexisting target-specific Stage-1 evidence text; identical in both arms."""
    if context_dir is None:
        return ""
    path = context_dir / f"{target}.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if len(text) > 20000:
        raise ValueError(f"Context file too long (>20k chars): {path}")
    return text


PROMPT_VERSION = "baseline_homo_max_v1"
# Retain the baseline size-interpretation guidance, changing only the AF3 inputs
# and score-specific instructions for targets containing one protein type.
HOMO_AF3_RULES = """ALPHAFOLD3 EVIDENCE RULES:
- Only max_ranking_score is supplied as AF3 evidence for this target.
- This is predicted-structure confidence, not the probability that a biological
  stoichiometry is correct. Weigh it with the supplied Stage-1 support and
  biological evidence; it does not automatically determine the final ranking.
- Missing AF3 scores mean unavailable structural evidence, NOT a score of zero
  or negative biological evidence, including when a complex is too large to model.
- Do not infer unprovided AF3 statistics, structural observations, or consistency
  across predictions from the maximum score alone.
""" + "\nEXPLICIT ASSEMBLY-SIZE INTERPRETATION RULES:" + AF3_RULES.split(
    "EXPLICIT ASSEMBLY-SIZE INTERPRETATION RULES:", 1)[1]


def is_homomeric_pool(candidates: list[dict]) -> bool:
    """One protein type across the pool, including a possible monomer candidate."""
    symbols = set()
    for row in candidates:
        sto = norm_stoic(row["stoic"])
        if not STOICH_RE.fullmatch(sto):
            return False
        symbols.update(re.findall(r"([A-Z])\d+", sto))
    return len(symbols) == 1


def make_messages(
    target_info: dict, candidates: list[dict], condition: str,
    target_context: str = "",
) -> list[dict]:
    """Only condition-specific change is AF3 info/rules; candidate set is fixed."""
    if condition not in {"noaf3", "af3"}:
        raise ValueError(condition)
    homo = is_homomeric_pool(candidates)
    rules = HOMO_AF3_RULES if homo else AF3_RULES
    s = SYSTEM + "\n" + (rules if condition == "af3" else NO_AF3_RULES)
    lines = [
        "Target information (do not infer a benchmark ID):",
        "Description: " + (target_info.get("description") or "not available"),
        "Organism/source: " + (target_info.get("organism") or "not available"),
        "Subunit sequences (each letter identifies a distinct protein):",
    ]
    for su, sequence in sorted(target_info["subunits"].items()):
        lines.append(f"{su} ({len(sequence)} residues): {sequence}")
    if target_context:
        lines += ["", "Stage-1 research/template evidence (provided, do not browse):", target_context]
    lines += ["", "Stage-1 candidate pool (ONLY these candidates are allowed):"]
    for row in candidates:
        msg = (
            f'- {row["stoic"]}: Stage-1 rank={row["stage1_rank"]}; '
            f'Stage-1 support={row["stage1_support"] if row["stage1_support"] is not None else "N/A"}'
        )
        if row.get("support_evidence"):
            msg += f'; Stage-1 support evidence={row["support_evidence"]}'
        if condition == "af3":
            if homo:
                if row["af3_available"]:
                    msg += f'; AF3 max_ranking_score={row["max_ranking_score"]:.6f}'
                else:
                    msg += "; AF3 max_ranking_score=N/A"
                    msg += "; AF3 unavailable (reason not a negative biological signal)"
            elif row["af3_available"]:
                msg += (
                    f'; AF3 max_ranking_score={row["max_ranking_score"]:.6f}'
                    f'; AF3 avg_ranking_score={row["avg_ranking_score"]:.6f}'
                    f'; AF3 num_models={row["num_models"]}'
                )
            else:
                msg += "; AF3 max_ranking_score=N/A; AF3 avg_ranking_score=N/A; AF3 num_models=N/A"
                msg += "; AF3 unavailable (reason not a negative biological signal)"
        lines.append(msg)
    allowed = [r["stoic"] for r in candidates]
    lines += [
        "", "Use precisely this PLAIN-TEXT output format (not JSON):",
        "Candidates:",
        *[f"{idx}. {sto}" for idx, sto in enumerate(allowed, 1)],
        f"Final Answer: {allowed[0]}",
        "Support Evidence: A concise explanation of the most important evidence and uncertainty.",
        "The example order above is only the ORIGINAL Stage-1 order; independently decide how to re-rank it.",
        "Return every input candidate exactly once; list the ranking and Final Answer FIRST.",
        "Do not generate another numbered list in Support Evidence; prose there is free-form.",
        "Your ranking is valid even if the explanation is short; prioritize completing the ranking.",
    ]
    return [{"role": "system", "content": s}, {"role": "user", "content": "\n".join(lines)}]


def extract_text_ranking(raw: str) -> dict:
    """Parse Phase-1-style text; do not demand a complete/closed rationale.

    Only numbered lines *inside* Candidates: (and before Final Answer:) are
    considered candidates. Biological evidence may mention other stoichiometries
    or numbered items and must not contaminate the ranked candidate pool.
    """
    if not isinstance(raw, str):
        raise ValueError("Model response must be text")
    rows: list[tuple[int, str]] = []
    in_candidates = False
    final_answer = None
    rationale_lines = []
    in_evidence = False
    for line in raw.splitlines():
        if re.fullmatch(r"\s*Candidates\s*:\s*", line, flags=re.I):
            if in_candidates or rows:
                raise ValueError("Multiple Candidates sections")
            if final_answer is not None:
                raise ValueError("Candidates section follows Final Answer")
            in_candidates = True
            continue
        m_final = re.match(r"^\s*Final\s+Answer\s*:\s*(.*?)\s*$", line, flags=re.I)
        if m_final:
            if final_answer is not None:
                raise ValueError("Multiple Final Answer lines")
            final_answer = norm_stoic(m_final.group(1))
            in_candidates = False
            in_evidence = False
            continue
        m_evidence = re.match(r"^\s*(?:Support\s+Evidence|Rationale)\s*:\s*(.*)$", line, flags=re.I)
        if m_evidence:
            if final_answer is None:
                raise ValueError("Support Evidence must follow Final Answer")
            in_candidates = False
            in_evidence = True
            rationale_lines.append(m_evidence.group(1))
            continue
        if in_candidates:
            if not line.strip():
                continue
            m_cand = re.fullmatch(
                r"\s*(\d+)\s*[.)]\s*((?:[A-Za-z]\d+)+)\s*", line
            )
            if m_cand is None:
                raise ValueError(f"Invalid candidate line: {line[:150]!r}")
            rows.append((int(m_cand.group(1)), norm_stoic(m_cand.group(2))))
        elif in_evidence:
            rationale_lines.append(line)
    if not rows:
        raise ValueError("No numbered Candidates section found")
    if [i for i, _ in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Candidate numbering must be consecutive starting at 1")
    if final_answer is None or not STOICH_RE.fullmatch(final_answer):
        raise ValueError("Missing or invalid Final Answer")
    return {
        "ranking": [sto for _, sto in rows],
        "final_answer": final_answer,
        "rationale": "\n".join(rationale_lines).strip(),
    }


def validate_ranking(raw: str | dict, candidates: list[dict]) -> dict:
    """Validate the text output or previously cached structured metadata.

    Support Evidence is optional for structural validation. A long explanation
    truncated after a complete ranking + Final Answer must not invalidate the
    otherwise complete permutation.
    """
    obj = raw if isinstance(raw, dict) else extract_text_ranking(raw)
    if not isinstance(obj, dict):
        raise ValueError("Output must contain a ranked candidate list")
    for key in ["ranking", "final_answer"]:
        if key not in obj:
            raise ValueError(f"Missing {key!r}")
    ranking = obj["ranking"]
    expected = [x["stoic"] for x in candidates]
    if not isinstance(ranking, list) or not all(isinstance(x, str) for x in ranking):
        raise ValueError("Ranking must be a list of stoichiometry strings")
    if len(ranking) != len(expected):
        raise ValueError(f"Ranking must contain exactly {len(expected)} candidates")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Duplicate candidates in output")
    if set(ranking) != set(expected):
        raise ValueError(f"Output candidate mismatch: missing={sorted(set(expected)-set(ranking))}; extra={sorted(set(ranking)-set(expected))}")
    if obj["final_answer"] != ranking[0]:
        raise ValueError("Final Answer must equal the first-ranked candidate")
    evidence = obj.get("rationale", "")
    if not isinstance(evidence, str):
        raise ValueError("Support Evidence must be text when provided")
    return {
        "ranking": ranking,
        "final_answer": ranking[0],
        "rationale": evidence.strip(),
    }


class LocalHFLLM:
    """Same four backbones/backends as the Stage-1 HFLLM, with no tool loop."""
    def __init__(self, key: str, max_memory_cpu_gib: int = 200):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
        self.torch = torch
        self.key = key
        self.model_id, self.backend = MODEL_SPECS[key]
        max_memory = {i: f"{max(1, int(torch.cuda.get_device_properties(i).total_memory / (1024**3)) - 2)}GiB"
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = f"{max_memory_cpu_gib}GiB"
        kwargs = {"device_map": "balanced", "max_memory": max_memory,
                  "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        self.tok = None
        self.processor = None
        self.mistral_tok = None
        self._Request = None
        print(f"[model] Loading {key}: {self.model_id} ({self.backend})")
        if self.backend == "causal":
            self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        elif self.backend == "multimodal":
            from transformers import AutoModelForMultimodalLM
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tok = self.processor.tokenizer
            self.model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **kwargs)
        else:
            from transformers import Mistral3ForConditionalGeneration
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            self._Request = ChatCompletionRequest
            self.mistral_tok = MistralTokenizer.from_hf_hub(self.model_id)
            self.model = Mistral3ForConditionalGeneration.from_pretrained(self.model_id, **kwargs)

    def _device(self):
        import torch
        d = getattr(self.model, "device", None)
        if d is not None and str(d) != "meta":
            return d
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    def _encode(self, messages: list[dict]):
        if self.backend == "multimodal":
            mm = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
            kwargs = {"enable_thinking": False, "preserve_thinking": False} if self.key == "Qwen3.6-27B" else {}
            enc = self.processor.apply_chat_template(mm, add_generation_prompt=True,
                                                      tokenize=True, return_dict=True,
                                                      return_tensors="pt", **kwargs)
            return enc
        if self.backend == "mistral3":
            req = self._Request(messages=messages)
            encoded = self.mistral_tok.encode_chat_completion(req)
            input_ids = self.torch.tensor([encoded.tokens], dtype=self.torch.long)
            return {"input_ids": input_ids, "attention_mask": self.torch.ones_like(input_ids)}
        kwargs = {"enable_thinking": False, "preserve_thinking": False} if self.key == "Qwen3.6-27B" else {}
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        return self.tok(prompt, return_tensors="pt")

    def generate(self, messages: list[dict], *, max_new_tokens: int, seed: int, max_prompt_tokens: int) -> str:
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        enc = self._encode(messages)
        prompt_len = int(enc["input_ids"].shape[-1])
        if prompt_len > max_prompt_tokens:
            raise ValueError(f"Prompt tokens {prompt_len} > --max-prompt-tokens {max_prompt_tokens}; not truncating biological evidence")
        enc = enc.to(self._device()) if hasattr(enc, "to") else {k: v.to(self._device()) for k,v in enc.items()}
        with torch.inference_mode():
            ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        new_tokens = ids[0, prompt_len:]
        if self.backend == "multimodal":
            return self.processor.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if self.backend == "mistral3":
            return self.mistral_tok.decode(new_tokens.detach().cpu().tolist()).strip()
        return self.tok.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def write_csv(path: Path, data: list[dict], *, fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in data for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)


def prompt_digest(messages: list[dict], candidate_pool: list[str]) -> str:
    blob = json.dumps({"messages": messages, "candidate_pool": candidate_pool},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def create_jobs(args) -> list[dict]:
    root = args.root
    truth = read_truth(args.pairs)
    if args.targets:
        targets = [x.upper() for x in args.targets]
    elif args.pairs is not None:
        targets = sorted(truth)
    else:
        targets = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not targets:
        raise ValueError("No targets found. Provide --targets or --pairs.")
    if args.af3_csv is not None and len(targets) != 1:
        raise ValueError("--af3-csv is for exactly one --targets TARGET; use --af3-csv-template for batch jobs")
    if args.casp17_sequences is None:
        raise ValueError("--casp17-sequences is required to supply the same biological descriptions/sequences as Stage 1")
    catalog = parse_casp17_catalog(args.casp17_sequences)
    evidence = load_candidate_evidence(args.candidate_evidence_csv)
    jobs: list[dict] = []
    for model in args.models:
        for target in targets:
            source = find_stage1_file(root, target, model, args.variant)
            if source is None:
                jobs.append({"model":model,"target":target,"status":"missing_stage1", "source_file":""})
                continue
            parsed = parse_stage1_ranked(source)
            if not parsed:
                jobs.append({"model":model,"target":target,"status":"empty_stage1", "source_file":str(source)})
                continue
            candidates = [dict(row) for row in parsed[:args.k]]
            tinfo = target_info_from_catalog(target, candidates, catalog)
            if args.af3_csv is not None:
                af3_path = args.af3_csv
            else:
                af3_path = Path(args.af3_csv_template.format(
                    root=root, target=target, target_lower=target.lower(),
                    model=model, variant=args.variant
                ))
            if not af3_path.is_absolute():
                af3_path = Path.cwd() / af3_path
            if args.af3_csv is not None and not af3_path.is_file():
                raise FileNotFoundError(f"Explicit --af3-csv does not exist: {af3_path}")
            if not af3_path.is_file():
                warnings.warn(f"AF3 CSV missing for {target}: {af3_path}. Scores will be N/A.")
            af3_index = read_af3_csv(af3_path)
            candidates = annotate_af3(candidates, af3_index, af3_path.is_file())
            for r in candidates:
                r["support_evidence"] = evidence.get((target, model, r["stoic"]), "")
            extra_context = load_target_context(args.target_context_dir, target)
            for condition in args.conditions:
                messages = make_messages(tinfo, candidates, condition, extra_context)
                # Phase-1-compatible layout: store run-specific artifacts inside
                # the target's Agent/<MODEL>/S2_<VARIANT>_<CONDITION>/ folder.
                # Never touch the original Stage-1 directory.
                if getattr(args, "output_layout", "phase1") == "phase1":
                    stage2_dir = (source.parent.parent /
                                  f"{args.stage2_variant_prefix}_{condition}")
                    base_name = f"{target}_subunit"
                    prompt_path = stage2_dir / f"{base_name}_phase2_prompt.json"
                    result_path = stage2_dir / f"{base_name}_phase2_result.json"
                else:
                    stage2_dir = args.outdir / "rankings" / model / target
                    prompt_path = args.outdir / "prompts" / model / target / f"{condition}.json"
                    result_path = args.outdir / "results" / model / target / f"{condition}.json"
                prompt_hash = prompt_digest(messages, [r["stoic"] for r in candidates])
                job = {
                    "prompt_sha256":prompt_hash, "model":model, "target":target, "condition":condition, "status":"ready",
                    "source_file":str(source), "af3_csv":str(af3_path),
                    "total_af3_csv_rows":len(af3_index),
                    "stage1_pool_size":len(candidates),
                    "af3_matched_candidates":sum(int(r["af3_available"]) for r in candidates),
                    "n_af3_extra_rows_ignored":sum(int(k not in {r["stoic"] for r in candidates}) for k in af3_index),
                    "candidate_stoichiometries":";".join(r["stoic"] for r in candidates),
                    "truth":truth.get(target,""),
                    "prompt_path":str(prompt_path),"result_path":str(result_path),
                    "stage2_dir":str(stage2_dir),
                }
                dump_json(prompt_path, {
                    "target":target, "model":model, "variant":args.variant,
                    "condition":condition, "candidates":candidates,
                    "prompt_version":PROMPT_VERSION,
                    "af3_input_mode":("none" if condition == "noaf3" else
                                      "max_only" if is_homomeric_pool(candidates) else "baseline_globals"),
                    "prompt_sha256":prompt_hash, "messages":messages,
                    "source_stage1_file":str(source), "af3_csv":str(af3_path),
                })
                jobs.append(job)
    return jobs


def _trace_and_ranking_paths(args, job: dict) -> tuple[Path, Path]:
    """Readable Stage-2 files beside each target's Phase-1 files by default."""
    if getattr(args, "output_layout", "phase1") == "central" or not job.get("stage2_dir"):
        # Maintain compatibility with hand-constructed test jobs and old results.
        trace_path = (args.outdir / "traces" / job["model"] / job["target"] /
                      f"{job['target']}_subunit_{job['condition']}_phase2.trace")
        rank_path = (args.outdir / "rankings" / job["model"] / job["target"] /
                     f"{job['target']}_subunit_phase2_top{args.k}_{job['condition']}.out")
        return trace_path, rank_path
    out = Path(job["stage2_dir"])
    base = f"{job['target']}_subunit"
    return (out / f"{base}_sample01_phase2.trace",
            out / "{}_phase2_top5_k1.out".format(base))


def _all_ranking_path(args, job: dict) -> Path | None:
    if getattr(args, "output_layout", "phase1") == "central" or not job.get("stage2_dir"):
        return None
    return Path(job["stage2_dir"]) / f"{job['target']}_subunit_phase2_topall_k1.out"


def _atomic_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def _candidate_annotation(cand: dict, condition: str) -> str:
    """Report AF3 values only in the AF3 condition (never in NoAF3)."""
    parts = [f"stage1_rank={cand['stage1_rank']}"]
    score = cand.get("stage1_support")
    parts.append(f"stage1_support={score if score is not None else 'N/A'}")
    if condition == "af3":
        if is_homomeric_pool([cand]):
            if cand.get("af3_available"):
                parts.append(f"af3_max={cand.get('max_ranking_score')}")
            else:
                parts.append(f"af3=N/A ({cand.get('af3_status', 'not_available')})")
        elif cand.get("af3_available"):
            parts.extend([
                f"af3_max={cand.get('max_ranking_score')}",
                f"af3_avg={cand.get('avg_ranking_score')}",
                f"num_models={cand.get('num_models')}",
            ])
        else:
            parts.append(f"af3=N/A ({cand.get('af3_status', 'not_available')})")
    return " ".join(parts)


def write_stage2_artifacts(args, job: dict, prompt: dict, result: dict) -> None:
    """Persist full Stage-1-style turn trace and human-readable ranked .out.

    Called after EACH attempt, including failed attempts, to retain diagnostic
    material if a subsequent model call fails or the batch is interrupted.
    """
    trace_path, rank_path = _trace_and_ranking_paths(args, job)
    candidates = prompt["candidates"]
    by_sto = {r["stoic"]: r for r in candidates}
    logs = result.get("attempt_trace", [])
    if not logs:
        # Older cached Stage-2 results stored only raw_responses. We can recover
        # the original prompt and raw text, but not lost repair-turn prompts.
        for i, raw in enumerate(result.get("raw_responses", []), 1):
            logs.append({
                "turn": i, "messages": prompt["messages"] if i == 1 else [],
                "raw_response": raw, "validation": "unknown (legacy cached result)",
                "error": "Historical repair-turn history was not recorded; rerun --force for a complete trace" if i > 1 else "",
            })
    # Match the Phase-1 trace format: each turn lists the complete message
    # history, including assistant responses and repair instructions.
    trace = []
    for log in logs:
        trace.append(f"\n--- Turn {log['turn']} ---")
        for msg in log.get("messages", []):
            trace.append(f"{msg['role'].upper()}: {msg['content']}\n")
        response = log.get("raw_response")
        trace.append(f"ASSISTANT: {response if response is not None else '[generation failed: no response]'}\n")
        trace.append(f"VALIDATION: {log.get('validation', 'unknown')}")
        if log.get("error"):
            trace.append(f"ERROR: {log['error']}")
    trace += ["\n=== Final Stage-2 Result ===", f"Status: {result.get('status', 'unknown')}",
              f"Attempts: {result.get('attempts', len(logs))}"]
    ranking = result.get("ranking", []) if result.get("status") == "ok" else []
    if ranking:
        trace.append("Final Answer: " + result["final_answer"])
        trace.append("\nRanked candidates (Stage-2 order):")
        for idx, sto in enumerate(ranking, 1):
            trace.append(f"{idx}. {sto:<12} {_candidate_annotation(by_sto[sto], job['condition'])}")
        trace.append("\nSupport Evidence: " + result.get("rationale", ""))
    else:
        trace.append("Final Answer: UNAVAILABLE")
        if result.get("last_error"):
            trace.append("Last Error: " + result["last_error"])
    _atomic_text(trace_path, "\n".join(trace).rstrip() + "\n")

    # Exactly the Phase-1 basename pattern: *_phase2_top5_k1.out and
    # *_phase2_topall_k1.out. Stage-2 generates ONE permutation, not 15
    # stochastic support scores; its original Stage-1 scores are labeled so.
    base = f"{job['target']}_subunit"
    top_title = f"=== Top-5 Predictions for {base} (phase2, first 1 sample) ==="
    all_title = f"=== All Predictions for {base} (phase2, first 1 sample) ==="
    def ranked_text(title: str, ordered: list[str]) -> str:
        lines = [title, ""]
        if ordered:
            for idx, sto in enumerate(ordered, 1):
                lines.append(f"{idx}. {sto:<12} {_candidate_annotation(by_sto[sto], job['condition'])}")
            lines += ["", "Final Answer: " + result["final_answer"],
                      "Support Evidence: " + result.get("rationale", "")]
        else:
            lines += ["RESULT: STAGE2_NOT_VALIDATED",
                      "Reason: " + result.get("last_error", "not finished")]
        return "\n".join(lines).rstrip() + "\n"
    _atomic_text(rank_path, ranked_text(top_title, ranking[:5]))
    all_path = _all_ranking_path(args, job)
    if all_path is not None:
        _atomic_text(all_path, ranked_text(all_title, ranking))


def restore_cached_artifacts(args, job: dict, result: dict) -> None:
    """Materialize absent traces for legacy cached results without rerunning LLM."""
    trace_path, rank_path = _trace_and_ranking_paths(args, job)
    all_path = _all_ranking_path(args, job)
    if (trace_path.is_file() and rank_path.is_file() and
            (all_path is None or all_path.is_file())):
        return
    prompt = json.loads(Path(job["prompt_path"]).read_text(encoding="utf-8"))
    write_stage2_artifacts(args, job, prompt, result)


def import_legacy_results(args, jobs: list[dict]) -> int:
    """Safely reuse old central-layout results without rerunning 27B models.

    Only import successful JSON with the exact same condition, candidate pool,
    and prompt hash. Existing newer files are never overwritten.
    """
    if getattr(args, "output_layout", "phase1") != "phase1":
        return 0
    roots = [args.outdir]
    if getattr(args, "legacy_outdir", None) is not None:
        roots.insert(0, args.legacy_outdir)
    copied = 0
    for job in jobs:
        if job.get("status") != "ready":
            continue
        destination = Path(job["result_path"])
        if destination.exists():
            continue
        prompt = json.loads(Path(job["prompt_path"]).read_text(encoding="utf-8"))
        expected_pool = [r["stoic"] for r in prompt["candidates"]]
        for root in roots:
            old_path = (root / "results" / job["model"] / job["target"] /
                        f"{job['condition']}.json")
            if not old_path.is_file():
                continue
            try:
                result = json.loads(old_path.read_text(encoding="utf-8"))
                validate_ranking(result, prompt["candidates"])
                if (result.get("status") == "ok" and
                    result.get("condition") == job["condition"] and
                    result.get("candidate_pool") == expected_pool and
                    result.get("prompt_sha256") == job["prompt_sha256"]):
                    dump_json(destination, result)
                    write_stage2_artifacts(args, job, prompt, result)
                    print(f"[imported] {old_path} -> {destination}")
                    copied += 1
                    break
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
    return copied


def add_reranking_explanation(args, llm, job: dict, prompt: dict, result: dict) -> None:
    """Follow up on a saved ranking; only its rationale may be updated."""
    if result.get("status") != "ok" or (result.get("explanation_status") == "ok" and not args.force_explanations):
        return
    fixed = validate_ranking(result, prompt["candidates"])
    # Use the saved order even for cached results whose original reply is absent.
    saved_reply = "Candidates:\n" + "\n".join(
        f"{i}. {sto}" for i, sto in enumerate(fixed["ranking"], 1)
    ) + "\nFinal Answer: " + fixed["final_answer"]
    saved_reply += "\nSupport Evidence: " + result.get("original_rationale", fixed["rationale"])
    messages = prompt["messages"] + [
        {"role": "assistant", "content": saved_reply},
        {"role": "user", "content": (
            "Please give a more detailed explanation for each candidate's position "
            "in the ranking above, including candidates whose position did not change "
            "from Stage 1. The saved ranking and Final Answer are fixed. "
            "For this follow-up, return only the explanation, without repeating or "
            "changing the ranking or Final Answer."
        )},
    ]
    log = {"turn": len(result.get("attempt_trace", [])) + 1,
           "kind": "explanation", "messages": messages, "raw_response": None,
           "validation": "generation_pending", "error": "",
           "fixed_ranking": list(fixed["ranking"]), "fixed_final_answer": fixed["final_answer"]}
    result.setdefault("attempt_trace", []).append(log)
    result["explanation_status"] = "in_progress"
    result["explanation_attempts"] = result.get("explanation_attempts", 0) + 1
    # Ranking has already been persisted before this additional generation.
    # Never use this reply to modify the saved ranking, final answer, or ranking status.
    dump_json(Path(job["result_path"]), result)
    try:
        reply = llm.generate(messages, max_new_tokens=args.explanation_max_new_tokens,
                             seed=args.seed, max_prompt_tokens=args.max_prompt_tokens)
        log["raw_response"] = reply
        explanation = re.sub(r"^\s*Support Evidence\s*:\s*", "", reply,
                             count=1, flags=re.I).strip()
        if not explanation or re.search(r"^\s*(?:Candidates|Final Answer)\s*:",
                                        explanation, re.I | re.M):
            raise ValueError("Explanation must be nonempty prose without a replacement ranking")
        # Never parse the follow-up as a ranking or merge its fields into result.
        result.setdefault("original_rationale", fixed["rationale"])
        result["followup_explanation"] = explanation
        result["rationale"] = explanation
        result["explanation_status"] = "ok"
        result.pop("explanation_error", None)
        log["validation"] = "explanation only; saved ranking unchanged"
    except Exception as exc:
        result["explanation_status"] = "failed"
        result["explanation_error"] = f"{type(exc).__name__}: {exc}"
        log["validation"] = "failed; saved ranking and rationale retained"
        log["error"] = result["explanation_error"]
        print(f"[explanation] {job['target']}/{job['condition']}: {exc}")
    write_stage2_artifacts(args, job, prompt, result)
    dump_json(Path(job["result_path"]), result)



def run_jobs(args, jobs: list[dict]) -> None:
    for model in args.models:
        model_jobs = [j for j in jobs if j.get("status") == "ready" and j.get("model") == model]
        if not model_jobs:
            continue
        pending = []
        explanation_jobs = []
        for j in model_jobs:
            output = Path(j["result_path"])
            if output.is_file() and not args.force:
                try:
                    old = json.loads(output.read_text(encoding="utf-8"))
                    prompt = json.loads(Path(j["prompt_path"]).read_text(encoding="utf-8"))
                    if (old.get("status") == "ok"
                        and old.get("candidate_pool") == [r["stoic"] for r in prompt["candidates"]]
                        and old.get("condition") == j["condition"]
                        and old.get("prompt_sha256") == j["prompt_sha256"]):
                        validate_ranking(old, prompt["candidates"])
                        restore_cached_artifacts(args, j, old)
                        if old.get("explanation_status") != "ok" or args.force_explanations:
                            explanation_jobs.append((j, prompt, old))
                        print(f"[resume] {model}/{j['target']}/{j['condition']}")
                        continue
                except (OSError, ValueError, KeyError):
                    pass
            if args.explain_only:
                print(f"[explain-only skip] {model}/{j['target']}/{j['condition']}: no valid saved prediction matching the current prompt; ranking not regenerated")
                continue
            pending.append(j)
        if not pending and not explanation_jobs:
            continue
        llm = LocalHFLLM(model, max_memory_cpu_gib=args.cpu_gib)
        for i, job in enumerate(pending):
            prompt = json.loads(Path(job["prompt_path"]).read_text(encoding="utf-8"))
            candidates, messages = prompt["candidates"], prompt["messages"]
            result = {
                "target":job["target"],"model":model,"variant":args.variant,
                "condition":job["condition"],"candidate_pool":[r["stoic"] for r in candidates],
                "stage1_ranking":[r["stoic"] for r in candidates],
                "af3_scored_candidates":job["af3_matched_candidates"],
                "af3_csv":job["af3_csv"], "prompt_sha256":job["prompt_sha256"],
                "raw_responses":[], "attempt_trace":[], "status":"in_progress",
            }
            for attempt in range(args.max_retries + 1):
                reply = None
                used_messages = [dict(m) for m in messages]
                attempt_log = {"turn": attempt + 1, "messages": used_messages,
                               "raw_response": None, "validation": "generation_pending", "error": ""}
                # Persist BEFORE generation as well, so OOM / process termination
                # still leaves an initial prompt and candidate evidence on disk.
                result["attempt_trace"].append(attempt_log)
                result["attempts"] = attempt + 1
                write_stage2_artifacts(args, job, prompt, result)
                dump_json(Path(job["result_path"]), result)
                try:
                    reply = llm.generate(messages, max_new_tokens=args.max_new_tokens,
                                         seed=args.seed, max_prompt_tokens=args.max_prompt_tokens)
                    result["raw_responses"].append(reply)
                    attempt_log["raw_response"] = reply
                    parsed = validate_ranking(reply, candidates)
                    result.update(parsed)
                    result["status"] = "ok"
                    attempt_log["validation"] = "ok"
                    result.pop("last_error", None)
                    break
                except Exception as e:  # includes OOM, decoding, validation
                    result["last_error"] = f"{type(e).__name__}: {e}"
                    attempt_log["error"] = result["last_error"]
                    attempt_log["validation"] = "failed"
                    if attempt < args.max_retries and isinstance(e, ValueError) and reply is not None:
                        messages = messages + [
                            {"role":"assistant", "content": reply},
                            {"role":"user", "content": (
                                f"Your previous response is invalid: {e}. "
                                "Use plain text with Candidates: followed by all numbered candidates "
                                "and Final Answer: matching the first candidate. "
                                "Write Support Evidence only AFTER the full ranking and Final Answer. "
                                "Do not output JSON."
                            )},
                        ]
                    else:
                        result["status"] = "failed"
                        break
                finally:
                    write_stage2_artifacts(args, job, prompt, result)
                    dump_json(Path(job["result_path"]), result)
            if result.get("status") == "ok":
                explanation_jobs.append((job, prompt, result))
            print(f"[{i+1}/{len(pending)}] {model}/{job['target']}/{job['condition']}: {result['status']} {result.get('final_answer', '')} {result.get('last_error', '')}")
        # Finish all ranking calls first, so extra explanation calls cannot
        # affect subsequent ranking generations for this loaded model.
        for job, prompt, result in explanation_jobs:
            add_reranking_explanation(args, llm, job, prompt, result)
        del llm
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def evaluate(args, jobs: list[dict]) -> None:
    """Expected-denominator scoring; never treat failed Stage-2 run as correct."""
    detailed = []
    # If --evaluate-only, regenerate manifest from current Stage1 and AF3 files.
    for job in jobs:
        if job.get("status") != "ready":
            continue
        path = Path(job["result_path"])
        row = dict(job)
        s1 = job["candidate_stoichiometries"].split(";")
        truth = job.get("truth", "")
        row.update({"stage1_top1":s1[0], "stage1_hit1":int(truth == s1[0]) if truth else "",
                    "stage1_hit3":int(truth in s1[:3]) if truth else "",
                    "stage1_hit5":int(truth in s1[:5]) if truth else "",
                    "stage1_pool_has_truth":int(truth in s1) if truth else ""})
        if path.is_file():
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                parsed = validate_ranking(result, [{"stoic":c} for c in s1]) if result.get("status") == "ok" else None
                if (parsed is None or result.get("condition") != job["condition"]
                        or result.get("prompt_sha256") != job["prompt_sha256"]):
                    raise ValueError("Missing/invalid result or condition mismatch")
                ranking = parsed["ranking"]
                row.update({"status":"ok","stage2_top1":ranking[0],
                            "stage2_ranking":";".join(ranking),"rationale":parsed["rationale"],
                            "stage2_hit1":int(truth == ranking[0]) if truth else "",
                            "stage2_hit3":int(truth in ranking[:3]) if truth else "",
                            "stage2_hit5":int(truth in ranking[:5]) if truth else ""})
            except (OSError, ValueError, json.JSONDecodeError):
                row["status"] = "failed"
        else:
            row["status"] = "not_run"
        detailed.append(row)
    write_csv(args.outdir / "stage2_per_target.csv", detailed)
    if not args.pairs:
        print("[eval] No --pairs supplied; predictions saved without accuracy metrics.")
        return
    summary = []
    for model in args.models:
        for condition in args.conditions:
            rows = [r for r in detailed if r["model"] == model and r["condition"] == condition]
            expected = len([t for t in (args.targets or sorted(read_truth(args.pairs)))])
            ok = [r for r in rows if r["status"] == "ok"]
            s1 = [r for r in rows if r.get("stage1_pool_has_truth") != ""]
            hits = {f"top{k}":sum(r.get(f"stage2_hit{k}", 0) == 1 for r in ok) for k in [1,3,5]}
            summary.append({
                "model":model,"condition":condition,"n_expected":expected,"n_stage1_loaded":len(rows),
                "n_stage2_success":len(ok),"n_stage2_failed_or_missing":expected-len(ok),
                "stage1_top1_hits":sum(r.get("stage1_hit1",0)==1 for r in s1),
                "stage1_top3_hits":sum(r.get("stage1_hit3",0)==1 for r in s1),
                "stage1_top5_hits":sum(r.get("stage1_hit5",0)==1 for r in s1),
                "stage1_pool_truth_hits":sum(r.get("stage1_pool_has_truth",0)==1 for r in s1),
                "top1_hits":hits["top1"],"top3_hits":hits["top3"],"top5_hits":hits["top5"],
                "top1_acc_expected":round(hits["top1"]/expected,6) if expected else "",
                "top3_acc_expected":round(hits["top3"]/expected,6) if expected else "",
                "top5_acc_expected":round(hits["top5"]/expected,6) if expected else "",
                "top1_acc_generated":round(hits["top1"]/len(ok),6) if ok else "",
                "conditional_top1_given_s1_truth_in_pool":(
                    round(sum(r.get("stage2_hit1",0)==1 for r in ok if r.get("stage1_pool_has_truth")==1) /
                          sum(r.get("stage1_pool_has_truth")==1 for r in ok),6)
                    if any(r.get("stage1_pool_has_truth")==1 for r in ok) else ""),
            })
    write_csv(args.outdir/"stage2_accuracy_summary.csv", summary)
    for row in summary:
        print(f"{row['model']:27} {row['condition']:6}  "
              f"S1={row['stage1_top1_hits']}/{row['n_expected']}  "
              f"S2 Top1={row['top1_hits']}/{row['n_expected']}  "
              f"Top3={row['top3_hits']}/{row['n_expected']}  "
              f"Top5={row['top5_hits']}/{row['n_expected']}  "
              f"success={row['n_stage2_success']}/{row['n_expected']}")
    # AF3-complete subset: compare both Stage-2 conditions on precisely the
    # same candidate pool and target set, with no structural-score missingness.
    complete_summary = []
    for model in args.models:
        candidate_groups = defaultdict(dict)
        for r in detailed:
            if r["model"] == model:
                candidate_groups[r["target"]][r["condition"]] = r
        both = []
        for target, per_condition in candidate_groups.items():
            a, b = per_condition.get("af3"), per_condition.get("noaf3")
            if not a or not b or a["status"] != "ok" or b["status"] != "ok":
                continue
            if int(a["af3_matched_candidates"]) != int(a["stage1_pool_size"]):
                continue
            both.append((a,b))
        n = len(both)
        for k in [1, 3, 5]:
            complete_summary.append({
                "model": model, "k":k,
                "n_paired_stage2_af3_complete_targets":n,
                "s1_hits":sum(int(a[f"stage1_hit{k}"]) for a,b in both),
                "noaf3_hits":sum(int(b[f"stage2_hit{k}"]) for a,b in both),
                "af3_hits":sum(int(a[f"stage2_hit{k}"]) for a,b in both),
                "noaf3_acc":round(sum(int(b[f"stage2_hit{k}"]) for a,b in both)/n,6) if n else "",
                "af3_acc":round(sum(int(a[f"stage2_hit{k}"]) for a,b in both)/n,6) if n else "",
            })
    write_csv(args.outdir/"stage2_af3_complete_subset_summary.csv", complete_summary)
    # Paired AF3 ablation: only valid outputs in both conditions for same target/model.
    by_key = {(r["model"],r["target"],r["condition"]):r for r in detailed}
    paired = []
    if {"af3", "noaf3"}.issubset(args.conditions):
        for model in args.models:
            comparisons=[]
            for target in (args.targets or sorted(read_truth(args.pairs))):
                a = by_key.get((model,target,"af3"))
                b = by_key.get((model,target,"noaf3"))
                if not a or not b or a["status"]!="ok" or b["status"]!="ok":
                    continue
                if not a.get("truth"):
                    continue
                r={"model":model,"target":target,"truth":a["truth"],
                   "af3_scored_candidates":a["af3_matched_candidates"]}
                for k in [1,3,5]:
                    a_hit=int(a[f"stage2_hit{k}"])
                    b_hit=int(b[f"stage2_hit{k}"])
                    r[f"af3_hit{k}"]=a_hit
                    r[f"noaf3_hit{k}"]=b_hit
                    r[f"top{k}_change"]="rescue" if a_hit>b_hit else "break" if a_hit<b_hit else "unchanged"
                paired.append(r)
                comparisons.append(r)
            print(f"[paired] {model}: {len(comparisons)} targets; "
                  f"Top1 rescues={sum(x['top1_change']=='rescue' for x in comparisons)}, "
                  f"breaks={sum(x['top1_change']=='break' for x in comparisons)}")
    write_csv(args.outdir/"stage2_af3_vs_noaf3_paired.csv", paired)


def af3_only_baselines(args, jobs: list[dict]) -> None:
    """Score-only upper-information controls, strictly on AF3-complete candidate pools.

    Do not score partially observed candidate pools: ranking a subset would be
    biased against hypotheses that could not feasibly be modeled.
    """
    one_per_pair = {(j["model"], j["target"]): j for j in jobs
                    if j.get("status") == "ready" and j.get("condition") == "af3"}
    # Can still run this if the requested LLM condition is only noaf3: build
    # annotated candidates from its prompt metadata (annotations stored in both).
    if not one_per_pair:
        one_per_pair = {(j["model"], j["target"]): j for j in jobs
                        if j.get("status") == "ready" and j.get("condition") == "noaf3"}
    rows = []
    for (model, target), job in sorted(one_per_pair.items()):
        prompt = json.loads(Path(job["prompt_path"]).read_text(encoding="utf-8"))
        cs = prompt["candidates"]
        all_scored = all(c["af3_available"] for c in cs)
        truth = job.get("truth", "")
        for metric, field in [("af3_max", "max_ranking_score"), ("af3_avg", "avg_ranking_score")]:
            row = {"model": model, "target": target, "metric": metric,
                   "truth": truth, "all_stage1_candidates_have_af3": int(all_scored),
                   "n_stage1_candidates":len(cs),
                   "n_scored_candidates":sum(c["af3_available"] for c in cs),
                   "stage1_top1":cs[0]["stoic"],
                   "stage1_hit1":int(cs[0]["stoic"] == truth) if truth else ""}
            if all_scored:
                ordered = sorted(cs, key=lambda c: (-c[field], c["stage1_rank"]))
                ranking = [c["stoic"] for c in ordered]
                row["ranking"]=";".join(ranking)
                for k in [1,3,5]:
                    row[f"hit{k}"] = int(truth in ranking[:k]) if truth else ""
            rows.append(row)
    write_csv(args.outdir/"af3_score_only_per_target.csv", rows)
    if not args.pairs:
        return
    summary = []
    for model in args.models:
        for metric in ["af3_max", "af3_avg"]:
            subset = [r for r in rows if r["model"]==model and r["metric"]==metric
                      and r["all_stage1_candidates_have_af3"]]
            n = len(subset)
            summary.append({"model":model,"metric":metric,"n_af3_complete_targets":n,
                            "top1_hits":sum(r["hit1"] for r in subset),
                            "top3_hits":sum(r["hit3"] for r in subset),
                            "top5_hits":sum(r["hit5"] for r in subset),
                            "stage1_top1_on_same_subset":sum(r["stage1_hit1"] for r in subset),
                            "top1_acc":round(sum(r["hit1"] for r in subset)/n,6) if n else "",
                            "top3_acc":round(sum(r["hit3"] for r in subset)/n,6) if n else "",
                            "top5_acc":round(sum(r["hit5"] for r in subset)/n,6) if n else ""})
    write_csv(args.outdir/"af3_score_only_complete_subset_summary.csv", summary)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="results/raw directory")
    p.add_argument("--pairs", type=Path, help="pairs.csv with phase1_target and stoichiometry (for evaluation only)")
    p.add_argument("--targets", nargs="+", help="Optional explicit target IDs")
    p.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    p.add_argument("--variant", default="S1_full_stoiagent_v9")
    p.add_argument("--k", type=int, default=5, help="Stage-1 candidates per model (default 5)")
    p.add_argument("--casp17-sequences", type=Path, required=True, help="Local CASP17 HT1 sequence FASTA used in Stage 1")
    p.add_argument("--af3-csv", type=Path, help="Exact AF3 CSV for a SINGLE selected target")
    p.add_argument("--af3-csv-template", default=DEFAULT_AF3_TEMPLATE,
                   help="Template with {root}, {target}, {target_lower}, {model}, {variant}; used for batches")
    p.add_argument("--candidate-evidence-csv", type=Path,
                   help="Optional target,model,stoichiometry,support_evidence table")
    p.add_argument("--target-context-dir", type=Path,
                   help="Optional TARGET.txt with Stage-1 research evidence (shared by both arms)")
    p.add_argument("--conditions", nargs="+", choices=["af3","noaf3"],
                   default=["noaf3","af3"])
    p.add_argument("--outdir", type=Path, default=Path("stage2_baseline_homo_max"),
                   help="Central summary CSV directory; Phase-1 layout saves per-target outputs under --root")
    p.add_argument("--output-layout", choices=["phase1", "central"], default="phase1",
                   help="phase1 (default): raw/TARGET/.../Agent/MODEL/S2_VARIANT_CONDITION; central: legacy traces/rankings layout")
    p.add_argument("--stage2-variant-prefix", default=None,
                   help="Per-target Stage-2 directory name prefix (default: S2 variant with _baseline_homo_max_kK suffix)")
    p.add_argument("--legacy-outdir", type=Path,
                   help="Optional older central Stage-2 directory to reuse valid cached results; no model rerun")
    p.add_argument("--prepare-only", action="store_true", help="Write prompts/manifest, no inference")
    p.add_argument("--evaluate-only", action="store_true", help="Re-evaluate existing results")
    p.add_argument("--force", action="store_true", help="Regenerate existing successful outputs")
    p.add_argument("--explain-only", action="store_true",
                   help="Explain matching saved predictions only; never run or rerun ranking")
    p.add_argument("--explanation-max-new-tokens", type=int, default=1536,
                   help="Output-token budget for the explanation follow-up only; ranking budget is unchanged")
    p.add_argument("--force-explanations", action="store_true",
                   help="Refresh follow-up explanations without forcing saved rankings to regenerate")
    p.add_argument("--max-retries", type=int, default=10, help="Format repair attempts (default 10)")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-prompt-tokens", type=int, default=16000)
    p.add_argument("--cpu-gib", type=int, default=200, help="Max HF device_map CPU GiB")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)
    if args.stage2_variant_prefix is None:
        args.stage2_variant_prefix = ("S2_" + args.variant[3:]
                                      if args.variant.startswith("S1_")
                                      else "S2_" + args.variant) + f"_baseline_homo_max_k{args.k}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.stage2_variant_prefix):
        p.error("--stage2-variant-prefix must contain only letters, digits, dot, dash, underscore")
    if args.stage2_variant_prefix == args.variant:
        p.error("Stage-2 output name must not overwrite Stage-1 variant")
    if args.k < 1 or args.max_retries < 0:
        p.error("--k >= 1 and --max-retries >= 0 required")
    if args.prepare_only and args.evaluate_only:
        p.error("--prepare-only and --evaluate-only are mutually exclusive")
    if args.explain_only and (args.force or args.prepare_only or args.evaluate_only):
        p.error("--explain-only cannot be combined with --force, --prepare-only, or --evaluate-only")
    if args.force_explanations and (args.prepare_only or args.evaluate_only):
        p.error("--force-explanations requires inference mode or --explain-only")
    if args.explanation_max_new_tokens < 1:
        p.error("--explanation-max-new-tokens must be >= 1")
    if not args.root.is_dir():
        p.error("--root is not a directory")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    jobs = create_jobs(args)
    write_csv(args.outdir/"stage2_manifest.csv", jobs)
    ready = [j for j in jobs if j.get("status")=="ready"]
    print(f"Prepared {len(ready)} Stage-2 prompts, {len(jobs)-len(ready)} missing/empty Stage-1 runs")
    print(f"AF3 condition(s): {', '.join(args.conditions)}; each includes ONLY its own Stage-1 Top-{args.k} stoichiometries")
    print(f"Total out-of-pool AF3 rows excluded from prompts: {sum(int(j['n_af3_extra_rows_ignored']) for j in ready if j['condition']=='af3')}")
    if not args.prepare_only:
        reused = import_legacy_results(args, jobs)
        if reused:
            print(f"Reused {reused} validated older Stage-2 result(s); no inference required for these.")
    if not args.prepare_only and not args.evaluate_only:
        run_jobs(args, jobs)
    if args.evaluate_only:
        # Rebuild missing human-readable artifacts from cached JSON results.
        for job in ready:
            result_path = Path(job["result_path"])
            if result_path.is_file():
                try:
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if cached.get("prompt_sha256") == job["prompt_sha256"]:
                        restore_cached_artifacts(args, job, cached)
                except (OSError, json.JSONDecodeError):
                    pass
    if not args.prepare_only:
        evaluate(args, jobs)
        af3_only_baselines(args, jobs)
    if args.output_layout == "phase1":
        print(f"Per-target Stage-2 outputs under: {args.root}/<TARGET>/stoichiometry/<SUBUNIT>/Agent/<MODEL>/{args.stage2_variant_prefix}_<CONDITION>/")
    else:
        print(f"Saved full turn traces: {args.outdir / 'traces'}")
        print(f"Saved Stage-1-style ranked outputs: {args.outdir / 'rankings'}")
    print(f"Aggregate CSV summaries: {args.outdir}")


if __name__ == "__main__":
    main()
