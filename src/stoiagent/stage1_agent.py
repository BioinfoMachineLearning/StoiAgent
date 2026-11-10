#!/usr/bin/env python3
"""StoiAgent Stage 1: shared templates with a compact subunit-template fallback.

Run from this repository with a prepared target JSON and CASP17 sequence file.
"""
import re
import copy
import time
import os
import argparse
import random
import json
import gzip
import string
import urllib.request
from itertools import product
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, set_seed

try:
    from transformers import AutoModelForMultimodalLM
except ImportError:
    AutoModelForMultimodalLM = None

try:
    from transformers import Mistral3ForConditionalGeneration
except ImportError:
    Mistral3ForConditionalGeneration = None

from .agent_helper import *
from .casp17_three_level_common import ResearchToolExecutor, resolve_target_information_cutoff
from . import D_core_three_level as core

BASE_SEED = 1234

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

def build_max_memory(cpu_gib=200, reserve_gib=2):
    """Build the max_memory budget used with device_map='balanced' from the GPUs actually present."""
    mem = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        usable_gib = max(1, int(total_gib - reserve_gib))
        mem[i] = f"{usable_gib}GiB"
    mem["cpu"] = f"{cpu_gib}GiB"
    return mem


max_memory = build_max_memory()


def reseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


SEEDS_30 = [BASE_SEED + i for i in range(30)]


# backend:
#   causal     -> AutoTokenizer + AutoModelForCausalLM
#   multimodal -> AutoProcessor + AutoModelForMultimodalLM (text-only usage here)
#   mistral3   -> mistral-common tokenizer + Mistral3ForConditionalGeneration
MODEL_SPECS = {
    # =====================================================
    # Main pre-CASP17 benchmark panel
    # =====================================================
    # These benchmark backbones were publicly released before the CASP17
    # target-release period.  They are intentionally kept in a similar
    # ~24B-32B scale range and come from four different model families.

    "Qwen3.6-27B": {
        "id": "Qwen/Qwen3.6-27B",
        "backend": "multimodal",
        # Keep direct/instruction behavior so the generation budget is spent
        # on the visible agent response rather than hidden reasoning.
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    },

    "Gemma-3-27B-it": {
        "id": "google/gemma-3-27b-it",
        "backend": "multimodal",
    },

    "OLMo-2-32B-Instruct": {
        "id": "allenai/OLMo-2-0325-32B-Instruct",
        "backend": "causal",
    },

    "Mistral-Small-3.2-24B": {
        "id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "backend": "mistral3",
    },

    # =====================================================
    # Fixed literature-summary model -- NOT a benchmark model
    # =====================================================
    "Qwen3-4B": {
        "id": "Qwen/Qwen3-4B-Instruct-2507",
        "backend": "causal",
    },
}

BENCHMARK_MODEL_KEYS = (
    "Qwen3.6-27B",
    "Gemma-3-27B-it",
    "OLMo-2-32B-Instruct",
    "Mistral-Small-3.2-24B",
)

# Only these four models are exposed through --model.
# Qwen3-4B remains accessible internally through MODEL_SPECS for literature
# summarization, but cannot accidentally be selected as a benchmark backbone.
MODEL_MAP = {
    key: MODEL_SPECS[key]["id"]
    for key in BENCHMARK_MODEL_KEYS
}


# =========================================================
# Literature-summary configuration
# =========================================================
# All benchmark backbones consume the SAME paper summaries generated by the
# fixed 4B summary agent below. Missing summaries are generated with Qwen3-4B
# BEFORE the benchmark model is loaded, then the 4B model is unloaded so it
# does not consume VRAM during the benchmark itself.
SUMMARY_MODEL_KEY = "Qwen3-4B"  # Qwen/Qwen3-4B-Instruct-2507
FROZEN_SUMMARY_CACHE_DIR = None


def _article_cache_key(source: str) -> str:
    """Match DiscoveryCache's observed URL -> directory convention."""
    return "".join(
        ch if (ch.isalnum() or ch in ".-_") else "_"
        for ch in str(source)
    )


def load_frozen_literature_summary(source: str) -> str:
    """
    Load a precomputed Qwen3-4B summary and NEVER call the benchmark LLM.

    Expected layout, matching the existing cache:
      article_cache/Qwen3-4B/<sanitized-source>/summary.txt

    A missing summary is a hard error by design: silently regenerating it with
    the benchmark backbone itself would confound the cross-model comparison.
    """
    if not FROZEN_SUMMARY_CACHE_DIR:
        raise RuntimeError("FROZEN_SUMMARY_CACHE_DIR has not been initialized.")

    source = str(source).strip()
    source_variants = [source]
    if source.endswith("/"):
        source_variants.append(source.rstrip("/"))
    else:
        source_variants.append(source + "/")

    # Be tolerant to an http/https normalization difference in old caches.
    if source.startswith("https://"):
        source_variants.append("http://" + source[len("https://"):])
    elif source.startswith("http://"):
        source_variants.append("https://" + source[len("http://"):])

    checked = []
    seen = set()
    for candidate_source in source_variants:
        if candidate_source in seen:
            continue
        seen.add(candidate_source)
        summary_path = os.path.join(
            FROZEN_SUMMARY_CACHE_DIR,
            _article_cache_key(candidate_source),
            "summary.txt",
        )
        checked.append(summary_path)
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read().strip()
            if summary:
                return summary
            raise RuntimeError(f"Frozen summary is empty: {summary_path}")

    raise FileNotFoundError(
        "Missing literature summary from the fixed summary-agent cache. "
        f"Summary agent: {SUMMARY_MODEL_KEY}. Source: {source}\n"
        "Checked:\n  - " + "\n  - ".join(checked)
    )


def _summary_is_cached(source: str) -> bool:
    try:
        load_frozen_literature_summary(source)
        return True
    except (FileNotFoundError, RuntimeError):
        return False


def _summarize_literature_with_hfllm(citation, text, summary_llm, max_words=250):
    """
    Summarize a fetched literature record using the project's HFLLM wrapper.

    DiscoveryCache.summarize_text() was written for an older Transformers
    pipeline wrapper and calls ``llm.pipe(...)``.  The current benchmark uses
    HFLLM.generate(), so calling DiscoveryCache.summarize_text() with HFLLM
    raises ``AttributeError: 'HFLLM' object has no attribute 'pipe'``.

    Keep literature summarization independent of DiscoveryCache's model wrapper:
    DiscoveryCache fetches/caches the source; the fixed Qwen3-4B HFLLM produces
    the summary; this script persists the summary in the existing cache layout.
    """
    body = str(text or "").strip()
    if not body:
        raise RuntimeError(f"No literature text was fetched for {citation}")

    # PubMed abstracts are normally much shorter than this.  The cap protects
    # against an unexpectedly large fetched page while leaving ample room for
    # the biologically relevant abstract text.
    max_chars = 24000
    if len(body) > max_chars:
        body = body[:max_chars]

    messages = [
        {
            "role": "system",
            "content": (
                "Summarize biomedical literature for protein-complex stoichiometry analysis. "
                "Preserve explicit evidence about oligomeric state, biological assembly, "
                "subunit composition/copy number, symmetry, interaction partners, and whether "
                "the reported complex is full-length or partial. Do not infer facts not stated "
                "in the supplied text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source: {citation}\n\n"
                f"Literature text:\n{body}\n\n"
                f"Return a concise summary of at most {int(max_words)} words, focusing on "
                "evidence relevant to protein-complex stoichiometry."
            ),
        },
    ]

    result = summary_llm.generate(
        messages,
        max_new_tokens=512,
        do_sample=False,
    )
    return (result or "").replace("<end_of_turn>", "").strip()


def ensure_literature_summaries(
    templates,
    shared_templates_for_agent,
    dc,
):
    """
    Generate any missing literature summaries with the fixed Qwen3-4B summary
    agent, then unload that model before the benchmark backbone is loaded.

    Existing cached Qwen3-4B summaries are reused unchanged.
    """
    citation_items = collect_unique_citations(templates, shared_templates_for_agent)
    missing = [item for item in citation_items if not _summary_is_cached(item["url"])]

    print(
        f"Literature summaries: total={len(citation_items)}, "
        f"cached={len(citation_items) - len(missing)}, missing={len(missing)}"
    )

    if not missing:
        print(f"All literature summaries already cached with {SUMMARY_MODEL_KEY}.")
        return

    print(
        f"Loading fixed summary agent {SUMMARY_MODEL_KEY} only to generate "
        f"{len(missing)} missing summary/s."
    )
    summary_llm = HFLLM(SUMMARY_MODEL_KEY)

    try:
        for idx, item in enumerate(missing, start=1):
            source = item["url"]
            print(f"[summary {idx}/{len(missing)}] {item['template_id']} -> {source}")
            text, citation, status = dc.fetch_and_cache(source)
            if status == "error":
                raise RuntimeError(f"Failed to fetch literature source for summary: {source}")

            summary = _summarize_literature_with_hfllm(
                citation,
                text,
                summary_llm,
                max_words=250,
            )
            if not summary:
                raise RuntimeError(f"{SUMMARY_MODEL_KEY} returned an empty summary for {citation}")

            # Persist explicitly; do not depend on DiscoveryCache's legacy
            # pipeline-based summarizer to write the summary cache.
            _write_literature_summary_cache(source, summary)
            if str(citation).strip() != str(source).strip():
                _write_literature_summary_cache(citation, summary)

            loaded = load_frozen_literature_summary(citation)
            if not loaded:
                raise RuntimeError(f"Summary cache verification failed for {citation}")
    finally:
        del summary_llm
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished Qwen3-4B summary generation; summary agent unloaded.")


# Lazily instantiated fixed literature-summary model. The benchmark backbone is
# NEVER used to create literature summaries. A cache hit does not load Qwen3-4B;
# a cache miss loads Qwen3-4B on first use and reuses it for later misses.
_SUMMARY_LLM_INSTANCE = None


def _get_summary_llm():
    global _SUMMARY_LLM_INSTANCE
    if _SUMMARY_LLM_INSTANCE is None:
        print(
            f"[summary] Cache miss detected; loading fixed summary agent "
            f"{SUMMARY_MODEL_KEY} ({MODEL_SPECS[SUMMARY_MODEL_KEY]['id']})."
        )
        _SUMMARY_LLM_INSTANCE = HFLLM(SUMMARY_MODEL_KEY)
    return _SUMMARY_LLM_INSTANCE


def _write_literature_summary_cache(source: str, summary: str):
    """Persist a Qwen3-4B summary using the same cache layout as the loader."""
    if not FROZEN_SUMMARY_CACHE_DIR:
        raise RuntimeError("FROZEN_SUMMARY_CACHE_DIR has not been initialized.")
    source = str(source).strip()
    summary_dir = os.path.join(FROZEN_SUMMARY_CACHE_DIR, _article_cache_key(source))
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary.strip() + "\\n")
    return summary_path


def get_or_generate_literature_summary(source: str, dc) -> str:
    """
    Return the fixed Qwen3-4B literature summary for ``source``.

    1) Reuse an existing cached summary when available.
    2) If missing/empty, fetch the source and generate exactly that summary with
       Qwen3-4B-Instruct-2507 (never with the benchmark backbone).
    3) Persist it to the Qwen3-4B cache and return it.

    In B/C this function is called only after the benchmark LLM explicitly asks
    to read a template-associated abstract, so missing summaries are generated
    truly on demand.
    """
    source = str(source).strip()
    try:
        return load_frozen_literature_summary(source)
    except (FileNotFoundError, RuntimeError):
        pass

    print(f"[summary] Missing Qwen3-4B summary; generating on demand: {source}")
    text, citation, status = dc.fetch_and_cache(source)
    if status == "error":
        raise RuntimeError(f"Failed to fetch literature source for summary: {source}")

    summary_llm = _get_summary_llm()
    summary = _summarize_literature_with_hfllm(
        citation,
        text,
        summary_llm,
        max_words=250,
    )
    if not summary:
        raise RuntimeError(f"{SUMMARY_MODEL_KEY} returned an empty summary for {citation}")

    # DiscoveryCache normally writes summary.txt itself. Persist explicitly as
    # well so this behavior does not depend on its internal cache implementation.
    _write_literature_summary_cache(source, summary)
    if str(citation).strip() != source:
        _write_literature_summary_cache(citation, summary)

    print(f"[summary] Cached Qwen3-4B summary for: {source}")
    return summary

class HFLLM:
    """Unified text-only inference wrapper for causal and multimodal chat checkpoints."""

    def __init__(self, model_key):
        if model_key not in MODEL_SPECS:
            raise KeyError(f"Unknown model key: {model_key}. Available: {sorted(MODEL_SPECS)}")

        self.model_key = model_key
        self.spec = MODEL_SPECS[model_key]
        self.model_id = self.spec["id"]
        self.backend = self.spec.get("backend", "causal")
        self.chat_template_kwargs = dict(self.spec.get("chat_template_kwargs", {}))

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.mistral_tokenizer = None
        self._ChatCompletionRequest = None

        print(f"Loading {self.model_key}: {self.model_id} [backend={self.backend}]")
        print(f"device_map max_memory={max_memory}")

        if self.backend == "multimodal":
            self._load_multimodal()
        elif self.backend == "mistral3":
            self._load_mistral3()
        else:
            self._load_causal()

    def _load_causal(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="balanced",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    def _load_multimodal(self):
        if AutoModelForMultimodalLM is None:
            raise RuntimeError(
                f"{self.model_key} requires AutoModelForMultimodalLM. "
                "Upgrade Transformers (your 5.18.0 installation should provide it)."
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            # Defensive fallback for processors that do not expose `.tokenizer`.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)

        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            device_map="balanced",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    def _load_mistral3(self):
        if Mistral3ForConditionalGeneration is None:
            raise RuntimeError(
                "Mistral Small 3.2 requires Mistral3ForConditionalGeneration. "
                "Upgrade Transformers first."
            )
        try:
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Mistral Small 3.2 requires mistral-common >= 1.6.2. "
                "Install it with: pip install -U 'mistral-common>=1.6.2'"
            ) from e

        self._ChatCompletionRequest = ChatCompletionRequest
        self.mistral_tokenizer = MistralTokenizer.from_hf_hub(self.model_id)
        # Keep this attribute available for any external code that expects llm.tokenizer.
        self.tokenizer = self.mistral_tokenizer
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="balanced",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    def _input_device(self):
        """Return a sensible device for input_ids when the model is sharded."""
        try:
            d = self.model.device
            if d is not None and str(d) != "meta":
                return d
        except Exception:
            pass

        device_map = getattr(self.model, "hf_device_map", None) or {}
        for d in device_map.values():
            if isinstance(d, int):
                return torch.device(f"cuda:{d}")
            if isinstance(d, str) and d.startswith("cuda"):
                return torch.device(d)

        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cpu")

    @staticmethod
    def _multimodal_text_messages(messages):
        """Convert plain chat strings to the text-block form used by MM processors."""
        out = []
        for msg in sanitize_chat_messages(messages):
            content = msg.get("content", "")
            if isinstance(content, list):
                mm_content = content
            else:
                mm_content = [{"type": "text", "text": str(content)}]
            out.append({"role": msg.get("role", "user"), "content": mm_content})
        return out

    def _render_causal_prompt(self, messages):
        safe_messages = sanitize_chat_messages(messages)
        kwargs = dict(self.chat_template_kwargs)
        return self.tokenizer.apply_chat_template(
            safe_messages,
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    def _encode_multimodal_messages(self, messages):
        mm_messages = self._multimodal_text_messages(messages)
        kwargs = dict(self.chat_template_kwargs)
        return self.processor.apply_chat_template(
            mm_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        )

    def _encode_mistral_messages(self, messages):
        safe_messages = sanitize_chat_messages(messages)

        # mistral_common serving requests must end in User/Tool unless the
        # final Assistant message is explicitly marked as a prefix.  During
        # the agent loop an intermediate LLM response can legitimately leave
        # the stored conversation ending in Assistant (for example, a Thought
        # without a recognized Action or Final Answer).  Add a local
        # continuation request for Mistral encoding/generation only.  This
        # does NOT mutate the stored messages or trace.
        if safe_messages and safe_messages[-1].get("role") == "assistant":
            safe_messages = copy.deepcopy(safe_messages)
            safe_messages.append({
                "role": "user",
                "content": (
                    "Continue the task. If evidence gathering is complete, "
                    "provide the required Candidates section and Final Answer; "
                    "otherwise request the next Action."
                ),
            })

        request = self._ChatCompletionRequest(messages=safe_messages)
        return self.mistral_tokenizer.encode_chat_completion(request)

    def get_model_max_len(self, default=8192):
        cfg = getattr(self.model, "config", None)
        cfgs = [cfg, getattr(cfg, "text_config", None) if cfg is not None else None]
        for c in cfgs:
            if c is None:
                continue
            for attr in ("max_position_embeddings", "seq_length", "n_positions"):
                v = getattr(c, attr, None)
                if isinstance(v, int) and 0 < v < 10**9:
                    return v

        v = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(v, int) and 0 < v < 10**9:
            return v
        return default

    def count_prompt_tokens(self, messages):
        if self.backend == "multimodal":
            inputs = self._encode_multimodal_messages(messages)
            return int(inputs["input_ids"].shape[-1])

        if self.backend == "mistral3":
            tokenized = self._encode_mistral_messages(messages)
            return len(tokenized.tokens)

        prompt = self._render_causal_prompt(messages)
        return int(self.tokenizer(prompt, return_tensors="pt").input_ids.shape[-1])

    @staticmethod
    def _generation_kwargs(max_new_tokens, do_sample=False, temperature=None, top_p=None):
        kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "num_beams": 1,
        }
        if do_sample:
            if temperature is not None:
                kwargs["temperature"] = float(temperature)
            if top_p is not None:
                kwargs["top_p"] = float(top_p)
        return kwargs

    def generate(
        self,
        messages,
        max_new_tokens=1024,
        do_sample=False,
        temperature=None,
        top_p=None,
    ):
        gen_kwargs = self._generation_kwargs(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        device = self._input_device()

        with torch.inference_mode():
            if self.backend == "multimodal":
                inputs = self._encode_multimodal_messages(messages)
                input_len = int(inputs["input_ids"].shape[-1])
                inputs = inputs.to(device)
                outputs = self.model.generate(**inputs, **gen_kwargs)
                new_tokens = outputs[0, input_len:]
                return self.processor.decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

            if self.backend == "mistral3":
                tokenized = self._encode_mistral_messages(messages)
                input_ids = torch.tensor(
                    [tokenized.tokens], dtype=torch.long, device=device
                )
                attention_mask = torch.ones_like(input_ids)
                input_len = input_ids.shape[-1]
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
                new_tokens = outputs[0, input_len:].detach().cpu().tolist()
                return self.mistral_tokenizer.decode(new_tokens).strip()

            prompt = self._render_causal_prompt(messages)
            enc = self.tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = int(enc["input_ids"].shape[-1])
            outputs = self.model.generate(**enc, **gen_kwargs)
            new_tokens = outputs[0, input_len:]
            return self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()


# =========================================================
# Canonical chain order
# =========================================================
def generate_chain_ids(max_ids=500):
    base = string.ascii_uppercase
    if max_ids <= len(base):
        return list(base[:max_ids])
    extended = [''.join(p) for p in product(base, repeat=2)]
    return list(base) + extended[:max_ids - len(base)]


PDB_CHAIN_IDS = generate_chain_ids(512)


def sort_subunits(subunits):
    order = {cid: i for i, cid in enumerate(PDB_CHAIN_IDS)}
    return sorted(subunits, key=lambda x: order.get(x, 10**9))


def normalize_stoich(stoich: str) -> str:
    s = stoich.replace(",", "").replace(" ", "")
    pairs = re.findall(r"([A-Z]{1,2})(\d+)", s)
    if not pairs:
        raise ValueError(f"Invalid stoichiometry: {stoich}")

    order = {cid: i for i, cid in enumerate(PDB_CHAIN_IDS)}
    pairs = sorted(pairs, key=lambda p: order.get(p[0], 10**9))
    return "".join(f"{sym}{cnt if cnt else '1'}" for sym, cnt in pairs)


def simplest_stoich(subunits):
    subs = sort_subunits(subunits)
    return "".join(f"{s}1" for s in subs)


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


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1]
    return s


# =========================================================
# Utilities
# =========================================================
def get_model_max_len(tokenizer, default=8192):
    v = getattr(tokenizer, "model_max_length", None)
    if isinstance(v, int) and 0 < v < 10**9:
        return v
    return default


def sanitize_chat_messages(messages):
    """
    Keep the initial system message as the only system message.

    Newer Qwen chat templates require `system` to appear only at the
    beginning of the conversation. Runtime observations/corrections are
    therefore converted to user messages. Consecutive user messages are
    merged to keep the dialogue structure clean.

    This also makes old cached Phase-1 conversations safe to reuse.
    """
    cleaned = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "user")
        content = str(msg.get("content", ""))

        if role == "system" and cleaned:
            role = "user"

        if cleaned and role == "user" and cleaned[-1].get("role") == "user":
            cleaned[-1]["content"] += "\n\n" + content
        else:
            cleaned.append({"role": role, "content": content})

    return cleaned


def compute_generation_budget(messages, llm, margin=2000, hard_cap=4096):
    model_max_len = llm.get_model_max_len()
    prompt_len = llm.count_prompt_tokens(messages)
    available = model_max_len - prompt_len
    safe_new = max(256, available - margin)
    return int(min(safe_new, hard_cap))


OBS_LINE_RE = re.compile(r'^\s*Observation\b', re.IGNORECASE)
PMID_LINE_RE = re.compile(r'^\s*PMID\b', re.IGNORECASE)


def filter_llm_reply(raw: str) -> str:
    kept, seen = [], set()
    for line in raw.splitlines():
        if OBS_LINE_RE.match(line):
            continue
        if PMID_LINE_RE.match(line):
            continue
        sig = line.strip().lower()
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(line.strip())
    return "\n".join(kept).strip()


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


def parse_stoich(stoich: str) -> dict:
    pairs = re.findall(r"([A-Z]{1,2})(\d+)", stoich)
    return {k: int(v) for k, v in pairs}


def build_stoich_from_counts(counts, subunit_ids):
    return "".join(f"{su}{counts[su]}" for su in sort_subunits(subunit_ids))


def parse_stoich_counts_safe(stoich):
    try:
        return parse_stoich(normalize_stoich(stoich))
    except Exception:
        return {}


def get_partial_scaffold_pattern(shared_template_summary, subunit_ids):
    """
    Pick the best shared-template pattern if it is partial (missing subunits).
    Returns normalized pattern or None.
    """
    if not shared_template_summary:
        return None

    candidates = [
        shared_template_summary.get("top_ranked_shared_pattern"),
        shared_template_summary.get("top_shared_template_pattern"),
    ]

    for pat in candidates:
        if not pat:
            continue
        counts = parse_stoich_counts_safe(pat)
        if not counts:
            continue
        missing = [su for su in sort_subunits(subunit_ids) if su not in counts]
        if missing:
            return normalize_stoich(pat)
    return None


def get_plausible_copies_for_uncovered_subunit(
    su,
    templates,
    subunit_consensus_summary=None,
    *,
    evalue_thr=1.0,
    cov_thr=0.50,
    max_copy=3,
    decisive_freq=0.75,
):
    """
    Return a small list of plausible copy numbers for an uncovered subunit.
    Conservative rule:
    - if evidence is decisive, return only dominant copy
    - if evidence is not decisive, return up to [1,2,3]
    """
    observed = []
    for h in templates.get(su, []):
        try:
            ev = float(h.get("evalue", 1e99))
        except Exception:
            ev = 1e99
        try:
            cov = float(h.get("coverage", 0.0))
        except Exception:
            cov = 0.0

        n = (
            h.get("copies", None)
            or h.get("template_chain_count", None)
            or h.get("chain_count", None)
        )
        try:
            n = int(n)
        except Exception:
            continue

        if n < 1:
            continue

        if ev <= evalue_thr and cov >= cov_thr:
            observed.append(n)

    freq = Counter(observed)

    dominant_copy = None
    dominant_freq = None
    if subunit_consensus_summary:
        per_sub = subunit_consensus_summary.get("per_subunit", {})
        info = per_sub.get(su, {})
        if info:
            try:
                dominant_copy = int(info.get("dominant_copy"))
            except Exception:
                dominant_copy = None
            try:
                dominant_freq = float(info.get("dominant_frequency"))
            except Exception:
                dominant_freq = None

            for key in ("preferred_copies", "all_observed_copies"):
                vals = info.get(key, [])
                if isinstance(vals, list):
                    for x in vals:
                        try:
                            xi = int(x)
                            if 1 <= xi <= max_copy:
                                freq[xi] += 1
                        except Exception:
                            pass

    if dominant_copy is not None and dominant_freq is not None and dominant_freq >= decisive_freq:
        return [dominant_copy]

    plausible = sorted([c for c in freq if 1 <= c <= max_copy])

    if len(plausible) <= 1:
        plausible = list(range(1, max_copy + 1))
    else:
        plausible = sorted(set(plausible + list(range(1, min(max_copy, 3) + 1))))

    return plausible[:max_copy]


def generate_partial_scaffold_completions(
    templates,
    shared_template_summary,
    subunit_consensus_summary,
    *,
    max_copy=3,
    max_total_candidates=12,
):
    """
    If there is a partial shared scaffold, generate completed full-complex candidates
    by enumerating plausible copies for uncovered subunits.
    """
    subunit_ids = sort_subunits(list(templates.keys()))
    scaffold = get_partial_scaffold_pattern(shared_template_summary, subunit_ids)
    if not scaffold:
        return None, [], {}

    base_counts = parse_stoich_counts_safe(scaffold)
    missing = [su for su in subunit_ids if su not in base_counts]
    if not missing:
        return scaffold, [], {}

    copy_options = {}
    for su in missing:
        copy_options[su] = get_plausible_copies_for_uncovered_subunit(
            su,
            templates,
            subunit_consensus_summary=subunit_consensus_summary,
            max_copy=max_copy,
        )

    completed = []
    option_lists = [copy_options[su] for su in missing]
    for combo in product(*option_lists):
        counts = dict(base_counts)
        for su, c in zip(missing, combo):
            counts[su] = c
        completed.append(build_stoich_from_counts(counts, subunit_ids))

    completed = list(dict.fromkeys(completed))[:max_total_candidates]
    return scaffold, completed, copy_options


def infer_symmetry_target_copy_from_scaffold(scaffold: str):
    """
    Infer the dominant repeated copy number from the scaffold.
    For A2B2C2D2E2, returns 2.
    For mixed scaffolds, returns the most common copy number >1 if present.
    """
    counts = parse_stoich_counts_safe(scaffold)
    if not counts:
        return None

    repeated = [c for c in counts.values() if c > 1]
    if not repeated:
        return None

    mode_copy, mode_freq = Counter(repeated).most_common(1)[0]
    return int(mode_copy)


def classify_symmetry_completion_candidates(scaffold, candidates, subunit_ids):
    """
    Mark which completed candidates preserve the dominant scaffold symmetry.
    For unresolved subunits, a candidate is symmetry-preserving if all missing
    subunits receive the same dominant scaffold copy number (typically 2).
    """
    if not scaffold or not candidates:
        return {
            "symmetry_target_copy": None,
            "missing_subunits": [],
            "symmetry_preserving_candidates": [],
            "symmetry_breaking_candidates": [],
        }

    base_counts = parse_stoich_counts_safe(scaffold)
    missing = [su for su in sort_subunits(subunit_ids) if su not in base_counts]
    target_copy = infer_symmetry_target_copy_from_scaffold(scaffold)

    preserving = []
    breaking = []
    for cand in candidates:
        counts = parse_stoich_counts_safe(cand)
        if (
            target_copy is not None
            and missing
            and all(counts.get(su) == target_copy for su in missing)
        ):
            preserving.append(cand)
        else:
            breaking.append(cand)

    return {
        "symmetry_target_copy": target_copy,
        "missing_subunits": missing,
        "symmetry_preserving_candidates": preserving,
        "symmetry_breaking_candidates": breaking,
    }


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


STOICH_RE = r"([A-Z]{1,2}\d+(?:[A-Z]{1,2}\d+)*)"


def parse_candidate_stoichiometries(reply_text):
    out = []
    for line in reply_text.splitlines():
        line = line.strip()
        m = re.search(rf"^\s*\d+\s*[\.\)]\s*{STOICH_RE}\b", line)
        if m:
            try:
                stoich = normalize_stoich(m.group(1))
            except ValueError:
                continue
            out.append(stoich)
    return out


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


def validate_stoichiometry(final_answer, subunit_ids):
    if not final_answer:
        return False, "empty or missing Final Answer"

    pairs = re.findall(r"([A-Z]{1,2})(\d+)", final_answer)
    if not pairs:
        return False, "invalid stoichiometry format"

    symbols = [k for k, _ in pairs]
    duplicate_symbols = sorted([s for s, n in Counter(symbols).items() if n > 1])

    counts = {k: int(v) for k, v in pairs}
    expected = set(subunit_ids)

    missing = [s for s in subunit_ids if s not in counts]
    extra = [s for s in counts if s not in expected]
    nonpos = [s for s, c in counts.items() if c < 1]

    problems = []
    if duplicate_symbols:
        problems.append(f"duplicated subunits {duplicate_symbols}")
    if missing:
        problems.append(f"missing subunits {missing}")
    if extra:
        problems.append(f"unexpected subunits {extra}")
    if nonpos:
        problems.append(f"nonpositive copy numbers for {nonpos}")

    if len(subunit_ids) == 1:
        single = subunit_ids[0]
        if re.fullmatch(rf"{single}1", final_answer.strip()):
            problems.append(
                f"invalid monomeric stoichiometry ({final_answer}); single-subunit complexes cannot be X1"
            )

    if problems:
        return False, "; ".join(problems)
    return True, ""


def validate_candidate_set(candidates, subunit_ids, require_unique=True):
    problems = []
    seen = {}

    for i, stoich in enumerate(candidates, start=1):
        if len(subunit_ids) == 1 and stoich.strip() == f"{subunit_ids[0]}1":
            problems.append(
                f"{i}) {stoich}: invalid monomeric candidate for single-subunit system"
            )
            continue

        valid, msg = validate_stoichiometry(stoich, subunit_ids)
        if not valid:
            problems.append(f"{i}) {stoich}: {msg}")

        if require_unique:
            if stoich in seen:
                problems.append(f"{i}) {stoich}: duplicate of candidate {seen[stoich]}")
            else:
                seen[stoich] = i

    return len(problems) == 0, problems


def violates_length_caps(stoich: str, subunit_lengths: dict) -> bool:
    """Legacy compatibility hook. Stage-1 no longer enforces a total-residue cap."""
    return False


# =========================================================
# Required baseline candidates
# =========================================================
def get_required_baseline_candidates(
    templates,
    shared_template_summary=None,
    subunit_consensus_summary=None
):
    subunit_ids = sort_subunits(list(templates.keys()))
    required = []

    native_candidate = simplest_stoich(subunit_ids)
    if len(subunit_ids) == 1 and native_candidate == f"{subunit_ids[0]}1":
        native_candidate = f"{subunit_ids[0]}2"
    required.append(native_candidate)

    if subunit_consensus_summary:
        common_candidate = subunit_consensus_summary.get("common_subunit_stoichiometry_pattern")
        if common_candidate:
            try:
                common_candidate = normalize_stoich(common_candidate)
                if common_candidate not in required:
                    required.append(common_candidate)
            except Exception:
                pass

    return required


# =========================================================
# Structured JSON -> prompt formatting
# =========================================================
def format_template_line(hit, su=None, rank=None):
    bits = []
    if rank is not None:
        bits.append(f"rank={rank}")
    bits.append(f"id={hit.get('id', 'NA')}")
    bits.append(f"sum_probs={hit.get('sum_probs', 'NA')}")
    bits.append(f"evalue={hit.get('evalue', 'NA')}")
    bits.append(f"pvalue={hit.get('pvalue', 'NA')}")
    bits.append(f"aligned_cols={hit.get('aligned_cols', 'NA')}")
    bits.append(f"coverage={hit.get('coverage', 'NA')}")
    bits.append(f"coverage_label={hit.get('coverage_label', '')}")
    bits.append(f"template_chain_count={hit.get('template_chain_count', hit.get('copies', '?'))}")
    bits.append(f"template_chain_id={hit.get('template_chain_id', '')}")
    bits.append(f"template_complex_stoichiometry={hit.get('template_complex_stoichiometry', '?')}")
    if su is not None:
        bits.append(f"aligned_to_target_subunit={su}")
    if hit.get("shared_across_subunits"):
        bits.append(f"shared_across_subunits={hit['shared_across_subunits']}")
    if hit.get("num_target_subunits_matched") is not None:
        bits.append(f"num_target_subunits_matched={hit['num_target_subunits_matched']}")
    if hit.get("citation"):
        bits.append(f"citation={hit['citation']}")
    if hit.get("citation_doi"):
        bits.append(f"citation_doi={hit['citation_doi']}")
    if hit.get("title"):
        bits.append(f"title={hit['title']}")
    if hit.get("keywords"):
        bits.append(f"keywords={hit['keywords']}")
    if hit.get("experimental_methods"):
        bits.append(f"experimental_methods={hit['experimental_methods']}")
    if hit.get("resolution") not in (None, "", "N/A"):
        bits.append(f"resolution={hit['resolution']}")
    if hit.get("entity_description"):
        bits.append(f"entity_description={hit['entity_description']}")
    if hit.get("entity_source_organism"):
        bits.append(f"entity_source_organism={hit['entity_source_organism']}")
    if hit.get("entity_host_organism"):
        bits.append(f"entity_host_organism={hit['entity_host_organism']}")
    if hit.get("entity_uniprot"):
        bits.append(f"entity_uniprot={hit['entity_uniprot']}")
    return "; ".join(str(x) for x in bits)


def build_tb_prediction_block(subunit_consensus_summary, shared_template_summary, constraint_summary):
    lines = []
    lines.append("4. Template-based (TB) prediction guidance:")

    if subunit_consensus_summary:
        top_common = subunit_consensus_summary.get("common_subunit_stoichiometry_pattern")
        if top_common:
            lines.append(f"- Common per-subunit template pattern: {top_common}")
        per_sub = subunit_consensus_summary.get("per_subunit", {})
        for su in sort_subunits(per_sub.keys()):
            info = per_sub[su]
            lines.append(
                f"  - Subunit {su}: dominant_copy={info.get('dominant_copy')}, "
                f"dominant_frequency={info.get('dominant_frequency')}, "
                f"preferred_copies={info.get('preferred_copies')}, "
                f"all_observed_copies={info.get('all_observed_copies')}, "
                f"confidence_label={info.get('confidence_label')}"
            )

    if shared_template_summary:
        top_shared = shared_template_summary.get("top_ranked_shared_pattern")
        if top_shared:
            lines.append(f"- High-priority shared-template candidate pattern: {top_shared}")
        top_shared_common = shared_template_summary.get("top_shared_template_pattern")
        if top_shared_common:
            lines.append(f"- Most frequent shared-template pattern: {top_shared_common}")

        inferred = shared_template_summary.get("inferred_shared_patterns", [])
        if inferred:
            lines.append("- Shared-template candidate patterns and evidence:")
            for item in inferred[:10]:
                lines.append(
                    f"  - pattern={item.get('pattern')}, "
                    f"supporting_template_count={item.get('supporting_template_count')}, "
                    f"max_covered_count={item.get('max_covered_count')}, "
                    f"mean_covered_count={item.get('mean_covered_count')}, "
                    f"best_coverage={item.get('best_coverage')}, "
                    f"mean_best_coverage={item.get('mean_best_coverage')}, "
                    f"best_evalue={item.get('best_evalue')}, "
                    f"copy_consistency={item.get('copy_consistency')}, "
                    f"best_is_full_complex={item.get('best_is_full_complex')}, "
                    f"best_is_major_partial_shared={item.get('best_is_major_partial_shared')}"
                )

        if shared_template_summary.get("num_shared_templates", 0) == 0:
            lines.append(
                "- No shared-template evidence is available. In this case, the most-common per-subunit template candidate must be included in the candidate list."
            )

    if constraint_summary:
        combined = constraint_summary.get("combined_constraint_view", {})
        if combined:
            lines.append(
                f"- Combined view: top_common_subunit_pattern={combined.get('top_common_subunit_pattern')}, "
                f"top_shared_template_pattern={combined.get('top_shared_template_pattern')}, "
                f"agreement={combined.get('agreement')}"
            )
            if combined.get("resolution_rule"):
                lines.append(f"- Resolution rule: {combined['resolution_rule']}")

        shared_constraints = constraint_summary.get("shared_template_stoichiometry_constraints", [])
        if shared_constraints:
            lines.append("- Strong shared-template constraints:")
            for item in shared_constraints[:10]:
                lines.append(
                    f"  - pattern={item.get('pattern')}, "
                    f"covered_subunits={item.get('covered_subunits')}, "
                    f"copy_map={item.get('copy_map')}, "
                    f"supporting_template_count={item.get('supporting_template_count')}, "
                    f"max_covered_count={item.get('max_covered_count')}, "
                    f"constraint_type={item.get('constraint_type')}"
                )

    lines.append(
        "- Do not copy any numeric support score, rank, or priority label from the input.\n"
        "- The TB summaries above are priors, not final decisions.\n"
        "- You must assign fresh Support Scores only after integrating:\n"
        "  (1) per-subunit templates,\n"
        "  (2) shared templates,\n"
        "  (3) literature evidence, and\n"
        "  (4) TB prediction guidance.\n"
        "- If a shared-template pattern does not cover all target subunits, treat it as a partial scaffold.\n"
        "- Do not assign copy number 1 to missing subunits by default.\n"
        "- Complete missing subunits using per-subunit evidence, preferred copies, subunit AF3 evidence, or other explicit support.\n"
        "- If several completions are plausible, include multiple completed candidates."
    )

    return "\n".join(lines)


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
# Citation action helpers
# =========================================================
def collect_unique_citations(templates, shared_templates_for_agent):
    items = []
    seen = set()

    for su in sort_subunits(list(templates.keys())):
        for hit in templates.get(su, []):
            url = hit.get("url") or hit.get("citation")
            tid = hit.get("id")
            if not url or url == "N/A" or not tid:
                continue

            sig = (tid, url)
            if sig in seen:
                continue
            seen.add(sig)

            items.append({
                "template_id": tid,
                "url": url,
                "source_type": "subunit",
                "covered_subunits": [su],
            })

    for rec in shared_templates_for_agent:
        rec_id = rec.get("id")
        covered = rec.get("covered_subunits", [])
        per_sub = rec.get("per_subunit_hits", {})

        chosen_url = None
        for su in sort_subunits(per_sub.keys()):
            hit = per_sub[su]
            url = hit.get("url") or hit.get("citation")
            if url and url != "N/A":
                chosen_url = url
                break

        if not rec_id or not chosen_url:
            continue

        sig = (rec_id, chosen_url)
        if sig in seen:
            continue
        seen.add(sig)

        items.append({
            "template_id": rec_id,
            "url": chosen_url,
            "source_type": "shared",
            "covered_subunits": covered,
        })

    return items


def generate_read_template_citation_actions(templates, shared_templates_for_agent):
    items = collect_unique_citations(templates, shared_templates_for_agent)
    if not items:
        return "Thought: No literature citations are available from the input templates."

    lines = []
    lines.append("Thought: I will first analyze the literature associated with the templates.")
    lines.append("Literature analysis actions:")
    for item in items:
        tid = item["template_id"]
        url = item["url"]
        if item["source_type"] == "shared":
            covered = ",".join(item.get("covered_subunits", []))
            lines.append(
                f'Action [Literature for shared template {tid} covering {covered}]: PubMed.readAbstract("{url}")'
            )
        else:
            lines.append(
                f'Action [Literature for template {tid}]: PubMed.readAbstract("{url}")'
            )
    return "\n".join(lines)


# =========================================================
# TB / shared-template observations
# =========================================================
def build_shared_template_observation_from_summary(shared_template_summary):
    if not shared_template_summary:
        return None

    shared_templates = shared_template_summary.get("shared_templates", [])
    if not shared_templates:
        return None

    lines = []
    lines.append(
        "Observation: Shared template evidence is present because some template PDBs jointly support multiple target subunits."
    )
    lines.append(
        f"There are {shared_template_summary.get('num_shared_templates', 0)} shared template PDB(s), "
        f"including {shared_template_summary.get('num_strong_shared_templates', 0)} strong shared template(s)."
    )

    top_ranked = shared_template_summary.get("top_ranked_shared_pattern")
    if top_ranked:
        lines.append(f"One high-priority shared-template candidate pattern is {top_ranked}.")

    top_common = shared_template_summary.get("top_shared_template_pattern")
    if top_common:
        lines.append(f"The most frequently implied shared-template pattern is {top_common}.")

    top_items = shared_templates[:5]
    details = []
    for item in top_items:
        part = f"{item['id']} [{item.get('shared_template_type', 'NA')}] covers {item.get('covered_subunits', [])}"
        if item.get("missing_target_subunits"):
            part += f", missing {item['missing_target_subunits']}"
        pats = item.get("target_stoich_patterns_from_shared_template", [])
        if pats:
            part += f", suggests {pats}"
        part += (
            f", best_coverage={item.get('best_coverage', 0):.2f}, "
            f"best_evalue={item.get('best_evalue', 1e99):.2e}"
        )
        details.append(part)

    if details:
        lines.append("Top shared templates: " + "; ".join(details) + ".")

    lines.append(
        "Full-complex or high-coverage major-partial shared templates should be weighted strongly in TB prediction, "
        "but if they miss one or more target subunits they should be treated as partial scaffolds rather than final complete stoichiometries."
    )
    return " ".join(lines)


def compute_consensus_stoichiometry_by_subunit(
    templates,
    evalue_threshold: float = 1.0,
    ignore_monomer_for_single_subunit: bool = True,
    single_subunit_floor: int = 2,
):
    is_single_subunit_system = (len(templates) == 1)
    per_subunit_counts = {}

    for su, hits in templates.items():
        sig_counts = []
        all_counts = []

        for h in hits:
            e = h.get("evalue", None)
            if e is None:
                continue
            try:
                e = float(e)
            except Exception:
                continue

            n = (
                h.get("copies", None)
                or h.get("template_chain_count", None)
                or h.get("chain_count", None)
            )
            if n is None:
                continue
            try:
                n = int(n)
            except Exception:
                continue

            if is_single_subunit_system and ignore_monomer_for_single_subunit and n == 1:
                continue

            all_counts.append(n)
            if e <= evalue_threshold:
                sig_counts.append(n)

        if sig_counts:
            mode_count = Counter(sig_counts).most_common(1)[0][0]
        elif all_counts:
            mode_count = Counter(all_counts).most_common(1)[0][0]
        else:
            mode_count = single_subunit_floor if is_single_subunit_system else 1

        if is_single_subunit_system and mode_count < single_subunit_floor:
            mode_count = single_subunit_floor

        per_subunit_counts[su] = mode_count

    subunits_sorted = sort_subunits(per_subunit_counts.keys())
    consensus_str = "".join(f"{su}{per_subunit_counts[su]}" for su in subunits_sorted)
    return consensus_str, per_subunit_counts


# =========================================================
# Shared-template summary over templates (fallback/debug)
# =========================================================
def validates_shared_field(hit: dict) -> list:
    shared = hit.get("shared_across_subunits", [])
    if not isinstance(shared, list):
        return []
    return [s for s in shared if isinstance(s, str)]


def compute_shared_template_summary(
    templates,
    *,
    strong_cov: float = 0.5,
    strong_evalue: float = 1e-3,
):
    subunit_ids = sort_subunits(list(templates.keys()))
    id_to_subunits = defaultdict(set)
    id_to_best = {}

    for su in subunit_ids:
        for hit in templates.get(su, []):
            pdb_id = hit.get("id")
            if not pdb_id:
                continue

            declared_shared = validates_shared_field(hit)
            if declared_shared:
                for s in declared_shared:
                    id_to_subunits[pdb_id].add(s)
            else:
                id_to_subunits[pdb_id].add(su)

            prev = id_to_best.get(pdb_id)
            try:
                cov = float(hit.get("coverage", 0.0) or 0.0)
            except Exception:
                cov = 0.0
            try:
                ev = float(hit.get("evalue", 1e99) or 1e99)
            except Exception:
                ev = 1e99

            score = (len(id_to_subunits[pdb_id]), cov, -np.log10(max(ev, 1e-300)))
            if prev is None or score > prev["score"]:
                id_to_best[pdb_id] = {
                    "score": score,
                    "coverage": cov,
                    "evalue": ev,
                }

    shared_templates = []
    strong_shared_templates = []
    best_shared_pdb = None
    best_shared_cover = 0

    for pdb_id, covered_subunits in id_to_subunits.items():
        covered_subunits = sort_subunits(list(covered_subunits))
        if len(covered_subunits) < 2:
            continue

        best = id_to_best[pdb_id]
        item = {
            "id": pdb_id,
            "covered_subunits": covered_subunits,
            "covered_count": len(covered_subunits),
            "best_coverage": float(best["coverage"]),
            "best_evalue": float(best["evalue"]),
        }
        shared_templates.append(item)

        if item["covered_count"] > best_shared_cover:
            best_shared_cover = item["covered_count"]
            best_shared_pdb = pdb_id

        if item["best_coverage"] >= strong_cov and item["best_evalue"] <= strong_evalue:
            strong_shared_templates.append(item)

    shared_templates.sort(
        key=lambda x: (-x["covered_count"], -x["best_coverage"], x["best_evalue"], x["id"])
    )
    strong_shared_templates.sort(
        key=lambda x: (-x["covered_count"], -x["best_coverage"], x["best_evalue"], x["id"])
    )

    return {
        "num_shared_templates": len(shared_templates),
        "num_strong_shared_templates": len(strong_shared_templates),
        "best_shared_pdb": best_shared_pdb,
        "best_shared_cover": best_shared_cover,
        "shared_templates": shared_templates,
        "strong_shared_templates": strong_shared_templates,
    }


def has_candidate_header(reply_text):
    for line in reply_text.splitlines():
        s = line.strip().lower()
        if s.startswith("candidates") or s.startswith("candidate list") or s.startswith("ranked candidates"):
            return True
    return False


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



# =========================================================
# CASP17 original target/sequence information
# =========================================================
CASP17_SEQUENCE_URL = (
    "https://predictioncenter.org/download_area/CASP17/sequences/"
    "casp17.HT1.seq.txt"
)


def extract_casp17_target_id(file_path: str) -> str:
    """Extract a CASP17 H/T target ID from a StoiBench JSON filename/path."""
    name = os.path.basename(str(file_path))
    m = re.search(r"(?<![A-Za-z0-9])([HT]\d{4})(?!\d)", name, flags=re.IGNORECASE)
    if not m:
        # Fall back to the full path in case the basename is unusual.
        m = re.search(r"(?<![A-Za-z0-9])([HT]\d{4})(?!\d)", str(file_path), flags=re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Could not infer CASP17 target ID from JSON path: {file_path}"
        )
    return m.group(1).upper()


def ensure_casp17_sequence_file(
    seq_file: str,
    url: str = CASP17_SEQUENCE_URL,
) -> str:
    """
    Return a local copy of the original CASP17 HT1 sequence file.

    If seq_file already exists, it is reused. Otherwise the official
    Prediction Center file is downloaded once and cached at seq_file.
    """
    seq_file = os.path.abspath(seq_file)
    if os.path.isfile(seq_file) and os.path.getsize(seq_file) > 0:
        return seq_file

    parent = os.path.dirname(seq_file)
    if parent:
        os.makedirs(parent, exist_ok=True)

    print(f"[CASP17] sequence file not found locally: {seq_file}")
    print(f"[CASP17] downloading original sequence release from: {url}")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "StoiAgent-CASP17/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
    except Exception as e:
        raise RuntimeError(
            "Could not download the CASP17 sequence file. "
            f"Download it manually from {url} and provide it with "
            f"--casp17_seq_file. Original error: {e}"
        ) from e

    if not raw:
        raise RuntimeError(f"Downloaded CASP17 sequence file is empty: {url}")

    with open(seq_file, "wb") as f:
        f.write(raw)

    print(f"[CASP17] cached sequence file -> {seq_file}")
    return seq_file


def _parse_casp17_header(header: str, occurrence_index: int) -> dict:
    """
    Parse CASP17 HT1 headers.

    Examples:
      H1312 EXT1-EXT2-2BAV4, Human, subunit 1, 649 residues;
      T1313 P66, Borrelia burgdorferi B31, 618 residues|

    Parsing is deliberately right-to-left because descriptions may contain
    commas. The final comma-delimited field before the residue count is the
    source/organism; an optional 'subunit N' field may occur after it.
    """
    h = header.strip()
    if h.startswith(">"):
        h = h[1:].strip()

    # Remove the trailing CASP delimiter used by the file.
    h = h.rstrip(";|").strip()

    m = re.match(
        r"^(?P<target>[HT]\d{4})\s+(?P<body>.*?),\s*"
        r"(?P<length>\d+)\s+residues\s*$",
        h,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Unrecognized CASP17 sequence header: >{header}")

    target_id = m.group("target").upper()
    body = m.group("body").strip()
    declared_length = int(m.group("length"))

    subunit_number = None
    sm = re.search(r",\s*subunit\s+(\d+)\s*$", body, flags=re.IGNORECASE)
    if sm:
        subunit_number = int(sm.group(1))
        body = body[:sm.start()].rstrip()

    # After removing optional "subunit N", split only on the final comma:
    # everything before it is target/protein description, final field is
    # organism/source. Empty source fields are allowed in the CASP file.
    if "," in body:
        description, organism = body.rsplit(",", 1)
        description = description.strip()
        organism = organism.strip()
    else:
        description = body.strip()
        organism = ""

    return {
        "target_id": target_id,
        "description": description,
        "organism": organism,
        "source_subunit_number": subunit_number,
        "occurrence_index": int(occurrence_index),
        "declared_length": declared_length,
        "header": header.strip(),
    }


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


def load_casp17_sequence_catalog(seq_file: str) -> dict:
    local_path = ensure_casp17_sequence_file(seq_file)
    with open(local_path, "r", encoding="utf-8") as f:
        catalog = parse_casp17_sequence_catalog(f.read())
    print(
        f"[CASP17] loaded target/sequence metadata for "
        f"{len(catalog)} target IDs from {local_path}"
    )
    return catalog


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
    source_target_id = core.resolve_casp17_sequence_target(target_id, catalog)
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


def format_casp17_target_info_block(target_info: dict) -> str:
    """Format only original CASP17 sequence-release information for the LLM."""
    if not target_info:
        return ""

    # Deliberately do NOT expose the CASP target ID (e.g. H1312/T1342)
    # or its Phase-1/Phase-2 alias to the benchmark LLM.  Those identifiers
    # remain available internally for lookup, logging, and output metadata.
    lines = [
        "0. Original target information:",
    ]

    description = target_info.get("description", "")
    organism = target_info.get("organism", "")
    if description:
        lines.append(f"- Target/protein description: {description}")
    if organism:
        lines.append(f"- Organism/source: {organism}")

    lines.append("- Subunit sequences:")
    for su in sort_subunits(target_info.get("subunits", {}).keys()):
        rec = target_info["subunits"][su]
        src_n = rec.get("source_subunit_number")
        source_label = (
            f"CASP subunit {src_n}"
            if src_n is not None
            else f"CASP sequence record {rec.get('occurrence_index')}"
        )
        lines.append(
            f"  - Subunit {su}: {source_label}; "
            f"length={rec.get('sequence_length')} residues"
        )
        lines.append(f"    Sequence: {rec.get('sequence', '')}")

    return "\n".join(lines)



# =========================================================
# JSON loader
# =========================================================
def infer_subunit_lengths_from_json(templates):
    lengths = {}
    for su in templates:
        hits = templates[su]
        vals = []
        for h in hits:
            aligned = h.get("aligned_cols")
            cov = h.get("coverage")
            try:
                aligned = float(aligned)
                cov = float(cov)
                if cov > 0:
                    vals.append(int(round(aligned / cov)))
            except Exception:
                continue
        if vals:
            lengths[su] = int(np.median(vals))
    return lengths


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
    os.makedirs(FROZEN_SUMMARY_CACHE_DIR, exist_ok=True)

    print(f"Literature summary agent: {SUMMARY_MODEL_KEY} (Qwen/Qwen3-4B-Instruct-2507)")
    print("Pre-CASP17 benchmark panel: " + ", ".join(BENCHMARK_MODEL_KEYS))
    print(f"Summary cache: {FROZEN_SUMMARY_CACHE_DIR}")

    # Load/cache the original CASP17 target descriptions and sequences once.
    casp17_sequence_catalog = core.load_casp17_sequence_catalog(args.casp17_seq_file)

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

    from . import casp17_three_level_common as _three_common
    _three_common.core.FROZEN_SUMMARY_CACHE_DIR = FROZEN_SUMMARY_CACHE_DIR

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
