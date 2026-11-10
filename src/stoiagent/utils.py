"""Lightweight helpers shared across StoiAgent modules (no torch/transformers).

Stoichiometry strings use PDB-style chain IDs (A..Z, then AA, AB, ...) followed
by copy counts, e.g. ``A2B1``.
"""
import re
import string
from itertools import product


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
    s = str(stoich).replace(",", "").replace(" ", "")
    pairs = re.findall(r"([A-Z]{1,2})(\d+)", s)
    if not pairs:
        raise ValueError(f"Invalid stoichiometry: {stoich}")

    order = {cid: i for i, cid in enumerate(PDB_CHAIN_IDS)}
    pairs = sorted(pairs, key=lambda p: order.get(p[0], 10**9))
    return "".join(f"{sym}{cnt}" for sym, cnt in pairs)


def parse_stoich(stoich: str) -> dict:
    pairs = re.findall(r"([A-Z]{1,2})(\d+)", stoich)
    return {k: int(v) for k, v in pairs}


def simplest_stoich(subunits):
    subs = sort_subunits(subunits)
    return "".join(f"{s}1" for s in subs)


def get_model_max_len(tokenizer, default=8192):
    v = getattr(tokenizer, "model_max_length", None)
    if isinstance(v, int) and 0 < v < 10**9:
        return v
    return default


def strip_quotes(s) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def linkify_citation(citation: str) -> str:
    """Convert a DOI or known record identifier to a citation URL."""
    citation = str(citation).strip()
    if re.fullmatch(r"\d{5,8}", citation):
        return f"https://pubmed.ncbi.nlm.nih.gov/{citation}/"
    if citation.upper().startswith("PMID:"):
        pmid = re.sub(r"(?i)pmid:", "", citation).strip()
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if citation.upper().startswith("PDB"):
        pdb_id = re.sub(r"(?i)pdb[:\s]*", "", citation).strip()
        return f"https://www.rcsb.org/structure/{pdb_id}"
    if "/" in citation and "." in citation and not citation.startswith("https://doi.org/"):
        return f"https://doi.org/{citation}"
    return citation
