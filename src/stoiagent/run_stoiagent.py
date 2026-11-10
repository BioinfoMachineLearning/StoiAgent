#!/usr/bin/env python3
"""FASTA-to-Stage-2 StoiAgent command for one protein complex."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .af3_runner import run_af3_shortlist
from .config import load_config
from .stage2_rerank import parse_stage1_ranked
from .template_search import read_target_fasta


TARGET_ID = "H0001"  # Internal ID; withheld from both model prompts.
STAGE1_VARIANT = "S1_fasta"


def _run(args: list[str], *, cwd: Path) -> None:
    subprocess.run([sys.executable, *args], cwd=cwd, check=True)


def _write_sequence_catalog(path: Path, sequences: dict[str, str],
                            description: str, organism: str) -> None:
    clean_description = re.sub(r"[\r\n,]", " ", description).strip() or "Protein complex"
    clean_organism = re.sub(r"[\r\n,]", " ", organism).strip() or "unspecified organism"
    lines = []
    for index, sequence in enumerate(sequences.values(), 1):
        lines += [
            f">{TARGET_ID} {clean_description}, {clean_organism}, subunit {index}, {len(sequence)} residues;",
            sequence,
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_fasta", required=True, type=Path,
                        help="Target FASTA with one record per distinct protein subunit.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional JSON overriding the default database and AlphaFold3 locations "
                             "inside the StoiAgent folder.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--description", default=None,
                        help="Protein/complex description; defaults to the FASTA filename stem.")
    parser.add_argument("--organism", default="unspecified organism")
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num_models", type=int, default=100,
                        help="AlphaFold3 models per candidate; multiple of five.")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be at least one")
    source = args.input_fasta.resolve()
    if not source.is_file():
        parser.error(f"FASTA not found: {source}")
    config_path = args.config.resolve() if args.config else None
    if config_path and not config_path.is_file():
        parser.error(f"Config not found: {config_path}")
    config = load_config(config_path)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sequences = read_target_fasta(source)
    canonical_fasta = out / f"{TARGET_ID}.fasta"
    canonical_fasta.write_text(
        "".join(f">{su}\n{sequence}\n" for su, sequence in sequences.items()),
        encoding="utf-8",
    )
    catalog = out / "target_sequences.fasta"
    _write_sequence_catalog(catalog, sequences, args.description or source.stem, args.organism)

    target_dir = out / "raw" / TARGET_ID / "stoichiometry" / f"{TARGET_ID}_subunit"
    target_json = out / f"{TARGET_ID}.json"
    _run([
        "-m", "stoiagent.build_target_json",
        "--input_fasta", str(canonical_fasta),
        *(["--config", str(config_path)] if config_path else []),
        "--workdir", str(out / "template_search"),
        "--output_json", str(target_json), "--protein_id", TARGET_ID,
    ], cwd=out)

    stage1_dir = target_dir / "Agent" / args.model / STAGE1_VARIANT
    _run([
        "-m", "stoiagent.stage1_agent",
        "--json_file", str(target_json), "--outdir", str(stage1_dir),
        "--model", args.model, "--temperature", str(args.temperature),
        "--casp17_seq_file", str(catalog), "--information_cutoff", "",
        "--summary_cache_dir", str(out / "article_cache" / "Qwen3-4B"),
    ], cwd=out)
    stage1_file = stage1_dir / f"{TARGET_ID}_phase1_topall_k15.out"
    if not stage1_file.is_file():
        raise RuntimeError(f"Stage 1 did not produce {stage1_file}")
    candidates = [row["stoic"] for row in parse_stage1_ranked(stage1_file)[:args.k]]
    if not candidates:
        raise RuntimeError("Stage 1 produced no valid stoichiometry candidates")

    af3_dir = target_dir / "AlphaFold3" / TARGET_ID.lower()
    af3_csv = run_af3_shortlist(canonical_fasta, candidates, af3_dir, config,
                                num_models=args.num_models)
    _run([
        "-m", "stoiagent.stage2_rerank", "--root", str(out / "raw"),
        "--targets", TARGET_ID, "--models", args.model,
        "--variant", STAGE1_VARIANT, "--k", str(args.k),
        "--casp17-sequences", str(catalog), "--af3-csv", str(af3_csv),
        "--conditions", "af3", "--outdir", str(out / "stage2_summary"),
        "--stage2-variant-prefix", f"S2_fasta_k{args.k}",
    ], cwd=out)
    result_file = target_dir / "Agent" / args.model / f"S2_fasta_k{args.k}_af3" / f"{TARGET_ID}_subunit_phase2_result.json"
    if not result_file.is_file():
        raise RuntimeError(f"Stage 2 did not produce {result_file}")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if result.get("status") != "ok":
        raise RuntimeError(f"Stage 2 failed: {result.get('last_error', 'unknown error')}")
    final = {
        "input_fasta": str(source), "top1_stoichiometry": result["final_answer"],
        "ranked_stoichiometries": result["ranking"],
        "stage1_candidates": candidates, "af3_scores_csv": str(af3_csv),
        "stage2_result": str(result_file),
    }
    final_path = out / "final_prediction.json"
    final_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(f"Final prediction: {result['final_answer']} ({final_path})")


if __name__ == "__main__":
    main()
