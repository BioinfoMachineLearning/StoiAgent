"""Run AlphaFold3 for a fixed Stage-1 shortlist and collect ranking scores."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path

from .config import require_paths
from .template_search import read_target_fasta


def _counts(stoichiometry: str, subunits: list[str]) -> dict[str, int]:
    tokens = re.findall(r"([A-Z])(\d+)", stoichiometry.upper())
    counts = {key: int(value) for key, value in tokens}
    if not tokens or set(counts) != set(subunits) or len(tokens) != len(counts):
        raise ValueError(f"Invalid candidate for subunits {subunits}: {stoichiometry}")
    return counts


def _write_af3_json(path: Path, stoichiometry: str, sequences: dict[str, str],
                    seeds: list[int], common_data: dict | None = None) -> None:
    counts = _counts(stoichiometry, list(sequences))
    entries = []
    chain_index = 0
    for subunit, sequence in sequences.items():
        n = counts[subunit]
        if chain_index + n > 26:
            raise ValueError("AlphaFold3 chain count exceeds 26 for this runner")
        ids = [chr(ord("A") + i) for i in range(chain_index, chain_index + n)]
        protein = {"id": ids, "sequence": sequence}
        if common_data is not None:
            data = common_data.get(sequence)
            if data is None:
                raise ValueError(f"Preprocessed AF3 data missing for subunit {subunit}")
            protein.update({
                "unpairedMsa": data.get("unpairedMsa", ""),
                "pairedMsa": data.get("pairedMsa", ""),
                "templates": data.get("templates", []),
            })
        entries.append({"protein": protein})
        chain_index += n
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": path.stem, "sequences": entries, "modelSeeds": seeds,
        "dialect": "alphafold3", "version": 1,
    }, indent=2), encoding="utf-8")


def _load_common_data(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for entry in data.get("sequences", []):
        protein = entry.get("protein")
        if protein:
            result[protein["sequence"]] = protein
    return result


def _run_af3(config: dict, output: Path, job_name: str) -> None:
    runtime = config.get("af3_runtime", "docker")
    require_paths(config, ["af3_program_path", "af3_params_path", "af3_db_path"]
                  + (["af3_sif_path"] if runtime == "singularity" else []))
    program = Path(config["af3_program_path"]).resolve()
    params = Path(config["af3_params_path"]).resolve()
    database = Path(config["af3_db_path"]).resolve()
    common_args = [
        "python", "run_alphafold.py",
        f"--json_path=/root/af_output/input_jsons/{job_name}.json",
        "--model_dir=/root/models", "--output_dir=/root/af_output",
    ]
    if runtime == "docker":
        command = [
            "docker", "run", "--rm", "--volume", f"{output}:/root/af_output",
            "--volume", f"{params}:/root/models",
            "--volume", f"{database}:/root/public_databases",
            "--gpus", f"device={config.get('gpu_device', '0')}",
            config.get("af3_image", "alphafold3"),
            *common_args, "--run_inference=true",
        ]
    elif runtime == "singularity":
        command = [
            "singularity", "exec", "--nv",
            "--bind", f"{output}:/root/af_output",
            "--bind", f"{params}:/root/models",
            "--bind", f"{database}:/root/public_databases",
            str(Path(config["af3_sif_path"]).resolve()), *common_args,
        ]
    else:
        raise ValueError("af3_runtime must be docker or singularity")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_device", "0"))
    subprocess.run(command, cwd=program, env=env, check=True)


def _score_file(path: Path) -> tuple[float, float, int] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = [float(row["ranking_score"]) for row in rows if row.get("ranking_score")]
    if not scores:
        return None
    return max(scores), sum(scores) / len(scores), len(scores)


def run_af3_shortlist(input_fasta: str | Path, candidates: list[str],
                      output: str | Path, config: dict, num_models: int = 100,
                      max_residues: int = 5000) -> Path:
    """Run a default AF3 preprocessing job, then fixed-pool candidates.

    Candidates over the residue cap receive no score; Stage 2 retains them.
    """
    if num_models < 5 or num_models % 5:
        raise ValueError("num_models must be a positive multiple of five")
    sequences = read_target_fasta(input_fasta)
    target = Path(input_fasta).stem.lower()
    out = Path(output).resolve()
    (out / "input_jsons").mkdir(parents=True, exist_ok=True)
    default = "A2" if len(sequences) == 1 else "".join(f"{su}1" for su in sequences)
    seeds = list(range(1, num_models // 5 + 1))
    default_name = f"{target}_{default.lower()}"
    default_json = out / "input_jsons" / f"{default_name}.json"
    _write_af3_json(default_json, default, sequences, seeds)
    default_dir = out / default_name
    if not (default_dir / f"{default_name}_data.json").is_file():
        _run_af3(config, out, default_name)
    common_path = default_dir / f"{default_name}_data.json"
    if not common_path.is_file():
        raise RuntimeError(f"AlphaFold3 did not produce preprocessed data: {common_path}")
    common_data = _load_common_data(common_path)

    rows = []
    for candidate in dict.fromkeys(candidates):
        counts = _counts(candidate, list(sequences))
        residues = sum(len(sequences[su]) * count for su, count in counts.items())
        if residues > max_residues or sum(counts.values()) > 26:
            continue
        name = f"{target}_{candidate.lower()}"
        if candidate != default:
            _write_af3_json(out / "input_jsons" / f"{name}.json", candidate,
                            sequences, seeds, common_data)
        score_path = out / name / "ranking_scores.csv"
        if not score_path.is_file():
            _run_af3(config, out, name)
        values = _score_file(score_path)
        if values is not None:
            maximum, average, count = values
            rows.append((candidate, maximum, average, count))
    csv_path = out / "stoichiometry_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stoic", "max_ranking_score", "avg_ranking_score", "num_models"])
        writer.writerows(rows)
    return csv_path
