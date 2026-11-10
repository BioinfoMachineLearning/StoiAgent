"""Focused checks for FASTA intake, search conversion, and AF3 score collection."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from af3_runner import run_af3_shortlist
import run_stoiagent
from run_stoiagent import TARGET_ID, _write_sequence_catalog
from stage2_rerank import parse_casp17_catalog
from template_search import read_target_fasta, stockholm_to_a3m


class FastaPipelineTest(unittest.TestCase):
    def test_fasta_deduplication_and_local_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.fasta"
            path.write_text(">one\nACDE\n>duplicate\nACDE\n>two\nFGHI\n")
            sequences = read_target_fasta(path)
            self.assertEqual(sequences, {"A": "ACDE", "B": "FGHI"})
            catalog = Path(directory) / "catalog.fasta"
            _write_sequence_catalog(catalog, sequences, "example complex", "example organism")
            parsed = parse_casp17_catalog(catalog)
            self.assertEqual([r["sequence"] for r in parsed[TARGET_ID]], ["ACDE", "FGHI"])

    def test_stockholm_conversion_removes_query_gap_columns_and_duplicates(self):
        stockholm = (
            "# STOCKHOLM 1.0\n"
            "query AC-DE\n"
            "hit1  ATGDE\n"
            "hit2  ATGDE\n"
            "//\n"
        )
        self.assertEqual(stockholm_to_a3m(stockholm), ">query\nACDE\n>hit1\nATgDE\n")

    def test_af3_scores_only_available_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "test.fasta"
            fasta.write_text(">A\nACDE\n", encoding="utf-8")
            calls = []

            def fake_af3(config, output, job_name):
                calls.append(job_name)
                folder = output / job_name
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "ranking_scores.csv").write_text(
                    "ranking_score\n0.5\n0.7\n", encoding="utf-8"
                )
                if job_name.endswith("_a2"):
                    (folder / f"{job_name}_data.json").write_text(json.dumps({
                        "sequences": [{"protein": {
                            "sequence": "ACDE", "unpairedMsa": ">A\\nACDE",
                            "pairedMsa": "", "templates": [],
                        }}],
                    }), encoding="utf-8")

            with patch("af3_runner._run_af3", side_effect=fake_af3):
                csv_path = run_af3_shortlist(
                    fasta, ["A2", "A3", "A100"], root / "af3", {}, num_models=5
                )
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["stoic"] for row in rows], ["A2", "A3"])
            self.assertEqual(calls, ["test_a2", "test_a3"])
            self.assertEqual(float(rows[0]["max_ranking_score"]), 0.7)
            self.assertEqual(float(rows[0]["avg_ranking_score"]), 0.6)

    def test_one_command_connects_stage1_af3_and_stage2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / "complex.fasta"
            fasta.write_text(">A\nACDE\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text("{}", encoding="utf-8")
            output = root / "results"
            commands = []

            def fake_run(args, *, cwd):
                commands.append(args)
                if args[0].endswith("stage1_agent.py"):
                    stage1 = output / "raw" / TARGET_ID / "stoichiometry" / f"{TARGET_ID}_subunit" / "Agent" / "Qwen3.6-27B" / "S1_fasta"
                    stage1.mkdir(parents=True)
                    (stage1 / f"{TARGET_ID}_phase1_topall_k15.out").write_text(
                        "1. A2 avg_conf=0.8\n2. A3 avg_conf=0.6\n", encoding="utf-8"
                    )
                if args[0].endswith("stage2_rerank.py"):
                    stage2 = output / "raw" / TARGET_ID / "stoichiometry" / f"{TARGET_ID}_subunit" / "Agent" / "Qwen3.6-27B" / "S2_fasta_k5_af3"
                    stage2.mkdir(parents=True)
                    (stage2 / f"{TARGET_ID}_subunit_phase2_result.json").write_text(
                        json.dumps({"status": "ok", "final_answer": "A3", "ranking": ["A3", "A2"]}),
                        encoding="utf-8",
                    )

            with patch("run_stoiagent._run", side_effect=fake_run), \
                 patch("run_stoiagent.run_af3_shortlist", return_value=output / "scores.csv") as af3:
                run_stoiagent.main([
                    "--input_fasta", str(fasta), "--config", str(config),
                    "--output_dir", str(output),
                ])
            final = json.loads((output / "final_prediction.json").read_text())
            self.assertEqual(final["top1_stoichiometry"], "A3")
            self.assertEqual(af3.call_args.args[1], ["A2", "A3"])
            self.assertEqual([Path(cmd[0]).name for cmd in commands], [
                "build_single_finetune_json.py", "stage1_agent.py", "stage2_rerank.py"
            ])


if __name__ == "__main__":
    unittest.main()
