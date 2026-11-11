"""Focused checks for FASTA intake, search conversion, and AF3 score collection."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stoiagent.af3_runner import run_af3_shortlist
from stoiagent import run_stoiagent
from stoiagent.run_stoiagent import FALLBACK_TARGET_ID, _write_sequence_catalog, target_id_for
from stoiagent.stage2_rerank import parse_casp17_catalog
from stoiagent.template_search import read_target_fasta, stockholm_to_a3m


class FastaPipelineTest(unittest.TestCase):
    def test_fasta_deduplication_and_local_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.fasta"
            path.write_text(">one\nACDE\n>duplicate\nACDE\n>two\nFGHI\n")
            sequences = read_target_fasta(path)
            self.assertEqual(sequences, {"A": "ACDE", "B": "FGHI"})
            catalog = Path(directory) / "catalog.fasta"
            _write_sequence_catalog(catalog, "H0208", sequences, "example complex", "example organism")
            parsed = parse_casp17_catalog(catalog)
            self.assertEqual([r["sequence"] for r in parsed["H0208"]], ["ACDE", "FGHI"])

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
                (folder / f"{job_name}_ranking_scores.csv").write_text(
                    "ranking_score\n0.5\n0.7\n", encoding="utf-8"
                )
                if job_name.endswith("_a2"):
                    (folder / f"{job_name}_data.json").write_text(json.dumps({
                        "sequences": [{"protein": {
                            "sequence": "ACDE", "unpairedMsa": ">A\\nACDE",
                            "pairedMsa": "", "templates": [],
                        }}],
                    }), encoding="utf-8")

            with patch("stoiagent.af3_runner._run_af3", side_effect=fake_af3):
                csv_path = run_af3_shortlist(
                    fasta, ["A2", "A3", "A100"], root / "af3", {}, num_models=5
                )
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["stoic"] for row in rows], ["A2", "A3"])
            self.assertEqual(calls, ["test_a2", "test_a3"])
            self.assertEqual(float(rows[0]["max_ranking_score"]), 0.7)
            self.assertEqual(float(rows[0]["avg_ranking_score"]), 0.6)

    def test_af3_docker_runs_as_calling_user(self):
        import os
        from stoiagent.af3_runner import _run_af3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"af3_program_path": str(root), "af3_params_path": str(root),
                      "af3_db_path": str(root)}
            with patch("stoiagent.af3_runner.subprocess.run") as run:
                _run_af3(config, root, "job")
            command = run.call_args_list[0].args[0]
            self.assertEqual(command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
            self.assertIn("--db_dir=/public_databases", command)
            self.assertFalse(any("/root/" in part for part in command))

    def test_stage1_duplicate_candidate_keeps_first_row(self):
        # Sample 07 of the H0208 run listed A1B1 twice; the merge step keeps
        # the first row, so its support and evidence must come from that row.
        from stoiagent.stage1_agent import parse_candidate_support_evidence, parse_confidences
        from stoiagent.stage1_agent_full import parse_confidences as parse_confidences_full
        reply = (
            "Candidates:\n"
            "1) A1B1  support=0.85  support_evidence=Consistent A1B1 pattern in shared templates\n"
            "2) A2B2  support=0.10  support_evidence=Possible dimer of heterodimers\n"
            "5) A1B1  support=0.01  support_evidence=Baseline candidate for comparison\n"
            "Final Answer: A1B1\n"
        )
        self.assertEqual(parse_confidences(reply), {"A1B1": 0.85, "A2B2": 0.10})
        self.assertEqual(parse_confidences_full(reply), {"A1B1": 0.85, "A2B2": 0.10})
        self.assertEqual(parse_candidate_support_evidence(reply)["A1B1"],
                         "Consistent A1B1 pattern in shared templates")

    def test_target_id_follows_fasta_name(self):
        self.assertEqual(target_id_for(Path("examples/casp16/H0208.fasta")), "H0208")
        self.assertEqual(target_id_for(Path("t1312.fasta")), "T1312")
        self.assertEqual(target_id_for(Path("complex.fasta")), FALLBACK_TARGET_ID)

    def test_one_command_connects_stage1_af3_and_stage2(self):
        for fasta_name, tid in [("complex.fasta", FALLBACK_TARGET_ID), ("H0208.fasta", "H0208")]:
            with self.subTest(fasta=fasta_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fasta = root / fasta_name
                fasta.write_text(">A\nACDE\n", encoding="utf-8")
                output = root / "results"
                target_dir = output / "raw" / tid / "stoichiometry" / f"{tid}_subunit"
                commands = []

                def fake_run(args, *, cwd):
                    commands.append(args)
                    if args[1] == "stoiagent.stage1_agent":
                        stage1 = target_dir / "Agent" / "Qwen3.6-27B" / "S1_fasta"
                        stage1.mkdir(parents=True)
                        (stage1 / f"{tid}_phase1_topall_k15.out").write_text(
                            "1. A2 avg_conf=0.8\n2. A3 avg_conf=0.6\n", encoding="utf-8"
                        )
                    if args[1] == "stoiagent.stage2_rerank":
                        stage2 = target_dir / "Agent" / "Qwen3.6-27B" / "S2_fasta_k5_af3"
                        stage2.mkdir(parents=True)
                        (stage2 / f"{tid}_subunit_phase2_result.json").write_text(
                            json.dumps({"status": "ok", "final_answer": "A3", "ranking": ["A3", "A2"]}),
                            encoding="utf-8",
                        )

                with patch("stoiagent.run_stoiagent._run", side_effect=fake_run), \
                     patch("stoiagent.run_stoiagent.run_af3_shortlist", return_value=output / "scores.csv") as af3:
                    run_stoiagent.main([
                        "--input_fasta", str(fasta),
                        "--output_dir", str(output),
                    ])
                final = json.loads((output / "final_prediction.json").read_text())
                self.assertEqual(final["top1_stoichiometry"], "A3")
                self.assertEqual(af3.call_args.args[0], output / f"{tid}.fasta")
                self.assertEqual(af3.call_args.args[1], ["A2", "A3"])
                self.assertEqual(af3.call_args.args[2], target_dir / "AlphaFold3" / tid.lower())
                self.assertIn(tid, [cmd[cmd.index("--targets") + 1] for cmd in commands if "--targets" in cmd])
                self.assertEqual([cmd[1] for cmd in commands], [
                    "stoiagent.build_target_json", "stoiagent.stage1_agent", "stoiagent.stage2_rerank"
                ])

    def test_default_config_points_inside_stoiagent_folder(self):
        from stoiagent.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            override = home / "override.json"
            override.write_text(json.dumps({
                "af3_db_path": "/data/af3", "template_stoichiometry_csv": "",
                "template_fasta_dir": "custom/fasta",
            }), encoding="utf-8")
            with patch.dict("os.environ", {"STOIAGENT_HOME": str(home)}):
                default = load_config()
                config = load_config(override)
        self.assertEqual(default["hhdb_prefix"], str(home / "databases/sort90_hhsuitedb/hhsuitedb3"))
        self.assertEqual(default["af3_program_path"], str(home / "tools/alphafold3"))
        self.assertEqual(config["af3_db_path"], "/data/af3")
        self.assertEqual(config["template_stoichiometry_csv"],
                         str(home / "databases/templates/pdb_stoichiometry.csv"))
        self.assertEqual(config["template_entity_fasta_dir"], str(home / "custom/fasta"))

    def test_config_json_in_stoiagent_folder_is_read_automatically(self):
        from stoiagent.config import load_config
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / "config.json").write_text(json.dumps({"af3_db_path": "/big/af3"}), encoding="utf-8")
            with patch.dict("os.environ", {"STOIAGENT_HOME": str(home)}):
                config = load_config()
        self.assertEqual(config["af3_db_path"], "/big/af3")
        self.assertEqual(config["hhdb_prefix"], str(home / "databases/sort90_hhsuitedb/hhsuitedb3"))

    def test_missing_resources_are_reported_with_their_keys(self):
        from stoiagent.config import load_config, require_paths
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"STOIAGENT_HOME": directory}):
                config = load_config()
            with self.assertRaises(FileNotFoundError) as caught:
                require_paths(config, ["hhdb_prefix", "af3_db_path"])
        self.assertIn("hhdb_prefix", str(caught.exception))
        self.assertIn("af3_db_path", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
