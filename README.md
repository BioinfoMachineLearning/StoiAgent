# StoiAgent

StoiAgent predicts protein-complex stoichiometry from a target FASTA. It searches for structural templates, builds a Stage-1 evidence bundle, generates and ranks candidates with an LLM, models the shortlist with AlphaFold3, and reranks that fixed shortlist with the LLM. The final ranked prediction is written to `final_prediction.json`.

## Requirements

- Python 3.10 or newer and `pip install -r requirements.txt`.
- `jackhmmer` (HMMER), `hhmake` and `hhsearch` (HH-suite), a UniRef90 sequence database, and an HHsearch template database.
- Template metadata: `<PDB_ID>_initial.fasta` files, `<PDB_ID>.json` files, `<HHR_hit_name>.atom.gz` files, and a CSV with `pdb_id` and `final_stoichiometry` columns. An optional `release_date` column supports template-date filtering.
- A local AlphaFold3 installation, model parameters, sequence databases, and a working Docker or Singularity image.
- A machine capable of loading the chosen StoiAgent backbone and fixed Qwen3-4B literature summarizer. For the Mistral backbone, install `mistral-common` too.

These databases, model weights, and container images are not distributed here. Set their paths in a local JSON configuration file. Example:

```json
{
  "uniref90_path": "/data/uniref90.fasta",
  "hhdb_prefix": "/data/pdb70/pdb70",
  "jackhmmer_binary": "jackhmmer",
  "hhmake_binary": "hhmake",
  "hhsearch_binary": "hhsearch",
  "template_fasta_dir": "/data/template_fastas",
  "template_pdb_json_dir": "/data/template_jsons",
  "template_atom_dir": "/data/template_atoms",
  "template_label_csv": "/data/template_labels.csv",
  "af3_runtime": "docker",
  "af3_program_path": "/opt/alphafold3",
  "af3_params_path": "/data/af3_models",
  "af3_db_path": "/data/af3_databases",
  "af3_image": "alphafold3_stoiagent",
  "gpu_device": "0"
}
```

For Singularity, set `"af3_runtime": "singularity"` and `"af3_sif_path"` to the image file. Use absolute paths in the configuration.

## Run from FASTA

Supply one FASTA record per **distinct** protein subunit, in A/B/C order. Repeated identical sequences are collapsed automatically, so a homomer can be supplied as one record. The target FASTA is the only target-specific file required.

```bash
python run_stoiagent.py \
  --input_fasta /path/to/complex.fasta \
  --config /path/to/config.json \
  --output_dir /path/to/results/complex \
  --description "Protein complex name" \
  --organism "Organism name"
```

`--description` and `--organism` are optional; the FASTA filename and “unspecified organism” are used when omitted. `--model` selects the LLM backbone (default `Qwen3.6-27B`), `--k` selects the Stage-1 shortlist size (default 5), and `--num_models` selects AlphaFold3 models per feasible candidate (default 100, a multiple of five). Run each different target or configuration in its own output directory.

The command generates internal HHsearch result files from the FASTA; users do not provide HHR files. Stage 1 performs 15 stochastic samples. AlphaFold3 evaluates the first `k` Stage-1 candidates that fit within the 5,000-residue and 26-chain limits. Candidates that exceed the limits remain in the Stage-2 shortlist with unavailable AF3 scores. Stage 2 reranks exactly that shortlist. The final output is `final_prediction.json`, containing the top-1 prediction, complete ranking, and paths to detailed Stage-1, AF3, and Stage-2 results.

Template search and AlphaFold3 output directories are reused when rerunning the same output folder. Use a fresh output directory if databases, models, or configuration change. AF3 ranking scores are structural-confidence evidence, not probabilities that a stoichiometry is biologically correct.

## Main files

| File | Role |
| --- | --- |
| `run_stoiagent.py` | Runs FASTA → template search → Stage 1 → AlphaFold3 → Stage 2. |
| `template_search.py` | Runs jackhmmer, HHmake, and HHsearch per distinct subunit. |
| `build_single_finetune_json.py` | Builds Stage-1 template evidence JSON from FASTA and search results. |
| `stage1_agent.py` | Stage-1 candidate generation and ranking. |
| `af3_runner.py` | Generates AlphaFold3 jobs and collects candidate ranking scores. |
| `stage2_rerank.py` | Reranks the fixed candidate pool. |
| `agent_helper.py`, `casp17_three_level_common.py`, `D_core_three_level.py` | Evidence retrieval, historical benchmark support, and model utilities. |
| `parsers.py` | HH-suite and PDB parsing, adapted from the supplied DeepMind parser under Apache 2.0. |

The standalone Stage-1 and Stage-2 scripts retain their CASP17 research interfaces. `run_stoiagent.py` creates a compatible local sequence catalog from the supplied FASTA, so no CASP17 input file is needed for the end-to-end command. The internal target ID is withheld from the LLM prompts. For historical benchmark evaluation, use the original target-release cutoffs and sequence release file via the standalone scripts.

The adapted parser retains its original copyright and license notice. A copy of the Apache 2.0 license is in `APACHE-2.0-LICENSE.txt`.
