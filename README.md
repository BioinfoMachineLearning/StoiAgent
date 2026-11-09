# StoiAgent

StoiAgent predicts protein-complex stoichiometry in two stages. Stage 1 uses an LLM to generate and rank candidate stoichiometries from target descriptions, subunit sequences, structural-template evidence, and on-demand database and literature retrieval. Stage 2 reranks a fixed shortlist using Stage-1 information and AlphaFold3 ranking scores. The Stage-2 model cannot add or remove candidates.

This repository currently contains **the agent pipeline only**. Building the target JSON files and running AlphaFold3 are separate upstream steps; the JSON-building workflow will be added later.

## Files

| File | Role |
| --- | --- |
| `stage1_agent.py` | Main Stage-1 entry point: shared-complex-template evidence when available, otherwise individual-subunit-template fallback. |
| `stage2_rerank.py` | Main Stage-2 entry point: fixed-pool reranking with or without AF3 evidence. |
| `casp17_three_level_common.py` | Research tools and target-specific information cutoffs used by Stage 1. |
| `D_core_three_level.py` | Shared Stage-1 model and CASP17 sequence utilities imported by the main agent. It also retains its original standalone entry point. |
| `agent_helper.py` | Discovery, record retrieval, literature summarization, and stoichiometry helpers. |

The source files are assembled from the supplied research scripts. The current entry points have not been refactored into a package or validated by running the large language models. In particular, the default UniProt observations are current unless a historically frozen cache is supplied; the Stage-1 script prints this warning at run time.

## Setup

Use Python 3.10 or newer and an environment capable of loading the selected Hugging Face model. Install the common dependencies:

```bash
python -m pip install -r requirements.txt
```

For the Mistral backbone, also install `mistral-common`. Model downloads, GPU memory, and any model-access requirements depend on the selected backbone. The Stage-1 literature summarizer uses the fixed `Qwen/Qwen3-4B-Instruct-2507` model in addition to the chosen prediction backbone. The PubMed content helper defaults to structured title and abstract retrieval.

## Inputs and run order

1. Prepare one target JSON file with the `templates` mapping expected by `stage1_agent.py`. The loader can also use `shared_templates_for_agent`, `subunit_consensus_summary`, `shared_template_summary`, and `constraint_summary` when present. The target identifier is read from the JSON filename. JSON construction is outside the present repository scope.
2. Supply the original CASP17 HT1 sequence-release file. Stage 1 maps its unique-subunit sequences to the target JSON's subunit keys in record order. Stage 2 uses the same sequence file.
3. Run Stage 1. It performs 15 stochastic samples and writes `*_phase1_topall_k15.out`, `*_phase1_top5_k15.out`, sample traces, and caches to the selected output directory.
4. Generate AlphaFold3 models and a `stoichiometry_results.csv` with `stoic`, `max_ranking_score`, `avg_ranking_score`, and `num_models` for the candidates to be structurally evaluated. Stage 2 reads the scores available for the Stage-1 top-`k` candidates; missing scores remain unavailable evidence.
5. Run Stage 2 with the Stage-1 output directory in the expected `results/raw/<TARGET>/stoichiometry/<TARGET>_subunit/Agent/<MODEL>/<VARIANT>/` hierarchy. The default `k` is 5. Results are written beside Stage 1 by default, with aggregate CSV summaries under `--outdir`.

Example commands (replace the target, paths, and model with your data):

```bash
python stage1_agent.py --json_file /path/to/H1389.json --outdir results/raw/H1389/stoichiometry/H1389_subunit/Agent/Qwen3.6-27B/S1_full_stoiagent_classaware_strict --model Qwen3.6-27B --temperature 0.7 --casp17_seq_file /path/to/casp17.HT1.seq.txt --target-release-date YYYY-MM-DD

python stage2_rerank.py --root results/raw --targets H1389 --models Qwen3.6-27B --variant S1_full_stoiagent_classaware_strict --k 5 --casp17-sequences /path/to/casp17.HT1.seq.txt --conditions af3 --outdir results/stage2_summary
```

Stage 2 expects the AF3 table at `results/raw/<TARGET>/stoichiometry/<TARGET>_subunit/AlphaFold3/<target_lower>/stoichiometry_results.csv` unless `--af3-csv` or `--af3-csv-template` is supplied. Use `--prepare-only` to inspect its job manifest and prompts before loading a model. `--pairs` is needed only for accuracy evaluation against reference stoichiometries. `--candidate-evidence-csv` and `--target-context-dir` are optional additional inputs; they are not supplied by the commands above.

For Stage 1, `--target-release-date` or `--target-release-dates` sets a target-specific historical cutoff for PDB and PubMed evidence. Otherwise the script uses its conservative default cutoff. The benchmark target identifier is used for lookup and output metadata but is withheld from the LLM prompt.

## Reproducibility notes

- Stage-1 candidate support scores are heuristic scores averaged across the 15 samples, counting a missing candidate as zero in that sample. They are not calibrated probabilities.
- Stage 2 reranks exactly the selected Stage-1 top-`k` pool. The AF3 condition uses the available score rows for that pool; the `noaf3` condition provides a matched comparison without AF3 scores.
- Runtime caches and prediction outputs are ignored by Git. Start with empty caches when reproducing the original experiment; clear cached literature summaries if the summarization model or prompt changes.
- No reference stoichiometry should be supplied as an agent input. The optional `--pairs` file is for evaluation after prediction.

For reproducible results, record the exact model revisions, Python environment, target JSON generation procedure, and dataset provenance used for a published result.
