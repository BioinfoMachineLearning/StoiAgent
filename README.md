# StoiAgent

StoiAgent predicts protein-complex stoichiometry from a FASTA file. It searches for structural templates, builds Stage-1 evidence, ranks candidate stoichiometries with an LLM, models the shortlist with AlphaFold 3 (AF3), and reranks the fixed shortlist. The final ranking is saved as `final_prediction.json`.

## Install StoiAgent

Use a Linux machine with a GPU suitable for the selected LLM and AF3. From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
stoiagent --help
```

`python -m stoiagent` is equivalent to `stoiagent`. The optional Mistral backbone requires `python -m pip install -e '.[mistral]'`. StoiAgent's Python dependencies are declared in `pyproject.toml`; `requirements.txt` is a convenience alias for an editable install.

The template-search stage also needs host-accessible `jackhmmer` (HMMER), `hhmake`, `hhsearch` (HH-suite), UniRef90, and an HHsearch template database. Install HMMER/HH-suite through your system package manager or their upstream instructions, and verify that the three executables run. In addition, provide the template metadata described in the configuration below. These resources are separate from the AF3 installation and are not bundled with this package.

## Install AlphaFold 3

Follow the [official AlphaFold 3 installation guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) for your machine. In particular, the standard GPU installation requires Linux, an NVIDIA GPU with Compute Capability 8.0 or higher, Docker with NVIDIA GPU support, substantial storage for the genetic databases, and sufficient RAM. The official guide recommends at least 64 GB RAM and documents a database footprint of approximately 630 GB after extraction.

For the Docker runtime used by StoiAgent, the essential steps are:

1. Install Docker, the NVIDIA driver, and the NVIDIA Container Toolkit as described in the official guide. Check `nvidia-smi` and that a GPU-enabled Docker container can see the GPU.
2. Clone `https://github.com/google-deepmind/alphafold3.git`. From that directory, run `./fetch_databases.sh /absolute/path/to/af3_databases`. Keep the database directory outside the source checkout.
3. Obtain the [AF3 model parameters](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-model-parameters) separately and place them in `/absolute/path/to/af3_models`. Review the linked model terms. Model parameters are not included in this repository.
4. From the AF3 source directory, build the official image: `docker build -t alphafold3 -f docker/Dockerfile .`. Run the official sample prediction first to verify the installation.
5. Set `af3_program_path` to the AF3 source checkout, `af3_params_path` to the parameter directory, `af3_db_path` to the database directory, and `af3_image` to `alphafold3` in the StoiAgent configuration. The runner mounts the models at `/root/models` and databases at `/root/public_databases` inside the container.

The [official Singularity instructions](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#running-using-singularity-instead-of-docker) are an alternative: build an `alphafold3.sif` from the Docker image, set `af3_runtime` to `singularity`, and set `af3_sif_path` to that file. StoiAgent invokes the installed AF3 image; it does not install AF3 or download its databases or parameters.

## Configure template search and AF3

Create a local JSON file with absolute paths. The `uniref90_path` may point to the UniRef90 FASTA downloaded for AF3. The `hhdb_prefix` must identify a separately prepared HHsearch database. The template metadata must include `<PDB_ID>_initial.fasta` files, `<PDB_ID>.json` files, `<HHR_hit_name>.atom.gz` files, and a CSV with `pdb_id` and `final_stoichiometry` columns. An optional `release_date` column supports template-date filtering in the standalone builder.

```json
{
  "uniref90_path": "/data/af3_databases/uniref90_2022_05.fa",
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
  "af3_image": "alphafold3",
  "gpu_device": "0"
}
```

## Run from FASTA

Supply one FASTA record per **distinct** protein subunit in A/B/C order. Repeated identical sequences are collapsed automatically, so a homomer can be supplied as one record. No HHR file is required: the package generates its own HHsearch results.

```bash
stoiagent \
  --input_fasta /path/to/complex.fasta \
  --config /path/to/config.json \
  --output_dir /path/to/results/complex \
  --description "Protein complex name" \
  --organism "Organism name"
```

`--description` and `--organism` are optional. `--model` selects the LLM backbone (default `Qwen3.6-27B`), `--k` selects the Stage-1 shortlist size (default 5), and `--num_models` selects AF3 models per feasible candidate (default 100, a multiple of five). Run each target or configuration in its own output directory.

Stage 1 performs 15 stochastic samples. AF3 evaluates Stage-1 candidates that fit within the 5,000-residue and 26-chain limits. Candidates beyond those limits remain in the Stage-2 shortlist without AF3 scores. Stage 2 reranks exactly that shortlist. `final_prediction.json` contains the top-1 prediction, complete ranking, and paths to detailed results. AF3 ranking scores are structural-confidence evidence, not probabilities that a stoichiometry is biologically correct.

The template-search and AF3 output directories are reused when rerunning the same output folder. Use a fresh output directory if databases, models, or configuration change.

## Repository layout

```text
src/stoiagent/       Installable Python package and command-line pipeline
  run_stoiagent.py   FASTA → template search → Stage 1 → AF3 → Stage 2
  template_search.py Host-side jackhmmer, HHmake, and HHsearch
  build_target_json.py  Stage-1 template evidence builder
  stage1_agent.py    Stage-1 candidate generation and ranking
  af3_runner.py      AF3 jobs and ranking-score collection
  stage2_rerank.py   Fixed-shortlist reranking
  agent_common.py    Shared Stage-1 code: model loading, summary cache, validation
  utils.py           Lightweight chain-ID and stoichiometry helpers
  discovery_cache.py DiscoveryCache (RCSB/UniProt/PubMed retrieval)
  research_tools.py  RCSB/UniProt/PubMed research actions for the agent
  stage1_agent_full.py  Stage-1 variant that sees all template evidence
  parsers.py         HH-suite and PDB parsing
tests/               Focused pipeline tests
pyproject.toml       Package metadata, dependencies, and CLI entry point
```

Advanced research interfaces remain available with `python -m stoiagent.stage1_agent` and `python -m stoiagent.stage2_rerank`. The end-to-end command creates a local sequence catalog from the FASTA, so no CASP17 input file is needed. Its internal target ID is withheld from LLM prompts. For historical benchmark evaluation, supply the original target-release cutoffs and sequence release file to the standalone stage modules.

The adapted parser retains its original DeepMind copyright and Apache 2.0 notice. A copy of that license is in `APACHE-2.0-LICENSE.txt`.
