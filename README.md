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

## Set up databases and tools

StoiAgent looks for its databases and tools inside this folder by default. If every resource is in its default location, no configuration file is needed. Git ignores `databases/` and `tools/`.

```text
StoiAgent/
  tools/
    alphafold3/             AF3 source checkout
      models/af3.bin        AF3 model parameters
    alphafold3.sif          AF3 Singularity image (Singularity runtime only)
  databases/
    af3/                    AF3 genetic databases, including uniref90_2022_05.fa
    pdb70/pdb70_*           HHsearch template database (prefix databases/pdb70/pdb70)
    templates/
      fasta/                <PDB_ID>_initial.fasta
      json/                 <PDB_ID>.json
      atom/                 <HHR_hit_name>.atom.gz
      labels.csv            pdb_id, final_stoichiometry, optional release_date
```

If a resource is missing, StoiAgent stops before the step that needs it and lists the missing paths.

### AlphaFold 3

Follow the [official AlphaFold 3 installation guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) for your machine. The standard GPU installation requires Linux, an NVIDIA GPU with Compute Capability 8.0 or higher, Docker with NVIDIA GPU support, at least 64 GB RAM, and about 630 GB of disk for the databases after extraction. Run these steps from this folder:

1. Install Docker, the NVIDIA driver, and the NVIDIA Container Toolkit as described in the official guide. Check `nvidia-smi` and that a GPU-enabled Docker container can see the GPU.
2. Clone AF3 and download its databases:

   ```bash
   git clone https://github.com/google-deepmind/alphafold3.git tools/alphafold3
   mkdir -p databases/af3
   tools/alphafold3/fetch_databases.sh "$PWD/databases/af3"
   ```

3. Request the [AF3 model parameters](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-model-parameters) from Google DeepMind and review their terms of use. Put them in `tools/alphafold3/models/`, so that the folder contains `af3.bin`. The parameters are not included in this repository.
4. Build the official image and run the official sample prediction to verify the installation:

   ```bash
   docker build -t alphafold3 -f tools/alphafold3/docker/Dockerfile tools/alphafold3
   ```

StoiAgent mounts `tools/alphafold3/models` at `/root/models` and `databases/af3` at `/root/public_databases` inside the container. To use Singularity instead, follow the [official Singularity instructions](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#running-using-singularity-instead-of-docker) to build `tools/alphafold3.sif`, and set `"af3_runtime": "singularity"` in a config file.

### Template search

Template search needs `jackhmmer` (HMMER) and `hhmake` and `hhsearch` (HH-suite) on the `PATH`. Install them with your system package manager or by following their upstream instructions, and check that all three run. Template search uses the UniRef90 file from the AF3 databases, so it needs no separate download. Put the HHsearch database in `databases/pdb70/` and the template metadata in `databases/templates/`, as shown in the layout above.

## Override default locations (optional)

To keep a resource outside this folder, for example a shared AF3 database, pass `--config` with a JSON file that sets only the paths that differ. Relative paths are resolved against this folder. Set `STOIAGENT_HOME` to resolve the default locations against another folder instead.

```json
{
  "af3_db_path": "/shared/af3_databases",
  "uniref90_path": "/shared/af3_databases/uniref90_2022_05.fa",
  "af3_runtime": "singularity",
  "gpu_device": "1"
}
```

The available keys are `uniref90_path`, `hhdb_prefix`, `template_fasta_dir`, `template_pdb_json_dir`, `template_atom_dir`, `template_label_csv`, `af3_program_path`, `af3_params_path`, `af3_db_path`, `af3_sif_path`, `af3_runtime` (`docker` or `singularity`), `af3_image`, `gpu_device`, `jackhmmer_binary`, `hhmake_binary`, `hhsearch_binary` and `search_cpu`. The defaults are in `src/stoiagent/config.py`.

## Run from FASTA

Supply one FASTA record per **distinct** protein subunit in A/B/C order. Repeated identical sequences are collapsed automatically, so a homomer can be supplied as one record. No HHR file is required: the package generates its own HHsearch results.

```bash
stoiagent \
  --input_fasta /path/to/complex.fasta \
  --output_dir /path/to/results/complex \
  --description "Protein complex name" \
  --organism "Organism name"
```

`--description`, `--organism` and `--config` are optional. `--model` selects the LLM backbone (default `Qwen3.6-27B`), `--k` selects the Stage-1 shortlist size (default 5), and `--num_models` selects AF3 models per feasible candidate (default 100, a multiple of five). Run each target or configuration in its own output directory.

Stage 1 performs 15 stochastic samples. AF3 evaluates Stage-1 candidates that fit within the 5,000-residue and 26-chain limits. Candidates beyond those limits remain in the Stage-2 shortlist without AF3 scores. Stage 2 reranks exactly that shortlist. `final_prediction.json` contains the top-1 prediction, complete ranking, and paths to detailed results. AF3 ranking scores are structural-confidence evidence, not probabilities that a stoichiometry is biologically correct.

The template-search and AF3 output directories are reused when rerunning the same output folder. Use a fresh output directory if databases, models, or configuration change.

## Notes

Advanced research interfaces remain available with `python -m stoiagent.stage1_agent` and `python -m stoiagent.stage2_rerank`. The end-to-end command creates a local sequence catalog from the FASTA, so no CASP17 input file is needed. Its internal target ID is withheld from LLM prompts. For historical benchmark evaluation, supply the original target-release cutoffs and sequence release file to the standalone stage modules.

The adapted parser retains its original DeepMind copyright and Apache 2.0 notice. A copy of that license is in `APACHE-2.0-LICENSE.txt`.
