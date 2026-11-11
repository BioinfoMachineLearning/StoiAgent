#!/usr/bin/env bash
# Download StoiAgent's databases and tools and set up their paths.
#
# By default everything goes to the locations StoiAgent reads without any
# configuration (databases/ and tools/ in this repository). With --db-dir, the
# databases go elsewhere and the script writes config.json in this repository,
# which StoiAgent then reads automatically.
#
# Downloads are resumable: rerun the script after an interruption and it skips
# whatever is already complete.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_URL="${STOIAGENT_DB_URL:-http://sysbio.rnet.missouri.edu/multicom_cluster/stoiagent_db_tools/databases}"
AF3_REPO="https://github.com/google-deepmind/alphafold3.git"
AF3_PARAMS_URL="https://storage.googleapis.com/alphafold3/af3.bin.zst"

DB_DIR="$REPO/databases"
DO_TEMPLATES=1
DO_AF3=1

usage() {
  cat <<EOF
Usage: scripts/setup_databases.sh [--db-dir DIR] [--only templates|af3]

  --db-dir DIR   Put the databases in DIR instead of $REPO/databases
                 and record DIR in $REPO/config.json.
  --only NAME    Set up only the HHsearch database and template metadata (templates, ~33 GB)
                 or only AlphaFold 3 and its databases (af3, ~630 GB).
  -h, --help     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db-dir) DB_DIR="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
    --only)
      case "$2" in
        templates) DO_AF3=0 ;;
        af3) DO_TEMPLATES=0 ;;
        *) echo "--only must be templates or af3" >&2; exit 2 ;;
      esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "Required command not found: $1" >&2; exit 1; }; }

download() {  # download URL DEST, resuming a partial file
  local url="$1" dest="$2"
  if command -v wget >/dev/null; then
    wget -c -q --show-progress -O "$dest.part" "$url"
  else
    need curl
    curl -fL -C - -o "$dest.part" "$url"
  fi
  mv "$dest.part" "$dest"
}

extract_flat() {  # extract_flat ARCHIVE DIR: unpack the archive's files directly into DIR
  local archive="$1" dir="$2" first depth
  # The archives store files under the path they were packed from; strip it.
  first="$(tar -tzf "$archive" 2>/dev/null | grep -v '/$' | head -n 1 || true)"
  [[ -n "$first" ]] || { echo "Empty or unreadable archive: $archive" >&2; exit 1; }
  depth="$(awk -F/ '{print NF-1}' <<<"$first")"
  mkdir -p "$dir.tmp"
  tar -xzf "$archive" -C "$dir.tmp" --strip-components="$depth"
  rm -rf "$dir"
  mv "$dir.tmp" "$dir"
}

setup_hhsuitedb() {
  local dir="$DB_DIR/sort90_hhsuitedb"
  log "HHsearch template database -> $dir"
  if [[ -s "$dir/hhsuitedb3_hhm.ffindex" && ! -e "$DB_DIR/sort90_hhsuitedb.tar.gz" ]]; then
    echo "already present"
    return
  fi
  mkdir -p "$DB_DIR"
  [[ -s "$DB_DIR/sort90_hhsuitedb.tar.gz" ]] || \
    download "$TEMPLATE_URL/sort90_hhsuitedb.tar.gz" "$DB_DIR/sort90_hhsuitedb.tar.gz"
  echo "Extracting sort90_hhsuitedb ..."
  extract_flat "$DB_DIR/sort90_hhsuitedb.tar.gz" "$dir"
  rm -f "$DB_DIR/sort90_hhsuitedb.tar.gz"
}

setup_templates() {
  local dir="$DB_DIR/templates" name
  setup_hhsuitedb
  log "Template metadata -> $dir"
  mkdir -p "$dir"
  if [[ -s "$dir/pdb_stoichiometry.csv" ]]; then
    echo "pdb_stoichiometry.csv already present"
  else
    download "$TEMPLATE_URL/pdb_stoichiometry.csv" "$dir/pdb_stoichiometry.csv"
  fi
  for name in entity_fasta entity_metadata chain_atoms; do
    if [[ -d "$dir/$name" && -n "$(ls -A "$dir/$name")" && ! -e "$dir/$name.tar.gz" ]]; then
      echo "$name already present"
      continue
    fi
    [[ -s "$dir/$name.tar.gz" ]] || download "$TEMPLATE_URL/$name.tar.gz" "$dir/$name.tar.gz"
    echo "Extracting $name ..."
    extract_flat "$dir/$name.tar.gz" "$dir/$name"
    rm -f "$dir/$name.tar.gz"
  done
}

setup_af3() {
  local src="$REPO/tools/alphafold3" dbs="$DB_DIR/af3"
  need git
  log "AlphaFold 3 source -> $src"
  if [[ -d "$src/.git" ]]; then
    echo "already cloned"
  else
    git clone "$AF3_REPO" "$src"
  fi
  mkdir -p "$src/models"

  log "AlphaFold 3 model parameters -> $src/models"
  echo "Use is subject to the AF3 weights terms of use:"
  echo "https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md"
  if [[ -s "$src/models/af3.bin" || -s "$src/models/af3.bin.zst" ]]; then
    echo "already present"
  else
    # AF3 reads the zstd-compressed file directly, so it is kept compressed.
    download "$AF3_PARAMS_URL" "$src/models/af3.bin.zst"
  fi

  log "AlphaFold 3 databases (~630 GB) -> $dbs"
  if [[ -e "$dbs/.stoiagent_complete" ]]; then
    echo "already downloaded"
  else
    need zstd
    mkdir -p "$dbs"
    bash "$src/fetch_databases.sh" "$dbs"  # upstream does not always mark it executable
    touch "$dbs/.stoiagent_complete"
  fi
}

write_config() {
  local config="$REPO/config.json"
  if [[ "$DB_DIR" == "$REPO/databases" ]]; then
    return
  fi
  log "Recording database locations in $config"
  python3 - "$config" "$DB_DIR" "$DO_TEMPLATES" "$DO_AF3" <<'EOF'
import json, sys
from pathlib import Path
config_path, db = Path(sys.argv[1]), Path(sys.argv[2])
do_templates, do_af3 = sys.argv[3] == "1", sys.argv[4] == "1"
config = json.loads(config_path.read_text()) if config_path.is_file() else {}
if do_templates:
    config.update({
        "hhdb_prefix": str(db / "sort90_hhsuitedb/hhsuitedb3"),
        "template_entity_fasta_dir": str(db / "templates/entity_fasta"),
        "template_entity_metadata_dir": str(db / "templates/entity_metadata"),
        "template_chain_atoms_dir": str(db / "templates/chain_atoms"),
        "template_stoichiometry_csv": str(db / "templates/pdb_stoichiometry.csv"),
    })
if do_af3:
    config.update({
        "uniref90_path": str(db / "af3/uniref90_2022_05.fa"),
        "af3_db_path": str(db / "af3"),
    })
config_path.write_text(json.dumps(config, indent=2) + "\n")
EOF
}

(( DO_TEMPLATES )) && setup_templates
(( DO_AF3 )) && setup_af3
write_config

log "Remaining manual steps"
missing=0
if (( DO_AF3 )) && [[ ! -e "$REPO/tools/alphafold3/models/af3.bin" && ! -e "$REPO/tools/alphafold3/models/af3.bin.zst" ]]; then
  echo "- Download the AF3 model parameters into $REPO/tools/alphafold3/models/:"
  echo "  $AF3_PARAMS_URL"
  missing=1
fi
if (( DO_AF3 )) && ! { command -v docker >/dev/null && docker image inspect stoiagent-alphafold3 >/dev/null 2>&1; }; then
  echo "- Build the AF3 Docker image:"
  echo "  docker build -t stoiagent-alphafold3 -f tools/alphafold3/docker/Dockerfile tools/alphafold3"
  missing=1
fi
for tool in jackhmmer hhmake hhsearch; do
  if ! command -v "$tool" >/dev/null; then
    echo "- $tool not found: activate the stoiagent environment (mamba activate stoiagent)."
    missing=1
  fi
done
(( missing )) || echo "None. StoiAgent is ready to run."
