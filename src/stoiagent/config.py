"""Default locations of databases and tools, and config-file loading.

Every path defaults to a fixed location inside the StoiAgent folder, so a
config file is only needed to override some of them. Relative paths, in the
defaults or in a user config, are resolved against the StoiAgent folder.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    # Template search
    "uniref90_path": "databases/af3/uniref90_2022_05.fa",
    "hhdb_prefix": "databases/pdb70/pdb70",
    "jackhmmer_binary": "jackhmmer",
    "hhmake_binary": "hhmake",
    "hhsearch_binary": "hhsearch",
    "search_cpu": 8,
    # Template metadata
    "template_fasta_dir": "databases/templates/fasta",
    "template_pdb_json_dir": "databases/templates/json",
    "template_atom_dir": "databases/templates/atom",
    "template_label_csv": "databases/templates/labels.csv",
    # AlphaFold 3
    "af3_runtime": "docker",
    "af3_program_path": "tools/alphafold3",
    "af3_params_path": "tools/alphafold3/models",
    "af3_db_path": "databases/af3",
    "af3_image": "alphafold3",
    "af3_sif_path": "tools/alphafold3.sif",
    "gpu_device": "0",
}

PATH_KEYS = (
    "uniref90_path", "hhdb_prefix",
    "template_fasta_dir", "template_pdb_json_dir", "template_atom_dir", "template_label_csv",
    "af3_program_path", "af3_params_path", "af3_db_path", "af3_sif_path",
)

# Older config files used these names.
_LEGACY_KEYS = {
    "stoichiometry_pdb_fasta_folder": "template_fasta_dir",
    "pdb_stoichiometry_database": "template_label_csv",
}


def stoiagent_home() -> Path:
    """The folder that relative resource paths are resolved against.

    STOIAGENT_HOME overrides it. Otherwise it is the repository checkout this
    package was installed from (``pip install -e .``), else the working directory.
    """
    env = os.environ.get("STOIAGENT_HOME")
    if env:
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout
    return Path.cwd()


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Return the defaults, overridden by the JSON file at ``path`` if given."""
    config = dict(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as f:
            user = json.load(f)
        for old, new in _LEGACY_KEYS.items():
            if user.get(old) and not user.get(new):
                user[new] = user[old]
        config.update({k: v for k, v in user.items() if v not in (None, "")})
    home = stoiagent_home()
    for key in PATH_KEYS:
        p = Path(str(config[key])).expanduser()
        config[key] = str(p if p.is_absolute() else home / p)
    return config


def require_paths(config: dict, keys) -> None:
    """Raise FileNotFoundError naming every configured path that does not exist.

    ``hhdb_prefix`` is a filename prefix, so it counts as present when any file
    starts with it.
    """
    missing = []
    for key in keys:
        p = Path(config[key])
        exists = any(p.parent.glob(p.name + "*")) if key == "hhdb_prefix" else p.exists()
        if not exists:
            missing.append(f"  {key}: {p}")
    if missing:
        raise FileNotFoundError(
            "These StoiAgent resources were not found. Download them to these "
            "locations (see README) or set the paths in --config:\n" + "\n".join(missing)
        )
