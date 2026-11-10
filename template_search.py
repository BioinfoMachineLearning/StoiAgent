"""Generate per-subunit HHsearch results from a multi-record target FASTA.

HHR files are internal intermediates. The public target input is FASTA.
"""

from __future__ import annotations

import re
import subprocess
import json
import hashlib
from collections import OrderedDict
from pathlib import Path


def read_target_fasta(path: str | Path) -> dict[str, str]:
    """Assign canonical A/B/... subunit IDs in FASTA record order."""
    records = []
    current_header = None
    current_sequence = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_sequence)))
            current_header = line[1:].strip()
            current_sequence = []
        else:
            if current_header is None:
                raise ValueError("FASTA sequence appears before a header")
            current_sequence.append(line)
    if current_header is not None:
        records.append((current_header, "".join(current_sequence)))
    if not records:
        raise ValueError(f"No sequences found in {path}")
    sequences = {}
    seen = set()
    for header, raw_sequence in records:
        sequence = raw_sequence.upper().replace(" ", "")
        if not sequence or not re.fullmatch(r"[A-Z]+", sequence):
            raise ValueError(f"Invalid protein sequence in FASTA record {header}")
        if sequence in seen:
            continue
        if len(sequences) >= 26:
            raise ValueError("At most 26 distinct subunits are supported")
        seen.add(sequence)
        sequences[chr(ord("A") + len(sequences))] = sequence
    return sequences


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def stockholm_to_a3m(stockholm: str) -> str:
    """Convert jackhmmer's Stockholm alignment to a deduplicated A3M."""
    rows = OrderedDict()
    for raw in stockholm.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "//":
            continue
        name, segment = line.split(maxsplit=1)
        rows[name] = rows.get(name, "") + segment.replace(".", "-")
    if not rows:
        raise ValueError("Jackhmmer returned an empty Stockholm alignment")
    lengths = {len(value) for value in rows.values()}
    if len(lengths) != 1:
        raise ValueError("Jackhmmer Stockholm rows have inconsistent lengths")
    names = list(rows)
    query = rows[names[0]]
    keep = [i for i in range(len(query)) if any(row[i] != "-" for row in rows.values())]
    query = "".join(query[i] for i in keep)
    records = []
    seen = set()
    for name, full in rows.items():
        aligned = "".join(full[i] for i in keep)
        key = "".join(c for c, q in zip(aligned, query) if q != "-")
        if key in seen:
            continue
        seen.add(key)
        a3m = "".join(
            c if q != "-" else c.lower()
            for c, q in zip(aligned, query) if q != "-" or c != "-"
        )
        records.append(f">{name}\n{a3m}")
    return "\n".join(records) + "\n"


def generate_template_search_inputs(
    input_fasta: str | Path,
    workdir: str | Path,
    *,
    uniref90_db: str | Path,
    hhdb_prefix: str | Path,
    jackhmmer_binary: str = "jackhmmer",
    hhmake_binary: str = "hhmake",
    hhsearch_binary: str = "hhsearch",
    cpu: int = 8,
) -> dict[str, dict[str, str]]:
    """Run jackhmmer -> hhmake -> hhsearch for each distinct subunit.

    Results are cached in workdir. To avoid stale results, use a new workdir
    when changing the input FASTA or search databases.
    """
    sequences = read_target_fasta(input_fasta)
    output = Path(workdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not Path(uniref90_db).is_file():
        raise FileNotFoundError(f"UniRef90 database not found: {uniref90_db}")
    if not list(Path(hhdb_prefix).parent.glob(Path(hhdb_prefix).name + "_*")):
        raise FileNotFoundError(f"HHsearch database prefix not found: {hhdb_prefix}")
    result = {}
    for subunit, sequence in sequences.items():
        fasta = output / f"{subunit}.fasta"
        sto = output / f"{subunit}.sto"
        a3m = output / f"{subunit}.a3m"
        hmm = output / f"{subunit}.hmm"
        hhr = output / f"{subunit}.hhr"
        fingerprint = hashlib.sha256(json.dumps({
            "sequence": sequence, "uniref90_db": str(uniref90_db),
            "hhdb_prefix": str(hhdb_prefix), "jackhmmer": jackhmmer_binary,
            "hhmake": hhmake_binary, "hhsearch": hhsearch_binary,
        }, sort_keys=True).encode()).hexdigest()
        fingerprint_path = output / f"{subunit}.search.sha256"
        if not fingerprint_path.is_file() or fingerprint_path.read_text().strip() != fingerprint:
            for old in (sto, a3m, hmm, hhr):
                old.unlink(missing_ok=True)
        fasta.write_text(f">{subunit}\n{sequence}\n", encoding="utf-8")
        if not sto.is_file():
            _run([
                jackhmmer_binary, "-o", str(output / f"{subunit}.jackhmmer.log"),
                "-A", str(sto), "--noali", "--F1", "0.0005",
                "--F2", "0.00005", "--F3", "0.0000005",
                "--incE", "0.0001", "-E", "0.0001",
                "--cpu", str(cpu), "-N", "1",
                "--tblout", str(output / f"{subunit}.tblout"),
                str(fasta), str(uniref90_db),
            ])
        if not a3m.is_file():
            msa = sto.read_text(encoding="utf-8")
            a3m.write_text(stockholm_to_a3m(msa), encoding="utf-8")
        if not hmm.is_file():
            _run([hhmake_binary, "-i", str(a3m), "-o", str(hmm)])
        if not hhr.is_file():
            _run([hhsearch_binary, "-i", str(hmm), "-o", str(hhr),
                  "-maxseq", "1000000", "-d", str(hhdb_prefix)])
        if not hhr.is_file():
            raise RuntimeError(f"HHsearch did not produce {hhr}")
        fingerprint_path.write_text(fingerprint + "\n", encoding="utf-8")
        result[subunit] = {"fasta_file": str(fasta), "hhr_file": str(hhr)}
    return result
