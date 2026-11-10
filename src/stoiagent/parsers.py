# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HHR and gzipped PDB parsing functions used by the JSON builder.

Adapted from the parser implementation supplied with StoiAgent. Only the
functions used by ``build_target_json.py`` are included here.
"""

import dataclasses
import gzip
import re
from typing import Sequence


@dataclasses.dataclass
class TemplateHit:
    index: int
    name: str
    prob: float
    e_value: float
    p_value: float
    score: float
    ss: float
    aligned_cols: int
    query_range: str
    hit_range: str
    hit_length: int
    citation: str = ""
    complex_stoichiometry: str = ""
    number_of_copies_in_complex: int = 0
    coverage: float = 0.0
    identity: float = 0.0


_HHR_TABLE_ROW = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+([\d.]+)\s+([\d.Ee+-]+)\s+"
    r"([\d.Ee+-]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+"
    r"([\d-]+)\s+([\d-]+)\s*\((\d+)\)"
)


def parse_hhr_table(hhr_text: str) -> Sequence[TemplateHit]:
    hits = []
    in_table = False
    for line in hhr_text.splitlines():
        line = line.strip()
        if line.startswith("No Hit"):
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("Probab="):
            break
        match = _HHR_TABLE_ROW.match(line)
        if match is None:
            continue
        hits.append(
            TemplateHit(
                index=int(match.group(1)),
                name=match.group(2),
                prob=float(match.group(3)),
                e_value=float(match.group(4)),
                p_value=float(match.group(5)),
                score=float(match.group(6)),
                ss=float(match.group(7)),
                aligned_cols=int(match.group(8)),
                query_range=match.group(9),
                hit_range=match.group(10),
                hit_length=int(match.group(11)),
            )
        )
    return hits


_AA_MAP = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


def read_pdb_sequence(atom_gz_file: str, chain_id: str = None) -> str:
    """Extract a sequence from a gzipped PDB ATOM file."""
    sequence = []
    seen_residues = set()
    with gzip.open(atom_gz_file, "rt") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21].strip()
            if chain_id and chain != chain_id:
                continue
            residue_key = (chain, line[22:26].strip())
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            sequence.append(_AA_MAP.get(line[17:20].strip(), "X"))
    return "".join(sequence)


def _update_residue_indices(sequence: str, start_index: int, indices: list):
    position = start_index
    for symbol in sequence:
        if symbol == "-":
            indices.append(-1)
        else:
            indices.append(position)
            position += 1


def _parse_hhr_hit(lines: Sequence[str]) -> TemplateHit:
    hit_index = int(lines[0].split()[-1])
    hit_name = lines[1][1:].strip()
    pattern = (
        r"Probab=(.*?)[\t ]+E-value=(.*?)[\t ]+Score=(.*?)[\t ]+"
        r"Aligned_cols=(.*?)[\t ]+Identities=(.*?)%[\t ]+"
        r"Similarity=(.*?)[\t ]+Sum_probs=(.*?)[\t ]+Template_Neff=(.*)"
    )
    match = re.match(pattern, lines[2])
    if match is None:
        raise ValueError(f"Could not parse HHR summary line: {lines[2]}")
    prob, e_value, score, aligned_cols, identities, *_ = map(float, match.groups())

    query_indices = []
    hit_indices = []
    for line in lines[3:]:
        if line.startswith("Q ") and not line.startswith(
            ("Q ss_dssp", "Q ss_pred", "Q Consensus", "Q ss_conf")
        ):
            match = re.match(
                r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)",
                line[17:],
            )
            if match is None:
                raise ValueError(f"Could not parse query alignment line: {line}")
            _update_residue_indices(match.group(2), int(match.group(1)) - 1, query_indices)
        elif line.startswith("T ") and not line.startswith(
            ("T ss_dssp", "T ss_pred", "T Consensus", "T ss_conf")
        ):
            match = re.match(
                r"[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \(([0-9]*)\)",
                line[17:],
            )
            if match is None:
                raise ValueError(f"Could not parse template alignment line: {line}")
            _update_residue_indices(match.group(2), int(match.group(1)) - 1, hit_indices)

    query_valid = [index for index in query_indices if index >= 0]
    hit_valid = [index for index in hit_indices if index >= 0]
    query_range = (
        f"{min(query_valid) + 1}-{max(query_valid) + 1}" if query_valid else ""
    )
    hit_range = f"{min(hit_valid) + 1}-{max(hit_valid) + 1}" if hit_valid else ""
    return TemplateHit(
        index=hit_index,
        name=hit_name,
        prob=prob,
        e_value=e_value,
        p_value=0.0,
        score=score,
        ss=0.0,
        aligned_cols=int(aligned_cols),
        query_range=query_range,
        hit_range=hit_range,
        hit_length=0,
        identity=identities / 100.0,
    )


def parse_hhr(hhr_string: str) -> Sequence[TemplateHit]:
    """Combine an HH-suite HHR summary table with detailed hit alignments."""
    table_hits = {hit.index: hit for hit in parse_hhr_table(hhr_string)}
    lines = hhr_string.splitlines()
    block_starts = [
        index for index, line in enumerate(lines) if re.match(r"^No\s+\d+", line)
    ]
    hits = []
    for start, end in zip(block_starts, block_starts[1:] + [len(lines)]):
        block = lines[start:end]
        try:
            detail = _parse_hhr_hit(block)
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid HHR hit block at line {start + 1}: {exc}") from exc
        table = table_hits.get(detail.index)
        if table is not None:
            detail.name = table.name
            detail.prob = table.prob
            detail.e_value = table.e_value
            detail.p_value = table.p_value
            detail.score = table.score
            detail.ss = table.ss
            detail.aligned_cols = table.aligned_cols
            detail.hit_length = table.hit_length
            detail.query_range = detail.query_range or table.query_range
            detail.hit_range = detail.hit_range or table.hit_range
        hits.append(detail)
    return hits
