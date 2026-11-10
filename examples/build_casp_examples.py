#!/usr/bin/env python3
"""Build per-target FASTA inputs and a true-stoichiometry table for CASP16/17.

Inputs (in this folder):
  casp16.HT1.seq.txt, casp17.HT1.seq.txt   CASP sequence releases
  casp16.targetlist.csv, casp17.targetlist.csv
      Prediction Center target lists (targetlist.cgi?type=csv), ';'-separated.

Each benchmark target is a stoichiometry-blind protein target: one released
with oligomeric state "UNK". Cancelled targets, targets that turned out to be
A1 monomers, and targets whose stoichiometry CASP has not released are excluded. In CASP16 these are the Phase-0 targets (H0xxx/
T0xxx); in CASP17 they are the Phase-1 targets (H1xxx/T1xxx). The sequence
releases do not always use those IDs, so IDs are mapped the same way as in
stoiagent.agent_common.resolve_casp17_sequence_target and
stoiagent.stage2_rerank.target_info_from_catalog: the H/T letter and the last
three digits identify a target, and the phase digit is swapped
(CASP16 H1202 -> H0202, CASP17 H2346 -> H1346). Sequence entries with a suffix
(T1208s1, T1228v1) are single subunits or variants of another target and are
skipped, as the scripts' "[HT]\\d{4}\\b" header pattern does.

The true stoichiometry is the oligomeric state CASP released for the same
target in the next phase (CASP16 Phase 1 H1xxx/T1xxx, CASP17 Phase 2
H2xxx/T2xxx), which is taken from the experimental structure.

Outputs:
  casp16/<ID>.fasta, casp17/<ID>.fasta   one record per distinct subunit, A/B/C order
  casp16_stoichiometry.csv, casp17_stoichiometry.csv
"""

from __future__ import annotations

import csv
import re
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# CASP edition -> (blind phase digit, truth phase digit)
EDITIONS = {16: ("0", "1"), 17: ("1", "2")}
HEADER = re.compile(
    r"^(?P<id>[HT]\d{4})\s+(?P<body>.*?),\s*(?P<length>\d+)\s+residues\s*[;|]?\s*$")


def read_target_list(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        columns = [c.strip() for c in next(reader)]
        out = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            # Descriptions may themselves contain ';': keep the overflow in the last column.
            row = row[:len(columns) - 1] + [";".join(row[len(columns) - 1:])]
            out[row[0].strip()] = {c: v.strip() for c, v in zip(columns, row)}
        return out


def read_sequence_release(path: Path) -> "OrderedDict[str, list[dict]]":
    """Records per target in file order; suffixed sub-entries are skipped."""
    targets: OrderedDict[str, list[dict]] = OrderedDict()
    header, seq = None, []

    def flush():
        if header is None:
            return
        m = HEADER.match(header)
        if not m:  # e.g. T1208s1, T1228v1
            return
        body = re.sub(r",\s*subunit\s+\d+\s*$", "", m["body"].strip(), flags=re.I)
        desc, org = ([p.strip() for p in body.rsplit(",", 1)] if "," in body else [body, ""])
        sequence = "".join(seq).upper()
        if len(sequence) != int(m["length"]):
            raise ValueError(f"{m['id']}: header says {m['length']} residues, sequence has {len(sequence)}")
        targets.setdefault(m["id"], []).append(
            {"description": desc, "organism": org, "sequence": sequence})

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(">"):
            flush()
            header, seq = line[1:], []
        elif line:
            seq.append(line)
    flush()
    return targets


def stoich_subunits(stoich: str) -> int:
    return len(re.findall(r"[A-Z]\d+", stoich))


def build(edition: int) -> list[dict]:
    blind_digit, truth_digit = EDITIONS[edition]
    target_list = read_target_list(HERE / f"casp{edition}.targetlist.csv")
    release = read_sequence_release(HERE / f"casp{edition}.HT1.seq.txt")
    out_dir = HERE / f"casp{edition}"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*.fasta"):
        old.unlink()

    rows = []
    for source_id, records in release.items():
        kind, number = source_id[0], source_id[2:]
        target_id = f"{kind}{blind_digit}{number}"
        blind = target_list.get(target_id)
        if not blind or blind["Oligo.State"] != "UNK":
            continue  # stoichiometry was given at release: not a blind target
        if blind.get("Cancellation Date", "-") not in ("", "-"):
            continue  # cancelled target

        # Distinct subunits in release order (A, B, C, ...).
        distinct, seen = [], set()
        for r in records:
            if r["sequence"] not in seen:
                seen.add(r["sequence"])
                distinct.append(r)

        truth_id = f"{kind}{truth_digit}{number}"
        truth = target_list.get(truth_id, {}).get("Oligo.State", "")
        if truth == "A1":
            continue  # turned out to be a monomer: no stoichiometry to predict
        if not re.fullmatch(r"([A-Z]\d+)+", truth):
            continue  # stoichiometry not released (or not a definite one, e.g. "An")
        if stoich_subunits(truth) != len(distinct):
            raise ValueError(f"{target_id}: {truth} has {stoich_subunits(truth)} subunits, "
                             f"FASTA has {len(distinct)}")

        lines = []
        for letter, r in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", distinct):
            fields = [r["description"]] + ([r["organism"]] if r["organism"] else [])
            lines += [f">{target_id}_{letter} {', '.join(fields)}, {len(r['sequence'])} residues",
                      r["sequence"]]
        (out_dir / f"{target_id}.fasta").write_text("\n".join(lines) + "\n", encoding="utf-8")

        first = next((r for r in distinct if r["description"]), distinct[0])
        rows.append({
            "target": target_id,
            "stoichiometry": truth,
            "num_subunits": len(distinct),
            "description": first["description"],
            "organism": next((r["organism"] for r in distinct if r["organism"]), ""),
            "sequence_release_id": source_id,
            "stoichiometry_source": truth_id,
        })

    rows.sort(key=lambda r: (r["target"][0], r["target"]))
    with (HERE / f"casp{edition}_stoichiometry.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    for edition in EDITIONS:
        rows = build(edition)
        print(f"CASP{edition}: {len(rows)} targets")


if __name__ == "__main__":
    main()
