#!/usr/bin/env python3
"""Freeze 25/50/75-percentile representatives from an exporter cases.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from qualitative_common import write_csv, write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--score-column", default="auto", help="column name or auto hierarchy")
    args = p.parse_args()

    with args.cases_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 3:
        raise RuntimeError("at least three audited cases are required for 25/50/75 selection")
    if args.score_column == "auto":
        priority = ("seq_ddfp_diff", "boundary_diff", "raw_ddfp_change", "score")
        available = [c for c in priority if c in rows[0]]
        if not available:
            raise RuntimeError(f"no supported burden column; found {list(rows[0])}")
        score_column = next(
            (c for c in available if any(float(r[c]) > 0 for r in rows)),
            available[-1],
        )
    else:
        score_column = args.score_column
    for row in rows:
        row["_score"] = float(row[score_column])
    ranked = sorted(rows, key=lambda r: (r["_score"], r["case_id"]))
    targets = (("lower", 0.25), ("median", 0.50), ("higher", 0.75))
    selected = []
    used = set()
    for label, percentile in targets:
        target_index = percentile * (len(ranked) - 1)
        candidates = sorted(
            ((abs(i - target_index), i, row) for i, row in enumerate(ranked) if row["case_id"] not in used),
            key=lambda item: (item[0], item[1], item[2]["case_id"]),
        )
        _, index, row = candidates[0]
        used.add(row["case_id"])
        clean = {k: v for k, v in row.items() if k != "_score"}
        clean.update({
            "selection_metric": score_column,
            "selection_score": row[score_column],
            "selection_label": label,
            "target_percentile": percentile,
            "target_zero_based_rank": target_index,
            "actual_rank_one_based": index + 1,
            "total_cases": len(ranked),
        })
        selected.append(clean)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selected_cases.csv", selected)
    write_json(args.output_dir / "selected_cases.json", selected)
    protocol = {
        "population_file": str(args.cases_csv.resolve()),
        "score_column": score_column,
        "metric_hierarchy": ["seq_ddfp_diff", "boundary_diff", "raw_ddfp_change", "score"],
        "auto_metric_rule": "first available metric with any positive value; if none positive, last available proxy",
        "ordering": ["numeric score ascending", "case_id lexicographic ascending"],
        "targets": [0.25, 0.50, 0.75],
        "target_rank_formula": "p * (N - 1), zero-based",
        "nearest_rule": "minimum absolute rank distance; lower rank then case_id on ties",
        "duplicate_rule": "exclude already selected case and choose nearest remaining rank",
        "visual_inspection_used": False,
    }
    write_json(args.output_dir / "selection_protocol_frozen.json", protocol)
    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
