"""Audit physical overlap between patch windows assigned to different splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for index, raw in enumerate(csv.DictReader(handle)):
            rows.append(
                {
                    "index": index,
                    "scene_id": raw["scene_id"],
                    "split": raw["split"],
                    "x0": int(raw["x0"]),
                    "y0": int(raw["y0"]),
                    "tile": int(raw["tile"]),
                }
            )
    return rows


def overlap_area(left: dict[str, Any], right: dict[str, Any]) -> int:
    width = min(left["x0"] + left["tile"], right["x0"] + right["tile"]) - max(
        left["x0"], right["x0"]
    )
    height = min(left["y0"] + left["tile"], right["y0"] + right["tile"]) - max(
        left["y0"], right["y0"]
    )
    return max(0, width) * max(0, height)


def audit(rows: list[dict[str, Any]], *, max_examples: int = 20) -> dict[str, Any]:
    if not rows:
        raise ValueError("Patch CSV is empty")
    bucket_size = min(row["tile"] for row in rows)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()
    pair_counts: Counter[str] = Counter()
    impacted: dict[str, set[int]] = defaultdict(set)
    overlap_areas: list[int] = []
    examples: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        bx0 = row["x0"] // bucket_size
        bx1 = (row["x0"] + row["tile"] - 1) // bucket_size
        by0 = row["y0"] // bucket_size
        by1 = (row["y0"] + row["tile"] - 1) // bucket_size
        row_buckets = [
            (bx, by)
            for bx in range(bx0, bx1 + 1)
            for by in range(by0, by1 + 1)
        ]
        candidates: set[int] = set()
        for bucket in row_buckets:
            candidates.update(buckets[bucket])

        for other_index in candidates:
            other = rows[other_index]
            if row["split"] == other["split"]:
                continue
            pair = (other_index, index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            area = overlap_area(row, other)
            if area <= 0:
                continue

            pair_name = "<->".join(sorted((row["split"], other["split"])))
            pair_counts[pair_name] += 1
            impacted[row["split"]].add(index)
            impacted[other["split"]].add(other_index)
            overlap_areas.append(area)
            if len(examples) < max_examples:
                examples.append(
                    {
                        "pair": pair_name,
                        "overlap_px2": area,
                        "left": other,
                        "right": row,
                    }
                )

        for bucket in row_buckets:
            buckets[bucket].append(index)

    split_counts = Counter(row["split"] for row in rows)
    overlap_count = sum(pair_counts.values())
    return {
        "rows": len(rows),
        "scenes": dict(Counter(row["scene_id"] for row in rows)),
        "split_counts": dict(split_counts),
        "cross_split_overlap_pairs": overlap_count,
        "pair_counts": dict(pair_counts),
        "impacted_windows": {
            split: {
                "count": len(indices),
                "fraction": len(indices) / split_counts[split],
            }
            for split, indices in sorted(impacted.items())
        },
        "overlap_area_px2": {
            "mean": (
                sum(overlap_areas) / len(overlap_areas) if overlap_areas else 0.0
            ),
            "max": max(overlap_areas) if overlap_areas else 0,
        },
        "examples": examples,
        "passed": overlap_count == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--fail-on-overlap", action="store_true")
    args = parser.parse_args()

    report = audit(load_rows(args.patch_csv))
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.fail_on_overlap and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
