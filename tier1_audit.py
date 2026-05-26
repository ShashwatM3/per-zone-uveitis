#!/usr/bin/env python3
"""Audit raw tier-0/1/2 zone labels per train/val/test split."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pre_processing import (
    build_annotation_index,
    find_default_xlsx,
    parse_eye_and_modality,
)
from zone_dataset import load_or_create_split, load_zone_records

RUN_DIR = Path("runs") / "clip_convnext_ce_zone64_v3"
SPLIT_JSON = RUN_DIR / "split.json"
CSV_PATH = Path("processed_image_arrays") / "zone_training_table.csv"
DATA_ROOT = Path("processed_image_arrays")
DATA_DIR = Path("data")
SPLIT_SEED = 13
SPLIT_ATTEMPTS = 10_000

SPLIT_NAMES = ("train", "val", "test")


def load_split(records: list) -> dict[str, list[int]]:
    if SPLIT_JSON.exists():
        with SPLIT_JSON.open(encoding="utf-8") as f:
            data = json.load(f)
        return {name: [int(pid) for pid in data[name]] for name in SPLIT_NAMES}

    print(
        f"Note: {SPLIT_JSON} not found; regenerating split with seed={SPLIT_SEED} "
        "(same as training when no split JSON was provided)."
    )
    return load_or_create_split(
        records,
        split_json=None,
        seed=SPLIT_SEED,
        attempts=SPLIT_ATTEMPTS,
    )


def lookup_raw_labels(
    label_index: dict[tuple[int, str, str, str], tuple[int, ...]],
    patient_id: int,
    cleaned_rel: str,
) -> tuple[int, ...]:
    path = Path(str(cleaned_rel).replace("\\", "/"))
    visit = path.parts[2]
    fp_stem = path.stem.lower()
    eye, _ = parse_eye_and_modality(path.name)
    if eye not in {"OD", "OS"}:
        raise ValueError(f"Could not parse eye from cleaned image path: {cleaned_rel}")

    key = (patient_id, visit, eye, fp_stem)
    labels = label_index.get(key)
    if labels is None:
        fallback_key = next(
            (
                candidate
                for candidate in label_index
                if candidate[0] == patient_id
                and candidate[1] == visit
                and candidate[2] == eye
            ),
            None,
        )
        if fallback_key is not None:
            labels = label_index[fallback_key]
    if labels is None:
        raise KeyError(f"No annotation labels found for {key}")
    return labels


def load_zone_tiers(
    label_index: dict[tuple[int, str, str, str], tuple[int, ...]],
) -> list[tuple[int, int]]:
    zones: list[tuple[int, int]] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = int(float(row["Patient_ID"]))
            zone_number = int(float(row["Zone_Number"]))
            labels = lookup_raw_labels(label_index, patient_id, row["Cleaned_Image"])
            tier = labels[zone_number - 1]
            if tier not in (0, 1, 2):
                raise ValueError(f"Unexpected raw tier {tier!r} for patient {patient_id}")
            zones.append((patient_id, tier))
    return zones


def summarize_split(
    zones: list[tuple[int, int]],
    patient_ids: set[int],
) -> dict[str, object]:
    split_zones = [(pid, tier) for pid, tier in zones if pid in patient_ids]
    tier_counts = Counter(tier for _, tier in split_zones)
    kept = [(pid, tier) for pid, tier in split_zones if tier != 1]
    kept_counts = Counter(tier for _, tier in kept)

    per_patient_tiers: dict[int, set[int]] = defaultdict(set)
    for pid, tier in split_zones:
        per_patient_tiers[pid].add(tier)
    fully_excluded_patients = sorted(
        pid for pid in patient_ids if per_patient_tiers.get(pid, set()) == {1}
    )

    tier0 = kept_counts.get(0, 0)
    tier2 = kept_counts.get(2, 0)
    kept_total = tier0 + tier2
    if kept_total:
        balance = f"{tier0}/{tier2} ({100.0 * tier0 / kept_total:.1f}% / {100.0 * tier2 / kept_total:.1f}%)"
    else:
        balance = "n/a"

    return {
        "total_rows": len(split_zones),
        "tier_0": tier_counts.get(0, 0),
        "tier_1": tier_counts.get(1, 0),
        "tier_2": tier_counts.get(2, 0),
        "rows_after_exclusion": len(kept),
        "tier_0_after": tier0,
        "tier_2_after": tier2,
        "balance_after": balance,
        "patients_fully_excluded": len(fully_excluded_patients),
        "fully_excluded_patient_ids": fully_excluded_patients,
    }


def print_table(rows: list[tuple[str, dict[str, object]]]) -> None:
    columns = [name for name, _ in rows]

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    metrics = [
        ("Total zone rows", "total_rows"),
        ("Tier 0", "tier_0"),
        ("Tier 1 (ambiguous)", "tier_1"),
        ("Tier 2", "tier_2"),
        ("Rows after tier-1 exclusion", "rows_after_exclusion"),
        ("Tier 0 after exclusion", "tier_0_after"),
        ("Tier 2 after exclusion", "tier_2_after"),
        ("Class balance (tier 0 vs tier 2)", "balance_after"),
        ("Patients fully excluded", "patients_fully_excluded"),
    ]

    col_widths = []
    for col_idx, name in enumerate(columns):
        width = len(name)
        _, stats = rows[col_idx]
        for _, key in metrics:
            width = max(width, len(fmt(stats[key])))
        col_widths.append(width + 2)

    header = f"{'Metric':<34}" + "".join(
        f"{name:>{col_widths[idx]}}" for idx, name in enumerate(columns)
    )
    print(header)
    print("-" * len(header))
    for label, key in metrics:
        line = f"{label:<34}"
        for col_idx, (_, stats) in enumerate(rows):
            line += f"{fmt(stats[key]):>{col_widths[col_idx]}}"
        print(line)


def main() -> int:
    xlsx_path = find_default_xlsx(DATA_DIR)
    if xlsx_path is None:
        raise FileNotFoundError(f"No annotation spreadsheet found under {DATA_DIR}")

    label_index = build_annotation_index(xlsx_path)
    zones = load_zone_tiers(label_index)

    records = load_zone_records(CSV_PATH, DATA_ROOT)
    split = load_split(records)

    print("Tier-1 zone label audit")
    print(f"Annotations: {xlsx_path}")
    print(f"Zone table:  {CSV_PATH}")
    print(f"Split source: {SPLIT_JSON if SPLIT_JSON.exists() else f'seed={SPLIT_SEED} (in-memory)'}")
    print(f"Total zone rows in table: {len(zones)}")
    print()

    split_rows: list[tuple[str, dict[str, object]]] = []
    all_patient_ids = set().union(*(split[name] for name in SPLIT_NAMES))
    all_stats = summarize_split(zones, all_patient_ids)

    for name in SPLIT_NAMES:
        stats = summarize_split(zones, set(split[name]))
        split_rows.append((name, stats))

    split_rows.append(("all", all_stats))
    print_table(split_rows)
    print()

    print("Fully excluded patients (all zones are tier 1):")
    for name in SPLIT_NAMES:
        stats = summarize_split(zones, set(split[name]))
        ids = stats["fully_excluded_patient_ids"]
        if ids:
            rendered = ", ".join(str(pid) for pid in ids)
            print(f"  {name:>5}: {len(ids)} patient(s) -> {rendered}")
        else:
            print(f"  {name:>5}: 0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
