from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset

# Spreadsheet / legacy table uses three severity tiers {0, 1, 2}. Training collapses
# the two non-zero tiers into a single positive class for binary classification.
MULTICLASS_ZONE_LABELS = frozenset({0, 1, 2})


def zone_label_to_binary(raw_label: int) -> int:
    """Map 0 -> 0 and 1 or 2 -> 1. Idempotent when raw_label is already 0 or 1."""
    if raw_label not in MULTICLASS_ZONE_LABELS:
        raise ValueError(
            f"Zone_Label must be in {sorted(MULTICLASS_ZONE_LABELS)}, got {raw_label!r}"
        )
    return 0 if raw_label == 0 else 1


@dataclass(frozen=True)
class ZoneRecord:
    patient_id: int
    image_path: Path
    zone_number: int
    label: int


def load_zone_records(csv_path: Path, data_root: Path) -> list[ZoneRecord]:
    records: list[ZoneRecord] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Patient_ID", "Zone_Image", "Zone_Number", "Zone_Label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        for row in reader:
            image_rel = Path(str(row["Zone_Image"]).replace("\\", "/"))
            raw_label = int(float(row["Zone_Label"]))
            records.append(
                ZoneRecord(
                    patient_id=int(float(row["Patient_ID"])),
                    image_path=data_root / image_rel,
                    zone_number=int(float(row["Zone_Number"])),
                    label=zone_label_to_binary(raw_label),
                )
            )
    return records


def labels_for(records: Iterable[ZoneRecord]) -> list[int]:
    return [record.label for record in records]


def records_for_patients(
    records: Iterable[ZoneRecord], patient_ids: Iterable[int]
) -> list[ZoneRecord]:
    patient_set = set(patient_ids)
    return [record for record in records if record.patient_id in patient_set]


def make_patient_split(
    records: list[ZoneRecord],
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 13,
    attempts: int = 10_000,
) -> dict[str, list[int]]:
    """Choose a deterministic patient-level split with roughly balanced labels."""
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    patient_ids = sorted({record.patient_id for record in records})
    if len(patient_ids) < 3:
        raise ValueError("Need at least three patients for train/val/test splitting")

    total_counts = Counter(record.label for record in records)
    total_n = sum(total_counts.values())
    labels = sorted(total_counts)
    target_props = {label: total_counts[label] / total_n for label in labels}

    per_patient: dict[int, Counter[int]] = defaultdict(Counter)
    for record in records:
        per_patient[record.patient_id][record.label] += 1

    n_patients = len(patient_ids)
    n_val = max(1, round(n_patients * val_fraction))
    n_test = max(1, round(n_patients * test_fraction))
    n_train = n_patients - n_val - n_test
    if n_train < 1:
        raise ValueError("Split fractions leave no training patients")

    target_sizes = {
        "train": train_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }

    def count_for(split_patients: Iterable[int]) -> Counter[int]:
        counts: Counter[int] = Counter()
        for patient_id in split_patients:
            counts.update(per_patient[patient_id])
        return counts

    def score(split: dict[str, list[int]]) -> float:
        value = 0.0
        for name, split_patients in split.items():
            counts = count_for(split_patients)
            n = sum(counts.values())
            value += 2.0 * abs((n / total_n) - target_sizes[name])
            if name in {"val", "test"}:
                value += 0.5 * sum(1 for label in labels if counts[label] == 0)
            for label in labels:
                prop = counts[label] / n if n else 0.0
                value += abs(prop - target_props[label])
        return value

    rng = random.Random(seed)
    best_split: dict[str, list[int]] | None = None
    best_score = float("inf")
    for _ in range(attempts):
        shuffled = patient_ids[:]
        rng.shuffle(shuffled)
        candidate = {
            "val": sorted(shuffled[:n_val]),
            "test": sorted(shuffled[n_val : n_val + n_test]),
            "train": sorted(shuffled[n_val + n_test :]),
        }
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best_score = candidate_score
            best_split = candidate

    if best_split is None:
        raise RuntimeError("Failed to generate a patient split")
    return {
        "train": best_split["train"],
        "val": best_split["val"],
        "test": best_split["test"],
    }


def load_or_create_split(
    records: list[ZoneRecord],
    split_json: Path | None,
    seed: int,
    attempts: int,
) -> dict[str, list[int]]:
    if split_json is not None and split_json.exists():
        with split_json.open(encoding="utf-8") as f:
            data = json.load(f)
        return {name: [int(pid) for pid in data[name]] for name in ("train", "val", "test")}

    split = make_patient_split(records, seed=seed, attempts=attempts)
    if split_json is not None:
        split_json.parent.mkdir(parents=True, exist_ok=True)
        with split_json.open("w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
    return split


class ZoneImageDataset(Dataset):
    def __init__(
        self,
        records: list[ZoneRecord],
        transform: Callable | None = None,
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": record.label,
            "patient_id": record.patient_id,
            "zone_number": record.zone_number,
            "image_path": str(record.image_path),
        }
