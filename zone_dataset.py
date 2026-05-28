from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SOFT_TARGETS = {
    0: [1.0, 0.0],  # tier 0 → healthy
    1: [0.5, 0.5],  # tier 1 → ambiguous
    2: [0.0, 1.0],  # tier 2 → active disease
}

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
    raw_label: int
    # Populated only for the new on-the-fly schema (Cleaned_Image column).
    # When ``cx`` is None the record is "legacy": ``image_path`` already points
    # at a pre-cropped zone PNG and is loaded directly by ``ZoneImageDataset``.
    cx: int | None = None
    cy: int | None = None
    angle_deg: float | None = None


def _read_zone_meta(json_path: Path) -> tuple[int, int, float]:
    with json_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta["cx"]), int(meta["cy"]), float(meta["angle_deg"])


def load_zone_records(csv_path: Path, data_root: Path) -> list[ZoneRecord]:
    """Load zone records from either the new (Cleaned_Image) or legacy (Zone_Image) CSV.

    New schema columns: ``Patient_ID, Cleaned_Image, Zone_Number, Zone_Label``
    Legacy columns:     ``Patient_ID, Zone_Image,    Zone_Number, Zone_Label``

    For the new schema each row's ``Cleaned_Image`` points at a ``.npy`` cached
    by ``pre_processing.py``; the matching ``.json`` sidecar stores the geometry
    (``cx``, ``cy``, ``angle_deg``) used to mask + crop the requested zone at
    ``__getitem__`` time.
    """
    records: list[ZoneRecord] = []
    meta_cache: dict[Path, tuple[int, int, float]] = {}
    missing_meta: list[Path] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "Cleaned_Image" in fieldnames:
            image_col = "Cleaned_Image"
            schema = "cleaned"
        elif "Zone_Image" in fieldnames:
            image_col = "Zone_Image"
            schema = "legacy"
        else:
            raise ValueError(
                f"{csv_path} must contain either Cleaned_Image or Zone_Image column "
                f"(got {sorted(fieldnames)})"
            )
        required = {"Patient_ID", image_col, "Zone_Number", "Zone_Label"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        for row in reader:
            image_rel = Path(str(row[image_col]).replace("\\", "/"))
            abs_path = data_root / image_rel
            raw_label = int(float(row["Zone_Label"]))
            patient_id = int(float(row["Patient_ID"]))
            zone_number = int(float(row["Zone_Number"]))
            label = zone_label_to_binary(raw_label)

            if schema == "legacy":
                records.append(
                    ZoneRecord(
                        patient_id=patient_id,
                        image_path=abs_path,
                        zone_number=zone_number,
                        label=label,
                        raw_label=raw_label,
                    )
                )
                continue

            meta_path = abs_path.with_suffix(".json")
            if meta_path in meta_cache:
                cx, cy, angle = meta_cache[meta_path]
            else:
                try:
                    cx, cy, angle = _read_zone_meta(meta_path)
                except FileNotFoundError:
                    missing_meta.append(meta_path)
                    continue
                meta_cache[meta_path] = (cx, cy, angle)

            records.append(
                ZoneRecord(
                    patient_id=patient_id,
                    image_path=abs_path,
                    zone_number=zone_number,
                    label=label,
                    raw_label=raw_label,
                    cx=cx,
                    cy=cy,
                    angle_deg=angle,
                )
            )

    if missing_meta:
        sample = ", ".join(str(p) for p in missing_meta[:5])
        raise FileNotFoundError(
            f"{len(missing_meta)} sidecar .json files were missing under {data_root}. "
            f"First few: {sample}. Re-run pre_processing.py to (re)build the cache."
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


def _load_zone_image(record: ZoneRecord) -> Image.Image:
    """Load the requested zone as an RGB PIL Image.

    Two code paths:
      * legacy (``record.cx is None``) -- the path already points at a pre-cropped
        zone PNG; just open + convert.
      * new   (``record.cx`` populated) -- the path points at a cleaned RGBA
        ``.npy`` for the full FP image; compute the one zone mask we need,
        zero alpha outside it, crop to content, then return RGB.
    """
    if record.cx is None:
        return Image.open(record.image_path).convert("RGB")

    # Lazy import so non-training code paths (e.g. label inspection) don't pay
    # the cv2 import cost.
    from extract_zones import apply_zone_and_crop, make_zone_mask

    # mmap so we only page in the (typically small) zone bbox region, not the
    # full 60+ MB FP array. ``apply_zone_and_crop`` does a ``.copy()`` of the
    # sub-slice, materialising it in RAM as needed.
    cleaned = np.load(record.image_path, mmap_mode="r")
    if cleaned.ndim != 3 or cleaned.shape[2] != 4:
        raise ValueError(
            f"Expected HxWx4 RGBA array at {record.image_path}, got {cleaned.shape}"
        )
    h, w = cleaned.shape[:2]
    mask = make_zone_mask(
        width=w,
        height=h,
        cx=record.cx,
        cy=record.cy,
        zone_number=record.zone_number,
        angle_deg=record.angle_deg or 0.0,
    )
    zone_rgba = apply_zone_and_crop(cleaned, mask)
    return Image.fromarray(zone_rgba, mode="RGBA").convert("RGB")


TIER_CONFIDENCE_WEIGHTS = {0: 1.0, 1: 0.3, 2: 1.0}


class ZoneImageDataset(Dataset):
    def __init__(
        self,
        records: list[ZoneRecord],
        transform: Callable | None = None,
        soft_labels: bool = False,
        tier_confidence_weights: bool = False,
    ) -> None:
        self.records = records
        self.transform = transform
        self.soft_labels = soft_labels
        self.tier_confidence_weights = tier_confidence_weights

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = _load_zone_image(record)
        if self.transform is not None:
            image = self.transform(image)
        if self.soft_labels:
            label = torch.tensor(SOFT_TARGETS[record.raw_label], dtype=torch.float32)
        else:
            label = record.label
        out = {
            "image": image,
            "label": label,
            "patient_id": record.patient_id,
            "zone_number": record.zone_number,
            "image_path": str(record.image_path),
        }
        if self.tier_confidence_weights:
            out["sample_weight"] = TIER_CONFIDENCE_WEIGHTS[record.raw_label]
        return out
