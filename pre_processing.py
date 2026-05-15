"""
Preprocess retinal images into patient-wise, visit-wise NumPy data, and build a
zone-level training table from fundus (FP) crops + spreadsheet labels.

Behavior:
  - Walks `data/Patient*/<visit>/`.
  - Loads all image files inside each visit.
  - Flips images whose filename indicates OS horizontally (OD-axis convention).
  - Saves arrays per visit as `.npz`, plus metadata JSON (unchanged).
  - For each FP image that matches a row in the annotations `.xlsx`, runs
    `extract_zones.extract` on the standardized (flipped-if-OS) image, saves
    each zone crop as a PNG, and appends rows to `zone_training_table.csv`.

Table columns: Patient_ID, Zone_Image, Zone_Number, Zone_Label

By default each Zone_Label is written in binary form: spreadsheet tiers 0, 1, and 2
become 0, 1, and 1 respectively (classes 1 and 2 merged). Pass
``--multiclass-zone-labels`` to write raw 0/1/2 instead; training still maps to
binary when loading the CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from extract_zones import extract
from zone_dataset import zone_label_to_binary


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
EYE_PATTERN = re.compile(r"(?:^|[_-])(OD|OS)(?:[_-]|$)", re.IGNORECASE)
MODALITY_PATTERN = re.compile(r"(?:^|[_-])(FA|FP)(?:[_-]|$)", re.IGNORECASE)


def parse_eye_and_modality(filename: str) -> Tuple[Optional[str], Optional[str]]:
    stem = Path(filename).stem
    eye_match = EYE_PATTERN.search(stem)
    modality_match = MODALITY_PATTERN.search(stem)
    eye = eye_match.group(1).upper() if eye_match else None
    modality = modality_match.group(1).upper() if modality_match else None
    return eye, modality


def patient_dir_to_id(patient_dir_name: str) -> Optional[int]:
    m = re.match(r"^Patient(\d+)$", patient_dir_name, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def visit_key_from_cell(visit_date) -> str:
    """Normalize Excel visit date to YYYYMMDD folder token."""
    if visit_date is None:
        return ""
    if hasattr(visit_date, "strftime"):
        return visit_date.strftime("%Y%m%d")
    s = str(visit_date).strip().split()[0]
    parts = s.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
        return f"{y:04d}{mo:02d}{d:02d}"
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else digits


def load_and_standardize(image_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    eye, modality = parse_eye_and_modality(image_path.name)
    arr = np.array(Image.open(image_path))
    transformed = False

    if eye == "OS":
        arr = np.flip(arr, axis=1)
        transformed = True

    metadata = {
        "filename": image_path.name,
        "eye_original": eye,
        "eye_standardized": "OD" if eye in {"OD", "OS"} else None,
        "modality": modality,
        "flipped_horizontal": transformed,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    return arr, metadata


def _array_to_rgba_u8(arr: np.ndarray) -> Image.Image:
    """Convert HxW or HxWx3/4 ndarray to RGBA PIL image for extract_zones."""
    if arr.ndim == 2:
        rgb = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr
    elif arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    else:
        raise ValueError(f"Unsupported array shape for staging: {arr.shape}")
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB").convert("RGBA")


def build_annotation_index(xlsx_path: Path) -> Dict[Tuple[int, str, str, str], Tuple[int, ...]]:
    """Map (patient_id, visit_yyyymmdd, eye, fp_stem_lower) -> 10 zone labels."""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet_name = "Data" if "Data" in wb.sheetnames else wb.sheetnames[0]
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    header_row = next(rows_iter, None)
    if not header_row:
        return {}

    header = [str(h).strip() if h is not None else "" for h in header_row]
    col = {name: i for i, name in enumerate(header) if name}

    def col_val(row, name: str):
        idx = col.get(name)
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    index: Dict[Tuple[int, str, str, str], Tuple[int, ...]] = {}
    for row in rows_iter:
        if not row or all(v is None for v in row):
            continue
        pid = col_val(row, "Patient_ID")
        eye = col_val(row, "Eye")
        visit_date = col_val(row, "Visit_Date")
        uwffp = col_val(row, "UWFFP")
        if pid is None or eye is None or visit_date is None or uwffp is None:
            continue
        try:
            patient_id = int(pid)
        except (TypeError, ValueError):
            continue
        visit_key = visit_key_from_cell(visit_date)
        if len(visit_key) != 8:
            continue
        eye_u = str(eye).strip().upper()
        if eye_u not in {"OD", "OS"}:
            continue
        fp_stem = Path(str(uwffp).replace("\\", "/")).stem.lower()
        labels: List[int] = []
        ok = True
        for zi in range(1, 11):
            key = f"Zone{zi}_label"
            v = col_val(row, key)
            if v is None:
                ok = False
                break
            try:
                labels.append(int(v))
            except (TypeError, ValueError):
                ok = False
                break
        if not ok or len(labels) != 10:
            continue
        k = (patient_id, visit_key, eye_u, fp_stem)
        index[k] = tuple(labels)

    return index


def find_default_xlsx(data_dir: Path) -> Optional[Path]:
    candidates = sorted(data_dir.glob("*.xlsx"))
    return candidates[0] if candidates else None


def visit_images(visit_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in visit_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def preprocess_dataset(
    data_dir: Path,
    output_dir: Path,
    annotations_xlsx: Optional[Path] = None,
    binary_zone_labels: bool = True,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "input_data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "patients": {},
        "zone_table": None,
        "zone_table_stats": {},
        "binary_zone_labels": binary_zone_labels,
    }

    label_index: Dict[Tuple[int, str, str, str], Tuple[int, ...]] = {}
    if annotations_xlsx is not None and annotations_xlsx.exists():
        label_index = build_annotation_index(annotations_xlsx)
        summary["annotations_xlsx"] = str(annotations_xlsx.resolve())
    elif annotations_xlsx is not None:
        summary["annotations_xlsx"] = str(annotations_xlsx)
        summary["zone_table_stats"]["xlsx_missing"] = True

    crops_root = output_dir / "zone_crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "zone_training_table.csv"
    zone_stats = {
        "fp_matched": 0,
        "fp_unmatched": 0,
        "zones_written": 0,
        "extract_errors": 0,
    }

    csv_file = open(table_path, "w", newline="", encoding="utf-8")
    try:
        writer = csv.writer(csv_file)
        writer.writerow(["Patient_ID", "Zone_Image", "Zone_Number", "Zone_Label"])

        patient_dirs = sorted(
            [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient")]
        )

        for patient_dir in patient_dirs:
            patient_name = patient_dir.name
            pid = patient_dir_to_id(patient_name)
            summary["patients"][patient_name] = {}

            visit_dirs = sorted([v for v in patient_dir.iterdir() if v.is_dir()])
            for visit_dir in visit_dirs:
                visit_token = visit_dir.name
                image_paths = visit_images(visit_dir)
                if not image_paths:
                    continue

                arrays_to_save: Dict[str, np.ndarray] = {}
                records: List[Dict[str, object]] = []

                for idx, image_path in enumerate(image_paths):
                    arr, record = load_and_standardize(image_path)
                    key = f"img_{idx:04d}"
                    arrays_to_save[key] = arr
                    record["array_key"] = key
                    records.append(record)

                    eye, modality = parse_eye_and_modality(image_path.name)
                    if (
                        modality == "FP"
                        and label_index
                        and pid is not None
                        and eye in {"OD", "OS"}
                    ):
                        lookup_key = (pid, visit_token, eye, image_path.stem.lower())
                        labels = label_index.get(lookup_key)
                        if labels is None:
                            zone_stats["fp_unmatched"] += 1
                            continue

                        rel_crop_dir = Path("zone_crops") / patient_name / visit_token
                        abs_crop_dir = output_dir / rel_crop_dir
                        abs_crop_dir.mkdir(parents=True, exist_ok=True)

                        try:
                            rgba = _array_to_rgba_u8(arr)
                            fd, tmp_name = tempfile.mkstemp(
                                suffix=".png", dir=str(output_dir), text=False
                            )
                            os.close(fd)
                            tmp_path = Path(tmp_name)
                            try:
                                rgba.save(tmp_path, format="PNG")
                                zone_arrays = extract(str(tmp_path))
                            finally:
                                tmp_path.unlink(missing_ok=True)

                            for zi, zone_arr in enumerate(zone_arrays, start=1):
                                crop_name = f"{image_path.stem}_zone{zi:02d}.png"
                                crop_path = abs_crop_dir / crop_name
                                Image.fromarray(
                                    zone_arr.astype(np.uint8), mode="RGBA"
                                ).save(crop_path)
                                zone_rel = rel_crop_dir / crop_name
                                raw_zone_label = labels[zi - 1]
                                zone_label = (
                                    zone_label_to_binary(raw_zone_label)
                                    if binary_zone_labels
                                    else raw_zone_label
                                )
                                writer.writerow(
                                    [
                                        pid,
                                        str(zone_rel).replace("\\", "/"),
                                        zi,
                                        zone_label,
                                    ]
                                )
                                zone_stats["zones_written"] += 1
                            zone_stats["fp_matched"] += 1
                        except Exception:
                            zone_stats["extract_errors"] += 1

                patient_out_dir = output_dir / patient_name
                patient_out_dir.mkdir(parents=True, exist_ok=True)

                visit_out_dir = patient_out_dir / visit_dir.name
                visit_out_dir.mkdir(parents=True, exist_ok=True)

                npz_path = visit_out_dir / "images.npz"
                np.savez_compressed(npz_path, **arrays_to_save)

                metadata_path = visit_out_dir / "metadata.json"
                with metadata_path.open("w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)

                summary["patients"][patient_name][visit_dir.name] = {
                    "num_images": len(records),
                    "npz_path": str(npz_path),
                    "metadata_path": str(metadata_path),
                }
    finally:
        csv_file.close()
    summary["zone_table"] = str(table_path.resolve())
    summary["zone_table_stats"] = zone_stats

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize OS→horizontal flip, save visit npz metadata, and build "
            "FP zone crop table from annotations xlsx."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Input root containing Patient* folders (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_image_arrays"),
        help="Output root for npz, zone crops, and zone_training_table.csv",
    )
    parser.add_argument(
        "--annotations-xlsx",
        type=Path,
        default=None,
        help="Path to annotations workbook (default: first *.xlsx under data-dir)",
    )
    parser.add_argument(
        "--multiclass-zone-labels",
        action="store_true",
        help="Write raw Zone_Label 0/1/2 from the spreadsheet instead of binary 0/1.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Input data directory not found: {args.data_dir}")

    xlsx_path = args.annotations_xlsx
    if xlsx_path is None:
        xlsx_path = find_default_xlsx(args.data_dir)

    summary = preprocess_dataset(
        args.data_dir,
        args.output_dir,
        xlsx_path,
        binary_zone_labels=not args.multiclass_zone_labels,
    )
    num_patients = len(summary["patients"])
    num_visits = sum(len(v) for v in summary["patients"].values())
    print(f"Done. Processed {num_patients} patients and {num_visits} visits.")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    if summary.get("zone_table"):
        print(f"Zone table: {summary['zone_table']}")
        st = summary.get("zone_table_stats", {})
        print(
            f"Zone stats: matched_fp={st.get('fp_matched', 0)}, "
            f"unmatched_fp={st.get('fp_unmatched', 0)}, "
            f"zones_written={st.get('zones_written', 0)}, "
            f"extract_errors={st.get('extract_errors', 0)}"
        )


if __name__ == "__main__":
    main()
