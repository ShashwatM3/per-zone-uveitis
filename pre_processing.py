"""Preprocess retinal images into a per-FP cleaned cache + zone training table.

What this script does:
  - Walks ``data/Patient*/<visit>/`` for fundus (FP) images with annotations
    in the ``.xlsx`` workbook.
  - For each annotated FP image, runs the expensive steps ONCE and caches them:
        1. Flip horizontally if filename indicates OS (standardize to OD axis).
        2. Detect yellow crosshair to recover the fovea ``(cx, cy)`` and the
           horizontal-axis angle ``angle_deg``.
        3. Inpaint the yellow overlay with OpenCV (the slow step).
    The cleaned RGBA array is written to
        ``processed_image_arrays/cleaned/<Patient>/<visit>/<fp_stem>.npy``
    and the geometry metadata to
        ``processed_image_arrays/cleaned/<Patient>/<visit>/<fp_stem>.json``
  - Writes ``processed_image_arrays/zone_training_table.csv`` with the columns
        Patient_ID, Cleaned_Image, Zone_Number, Zone_Label
    Zone masks/crops are NOT materialised on disk anymore: ``zone_dataset.py``
    computes them on the fly from the cached cleaned arrays (those operations
    are fast vectorised numpy / PIL calls).

Why this is faster than the previous version:
  - Per-FP work runs across CPU cores via ``multiprocessing`` instead of a
    single-threaded loop. ``cv2.inpaint`` releases the GIL, so we get
    near-linear scaling on this machine.
  - Each FP is cached as a single uncompressed ``.npy`` (one fast write +
    one fast read) instead of a compressed ``.npz`` plus 10 PNG crops.
  - Re-runs skip any FP whose cleaned ``.npy`` + sidecar already exist, so
    interrupting the script is safe.

By default each Zone_Label is written in binary form: spreadsheet tiers 0, 1,
and 2 become 0, 1, and 1 respectively (classes 1 and 2 merged). Pass
``--multiclass-zone-labels`` to write raw 0/1/2 instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


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


def _zone_label_to_binary(raw_label: int) -> int:
    """Map raw 0/1/2 tiers to binary 0/1 (idempotent for 0/1)."""
    if raw_label not in (0, 1, 2):
        raise ValueError(f"Zone_Label must be 0, 1, or 2; got {raw_label!r}")
    return 0 if raw_label == 0 else 1


VISIT_DATE_FROM_PATH_RE = re.compile(r"(\d{8})")


def build_annotation_index(
    xlsx_path: Path,
) -> Dict[Tuple[int, str, str, str], Tuple[int, ...]]:
    """Map ``(patient_id, visit_yyyymmdd, eye, fp_stem_lower) -> 10 zone labels``.

    The Excel ``Visit_Date`` cell can be missing, malformed, or stored as a
    string that ``visit_key_from_cell`` cannot collapse to an 8-digit token.
    To recover those rows we additionally try to extract a ``YYYYMMDD`` token
    directly from the ``UWFFP`` path (the file names are reliably stamped as
    ``PatientXXX_YYYYMMDD_<eye>_FP_NNNN.png``). When the two sources differ
    we register the row under BOTH visit keys so the on-disk visit token has
    a chance of matching whichever one the source-of-truth turns out to be.
    """
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
        if pid is None or eye is None or uwffp is None:
            continue
        try:
            patient_id = int(pid)
        except (TypeError, ValueError):
            continue
        eye_u = str(eye).strip().upper()
        if eye_u not in {"OD", "OS"}:
            continue

        uwffp_norm = str(uwffp).replace("\\", "/")
        fp_stem = Path(uwffp_norm).stem.lower()

        primary_visit = visit_key_from_cell(visit_date)
        path_match = VISIT_DATE_FROM_PATH_RE.search(uwffp_norm)
        path_visit = path_match.group(1) if path_match else ""

        visit_keys: List[str] = []
        if len(primary_visit) == 8:
            visit_keys.append(primary_visit)
        if len(path_visit) == 8 and path_visit not in visit_keys:
            visit_keys.append(path_visit)
        if not visit_keys:
            continue

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

        labels_t = tuple(labels)
        for vk in visit_keys:
            k = (patient_id, vk, eye_u, fp_stem)
            index.setdefault(k, labels_t)

    return index


def find_default_xlsx(data_dir: Path) -> Optional[Path]:
    candidates = sorted(data_dir.glob("*.xlsx"))
    return candidates[0] if candidates else None


def visit_images(visit_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in visit_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Worker-side processing of one FP image.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FpTask:
    fp_path: str
    patient_name: str
    visit_token: str
    pid: int
    eye: str  # "OD" or "OS"
    fp_stem: str  # original stem (case preserved)
    labels: Tuple[int, ...]  # 10 raw zone labels
    output_dir: str
    force: bool


def _worker_init() -> None:
    """Limit per-worker thread fan-out so 64 workers don't oversubscribe BLAS/OpenCV."""
    try:
        import cv2

        cv2.setNumThreads(1)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass


def _process_fp(task: FpTask) -> dict:
    """Run the heavy per-FP pipeline (or skip if cache is valid) and return CSV rows."""
    fp_path = Path(task.fp_path)
    output_dir = Path(task.output_dir)

    cleaned_dir = output_dir / "cleaned" / task.patient_name / task.visit_token
    npy_path = cleaned_dir / f"{task.fp_stem}.npy"
    meta_path = cleaned_dir / f"{task.fp_stem}.json"
    cleaned_rel = (
        Path("cleaned") / task.patient_name / task.visit_token / f"{task.fp_stem}.npy"
    )
    cleaned_rel_str = str(cleaned_rel).replace("\\", "/")

    rows = []
    for zi in range(1, 11):
        raw_label = task.labels[zi - 1]
        rows.append((task.pid, cleaned_rel_str, zi, raw_label))

    if (not task.force) and npy_path.exists() and meta_path.exists():
        try:
            json.loads(meta_path.read_text())
            return {
                "status": "skipped",
                "rows": rows,
                "fp_path": str(fp_path),
            }
        except Exception:
            pass

    try:
        from extract_zones import detect_crosshair_from_yellow, remove_yellow_overlay

        arr = np.array(Image.open(fp_path).convert("RGBA"))
        flipped = False
        if task.eye == "OS":
            arr = np.ascontiguousarray(np.flip(arr, axis=1))
            flipped = True

        fovea_fallback = False
        try:
            cx, cy, angle_deg, yellow_count = detect_crosshair_from_yellow(
                arr, output_dir=None, save_debug=False
            )
        except ValueError:
            # Faint / clipped / ambiguous crosshair. Fall back to image-center
            # geometry (cx, cy at the middle of the frame, no rotation) so the
            # FP is retained for training; downstream zone masks will be
            # geometrically off but the image isn't lost. Flagged in metadata.
            h, w = arr.shape[:2]
            cx, cy, angle_deg, yellow_count = w // 2, h // 2, 0.0, 0
            fovea_fallback = True
        cleaned = remove_yellow_overlay(arr, inpaint_radius=3, dilate_iterations=1)

        cleaned_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: numpy.save auto-appends ".npy" when the path does not already
        # end with it, so a tmp name like "foo.npy.tmp" would actually become
        # "foo.npy.tmp.npy". Use a tmp name that already ends in ".npy".
        tmp_npy = cleaned_dir / f"{task.fp_stem}.tmp.npy"
        np.save(tmp_npy, cleaned)
        os.replace(tmp_npy, npy_path)

        meta = {
            "fp_filename": fp_path.name,
            "patient_id": task.pid,
            "visit": task.visit_token,
            "eye_original": task.eye,
            "eye_standardized": "OD",
            "flipped_horizontal": flipped,
            "cx": int(cx),
            "cy": int(cy),
            "angle_deg": float(angle_deg),
            "yellow_pixels": int(yellow_count),
            "fovea_fallback": bool(fovea_fallback),
            "shape": [int(x) for x in cleaned.shape],
            "dtype": str(cleaned.dtype),
        }
        tmp_meta = cleaned_dir / f"{task.fp_stem}.json.tmp"
        tmp_meta.write_text(json.dumps(meta, indent=2))
        os.replace(tmp_meta, meta_path)
        return {"status": "ok", "rows": rows, "fp_path": str(fp_path)}
    except Exception as exc:
        return {
            "status": "error",
            "fp_path": str(fp_path),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=4),
        }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def _enumerate_fp_tasks(
    data_dir: Path,
    output_dir: Path,
    label_index: Dict[Tuple[int, str, str, str], Tuple[int, ...]],
    force: bool,
) -> Tuple[List[FpTask], int, int]:
    """Walk ``data_dir`` and build the list of per-FP work items.

    Matching strategy per FP:
      1. Primary 4-tuple ``(pid, visit, eye, stem_lower)`` -- exact hit.
      2. Fallback 3-tuple ``(pid, visit, eye)`` -- recovers FPs whose Excel
         ``UWFFP`` cell points at a filename that doesn't match the on-disk
         stem (different ``_NNNN`` suffix, missing patient prefix, etc.).
    A diagnostic ``unmatched_fps.txt`` is written into ``output_dir`` listing
    every FP that still failed to match after both passes, with the primary
    key that was tried.

    Returns ``(tasks, fp_unmatched_count, fp_total_count)``.
    """
    tasks: List[FpTask] = []
    fp_total = 0
    fp_unmatched = 0
    unmatched_log: List[Tuple[str, Tuple[object, ...]]] = []

    patient_dirs = sorted(
        p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Patient")
    )
    for patient_dir in patient_dirs:
        pid = patient_dir_to_id(patient_dir.name)
        if pid is None:
            continue
        visit_dirs = sorted(v for v in patient_dir.iterdir() if v.is_dir())
        for visit_dir in visit_dirs:
            for image_path in visit_images(visit_dir):
                eye, modality = parse_eye_and_modality(image_path.name)
                if modality != "FP":
                    continue
                fp_total += 1
                if eye not in {"OD", "OS"}:
                    fp_unmatched += 1
                    unmatched_log.append(
                        (str(image_path), (pid, visit_dir.name, eye, image_path.stem.lower()))
                    )
                    continue
                key = (pid, visit_dir.name, eye, image_path.stem.lower())
                labels = label_index.get(key)
                if labels is None:
                    fallback_key = next(
                        (
                            k
                            for k in label_index
                            if k[0] == pid and k[1] == visit_dir.name and k[2] == eye
                        ),
                        None,
                    )
                    if fallback_key is not None:
                        labels = label_index[fallback_key]
                if labels is None:
                    fp_unmatched += 1
                    unmatched_log.append((str(image_path), key))
                    continue
                tasks.append(
                    FpTask(
                        fp_path=str(image_path),
                        patient_name=patient_dir.name,
                        visit_token=visit_dir.name,
                        pid=pid,
                        eye=eye,
                        fp_stem=image_path.stem,
                        labels=tuple(labels),
                        output_dir=str(output_dir),
                        force=force,
                    )
                )

    unmatched_path = output_dir / "unmatched_fps.txt"
    if unmatched_log:
        output_dir.mkdir(parents=True, exist_ok=True)
        with unmatched_path.open("w", encoding="utf-8") as f:
            for fp_path_str, tried_key in unmatched_log:
                f.write(f"UNMATCHED: {fp_path_str}  tried_key={tried_key}\n")
    elif unmatched_path.exists():
        # Stale file from a prior run -- a clean run should leave no leftovers.
        try:
            unmatched_path.unlink()
        except OSError:
            pass

    return tasks, fp_unmatched, fp_total


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or not np.isfinite(seconds):
        return "?"
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m{seconds % 60:02d}s"
    hours = minutes // 60
    return f"{hours}h{minutes % 60:02d}m"


def preprocess_dataset(
    data_dir: Path,
    output_dir: Path,
    annotations_xlsx: Optional[Path] = None,
    binary_zone_labels: bool = True,
    workers: int = 1,
    force: bool = False,
    progress_every: int = 10,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cleaned").mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "input_data_dir": str(data_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "binary_zone_labels": binary_zone_labels,
        "workers": workers,
        "force": force,
    }

    label_index: Dict[Tuple[int, str, str, str], Tuple[int, ...]] = {}
    if annotations_xlsx is not None and annotations_xlsx.exists():
        print(f"[preproc] loading annotations from {annotations_xlsx} ...", flush=True)
        label_index = build_annotation_index(annotations_xlsx)
        summary["annotations_xlsx"] = str(annotations_xlsx.resolve())
        print(f"[preproc] indexed {len(label_index)} annotated FP rows", flush=True)
    elif annotations_xlsx is not None:
        summary["annotations_xlsx"] = str(annotations_xlsx)
        summary["xlsx_missing"] = True

    print(f"[preproc] enumerating FP images under {data_dir} ...", flush=True)
    tasks, fp_unmatched, fp_total = _enumerate_fp_tasks(
        data_dir, output_dir, label_index, force=force
    )
    print(
        f"[preproc] found {fp_total} FP images, "
        f"{len(tasks)} matched annotations, {fp_unmatched} unmatched (will be skipped).",
        flush=True,
    )

    stats = {
        "fp_total_in_data": fp_total,
        "fp_matched": 0,
        "fp_unmatched": fp_unmatched,
        "fp_cached_hits": 0,
        "fp_processed": 0,
        "fp_errors": 0,
        "zones_written": 0,
    }
    errors: List[dict] = []

    table_path = output_dir / "zone_training_table.csv"
    tmp_table = output_dir / "zone_training_table.csv.tmp"

    start = time.monotonic()
    with tmp_table.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Patient_ID", "Cleaned_Image", "Zone_Number", "Zone_Label"])

        if not tasks:
            print("[preproc] no annotated FP images to process.", flush=True)
        else:
            n_workers = max(1, min(workers, len(tasks)))
            print(
                f"[preproc] processing {len(tasks)} FP images "
                f"with {n_workers} workers ...",
                flush=True,
            )

            if n_workers == 1:
                _worker_init()
                result_iter = (_process_fp(t) for t in tasks)
            else:
                ctx = mp.get_context("fork")
                pool = ctx.Pool(processes=n_workers, initializer=_worker_init)
                result_iter = pool.imap_unordered(_process_fp, tasks, chunksize=1)

            try:
                done = 0
                for result in result_iter:
                    done += 1
                    status = result.get("status")
                    if status in ("ok", "skipped"):
                        for pid, rel, zi, raw_label in result["rows"]:
                            zone_label = (
                                _zone_label_to_binary(raw_label)
                                if binary_zone_labels
                                else raw_label
                            )
                            writer.writerow([pid, rel, zi, zone_label])
                            stats["zones_written"] += 1
                        stats["fp_matched"] += 1
                        if status == "ok":
                            stats["fp_processed"] += 1
                        else:
                            stats["fp_cached_hits"] += 1
                    else:
                        stats["fp_errors"] += 1
                        errors.append(
                            {
                                "fp_path": result.get("fp_path"),
                                "error": result.get("error"),
                            }
                        )
                        print(
                            f"[preproc] ERROR  {result.get('fp_path')}: "
                            f"{result.get('error')}",
                            flush=True,
                        )

                    if (
                        done == 1
                        or done == len(tasks)
                        or (done % progress_every == 0)
                    ):
                        elapsed = time.monotonic() - start
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remaining = (len(tasks) - done) / rate if rate > 0 else 0.0
                        print(
                            f"[preproc] {done:>4d}/{len(tasks)}  "
                            f"({100.0 * done / len(tasks):5.1f}%)  "
                            f"{rate:5.2f} fp/s  "
                            f"ok={stats['fp_processed']} "
                            f"cached={stats['fp_cached_hits']} "
                            f"err={stats['fp_errors']}  "
                            f"eta={_format_eta(remaining)}",
                            flush=True,
                        )
            finally:
                if n_workers > 1:
                    pool.close()
                    pool.join()

    os.replace(tmp_table, table_path)

    summary["zone_table"] = str(table_path.resolve())
    summary["stats"] = stats
    if errors:
        summary["errors_sample"] = errors[:20]
        summary["error_count"] = len(errors)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    total_elapsed = time.monotonic() - start
    print(
        f"[preproc] done in {_format_eta(total_elapsed)}: "
        f"processed={stats['fp_processed']} "
        f"cached={stats['fp_cached_hits']} "
        f"unmatched={stats['fp_unmatched']} "
        f"errors={stats['fp_errors']} "
        f"zones={stats['zones_written']}",
        flush=True,
    )
    print(f"[preproc] zone table: {table_path}", flush=True)
    print(f"[preproc] summary:    {summary_path}", flush=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cache cleaned (yellow-removed, OD-standardized) RGBA arrays for each "
            "annotated FP image and emit a zone-level training table. Zone masks "
            "and crops are generated on the fly by zone_dataset.py at training time."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Input root containing Patient* folders (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_image_arrays"),
        help="Output root for cleaned/ cache and zone_training_table.csv.",
    )
    parser.add_argument(
        "--annotations-xlsx",
        type=Path,
        default=None,
        help="Path to annotations workbook (default: first *.xlsx under data-dir).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 1) - 2)),
        help="Parallel worker processes (default: min(32, cpu-2)).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cleaned arrays even if cache hit on disk.",
    )
    parser.add_argument(
        "--multiclass-zone-labels",
        action="store_true",
        help="Write raw Zone_Label 0/1/2 from the spreadsheet instead of binary 0/1.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a progress line every N completed FP images (default: 10).",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Input data directory not found: {args.data_dir}")

    xlsx_path = args.annotations_xlsx
    if xlsx_path is None:
        xlsx_path = find_default_xlsx(args.data_dir)
        if xlsx_path is None:
            print(
                f"[preproc] WARNING: no .xlsx annotations found under {args.data_dir}; "
                "nothing will be processed (FPs without labels are skipped).",
                flush=True,
            )

    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        annotations_xlsx=xlsx_path,
        binary_zone_labels=not args.multiclass_zone_labels,
        workers=args.workers,
        force=args.force,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
