#!/usr/bin/env python3
"""fovea_audit.py -- Phase 0 (P0.3): audit fovea-fallback geometry in the zone table.

When pre_processing.py cannot detect the yellow crosshair, it falls back to
image-center geometry (writes ``fovea_fallback: true`` in the sidecar .json) and
keeps the image. Those images have their 10 zone masks attached to the WRONG
pixels -- every zone label is geometrically misaligned -- which corrupts both the
per-zone labels and any (future) masked-pooling architecture.

This script counts fovea_fallback status per image / zone-row / patient, overall
and (optionally) per canonical split, and writes the list of confirmed-fallback
images so training can drop them (``train_convnext.py --exclude-fovea-fallback``).

Status categories (read directly from each sidecar):
  fallback     -- fovea_fallback == True  (geometrically wrong, should be excluded)
  ok           -- fovea_fallback == False (crosshair detected)
  missing_key  -- sidecar exists but predates the field (geometry UNVERIFIED)
  no_sidecar   -- .npy has no .json sidecar
  unreadable   -- sidecar could not be parsed

Standalone: needs only the CSV + cleaned/ sidecars (+ optional split JSON). Mirrors
the one-off-audit precedent of diagnose_unmatched.py.

Usage:
    python fovea_audit.py \
        --csv processed_image_arrays_multiclass/zone_training_table.csv \
        --data-root processed_image_arrays_multiclass \
        --split-json splits/canonical_split.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

STATUS_ORDER = ("fallback", "ok", "missing_key", "no_sidecar", "unreadable")


def sidecar_status(npy_abs: Path) -> str:
    meta_path = npy_abs.with_suffix(".json")
    if not meta_path.exists():
        return "no_sidecar"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return "unreadable"
    if "fovea_fallback" not in meta:
        return "missing_key"
    return "fallback" if meta["fovea_fallback"] else "ok"


def load_split(split_json: Path | None) -> dict[int, str] | None:
    if split_json is None or not split_json.exists():
        return None
    data = json.loads(split_json.read_text())
    pid2split: dict[int, str] = {}
    for name in ("train", "val", "test"):
        for pid in data.get(name, []):
            pid2split[int(pid)] = name
    return pid2split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("processed_image_arrays_multiclass/zone_training_table.csv"),
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("processed_image_arrays_multiclass"),
    )
    ap.add_argument(
        "--split-json",
        type=Path,
        default=Path("splits/canonical_split.json"),
        help="Optional patient split for a per-split breakdown.",
    )
    ap.add_argument(
        "--write-list",
        type=Path,
        default=Path("processed_image_arrays_multiclass/fovea_fallback_images.txt"),
        help="Where to write the list of confirmed-fallback Cleaned_Image paths.",
    )
    args = ap.parse_args()

    pid2split = load_split(args.split_json)

    img_status: dict[str, str] = {}
    img_pid: dict[str, int] = {}
    zone_rows_by_img: Counter = Counter()

    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = str(row["Cleaned_Image"]).replace("\\", "/")
            zone_rows_by_img[rel] += 1
            img_pid[rel] = int(float(row["Patient_ID"]))
            if rel not in img_status:
                img_status[rel] = sidecar_status(args.data_root / rel)

    images_by_status: Counter = Counter(img_status.values())
    zonerows_by_status: Counter = Counter()
    for rel, st in img_status.items():
        zonerows_by_status[st] += zone_rows_by_img[rel]
    patients_affected = sorted(
        {img_pid[rel] for rel, st in img_status.items() if st == "fallback"}
    )

    total_imgs = len(img_status)
    total_rows = sum(zone_rows_by_img.values())

    print("=" * 72)
    print("FOVEA-FALLBACK AUDIT  (Phase 0 / P0.3)")
    print(f"CSV:        {args.csv}")
    print(f"Data root:  {args.data_root}")
    print(f"Split:      {args.split_json if pid2split else '(none)'}")
    print("=" * 72)
    print(f"\nTotal matched images: {total_imgs}    total zone rows: {total_rows}\n")
    print(f"{'status':<13}{'images':>10}{'zone_rows':>12}")
    print("-" * 35)
    for st in STATUS_ORDER:
        if images_by_status.get(st):
            print(f"{st:<13}{images_by_status.get(st, 0):>10}{zonerows_by_status.get(st, 0):>12}")

    unknown_imgs = images_by_status.get("missing_key", 0) + images_by_status.get("no_sidecar", 0)
    print(
        f"\nConfirmed fallback: {images_by_status.get('fallback', 0)} images "
        f"({zonerows_by_status.get('fallback', 0)} zone rows) across "
        f"{len(patients_affected)} patients."
    )
    if patients_affected:
        print(f"  affected patient_ids: {patients_affected}")
    print(
        f"Unknown geometry (missing_key / no_sidecar): {unknown_imgs} images -- "
        "sidecars predate the fovea_fallback field; status is unverifiable without\n"
        "  re-running pre_processing.py --force on those images. NOT auto-excluded."
    )

    if pid2split:
        in_split = [rel for rel in img_status if img_pid[rel] in pid2split]
        print("\nPer canonical split (patient-level):")
        print(f"{'split':<8}{'fallback_imgs':>15}{'fallback_rows':>15}{'unknown_imgs':>15}")
        print("-" * 53)
        for name in ("train", "val", "test"):
            fi = fr = ui = 0
            for rel, st in img_status.items():
                if pid2split.get(img_pid[rel]) != name:
                    continue
                if st == "fallback":
                    fi += 1
                    fr += zone_rows_by_img[rel]
                elif st in ("missing_key", "no_sidecar"):
                    ui += 1
            print(f"{name:<8}{fi:>15}{fr:>15}{ui:>15}")
        print(
            f"\n(note: {len(in_split)}/{total_imgs} images belong to the current "
            f"{len(set(pid2split))}-patient split; the rest are unused by it.)"
        )

    fallback_imgs = sorted(rel for rel, st in img_status.items() if st == "fallback")
    args.write_list.parent.mkdir(parents=True, exist_ok=True)
    args.write_list.write_text("\n".join(fallback_imgs) + ("\n" if fallback_imgs else ""))
    print(f"\nWrote {len(fallback_imgs)} confirmed-fallback image paths -> {args.write_list}")
    print("Exclude them in training with:  train_convnext.py --exclude-fovea-fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
