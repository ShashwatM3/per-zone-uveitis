#!/usr/bin/env python3
"""
Validate that every Zone_Label in zone_training_table.csv is in {0, 1, 2}.

Usage:
  python3 tests/check_zone_labels.py [--csv PATH]

Default CSV: processed_image_arrays/zone_training_table.csv (relative to cwd).
Exit code: 0 if all valid, 1 if any invalid or file missing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


VALID_LABELS = frozenset({0, 1, 2})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("processed_image_arrays") / "zone_training_table.csv",
        help="Path to zone_training_table.csv",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv.resolve()}", file=sys.stderr)
        return 1

    bad_rows: list[tuple[int, str, object]] = []
    data_rows = 0

    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: CSV has no header row.", file=sys.stderr)
            return 1

        required = {"Patient_ID", "Zone_Image", "Zone_Number", "Zone_Label"}
        missing = required - set(reader.fieldnames)
        if missing:
            print(f"ERROR: CSV missing columns: {sorted(missing)}", file=sys.stderr)
            return 1

        for i, row in enumerate(reader, start=2):
            raw = row.get("Zone_Label", "")
            if raw is None or str(raw).strip() == "":
                bad_rows.append((i, "empty Zone_Label", raw))
                continue
            try:
                # Accept "2", "2.0" from Excel re-exports, but only if value is integer 0/1/2
                v = int(float(str(raw).strip()))
            except (TypeError, ValueError):
                bad_rows.append((i, "not an integer", raw))
                continue
            if v not in VALID_LABELS:
                bad_rows.append((i, "not in {0,1,2}", v))
            else:
                data_rows += 1

    print(f"CSV: {args.csv.resolve()}")
    print(f"Rows with valid Zone_Label (0/1/2): {data_rows}")
    print(f"Rows with invalid Zone_Label: {len(bad_rows)}")

    if bad_rows:
        print("\nInvalid rows (line_number = 1-based file line, header is line 1):")
        for line_no, reason, value in bad_rows[:50]:
            print(f"  line {line_no}: {reason!r} value={value!r}")
        if len(bad_rows) > 50:
            print(f"  ... and {len(bad_rows) - 50} more")
        return 1

    print("OK: all Zone_Label values are in {0, 1, 2}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
