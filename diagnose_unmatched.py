#!/usr/bin/env python3
"""
diagnose_unmatched.py

Diagnose why FP images in unmatched_fps.txt failed to match the annotation index.
Runs against the Excel file + unmatched_fps.txt only — does NOT need the data/ folder.

Usage:
    python diagnose_unmatched.py \
        --xlsx "data/UWFAFP_Annotations_Mo_4.5.2026 (Uveitis).xlsx" \
        --unmatched processed_image_arrays_multiclass/unmatched_fps.txt
"""

import re
import ast
import argparse
from pathlib import Path
from collections import defaultdict

import openpyxl

VISIT_DATE_RE = re.compile(r"(\d{8})")


def visit_key_from_cell(visit_date) -> str:
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


def load_xlsx_index(xlsx_path: Path):
    """Returns two things:
    - full index: (pid, visit, eye, fp_stem_lower) -> labels
    - three_key index: (pid, visit, eye) -> list of (fp_stem, labels)
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet_name = "Data" if "Data" in wb.sheetnames else wb.sheetnames[0]
    ws = wb[sheet_name]
    rows_iter = ws.iter_rows(values_only=True)
    header_row = next(rows_iter, None)
    header = [str(h).strip() if h is not None else "" for h in header_row]
    col = {name: i for i, name in enumerate(header) if name}

    def col_val(row, name):
        idx = col.get(name)
        if idx is None or idx >= len(row):
            return None
        return row[idx]

    full_index = {}
    three_index = defaultdict(list)

    for row in rows_iter:
        if not row or all(v is None for v in row):
            continue
        pid_raw = col_val(row, "Patient_ID")
        eye = col_val(row, "Eye")
        visit_date = col_val(row, "Visit_Date")
        uwffp = col_val(row, "UWFFP")
        if pid_raw is None or eye is None or uwffp is None:
            continue
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue
        eye_u = str(eye).strip().upper()
        if eye_u not in {"OD", "OS"}:
            continue

        uwffp_norm = str(uwffp).replace("\\", "/")
        fp_stem = Path(uwffp_norm).stem.lower()
        primary_visit = visit_key_from_cell(visit_date)
        path_match = VISIT_DATE_RE.search(uwffp_norm)
        path_visit = path_match.group(1) if path_match else ""

        visit_keys = []
        if len(primary_visit) == 8:
            visit_keys.append(primary_visit)
        if len(path_visit) == 8 and path_visit not in visit_keys:
            visit_keys.append(path_visit)
        if not visit_keys:
            continue

        labels_ok = True
        labels = []
        for zi in range(1, 11):
            v = col_val(row, f"Zone{zi}_label")
            if v is None:
                labels_ok = False
                break
            try:
                labels.append(int(v))
            except (TypeError, ValueError):
                labels_ok = False
                break
        if not labels_ok:
            labels = None  # still report the row, just flag missing labels

        for vk in visit_keys:
            k4 = (pid, vk, eye_u, fp_stem)
            if labels:
                full_index[k4] = tuple(labels)
            k3 = (pid, vk, eye_u)
            three_index[k3].append({
                "fp_stem_in_xlsx": fp_stem,
                "uwffp_raw": str(uwffp),
                "visit_key_from_date": primary_visit,
                "visit_key_from_path": path_visit,
                "has_labels": labels is not None,
            })

    return full_index, three_index


def parse_unmatched(unmatched_path: Path):
    """Parse unmatched_fps.txt into list of (fp_path_str, tried_key_tuple)."""
    entries = []
    for line in unmatched_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("UNMATCHED:"):
            continue
        # Format: UNMATCHED: <path>  tried_key=<tuple>
        m = re.match(r"UNMATCHED:\s+(\S+)\s+tried_key=(.+)$", line)
        if not m:
            continue
        fp_path = m.group(1)
        tried_key = ast.literal_eval(m.group(2))
        entries.append((fp_path, tried_key))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--unmatched", required=True)
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx)
    unmatched_path = Path(args.unmatched)

    print(f"Loading Excel: {xlsx_path}")
    full_index, three_index = load_xlsx_index(xlsx_path)
    print(f"  → {len(full_index)} rows in full index (pid, visit, eye, fp_stem)")
    print(f"  → {len(three_index)} unique (pid, visit, eye) combos\n")

    entries = parse_unmatched(unmatched_path)
    print(f"Unmatched FPs: {len(entries)}\n")
    print("=" * 80)

    recovered = 0
    truly_missing = 0

    for fp_path, tried_key in entries:
        pid, visit, eye, disk_stem = tried_key
        print(f"\nFILE : {fp_path}")
        print(f"  tried key : {tried_key}")

        # Check exact match
        if tried_key in full_index:
            print(f"  ✅ EXACT MATCH EXISTS — bug is elsewhere (race condition?)")
            continue

        # Check 3-key (pid, visit, eye) — what stems does Excel have?
        k3 = (pid, visit, eye)
        rows_for_visit = three_index.get(k3, [])

        if not rows_for_visit:
            # Maybe visit date mismatch — search by pid+eye only
            all_rows_for_pid_eye = []
            for (p, v, e), rows in three_index.items():
                if p == pid and e == eye:
                    for r in rows:
                        all_rows_for_pid_eye.append((v, r))

            if all_rows_for_pid_eye:
                print(f"  ❌ NO ROWS for (pid={pid}, visit={visit}, eye={eye})")
                print(f"     But Excel has these visits for pid={pid}, eye={eye}:")
                for v, r in all_rows_for_pid_eye:
                    print(f"       visit_key={v}  xlsx_stem={r['fp_stem_in_xlsx']!r}")
                    print(f"         UWFFP raw: {r['uwffp_raw']!r}")
                    print(f"         visit from date cell: {r['visit_key_from_date']!r}  from path: {r['visit_key_from_path']!r}")
            else:
                print(f"  ❌ NO ROWS AT ALL for pid={pid}, eye={eye} in Excel")
            truly_missing += 1
        else:
            print(f"  ⚠️  (pid, visit, eye) EXISTS in Excel — stem mismatch:")
            print(f"     disk stem  : {disk_stem!r}")
            for r in rows_for_visit:
                xlsx_stem = r["fp_stem_in_xlsx"]
                match_indicator = "✅ MATCH" if xlsx_stem == disk_stem else "❌ DIFFERS"
                print(f"     xlsx stem  : {xlsx_stem!r}  {match_indicator}")
                print(f"       UWFFP raw: {r['uwffp_raw']!r}")
                print(f"       has_labels: {r['has_labels']}")
            recovered += 1

    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Stem mismatch (recoverable via fuzzy match): {recovered}")
    print(f"  Not in Excel at all (truly missing data):   {truly_missing}")
    print(f"  Total unmatched:                            {len(entries)}")

    if recovered > 0:
        print("\nFix: add fallback matching in build_annotation_index on (pid, visit, eye) when unique.")
    if truly_missing > 0:
        print("\nFix: those patients/visits need to be added to the Excel file.")


if __name__ == "__main__":
    main()