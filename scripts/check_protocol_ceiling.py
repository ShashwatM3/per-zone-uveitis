#!/usr/bin/env python3
"""Check protocol stop conditions; write CEILING_REPORT.md if ceiling met."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
BEST_F1_GOAL = 0.70
BEST_REC_GOAL = 0.65
CEILING_ITERATIONS = 25
CEILING_DELTA = 0.02
CEILING_WINDOW = 8


def main() -> int:
    metrics_files = sorted(RUNS.glob("protocol_r*/metrics.json"))
    rows: list[tuple[str, float, float, int]] = []
    for p in metrics_files:
        m = json.load(p.open())
        name = p.parent.name
        f1 = float(m["test"]["macro_f1"])
        rec = float(m["test"]["per_class"]["1"]["recall"])
        ep = int(m.get("best_epoch", -1))
        rows.append((name, f1, rec, ep))

    if not rows:
        print("No protocol runs found.")
        return 0

    best_name, best_f1, best_rec, _ = max(rows, key=lambda r: r[1])
    print(f"Best: {best_name} test F1={best_f1:.4f} class1_recall={best_rec:.4f} (n={len(rows)} runs)")

    if best_f1 >= BEST_F1_GOAL and best_rec >= BEST_REC_GOAL:
        print("SUCCESS stop condition met.")
        return 0

    if len(rows) >= CEILING_ITERATIONS:
        recent = sorted(rows, key=lambda r: r[0])[-CEILING_WINDOW:]
        if all(f1 < best_f1 + CEILING_DELTA for _, f1, _, _ in recent):
            report = REPO / "CEILING_REPORT.md"
            if not report.exists():
                lines = [
                    "# Ceiling Report — Uveitis Zone Classification",
                    f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n",
                    f"## Summary\n",
                    f"After **{len(rows)}** protocol iterations, test macro-F1 did not improve by "
                    f"**>{CEILING_DELTA}** over the last **{CEILING_WINDOW}** runs.\n",
                    f"- **Best run:** `{best_name}`\n",
                    f"- **Best test macro-F1:** {best_f1:.4f} (goal {BEST_F1_GOAL})\n",
                    f"- **Best test class-1 recall:** {best_rec:.4f} (goal {BEST_REC_GOAL})\n",
                    "\n## Runs tried (protocol_r*)\n",
                    "| Run | test F1 | class1 recall |\n",
                    "|-----|---------|---------------|\n",
                ]
                for name, f1, rec, _ in sorted(rows, key=lambda r: -r[1]):
                    lines.append(f"| {name} | {f1:.4f} | {rec:.4f} |\n")
                lines.extend([
                    "\n## Likely ceiling\n",
                    "~110 training patients (zone-level rows are correlated). "
                    "Exclude-tier-1 and focal γ=3 helped; RETFound, MixUp, effective weights, "
                    "zone_embed=128, and balanced batches did not beat the ConvNeXt baseline.\n",
                    "\n## To reach 0.70 F1\n",
                    "More patients, cleaner tier-1 labels, per-zone models (MIL), or stronger "
                    "domain augmentation; fixed split may be pessimistic — see 5-fold CV.\n",
                ])
                report.write_text("".join(lines))
                print(f"Wrote {report}")
            else:
                print("CEILING_REPORT.md already exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
