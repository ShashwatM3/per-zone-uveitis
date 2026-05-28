#!/usr/bin/env python3
"""Execute research protocol runs 3..N with pre/post logging."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = "/home/shashwat/miniconda3/envs/venv/bin/python"
SPLIT = ROOT / "splits/canonical_split.json"
LOG = ROOT / "RESEARCH_LOG.md"
BEST_F1 = 0.5284623503393579  # protocol_r03_convnext_focal


def metrics_summary(path: Path) -> dict:
    m = json.loads(path.read_text())
    bv = m.get("best_val") or {}
    t = m.get("test") or {}
    pc = bv.get("per_class") or {}
    return {
        "best_epoch": m.get("best_epoch"),
        "val_macro_f1": bv.get("macro_f1"),
        "r0": (pc.get("0") or {}).get("recall"),
        "r1": (pc.get("1") or {}).get("recall"),
        "test_macro_f1": t.get("macro_f1"),
        "test_bal_acc": t.get("balanced_accuracy"),
    }


def git_commit(msg: str) -> None:
    import time

    subprocess.run(["git", "add", "RESEARCH_LOG.md"], cwd=ROOT, check=False)
    for attempt in range(5):
        r = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return
        err = (r.stderr or "") + (r.stdout or "")
        if "nothing to commit" in err:
            return
        if "index.lock" in err and attempt < 4:
            time.sleep(2)
            continue
        print("git commit note:", err)
        return


def append_log(row: str) -> None:
    text = LOG.read_text()
    if row not in text:
        LOG.write_text(text.rstrip() + "\n" + row + "\n")


def run_cmd(name: str, cmd: list[str], run_id: int, hypothesis: str) -> None:
    global BEST_F1
    out = ROOT / "runs" / name
    metrics_path = out / "metrics.json"
    if metrics_path.exists():
        s = metrics_summary(metrics_path)
        test_f1 = s["test_macro_f1"] or 0.0
        if test_f1 > BEST_F1:
            BEST_F1 = test_f1
        print(f"SKIP {name} (metrics exist) test_f1={test_f1:.4f}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    print(f"\n=== RUN {name} ===", flush=True)
    with log_path.open("w") as logf:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "3"
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise SystemExit(f"Run {name} failed with code {proc.returncode}")
    s = metrics_summary(out / "metrics.json")
    test_f1 = s["test_macro_f1"] or 0.0
    delta = test_f1 - BEST_F1
    improved = test_f1 > BEST_F1
    if improved:
        BEST_F1 = test_f1
    decision = "IMPROVED" if improved else "no gain"
    append_log(
        f"| {run_id} | {name} | {hypothesis} | "
        f"{test_f1:.4f} | {delta:+.4f} | {decision} |"
    )
    git_commit(f"result: {name} test.macro_f1={test_f1:.4f} ({decision})")
    print(
        f"best_epoch={s['best_epoch']} val_f1={s['val_macro_f1']:.4f} "
        f"recalls=({s['r0']:.3f},{s['r1']:.3f}) test_f1={test_f1:.4f} bal={s['test_bal_acc']:.4f}"
    )


HYPOTHESES = {
    "protocol_r03_convnext_focal": "ConvNeXt-Tiny focal vs CLIP on canonical split (exp 3)",
    "protocol_r04_clip_focal": "CLIP + focal loss",
    "protocol_r05_clip_freeze3": "CLIP unfreeze backbone after 3 epochs",
    "protocol_r06_clip_lr2e4": "CLIP head LR 2e-4",
    "protocol_r07_clip_bb_lr": "CLIP higher backbone LRs (stages23/head_norm)",
    "protocol_r08_clip_25ep": "CLIP full 25-epoch budget",
    "protocol_r09_convnext_zone128": "ConvNeXt zone_embed=128 head_hidden=512 (exp 5)",
    "protocol_r10_clip_softlabels": "CLIP soft_ce + multiclass labels",
}


def main() -> None:
    lock = ROOT / "runs" / ".protocol_batch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("Another protocol batch is already running (lock exists). Exiting.")
        raise SystemExit(1)

    experiments: list[tuple[str, list[str]]] = [
        (
            "protocol_r03_convnext_focal",
            [
                PY,
                str(ROOT / "train_convnext.py"),
                "--csv",
                "processed_image_arrays/zone_training_table.csv",
                "--data-root",
                "processed_image_arrays",
                "--loss",
                "focal",
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r03_convnext_focal",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r03_convnext_focal",
            ],
        ),
        (
            "protocol_r04_clip_focal",
            [
                PY,
                str(ROOT / "train_clip_convnext.py"),
                "--loss",
                "focal",
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r04_clip_focal",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r04_clip_focal",
            ],
        ),
        (
            "protocol_r05_clip_freeze3",
            [
                PY,
                str(ROOT / "train_clip_convnext.py"),
                "--freeze-backbone-epochs",
                "3",
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r05_clip_freeze3",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r05_clip_freeze3",
            ],
        ),
        (
            "protocol_r06_clip_lr2e4",
            [
                PY,
                str(ROOT / "train_clip_convnext.py"),
                "--lr",
                "2e-4",
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r06_clip_lr2e4",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r06_clip_lr2e4",
            ],
        ),
        (
            "protocol_r07_clip_bb_lr",
            [
                PY,
                str(ROOT / "train_clip_convnext.py"),
                "--backbone-lr-stages23",
                "6e-6",
                "--backbone-lr-head-norm",
                "1.2e-5",
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r07_clip_bb_lr",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r07_clip_bb_lr",
            ],
        ),
        (
            "protocol_r08_clip_25ep",
            [
                PY,
                str(ROOT / "train_clip_convnext.py"),
                "--epochs",
                "25",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r08_clip_25ep",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r08_clip_25ep",
            ],
        ),
        (
            "protocol_r09_convnext_zone128",
            [
                PY,
                str(ROOT / "train_convnext.py"),
                "--loss",
                "focal",
                "--zone-embed-dim",
                "128",
                "--head-hidden",
                "512",
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r09_convnext_zone128",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r09_convnext_zone128",
            ],
        ),
        (
            "protocol_r10_clip_softlabels",
            [
                PY,
                str(ROOT / "train_clip_convnext.py"),
                "--soft-labels",
                "--loss",
                "soft_ce",
                "--csv",
                "processed_image_arrays_multiclass/zone_training_table.csv",
                "--data-root",
                "processed_image_arrays_multiclass",
                "--epochs",
                "20",
                "--batch-size",
                "32",
                "--num-workers",
                "8",
                "--output-dir",
                "runs/protocol_r10_clip_softlabels",
                "--split-json",
                str(SPLIT),
                "--wandb-project",
                "uveitis-per-zone",
                "--wandb-run-name",
                "protocol_r10_clip_softlabels",
            ],
        ),
    ]

    try:
        start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        for i, (name, cmd) in enumerate(experiments[start:], start=3 + start):
            hyp = HYPOTHESES.get(name, name)
            append_log(f"| {i} | (running) {name} | {hyp} | — | — | — |")
            run_cmd(name, cmd, i, hyp)
        print(f"\nDone. Best test.macro_f1={BEST_F1:.4f}")
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
