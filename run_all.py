#!/usr/bin/env python3
"""Cross-platform version of run_all.sh (works in Windows PowerShell):

    python run_all.py            # rebuild test sets + all raw outputs + scores
    python run_all.py --skip-samples   # reuse existing samples/ and samples_heldout/
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(*args):
    print(">", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        sys.exit("ffmpeg/ffprobe not found on PATH. Install it (e.g. `winget install Gyan.FFmpeg`), "
                 "then open a NEW terminal and retry.")
    skip = "--skip-samples" in sys.argv
    shutil.rmtree(os.path.join(ROOT, "outputs"), ignore_errors=True)   # outputs_archive/ is kept
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    if not skip:
        run("tests/make_testset.py", "--out", "samples", "--set", "tuning")
        run("tests/make_testset.py", "--out", "samples_heldout", "--set", "heldout")
    # The synthetic episodes have flat cartoon faces that MediaPipe (trained on real faces) mostly can't see,
    # so they are scored with the skin-colour face finder they were designed for. Real footage uses MediaPipe.
    F = ["--faces", "skin"]
    run("qa.py", "samples", "--scripts", "samples", "--out", "outputs/tuning_with_script", *F)
    run("qa.py", "samples_heldout", "--scripts", "samples_heldout", "--out", "outputs/heldout_with_script", *F)
    run("qa.py", "samples", "--out", "outputs/tuning_no_script", *F)
    run("qa.py", "samples_heldout", "--out", "outputs/heldout_no_script", *F)
    for d in ("tuning_with_script", "tuning_no_script"):
        run("tests/evaluate.py", "--samples", "samples", "--reports", f"outputs/{d}", "--label", d)
    for d in ("heldout_with_script", "heldout_no_script"):
        run("tests/evaluate.py", "--samples", "samples_heldout", "--reports", f"outputs/{d}", "--label", d)


if __name__ == "__main__":
    main()
