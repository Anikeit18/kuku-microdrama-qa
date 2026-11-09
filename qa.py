#!/usr/bin/env python3
"""Pre-promo QA for a microdrama episode.

    python qa.py episode.mp4 --script episode.txt --out outputs/ep01
    python qa.py episodes/ --scripts scripts/ --out outputs/      # batch: matches by file stem
"""
import argparse
import glob
import json
import os
import sys

from qa import pipeline, report
from qa.config import load_config

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm")


def run_one(video, script, out, cfg, faces):
    rep = pipeline.run(video, script, cfg, face_provider=faces)
    path = report.write(rep, out)
    print(f"[qa] {rep['verdict']}  {rep['counts']}  -> {path}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file or folder of videos")
    ap.add_argument("--script", help="screenplay (.txt/.json) for a single video")
    ap.add_argument("--scripts", help="folder of screenplays, matched to videos by file name stem")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--config", help="JSON file overriding thresholds in qa/config.py")
    ap.add_argument("--faces", default="auto", choices=["auto", "yunet", "mediapipe", "skin"])
    ap.add_argument("--llm", choices=["gemini", "openai_compatible"], help="vision-LLM verifier for continuity flags")
    ap.add_argument("--llm-model")
    a = ap.parse_args()
    cfg = load_config(a.config, {"llm_provider": a.llm, "llm_model": a.llm_model})

    if os.path.isdir(a.video):
        vids = sorted(p for p in glob.glob(os.path.join(a.video, "*")) if p.lower().endswith(VIDEO_EXT))
        summary = []
        for v in vids:
            stem = os.path.splitext(os.path.basename(v))[0]
            scr = None
            if a.scripts:
                cands = [p for p in glob.glob(os.path.join(a.scripts, stem + ".*")) if p.endswith((".txt", ".json"))]
                scr = cands[0] if cands else None
            rep = run_one(v, scr, os.path.join(a.out, stem), cfg, a.faces)
            summary.append({"video": stem, "verdict": rep["verdict"], **rep["counts"]})
        with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    else:
        run_one(a.video, a.script, a.out, cfg, a.faces)


if __name__ == "__main__":
    sys.exit(main())
