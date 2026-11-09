#!/usr/bin/env python3
"""Build a 10-episode REAL test set from public AI-generated video.

Source: microsoft/AVGen-Bench on Hugging Face (MIT license). These are real outputs of
Veo 3.1, Sora 2, Kling, LTX-2 etc., with native speech. Each clip's generation prompt
is its script.

What is real vs injected:
  * Pictures, voices, lip sync and any continuity drift: real model output, nothing edited.
  * Black frames, frozen picture, silence, clipping, A/V offset, wrong audio track:
    INJECTED at known times. Published AI video has already been QC'd, so these faults
    can't be found "in the wild" with ground truth.
  * Outfit/background breaks: real. Two models' renders of the SAME prompt are cut
    together inside one scripted scene, which is what happens when one shot is regenerated
    with a different seed or model. Nothing is drawn or recoloured.

Run on your own machine (this needs internet):
    pip install huggingface_hub pandas pyarrow
    python tests/build_real_inputs.py --out real_inputs
    python qa.py real_inputs --scripts real_inputs --out outputs/real
    python tests/evaluate.py --samples real_inputs --reports outputs/real --label real

After watching the clips, add any NATURAL faults you see to real_inputs/natural_labels.json
(template is written for you). evaluate.py merges them into the ground truth.
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa import media  # noqa: E402

REPO = "microsoft/AVGen-Bench"
MAIN, ALT, WEAK = ["Veo_3.1_quality", "Sora_2"], ["Sora_2", "Veo_3.1_quality", "Kling_2.6", "Seedance_2"], \
    ["LTX-2", "Ovi_11", "MOVA_360p_Emu3.5", "LTX-2.3"]
CATEGORIES = ["movie_trailer", "news", "ads"]
W, H, FPS = 1280, 720, 25


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(" ".join(cmd) + "\n" + r.stderr[-1500:])


# ---------------------------------------------------------------- selection
def dialogue_lines(prompt):
    """Quoted speech in a prompt -> [(speaker, text)]. Speaker = capitalised name just before the quote, if any."""
    out = []
    for m in re.finditer(r'(?:([A-Z][A-Za-z\.]+(?: [A-Z][A-Za-z]+)?)[^"“”]{0,40}?)?["“]([^"”]{6,})["”]', prompt):
        who = (m.group(1) or "SPEAKER").upper()
        if who in ("A", "THE", "AN", "SHE", "HE", "THEY", "IT", "IN", "AS", "WITH"):
            who = "SPEAKER"
        out.append((who, m.group(2).strip()))
    return out


def pick(meta, n_main=12, n_pairs=2, n_weak=1):
    meta = meta[meta["category"].isin(CATEGORIES)].copy()
    meta["n_words"] = meta["prompt"].map(lambda p: sum(len(t.split()) for _, t in dialogue_lines(p)))
    meta = meta[meta["n_words"] >= 4]
    by_content = meta.groupby("content")
    # prompts rendered by a main model AND an alternative model (for regenerated-shot splices)
    pairs, mains, weak = [], [], []
    for content, g in sorted(by_content, key=lambda kv: -kv[1]["n_words"].max()):
        models = set(g["model"])
        m1 = next((m for m in MAIN if m in models), None)
        if not m1:
            continue
        m2 = next((m for m in ALT if m in models and m != m1), None)
        row = lambda m: g[g["model"] == m].iloc[0]  # noqa: E731
        if m2 and len(pairs) < n_pairs:
            pairs.append((row(m1), row(m2)))
        elif len(mains) < n_main:
            mains.append(row(m1))
        wm = next((m for m in WEAK if m in models), None)
        if wm and len(weak) < n_weak and len(mains) >= n_main:
            weak.append(row(wm))
        if len(pairs) >= n_pairs and len(mains) >= n_main and len(weak) >= n_weak:
            break
    if len(mains) < 7 or len(pairs) < n_pairs:  # need 7; up to n_main gives choice by face presence
        sys.exit(f"not enough dialogue prompts found (mains={len(mains)}, pairs={len(pairs)})")
    return mains, pairs, weak


# ---------------------------------------------------------------- media helpers
def fetch(row, cache):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, row["video_path"], repo_type="dataset", cache_dir=cache)


def normalise(src, dst):
    """Same size/fps/audio layout for every clip so they can be concatenated. Adds silent audio if missing."""
    info = media.probe(src)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src]
    if not info["has_audio"]:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest"]
    cmd += ["-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p",
            "-af", "aresample=48000,aformat=channel_layouts=stereo", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "18", "-c:a", "aac", "-b:a", "192k", dst]
    sh(cmd)
    return media.probe(dst)


def speech_segments(path, min_len=0.4):
    """Rough speech/sound activity from loudness: [(start, end)] in seconds."""
    y, sr = media.load_audio(path, 16000)
    hop = sr // 50
    n = len(y) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    act = db > max(np.percentile(db, 30) + 8, -45)
    k = 10  # bridge 0.2 s gaps
    act = np.array([act[max(0, i - k):i + k + 1].any() for i in range(n)])
    segs, s = [], None
    for i, a in enumerate(np.append(act, False)):
        if a and s is None:
            s = i
        elif not a and s is not None:
            if (i - s) / 50 >= min_len:
                segs.append((s / 50, i / 50))
            s = None
    return segs


def face_share(path, k=8):
    """Share of k sampled frames where YuNet sees a face (used only to decide which clips can test lip sync)."""
    import cv2
    from qa.faces import YUNET_URL, _cached
    yp = _cached(YUNET_URL, "face_detection_yunet_2023mar.onnx", "QA_YUNET_MODEL")
    if not yp:
        return 0.0
    det = cv2.FaceDetectorYN.create(yp, "", (320, 320), 0.6, 0.3, 5000)
    cap = cv2.VideoCapture(path)
    n, hits = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0
    for j in range(k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (j + 0.5) / k))
        ok, f = cap.read()
        if not ok:
            continue
        s = 480 / min(f.shape[:2])
        f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
        det.setInputSize((f.shape[1], f.shape[0]))
        hits += det.detect(f)[1] is not None
    cap.release()
    return hits / k


def concat(parts, dst):
    """parts = [(file, start, end)] -> one file."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for f, a, b in parts:
        cmd += ["-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", f]
    fc = "".join(f"[{i}:v][{i}:a]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=1[v][a]"
    sh(cmd + ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
              "-crf", "18", "-c:a", "aac", "-b:a", "192k", dst])


def reencode(src, dst, vf=None, af=None, extra_in=None, maps=None, fc=None):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src] + (extra_in or [])
    if fc:
        cmd += ["-filter_complex", fc]
    if vf:
        cmd += ["-vf", vf]
    if af:
        cmd += ["-af", af]
    for m in maps or []:
        cmd += ["-map", m]
    sh(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", dst])


# ---------------------------------------------------------------- scripts
def ts(x):
    return f"{int(x // 60):02d}:{x % 60:05.2f}"


def scene_block(sid, row, start, end, path_for_speech, offset=0.0, clip_start=0.0):
    """Scene with the prompt's dialogue spread over the detected speech in [clip_start, clip_start+(end-start)]."""
    lines = dialogue_lines(row["prompt"])
    segs = [(max(a, clip_start), min(b, clip_start + end - start)) for a, b in speech_segments(path_for_speech)]
    segs = [(a - clip_start + offset, b - clip_start + offset) for a, b in segs if b - a > 0.3]
    out = [f"SCENE {sid} | {row['category'].upper().replace('_', ' ')}: {row['content']} | {ts(start)} - {ts(end)}",
           # prompt kept as an unparsed note: its words ("fades", "walks into") must not switch checks off
           f"PROMPT ({row['model']}): {row['prompt'][:400].replace(chr(10), ' ')}"]
    if lines and segs:
        # distribute lines over speech by word count (approximate: timing comes from audio, words from the prompt)
        total = sum(b - a for a, b in segs)
        words = np.array([len(t.split()) for _, t in lines], float)
        bounds = np.concatenate([[0], np.cumsum(words / words.sum() * total)])
        flat = []
        for a, b in segs:
            flat.append((a, b))

        def at(x):  # speech-time -> clock time
            acc = 0.0
            for a, b in flat:
                if x <= acc + (b - a):
                    return a + (x - acc)
                acc += b - a
            return flat[-1][1]
        for (who, text), b0, b1 in zip(lines, bounds[:-1], bounds[1:]):
            out.append(f"[{ts(at(b0))} - {ts(at(b1))}] {who}: {text}")
    out.append("")
    return out


# ---------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="real_inputs")
    ap.add_argument("--cache", default=".hf_cache")
    a = ap.parse_args()
    import pandas as pd
    from huggingface_hub import hf_hub_download

    os.makedirs(a.out, exist_ok=True)
    norm = os.path.join(a.out, "_normalised")
    os.makedirs(norm, exist_ok=True)
    meta = pd.read_parquet(hf_hub_download(REPO, "metadata.parquet", repo_type="dataset", cache_dir=a.cache))
    mains, pairs, weak = pick(meta)

    clips = {}

    def get(row, tag):
        dst = os.path.join(norm, f"{tag}.mp4")
        info = normalise(fetch(row, a.cache), dst)
        clips[tag] = {"file": dst, "row": row, "dur": info["duration"], "speech": speech_segments(dst),
                      "faces": face_share(dst)}
        print(f"  {tag}: {row['model']}/{row['content']} ({info['duration']:.1f}s, "
              f"{len(clips[tag]['speech'])} speech segs, face in {100 * clips[tag]['faces']:.0f}% of sampled frames)")
        return clips[tag]

    print("downloading + normalising…")
    M = [get(r, f"m{i}") for i, r in enumerate(mains)]
    # Lip sync can only be tested where a face is on screen: most face-heavy clips first.
    # M[0], M[1] -> clean control + A/V offset + wrong-audio tests; M[2] donates the wrong audio.
    M.sort(key=lambda c: -c["faces"])
    print("face share by clip:", [f"{c['row']['content']}={100 * c['faces']:.0f}%" for c in M])
    P = [(get(r1, f"p{i}a"), get(r2, f"p{i}b")) for i, (r1, r2) in enumerate(pairs)]
    Wk = [get(r, f"w{i}") for i, r in enumerate(weak)]

    truth, sources = {}, []

    def first_speech(c, default=1.0):
        return c["speech"][0] if c["speech"] else (default, min(c["dur"], default + 2))

    def write(name, path_script_lines, faults, used, note):
        with open(os.path.join(a.out, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([f"TITLE: {name} (real AI clips from {REPO})", ""] + path_script_lines))
        truth[name] = [dict(f, source=f.get("source", "injected")) for f in faults]
        sources.append({"episode": name, "note": note,
                        "clips": [{"model": c["row"]["model"], "content": c["row"]["content"],
                                   "hf_path": c["row"]["video_path"]} for c in used]})
        print("built", name)

    # r01: clean control, 3 different prompts as 3 scenes (scene changes must NOT be flagged)
    a0, a1, a2 = M[0], M[1], M[2]
    concat([(c["file"], 0, c["dur"]) for c in (a0, a1, a2)], os.path.join(a.out, "r01_clean_3_scenes.mp4"))
    t, lines = 0.0, []
    for i, c in enumerate((a0, a1, a2), 1):
        lines += scene_block(i, c["row"], t, t + c["dur"], c["file"], offset=t)
        t += c["dur"]
    write("r01_clean_3_scenes", lines, [], [a0, a1, a2], "no injection; control for false alarms")

    def single(name, c, faults, note, **kw):
        reencode(c["file"], os.path.join(a.out, name + ".mp4"), **kw)
        write(name, scene_block(1, c["row"], 0, c["dur"], c["file"]), faults, [c], note)

    # r02 black 0.5 s in the middle of speech
    c = M[3]; s0, s1 = first_speech(c); tb = min(c["dur"] - 1.0, (s0 + s1) / 2)
    single("r02_black_frames", c, [{"check": "black_frames", "start": tb, "end": tb + 0.5}], "injected 0.5 s black",
           vf=f"drawbox=enable='between(t,{tb:.2f},{tb + 0.5:.2f})':x=0:y=0:w=iw:h=ih:color=black:t=fill")
    # r03 freeze 1.5 s
    c = M[4]; tf = min(max(1.0, first_speech(c)[0]), c["dur"] - 2.0)
    f0, f1 = int(tf * FPS), int((tf + 1.5) * FPS)
    single("r03_frozen_picture", c, [{"check": "frozen_picture", "start": tf, "end": tf + 1.5}], "injected 1.5 s freeze",
           fc=f"[0:v]split[x][y];[x][y]freezeframes=first={f0}:last={f1}:replace={f0}[v]", maps=["[v]", "0:a"])
    # r04 silence 1.5 s over the first speech
    c = M[5]; s0, _ = first_speech(c); s0 = min(s0, c["dur"] - 1.6)
    single("r04_silence_over_dialogue", c, [{"check": "silence", "start": s0, "end": s0 + 1.5}], "injected 1.5 s mute over speech",
           af=f"volume=enable='between(t,{s0:.2f},{s0 + 1.5:.2f})':volume=0")
    # r05 hard clipping for 2 s over speech
    c = M[6]; s0, _ = first_speech(c); s0 = min(s0, c["dur"] - 2.1)
    g = f"if(between(t,{s0:.2f},{s0 + 2:.2f}),12,1)"
    single("r05_clipped_audio", c, [{"check": "distorted_audio", "start": s0, "end": s0 + 2}], "injected x12 gain + hard clip",
           af=f"aeval='clip(val(0)*{g},-1,1)|clip(val(1)*{g},-1,1)',aformat=channel_layouts=stereo")
    # r06 audio 300 ms late for the whole clip
    c = M[0]
    single("r06_av_offset_300ms", c, [{"check": "lip_sync", "start": 0, "end": c["dur"]}], "injected +300 ms audio delay",
           af="adelay=300:all=1")
    # r07 wrong audio track: picture of one clip with another clip's speech (lips can't follow)
    c, donor = M[1], M[2]
    single("r07_wrong_audio_track", c, [{"check": "lip_sync", "start": 0, "end": c["dur"]}],
           "picture from one clip, speech from another (real dubbing/mux error)",
           extra_in=["-i", donor["file"]], maps=["0:v", "1:a"])
    # r08 regenerated shot: first half from model A, second half from model B, same prompt, one scene
    pa, pb = P[0]; cut = min(pa["dur"], pb["dur"]) / 2
    concat([(pa["file"], 0, cut), (pb["file"], cut, pb["dur"])], os.path.join(a.out, "r08_regenerated_second_half.mp4"))
    write("r08_regenerated_second_half", scene_block(1, pa["row"], 0, cut + pb["dur"] - cut, pa["file"]),
          [{"check": ["outfit_change", "background_change"], "start": cut, "end": pb["dur"], "source": "real (two renders spliced)"}],
          [pa, pb], f"{pa['row']['model']} then {pb['row']['model']} for the same prompt, inside one scripted scene")
    # r09 one regenerated 2 s shot in the middle (glitch insert)
    pa, pb = P[1]; d = min(pa["dur"], pb["dur"]); g0 = max(1.5, d / 2 - 1.0); g1 = g0 + 2.0
    concat([(pa["file"], 0, g0), (pb["file"], g0, g1), (pa["file"], g1, d)], os.path.join(a.out, "r09_regenerated_insert_shot.mp4"))
    write("r09_regenerated_insert_shot", scene_block(1, pa["row"], 0, d, pa["file"]),
          [{"check": ["outfit_change", "background_change"], "start": g0, "end": g1, "source": "real (two renders spliced)"}],
          [pa, pb], "2 s from another model's render of the same prompt, in the middle of one scene")
    # r10 weaker model, untouched: whatever is wrong with it is natural
    c = Wk[0] if Wk else M[2]
    single("r10_weaker_model_untouched", c, [], f"{c['row']['model']} output, no injection; label natural faults after watching")

    with open(os.path.join(a.out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)
    with open(os.path.join(a.out, "SOURCES.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": f"https://huggingface.co/datasets/{REPO}", "license": "MIT", "episodes": sources}, f, indent=2)
    nl = os.path.join(a.out, "natural_labels.json")
    if not os.path.exists(nl):
        with open(nl, "w", encoding="utf-8") as f:
            json.dump({"_how_to": "After watching each video, list real faults you see, e.g. "
                       "{\"check\": \"lip_sync\", \"start\": 3.2, \"end\": 5.0, \"note\": \"mouth closed while talking\"}. "
                       "Checks: black_frames, frozen_picture, silence, distorted_audio, outfit_change, background_change, lip_sync.",
                       **{k: [] for k in truth}}, f, indent=2)
    print(f"\n{len(truth)} episodes in {a.out}/  (sources in SOURCES.json, fill natural_labels.json after watching)")


if __name__ == "__main__":
    main()
