#!/usr/bin/env python3
"""Build a seeded test set: one clean synthetic episode + variants with known faults.

Why synthetic: the input pack has screenplays only (no video), and this build
environment cannot download real clips. Every fault here is injected at a known
time, so recall can be measured exactly. Real-footage validation is listed as
the next step in the doc.

    python tests/make_testset.py --out samples
Writes samples/<variant>.mp4, samples/<variant>.txt (screenplay) and samples/truth.json
"""
import argparse
import json
import os
import subprocess
import wave

import cv2
import numpy as np

W, H, FPS, SR, DUR = 540, 960, 25, 48000, 38.0

# --- scene plan (times in s) ---------------------------------------------------
SKIN = (150, 180, 225)
SHIRT = {"MAYA": (40, 40, 200), "RYAN": (60, 150, 60)}          # red, green (BGR)
F0 = {"MAYA": 210, "RYAN": 120}
SCENES = [
    {"id": "1", "heading": "INT. CAFE - DAY", "start": 0, "end": 12, "bg": "cafe", "shots": [
        (0, 4, "MAYA", "wide", "cafe"), (4, 8, "MAYA", "close", "cafe"), (8, 12, "MAYA", "wide", "cafe")],
     "lines": [("ACTION", 0.5, None, "Maya waits by the window."),
               ("MAYA", 1.0, 3.5, "I told myself I would never come back here."),
               ("MAYA", 5.0, 7.5, "But the letter had your name on it."),
               ("MAYA", 9.0, 11.2, "So tell me. Who sent it?")]},
    {"id": "2", "heading": "INT. OFFICE - NIGHT", "start": 12, "end": 26, "bg": "office", "shots": [
        (12, 15, "RYAN", "wide", "office"), (15, 18, "MAYA", "close", "office_rev"),
        (18, 21, "RYAN", "close", "office"), (21, 26, "MAYA", "wide", "office_rev")],
     "lines": [("RYAN", 12.4, 14.8, "You were never supposed to find that letter."),
               ("MAYA", 15.3, 17.6, "Then you should have burned it, Ryan."),
               ("RYAN", 18.3, 20.8, "I did. Someone made a copy."),
               ("MAYA", 21.5, 24.5, "Then we both know who is lying to us.")]},
    {"id": "3", "heading": "EXT. ROOFTOP - NIGHT", "start": 26, "end": 38, "bg": "roof", "shots": [
        (26, 31, "MAYA", "wide", "roof"), (31, 33.5, "MAYA", "close", "roof"), (33.5, 38, "MAYA", "wide", "roof")],
     "lines": [("MAYA", 27.0, 30.0, "Three years. Three years I trusted him."),
               ("ACTION", 31.0, None, "Silence. Maya stares at the city."),
               ("MAYA", 33.8, 36.2, "Not anymore. Tonight it ends."),
               ("ACTION", 37.2, None, "FADE TO BLACK.")]},
]
SCRIPTED_SILENCE = (31.0, 33.0)
SCRIPTED_FADE = (37.2, 38.0)


def make_bg(kind, rng):
    img = np.zeros((H, W, 3), np.uint8)
    if kind == "cafe":
        img[:] = (140, 140, 60)                                   # teal wall
        cv2.rectangle(img, (40, 120), (300, 420), (230, 220, 200), -1)    # window
        cv2.rectangle(img, (0, 700), (W, H), (70, 60, 60), -1)           # grey counter
        for x in range(340, 520, 45):
            cv2.circle(img, (x, 200), 18, (30, 120, 200), -1)            # hanging lamps
    elif kind == "office":
        img[:] = (120, 100, 90)                                   # blue-grey wall
        cv2.rectangle(img, (0, 680), (W, H), (60, 60, 60), -1)           # desk
        cv2.rectangle(img, (330, 150), (500, 330), (200, 200, 190), -1)   # whiteboard
    elif kind == "office_rev":                                    # reverse angle: same office, other wall
        img[:] = (110, 95, 92)
        cv2.rectangle(img, (0, 690), (W, H), (60, 60, 60), -1)
        for y in range(120, 600, 60):
            cv2.rectangle(img, (30, y), (160, y + 40), (90, 80, 70), -1)  # shelves
    elif kind == "roof":
        img[:] = (80, 40, 20)                                     # navy sky
        for x in range(0, W, 60):
            h = int(rng.integers(150, 380))
            cv2.rectangle(img, (x, H - h), (x + 50, H), (50, 50, 50), -1)
            for wy in range(H - h + 20, H - 20, 40):
                cv2.rectangle(img, (x + 10, wy), (x + 20, wy + 12), (120, 200, 230), -1)
    elif kind == "bedroom":                                       # wrong set used for continuity faults
        img[:] = (130, 70, 150)
        cv2.rectangle(img, (0, 640), (W, H), (200, 200, 230), -1)
        cv2.rectangle(img, (60, 200), (220, 380), (60, 160, 200), -1)
    return img


def speech(line_start, line_end, f0, rng):
    """Voice-like audio + matching 0..1 mouth envelope for one line."""
    n = int((line_end - line_start) * SR)
    t = np.arange(n) / SR
    env = np.zeros(n)
    pos = 0.05
    while pos < (line_end - line_start) - 0.15:
        d = rng.uniform(0.12, 0.24)
        a, b = int(pos * SR), min(n, int((pos + d) * SR))
        env[a:b] += rng.uniform(0.5, 1.0) * np.sin(np.linspace(0, np.pi, b - a)) ** 2
        pos += d + rng.uniform(0.02, 0.10)
    f = f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    voice = sum((0.6 / k) * np.sin(k * ph) for k in range(1, 12))
    voice += 0.15 * rng.standard_normal(n)
    return 0.22 * voice * env, env


def build_audio(rng, opts):
    total = int(DUR * SR)
    audio = 0.006 * rng.standard_normal(total)                         # room tone (~ -45 dBFS)
    mouth = np.zeros(total)
    speaker_at = np.empty(total, dtype=object)
    for sc in SCENES:
        for who, t0, t1, _ in sc["lines"]:
            if who == "ACTION":
                continue
            v, env = speech(t0, t1, F0[who], rng)
            a = int(t0 * SR)
            audio[a:a + len(v)] += v
            mouth[a:a + len(env)] = np.maximum(mouth[a:a + len(env)], env)
            speaker_at[a:a + len(env)] = who
    s, e = SCRIPTED_SILENCE
    audio[int(s * SR):int(e * SR)] = 0
    s, e = SCRIPTED_FADE
    audio[int(s * SR):] *= np.linspace(1, 0, total - int(s * SR))
    # ---- audio faults ----
    if "silence" in opts:
        s, e = opts["silence"]
        audio[int(s * SR):int(e * SR)] = 0
    if "clip" in opts:
        s, e = opts["clip"]
        seg = audio[int(s * SR):int(e * SR)] * opts.get("clip_gain", 12)
        audio[int(s * SR):int(e * SR)] = np.clip(seg, -1, 1)
    if "av_offset" in opts:                                            # audio late by N s
        k = int(opts["av_offset"] * SR)
        audio = np.concatenate([np.zeros(k), audio[:-k]])
    if "av_lead" in opts:                                              # audio early by N s
        k = int(opts["av_lead"] * SR)
        audio = np.concatenate([audio[k:], np.zeros(k)])
    return np.clip(audio, -1, 1), mouth, speaker_at


def draw_person(img, who, cx, cy, scale, mouth_open, shirt):
    fw, fh = int(46 * scale), int(60 * scale)
    cv2.rectangle(img, (cx - int(2.0 * fw), cy + int(0.9 * fh)), (cx + int(2.0 * fw), H + 50), shirt, -1)  # torso
    cv2.rectangle(img, (cx - int(0.35 * fw), cy + int(0.7 * fh)), (cx + int(0.35 * fw), cy + int(1.0 * fh)), SKIN, -1)  # neck
    cv2.ellipse(img, (cx, cy), (fw, fh), 0, 0, 360, SKIN, -1)
    cv2.ellipse(img, (cx, cy - int(0.75 * fh)), (int(1.05 * fw), int(0.45 * fh)), 0, 180, 360, (30, 30, 40), -1)  # hair
    for ex in (-0.38, 0.38):
        cv2.circle(img, (cx + int(ex * fw), cy - int(0.15 * fh)), max(2, int(0.09 * fw)), (40, 30, 30), -1)
    mh = int(1 + mouth_open * 0.22 * fh)
    cv2.ellipse(img, (cx, cy + int(0.45 * fh)), (int(0.32 * fw), mh), 0, 0, 360, (40, 30, 90), -1)


def render(name, opts, out_dir, seed=7):
    rng = np.random.default_rng(seed)
    bgs = {k: make_bg(k, np.random.default_rng(1)) for k in ("cafe", "office", "office_rev", "roof", "bedroom")}
    audio, mouth, _ = build_audio(rng, opts)
    nframes = int(DUR * FPS)
    vpath = os.path.join(out_dir, name + ".mp4")
    wpath = os.path.join(out_dir, name + "_tmp.wav")
    with wave.open(wpath, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-i", wpath, "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", vpath],
                          stdin=subprocess.PIPE)
    frozen = None
    nrng = np.random.default_rng(seed + 1)
    for fi in range(nframes):
        t = fi / FPS
        sc = next(s for s in SCENES if s["start"] <= t < s["end"])
        shot = next(s for s in sc["shots"] if s[0] <= t < s[1])
        _, _, who, size, bgk = shot
        shirt = SHIRT[who]
        for (a, b, k, v) in opts.get("bg_swap", []):
            if a <= t < b:
                bgk = v
        for (a, b, w_, col) in opts.get("shirt", []):
            if a <= t < b and w_ == who:
                shirt = col
        img = bgs[bgk].copy()
        m = mouth[int(t * SR)] if int(t * SR) < len(mouth) else 0
        for (a, b) in opts.get("static_mouth", []):
            if a <= t < b:
                m = 0.05
        sway = int(10 * np.sin(2 * np.pi * 0.25 * t))
        draw_person(img, who, W // 2 + sway, 470 + int(4 * np.sin(2 * np.pi * 0.4 * t)), 1.0, m, shirt)
        if size == "close":
            zx, zy, zw, zh = W // 2 - 170, 470 - 260, 340, 604
            img = cv2.resize(img[zy:zy + zh, zx:zx + zw], (W, H))
        img = np.clip(img.astype(np.int16) + nrng.integers(-4, 5, img.shape), 0, 255).astype(np.uint8)
        # ---- video faults / scripted fade ----
        fs, fe = SCRIPTED_FADE
        if t >= fs:
            img = (img * max(0.0, 1 - (t - fs) / 0.3)).astype(np.uint8)
        if "black" in opts and opts["black"][0] <= t < opts["black"][1]:
            img = np.zeros_like(img)
        if "freeze" in opts:
            a, b = opts["freeze"]
            if a <= t < b:
                frozen = img if frozen is None else frozen
                img = frozen
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    os.remove(wpath)
    return vpath


def write_script(path, title):
    def ts(x):
        return f"{int(x // 60):02d}:{x % 60:04.1f}"
    out = [f"TITLE: {title}", ""]
    for sc in SCENES:
        out.append(f"SCENE {sc['id']} | {sc['heading']} | {ts(sc['start'])} - {ts(sc['end'])}")
        for who, t0, t1, text in sc["lines"]:
            span = f"{ts(t0)} - {ts(t1)}" if t1 else ts(t0)
            out.append(f"[{span}] {who}: {text}")
        out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


BLUE = (180, 90, 40)
VARIANTS = {
    # name: (render options, ground-truth faults)
    "01_clean": ({}, []),
    "02_black_frames": ({"black": (6.0, 6.6)}, [("black_frames", 6.0, 6.6)]),
    "03_frozen_picture": ({"freeze": (16.0, 18.5)}, [("frozen_picture", 16.0, 18.5)]),
    "04_silence_over_dialogue": ({"silence": (19.0, 21.5)}, [("silence", 19.0, 21.5)]),
    "05_clipped_audio": ({"clip": (4.0, 6.5)}, [("distorted_audio", 4.0, 6.5)]),
    "06_av_offset_400ms": ({"av_offset": 0.4}, [("lip_sync", 0.0, 38.0)]),
    "07_outfit_change": ({"shirt": [(8.0, 12.0, "MAYA", BLUE)]}, [("outfit_change", 8.0, 12.0)]),
    "08_background_glitch_shot": ({"bg_swap": [(4.0, 8.0, "cafe", "bedroom")]}, [("background_change", 4.0, 8.0)]),
    "09_background_switch": ({"bg_swap": [(33.5, 38.0, "roof", "bedroom")]}, [("background_change", 33.5, 38.0)]),
    "10_mouth_not_moving": ({"static_mouth": [(12.0, 15.0)]}, [("lip_sync", 12.4, 14.8)]),
    "11_combo": ({"black": (13.0, 13.8), "silence": (27.5, 29.5), "shirt": [(21.0, 26.0, "MAYA", BLUE)]},
                 [("black_frames", 13.0, 13.8), ("silence", 27.5, 29.5), ("outfit_change", 21.0, 26.0)]),
}

# Held-out set: built AFTER thresholds were tuned on VARIANTS. Different random seed (speech,
# noise), different fault positions, and some smaller/harder faults near the thresholds.
GREY = (150, 150, 150)
HELDOUT = {
    "h01_clean": ({}, []),
    "h02_short_black_0.25s": ({"black": (22.0, 22.25)}, [("black_frames", 22.0, 22.25)]),
    "h03_freeze_1.0s": ({"freeze": (28.0, 29.0)}, [("frozen_picture", 28.0, 29.0)]),
    "h04_silence_1.5s": ({"silence": (1.5, 3.0)}, [("silence", 1.5, 3.0)]),
    "h05_mild_clipping": ({"clip": (21.5, 24.5), "clip_gain": 5}, [("distorted_audio", 21.5, 24.5)]),
    "h06_av_offset_250ms": ({"av_offset": 0.25}, [("lip_sync", 0.0, 38.0)]),
    "h07_audio_early_300ms": ({"av_lead": 0.3}, [("lip_sync", 0.0, 38.0)]),
    "h08_outfit_change_rooftop": ({"shirt": [(33.5, 38.0, "MAYA", GREY)]}, [("outfit_change", 33.5, 38.0)]),
    "h09_ryan_outfit_change": ({"shirt": [(18.0, 21.0, "RYAN", BLUE)]}, [("outfit_change", 18.0, 21.0)]),
    "h10_background_glitch_office": ({"bg_swap": [(18.0, 21.0, "office", "cafe")]}, [("background_change", 18.0, 21.0)]),
    "h11_mouth_static_rooftop": ({"static_mouth": [(33.5, 37.0)]}, [("lip_sync", 33.8, 36.2)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--set", default="tuning", choices=["tuning", "heldout"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    truth = {}
    variants, seed = (VARIANTS, 7) if a.set == "tuning" else (HELDOUT, 2024)
    for name, (opts, faults) in variants.items():
        if a.only and name not in a.only:
            continue
        render(name, opts, a.out, seed=seed)
        write_script(os.path.join(a.out, name + ".txt"), "The Letter (synthetic test episode)")
        truth[name] = [{"check": c, "start": s, "end": e} for c, s, e in faults]
        print("wrote", name)
    with open(os.path.join(a.out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)


if __name__ == "__main__":
    main()
