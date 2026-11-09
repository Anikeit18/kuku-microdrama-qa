"""Lip-sync check: does mouth opening follow the speech loudness envelope?

For every window where someone should be talking on screen (from the script, or
from non-silent audio when there's no script) we cross-correlate
  mouth(t)  = opening of the most-open visible mouth in each frame
  audio(t)  = log loudness of the soundtrack at the same frame
over lags of +-0.5 s. A clear peak away from 0 means audio and picture are
offset; no peak at any lag means the lips don't follow the words at all.
"""
import numpy as np

from . import media
from .script_parser import dialogue_windows


def _audio_env(path, fps, n_frames, start_offset=0.0):
    y, sr = media.load_audio(path, sr=16000)
    # decoded audio begins at the audio stream's first sample; put it on the video's clock
    k = int(round(abs(start_offset) * sr))
    if start_offset > 0:
        y = np.concatenate([np.zeros(k, np.float32), y])
    elif start_offset < 0:
        y = y[k:]
    hop = sr / fps
    env = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        seg = y[int(i * hop): int((i + 1) * hop)]
        env[i] = np.log(np.sqrt((seg ** 2).mean()) + 1e-4) if len(seg) else np.log(1e-4)
    return env


def _pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _windows(script, env, fps, cfg):
    if script:
        out = []
        for l in dialogue_windows(script):
            if l["offscreen"]:
                continue
            out.append((l["t"], l["end"], l["speaker"]))
        return out
    # no script: speech-like stretches = audio well above the quiet floor, chunked to 4 s
    floor = np.percentile(env, 20)
    k = max(1, int(0.3 * fps))  # bridge the short gaps between syllables/words
    smooth = np.array([env[max(0, i - k):i + k + 1].max() for i in range(len(env))])
    active = smooth > floor + 1.5
    out, start = [], None
    for i, a in enumerate(np.append(active, False)):
        if a and start is None:
            start = i
        elif not a and start is not None:
            t0, t1 = start / fps, i / fps
            while t1 - t0 >= cfg["lip_min_window"]:
                out.append((t0, min(t1, t0 + 4.0), None))
                t0 += 4.0
            start = None
    return out


def check_lipsync(path, fp, script, cfg, silences=(), start_offset=0.0):
    fps, n = fp["fps"], fp["n"]
    if n == 0:
        return [], {}
    env = _audio_env(path, fps, n, start_offset)
    mouth = np.full(n, np.nan, np.float32)
    for i, fl in enumerate(fp["faces"]):
        ms = [f["mouth"] for f in fl if f.get("mouth") is not None]
        if ms:
            mouth[i] = max(ms)

    K = int(round(cfg["lip_max_lag"] * fps))
    results, issues = [], []
    for t0, t1, who in _windows(script, env, fps, cfg):
        if t1 - t0 < cfg["lip_min_window"]:
            continue
        a, b = int(t0 * fps), min(n, int(t1 * fps))
        m = mouth[a:b]
        cov = float(np.mean(~np.isnan(m))) if len(m) else 0
        res = {"start": round(t0, 2), "end": round(t1, 2), "speaker": who, "face_coverage": round(cov, 2)}
        silent = sum(max(0.0, min(t1, e) - max(t0, s)) for s, e in silences)
        if silent > 0.5 * (t1 - t0):
            res["status"] = "skipped_silence_or_freeze"  # already reported by those checks
            results.append(res)
            continue
        if cov < cfg["lip_min_face_cov"]:
            res["status"] = "skipped_no_face"
            results.append(res)
            continue
        mm = np.where(np.isnan(m), np.nanmean(m), m)
        if np.std(mm) < cfg["lip_static_std"]:
            res["status"] = "mouth_static"
            results.append(res)
            issues.append(dict(check="lip_sync", start=res["start"], end=res["end"], severity="major",
                               confidence="medium", frame_time=round((t0 + t1) / 2, 2),
                               detail=f"{who or 'speech'} is heard but the visible mouth does not move"))
            continue
        rs = {}
        for L in range(-K, K + 1):
            lo, hi = a + L, b + L
            if lo < 0 or hi > n:
                continue
            rs[L] = _pearson(mm, env[lo:hi])
        best = max(rs, key=rs.get)
        res.update(r_at_0=round(rs.get(0, 0), 2), best_r=round(rs[best], 2), best_lag_ms=int(1000 * best / fps))
        if rs[best] < cfg["lip_min_corr"]:
            res["status"] = "no_match"
            # weak signal on real footage (off-screen speakers, profile faces): minor unless there is no relation at all
            issues.append(dict(check="lip_sync", start=res["start"], end=res["end"],
                               severity="major" if rs[best] < 0.1 else "minor",
                               confidence="low", frame_time=round((t0 + t1) / 2, 2),
                               detail=f"mouth movement doesn't follow {who or 'the'} speech (best correlation {rs[best]:.2f})"))
        elif abs(best) / fps >= cfg["lip_offset_flag"] and rs[best] - rs.get(0, 0) > 0.1:
            res["status"] = "offset"
        else:
            res["status"] = "ok"
        results.append(res)

    # Offsets: one episode-level issue if they agree, else per window
    off = [r for r in results if r.get("status") == "offset"]
    judged = [r for r in results if r.get("status") in ("ok", "offset", "no_match")]
    if off:
        lags = np.array([r["best_lag_ms"] for r in off])
        if len(off) >= max(2, 0.5 * len(judged)) and (np.all(lags > 0) or np.all(lags < 0)):
            med = int(np.median(lags))
            issues.append(dict(check="lip_sync", start=off[0]["start"], end=off[-1]["end"], severity="blocker",
                               confidence="high", frame_time=off[0]["start"],
                               detail=f"audio is {'behind' if med > 0 else 'ahead of'} the picture by ~{abs(med)} ms "
                                      f"across {len(off)}/{len(judged)} dialogue windows"))
        else:
            for r in off:
                issues.append(dict(check="lip_sync", start=r["start"], end=r["end"], severity="major",
                                   confidence="medium", frame_time=r["start"],
                                   detail=f"audio {'behind' if r['best_lag_ms'] > 0 else 'ahead'} by ~{abs(r['best_lag_ms'])} ms here"))
    summary = {"windows": len(results), "judged": len(judged),
               "skipped_no_face": sum(r.get("status") == "skipped_no_face" for r in results)}
    return issues, {"summary": summary, "windows": results}
