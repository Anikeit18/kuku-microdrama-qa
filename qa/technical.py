"""Technical checks: black frames, frozen picture, silence, distorted audio.

Detection is deterministic (ffmpeg filters + numpy). The script is only used to
decide whether a detection was *intended* (scripted fade, scripted silence) and
how severe it is (silence over dialogue is worse than silence between lines).
"""
import numpy as np

from . import media
from .script_parser import (FADE_WORDS, SILENT_WORDS, action_near, overlaps_dialogue)


def _issue(kind, start, end, severity, detail, confidence="high", **extra):
    return dict(check=kind, start=round(start, 2), end=round(end, 2), severity=severity,
                detail=detail, confidence=confidence, **extra)


def check_black(path, info, script, cfg):
    log = media.ffmpeg_filter_log(
        path, vf=f"blackdetect=d={cfg['black_min_dur']}:pix_th={cfg['black_pix_th']}")
    issues = []
    starts, ends = media.parse_intervals(log, "black_start", "black_end")
    for s, e in zip(starts, ends):
        dur = e - s
        at_edge = s < 0.05 or e > info["duration"] - 0.05
        scripted = action_near(script, s, FADE_WORDS, cfg["script_tol"] + dur) or \
            action_near(script, e, FADE_WORDS, cfg["script_tol"])
        if scripted:
            sev, note = "info", "matches a scripted fade/black in the screenplay"
        elif dur >= cfg["black_blocker_dur"] and not at_edge:
            sev, note = "blocker", "unscripted black mid-episode"
        elif dur >= 0.2 and not at_edge:
            sev, note = "major", "unscripted black flash mid-episode"
        else:
            sev, note = "minor", "short black" + (" at start/end of file" if at_edge else "")
        issues.append(_issue("black_frames", s, e, sev, f"{dur:.2f}s of black picture; {note}"))
    return issues


def check_freeze(path, info, script, cfg):
    log = media.ffmpeg_filter_log(
        path, vf=f"freezedetect=n={cfg['freeze_noise_db']}dB:d={cfg['freeze_min_dur']}")
    issues = []
    starts, _ = media.parse_intervals(log, "lavfi.freezedetect.freeze_start", "x")
    ends, _ = media.parse_intervals(log, "lavfi.freezedetect.freeze_end", "x")
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else info["duration"]
        dur = e - s
        sev = "blocker" if dur >= cfg["freeze_blocker_dur"] else "major"
        talking = overlaps_dialogue(script, s, e) if script else 0
        note = f"; {talking:.1f}s of scripted dialogue plays over the frozen picture" if talking > 0.3 else ""
        issues.append(_issue("frozen_picture", s, e, sev, f"picture frozen for {dur:.2f}s{note}"))
    return issues


def check_silence(path, info, script, cfg):
    if not info["has_audio"]:
        return [_issue("silence", 0, info["duration"], "blocker", "file has no audio track")]
    log = media.ffmpeg_filter_log(
        path, af=f"silencedetect=n={cfg['silence_db']}dB:d={cfg['silence_min_dur']}")
    starts, _ = media.parse_intervals(log, "silence_start", "x")
    ends, _ = media.parse_intervals(log, "silence_end", "x")
    issues = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else info["duration"]
        s = max(0.0, s)
        dur = e - s
        if script:
            over_dlg = overlaps_dialogue(script, s, e)
            if over_dlg >= 0.5:
                issues.append(_issue("silence", s, e, "blocker",
                                     f"{dur:.2f}s of silence while the script has {over_dlg:.1f}s of dialogue here"))
            elif action_near(script, s, SILENT_WORDS, cfg["script_tol"] + dur):
                issues.append(_issue("silence", s, e, "info", f"{dur:.2f}s silence; scripted"))
            else:
                sev = "major" if dur >= cfg["silence_no_script_major"] else "minor"
                issues.append(_issue("silence", s, e, sev,
                                     f"{dur:.2f}s of dead air (no dialogue scripted, but even room tone/music is missing)"))
        else:
            sev = "major" if dur >= cfg["silence_no_script_major"] else "minor"
            issues.append(_issue("silence", s, e, sev, f"{dur:.2f}s of silence (no script to tell if intended)",
                                 confidence="medium"))
    return issues


def _runs(mask, hop):
    """Contiguous True runs in a boolean array -> [(start_s, end_s)]."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start * hop, i * hop))
            start = None
    return out


def check_distortion(path, info, script, cfg):
    if not info["has_audio"]:
        return []
    y, sr = media.load_audio(path, sr=info["audio_rate"] or 48000)
    win = int(0.1 * sr)
    n = len(y) // win
    if n == 0:
        return []
    frames = np.abs(y[: n * win]).reshape(n, win)
    clip_frac = (frames >= cfg["clip_level"]).mean(axis=1)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)

    issues = []
    runs = []
    for s, e in _runs(clip_frac > cfg["clip_ratio"], 0.1):
        if runs and s - runs[-1][1] <= cfg["clip_merge_gap"]:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    for s, e in runs:
        if e - s >= cfg["clip_min_dur"]:
            pct = 100 * clip_frac[int(s / 0.1):int(e / 0.1)].mean()
            issues.append(_issue("distorted_audio", s, e, "major",
                                 f"audio clipping for {e - s:.1f}s (~{pct:.0f}% of samples at full scale)"))

    # sudden loudness spikes: louder than the episode's own loud speech by spike_db (pops, bursts, gain jumps)
    loud = np.percentile(rms_db, 95)
    spike = rms_db > loud + cfg["spike_db"]
    for s, e in _runs(spike, 0.1):
        if not any(i["start"] - 0.2 <= s <= i["end"] + 0.2 for i in issues):
            issues.append(_issue("distorted_audio", s, e, "minor",
                                 f"sudden loudness spike ({rms_db[int(s / 0.1)] - loud:.0f} dB above the episode's normal loud level)",
                                 confidence="medium"))
    return issues


def run_all(path, info, script, cfg, on_check=None):
    """on_check(name, new_issues) is called after each check (used by the live demo UI)."""
    def step(name, fn):
        if on_check:
            on_check(name, None)  # started
        new = fn()
        if on_check:
            on_check(name, new)   # finished
        return new

    issues = []
    if info["has_video"]:
        issues += step("black_frames", lambda: check_black(path, info, script, cfg))
        blacks = [(i["start"], i["end"]) for i in issues if i["check"] == "black_frames"]
        # a run of black frames is also "frozen"; report it once, as black
        issues += step("frozen_picture", lambda: [f for f in check_freeze(path, info, script, cfg)
                                                  if not any(b0 - 0.2 <= f["start"] and f["end"] <= b1 + 0.2 for b0, b1 in blacks)])
    issues += step("silence", lambda: check_silence(path, info, script, cfg))
    issues += step("distorted_audio", lambda: check_distortion(path, info, script, cfg))
    off = info.get("av_start_offset", 0.0)
    if info["has_audio"] and info["has_video"] and abs(off) >= 0.08:
        issues.append(_issue("lip_sync", 0.0, info["duration"], "blocker" if abs(off) >= 0.15 else "major",
                             f"audio stream starts {abs(off) * 1000:.0f} ms {'after' if off > 0 else 'before'} the video "
                             "(container offset; every line will look out of sync)"))
    if script and script["scenes"]:
        end = max(sc["end"] for sc in script["scenes"])
        if abs(end - info["duration"]) > 2.0:
            issues.append(_issue("script_mismatch", min(end, info["duration"]), max(end, info["duration"]),
                                 "major", f"video is {info['duration']:.1f}s but the script runs to {end:.1f}s "
                                 "(timestamps below may be offset)"))
    return issues
