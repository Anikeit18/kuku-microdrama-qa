"""Thin wrappers around ffprobe/ffmpeg."""
import json
import re
import subprocess

import numpy as np


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    fps = None
    if v:
        num, den = v.get("avg_frame_rate", "0/1").split("/")
        fps = float(num) / float(den) if float(den) else None
    return {
        "duration": float(info["format"].get("duration", 0)),
        "has_video": v is not None,
        "has_audio": a is not None,
        "fps": fps,
        "width": int(v["width"]) if v else None,
        "height": int(v["height"]) if v else None,
        "audio_rate": int(a["sample_rate"]) if a else None,
        # container-level A/V offset: audio stream starting later/earlier than video
        "av_start_offset": round(float(a.get("start_time", 0)) - float(v.get("start_time", 0)), 3) if (a and v) else 0.0,
    }


def ffmpeg_filter_log(path, vf=None, af=None):
    """Run a detection filter and return ffmpeg's stderr log."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", path]
    if vf:
        cmd += ["-vf", vf, "-an"]
    if af:
        cmd += ["-af", af, "-vn"]
    cmd += ["-f", "null", "-"]
    return subprocess.run(cmd, capture_output=True, text=True).stderr


def parse_intervals(log, start_key, end_key):
    """Pair up '<start_key>: t' and '<end_key>: t' entries from a filter log."""
    starts = [float(x) for x in re.findall(start_key + r"[:=]\s*([\d.]+)", log)]
    ends = [float(x) for x in re.findall(end_key + r"[:=]\s*([\d.]+)", log)]
    return starts, ends


def load_audio(path, sr=16000):
    """Decode audio to mono float32 in [-1, 1]."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr
