"""Parse a timestamped microdrama screenplay.

Accepted text format (one item per line; anything unrecognised is ignored):

    SCENE 1 | INT. COFFEE SHOP - DAY | 00:00 - 00:20
    [00:01] ACTION: Maya waits by the window in a red jacket.
    [00:03] MAYA: I knew you'd come back.
    [00:07 - 00:10] RYAN (V.O.): You were never supposed to see this.
    [00:19] ACTION: FADE TO BLACK.

Timestamps may be ss, mm:ss, hh:mm:ss, with optional .fraction.
A .json file with the same structure ({"scenes": [{"id","heading","start","end",
"lines": [{"t","end","speaker","text","kind"}]}]}) is also accepted.
"""
import json
import re

TS = r"\d{1,2}(?::\d{1,2}){0,2}(?:\.\d+)?"
SCENE_RE = re.compile(
    rf"^\s*SCENE\s+(\w+)\s*[|:\-–]\s*(.*?)\s*\|\s*({TS})\s*[-–]\s*({TS})\s*$", re.I)
LINE_RE = re.compile(rf"^\s*\[\s*({TS})(?:\s*[-–]\s*({TS}))?\s*\]\s*([^:]+?)\s*:\s*(.*)$")

OFFSCREEN = re.compile(r"\((V\.?O\.?|O\.?S\.?|O\.?C\.?|PHONE|CONT'D V\.?O\.?)\)", re.I)
FADE_WORDS = re.compile(r"\b(fade[s]? (to|out|in)|black screen|blackout|cut to black|lights? (go|goes) out)\b", re.I)
WARDROBE_WORDS = re.compile(r"\b(changes? (into|clothes|outfit)|now wearing|puts? on|takes? off|removes? (his|her|the) (jacket|coat|shirt))\b", re.I)
MOVE_WORDS = re.compile(r"\b(walks? (in|into|out|outside|to)|enters?|exits?|steps? (out|outside|into)|moves? to|arrives? at|flashback)\b", re.I)
SILENT_WORDS = re.compile(r"\b(silence|silent|no sound|music stops|mutes?)\b", re.I)


def ts_to_sec(s):
    parts = [float(p) for p in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def _estimate_end(text, start):
    words = max(1, len(text.split()))
    return start + max(1.0, words / 2.6)  # ~2.6 words/s typical delivery


def parse_script(path):
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return _finalise(data["scenes"])

    scenes, cur = [], None
    with open(path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip()
            m = SCENE_RE.match(line)
            if m:
                cur = {"id": m.group(1), "heading": m.group(2).strip(),
                       "start": ts_to_sec(m.group(3)), "end": ts_to_sec(m.group(4)), "lines": []}
                scenes.append(cur)
                continue
            m = LINE_RE.match(line)
            if m and cur is not None:
                who = m.group(3).strip()
                kind = "action" if who.upper() in ("ACTION", "SFX", "MUSIC", "TRANSITION") else "dialogue"
                cur["lines"].append({
                    "t": ts_to_sec(m.group(1)),
                    "end": ts_to_sec(m.group(2)) if m.group(2) else None,
                    "speaker": None if kind == "action" else OFFSCREEN.sub("", who).strip().upper(),
                    "offscreen": bool(OFFSCREEN.search(who)),
                    "text": m.group(4).strip(),
                    "kind": kind,
                })
    return _finalise(scenes)


def _finalise(scenes):
    scenes = sorted(scenes, key=lambda s: s["start"])
    for sc in scenes:
        lines = sorted(sc["lines"], key=lambda l: l["t"])
        dlg = [l for l in lines if l["kind"] == "dialogue"]
        for i, l in enumerate(dlg):
            l.setdefault("offscreen", False)
            if not l.get("end"):
                est = _estimate_end(l["text"], l["t"])
                nxt = dlg[i + 1]["t"] if i + 1 < len(dlg) else sc["end"]
                l["end"] = min(est, nxt, sc["end"])
        sc["lines"] = lines
        sc["characters"] = sorted({l["speaker"] for l in dlg if l["speaker"] and not l["offscreen"]})
    return {"scenes": scenes}


# ---------- queries used by the checks ----------

def scene_at(script, t):
    if not script:
        return None
    for sc in script["scenes"]:
        if sc["start"] <= t < sc["end"]:
            return sc
    return None


def dialogue_windows(script):
    if not script:
        return []
    return [dict(l, scene=sc["id"]) for sc in script["scenes"] for l in sc["lines"] if l["kind"] == "dialogue"]


def action_near(script, t, regex, tol):
    """True if an ACTION/SFX line matching regex is within tol seconds of t (or the scene says so)."""
    if not script:
        return False
    for sc in script["scenes"]:
        for l in sc["lines"]:
            if l["kind"] == "action" and regex.search(l["text"]) and abs(l["t"] - t) <= tol:
                return True
    return False


def scene_mentions(scene, regex):
    return bool(scene) and any(l["kind"] == "action" and regex.search(l["text"]) for l in scene["lines"])


def overlaps_dialogue(script, a, b):
    """Seconds of scripted on- or off-screen dialogue inside [a, b]."""
    tot = 0.0
    for l in dialogue_windows(script):
        tot += max(0.0, min(b, l["end"]) - max(a, l["t"]))
    return tot
