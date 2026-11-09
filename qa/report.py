"""Write report.json and a self-contained report.html (thumbnails embedded)."""
import base64
import html
import json
import os

from .pipeline import grab_frame_fn

SEV_COLOR = {"blocker": "#c62828", "major": "#ef6c00", "minor": "#9e9d24", "info": "#78909c"}
LABEL = {"black_frames": "Black frames", "frozen_picture": "Frozen picture", "silence": "Silence",
         "distorted_audio": "Distorted audio", "outfit_change": "Outfit change",
         "background_change": "Background change", "lip_sync": "Lip sync", "script_mismatch": "Script/video mismatch"}


def _ts(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def _img(b, w=200):
    return (f'<img src="data:image/jpeg;base64,{base64.b64encode(b).decode()}" '
            f'style="max-width:{w}px;max-height:220px">') if b else ""


def write(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    slim = {k: v for k, v in report.items()}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, default=float)

    grab = grab_frame_fn(report["video"], width=320)
    dur = max(report["media"]["duration"], 0.1)
    marks = "".join(
        f'<div title="#{i["id"]} {LABEL.get(i["check"], i["check"])} {_ts(i["start"])}" style="position:absolute;'
        f'left:{100 * i["start"] / dur:.2f}%;width:{max(0.4, 100 * (i["end"] - i["start"]) / dur):.2f}%;'
        f'top:0;bottom:0;background:{SEV_COLOR[i["severity"]]};opacity:.85"></div>'
        for i in report["issues"])
    rows = []
    for i in report["issues"]:
        if "ref_time" in i:
            thumb = (f'<div class="pair"><figure>{_img(grab(i["ref_time"]), 150)}<figcaption>earlier {_ts(i["ref_time"])}</figcaption></figure>'
                     f'<figure>{_img(grab(i["frame_time"]), 150)}<figcaption>flagged {_ts(i["frame_time"])}</figcaption></figure></div>')
        elif i["check"] in ("black_frames", "frozen_picture", "lip_sync", "outfit_change", "background_change"):
            t = i.get("frame_time", (i["start"] + i["end"]) / 2)
            thumb = _img(grab(t), 200)
        else:
            thumb = ""
        llm = f'<div class="llm">vision model: {html.escape(json.dumps(i["llm"]))}</div>' if i.get("llm") else ""
        rows.append(
            f'<tr><td>{i["id"]}</td><td><span class="sev" style="background:{SEV_COLOR[i["severity"]]}">{i["severity"]}</span></td>'
            f'<td>{LABEL.get(i["check"], i["check"])}</td><td class="mono">{_ts(i["start"])} → {_ts(i["end"])}</td>'
            f'<td>{html.escape(i["detail"])}{llm}</td><td>{i["confidence"]}</td><td>{thumb}</td></tr>')
    c = report["counts"]
    vcol = {"FAIL": "#c62828", "REVIEW": "#ef6c00", "PASS": "#2e7d32"}[report["verdict"]]
    page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>QA report: {html.escape(os.path.basename(report['video']))}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222;max-width:1200px}}
.verdict{{display:inline-block;padding:6px 14px;border-radius:6px;color:#fff;font-weight:700;background:{vcol}}}
.bar{{position:relative;height:22px;background:#eceff1;border-radius:4px;margin:8px 0 2px}}
table{{border-collapse:collapse;width:100%;margin-top:16px;font-size:14px}}td,th{{border-bottom:1px solid #e0e0e0;padding:8px;vertical-align:top;text-align:left}}
.sev{{color:#fff;border-radius:4px;padding:2px 6px;font-size:12px}}.mono{{font-family:ui-monospace,monospace;white-space:nowrap}}
.pair{{display:flex;gap:6px}}figure{{margin:0}}figcaption{{font-size:11px;color:#666}}.llm{{font-size:12px;color:#555;margin-top:4px}}
.meta{{color:#555;font-size:13px}}</style></head><body>
<h2>Pre-promo QA: {html.escape(os.path.basename(report['video']))}</h2>
<div><span class="verdict">{report['verdict']}</span>&nbsp; {c['blocker']} blocker · {c['major']} major · {c['minor']} minor · {c['info']} info</div>
<p class="meta">Duration {report['media']['duration']:.1f}s · {report['media']['fps'] or 0:.1f} fps ·
script: {html.escape(os.path.basename(report['script'])) if report['script'] else 'none'} · {len(report['shots'])} shots ·
face detector: {report['face_detector']} (face in {100 * (report.get('face_frame_share') or 0):.0f}% of frames) · runtime {report['runtime_s']}s</p>
<div class="bar">{marks}</div><div class="meta" style="display:flex;justify-content:space-between"><span>00:00</span><span>{_ts(dur)}</span></div>
<table><tr><th>#</th><th>Severity</th><th>Check</th><th>Time</th><th>What was found</th><th>Confidence</th><th>Frames</th></tr>
{''.join(rows) or '<tr><td colspan=7>No issues found.</td></tr>'}</table>
<p class="meta">Verdict rule: any blocker → FAIL; any major → REVIEW (a person checks the flagged timestamps); otherwise PASS.
"info" rows are detections the screenplay explains (e.g. scripted fade) and need no action.</p>
</body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return os.path.join(out_dir, "report.html")
