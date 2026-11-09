#!/usr/bin/env python3
"""Build the static presentation site (deploy the site/ folder to Vercel, Netlify or GitHub Pages).

    python tools/build_site.py --github https://github.com/<you>/<repo> --demo https://huggingface.co/spaces/<you>/<space>

Reads: real_results/summary.json, outputs/*/evaluation.json, docs/solution_doc.pdf,
       outputs/real/<episode>/report.html (if you ran the real set) and a few synthetic reports.
Writes: site/index.html, site/reports/..., site/solution_doc.pdf
"""
import argparse
import html
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH_SHOWCASE = []  # the site shows real AI clips only; synthetic results appear as a score table
REAL_NOTES = {
    "r01_clean_3_scenes": "3 Veo clips as 3 scenes. Tool found 2 real lip-sync faults in Veo's output (confirmed by watching).",
    "r02_black_frames": "Injected 0.5 s black.", "r03_frozen_picture": "Injected 1.5 s freeze.",
    "r04_silence_over_dialogue": "Injected 1.5 s mute over speech.", "r05_clipped_audio": "Injected clipping.",
    "r06_av_offset_300ms": "Injected 300 ms audio delay.", "r07_wrong_audio_track": "Picture of one clip, speech of another (missed).",
    "r08_regenerated_second_half": "Veo then Sora render of the same prompt in one scene.",
    "r09_regenerated_insert_shot": "2 s of another model's render inside the scene.",
    "r10_weaker_model_untouched": "LTX-2 output, untouched.",
}


def esc(x):
    return html.escape(str(x))


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "site"))
    ap.add_argument("--github", default="")
    ap.add_argument("--demo", default="")
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(os.path.join(a.out, "reports"), exist_ok=True)

    real = load(os.path.join(ROOT, "real_results", "summary.json"))
    ev = {}
    for d in ["tuning_with_script", "heldout_with_script", "tuning_no_script", "heldout_no_script"]:
        p = os.path.join(ROOT, "outputs", d, "evaluation.json")
        if os.path.exists(p):
            ev[d] = load(p)
    pdf = os.path.join(ROOT, "docs", "solution_doc.pdf")
    if os.path.exists(pdf):
        shutil.copy(pdf, os.path.join(a.out, "solution_doc.pdf"))

    def copy_report(run, ep):
        src = os.path.join(ROOT, "outputs", run, ep, "report.html")
        if not os.path.exists(src):
            return None
        rel = f"reports/{run}/{ep}.html"
        os.makedirs(os.path.join(a.out, "reports", run), exist_ok=True)
        shutil.copy(src, os.path.join(a.out, rel))
        verdict = load(os.path.join(os.path.dirname(src), "report.json"))["verdict"]
        return rel, verdict

    real_cards = []
    for ep, note in REAL_NOTES.items():
        r = copy_report("real", ep)
        if r:
            real_cards.append((ep, note, *r))
    synth_cards = []
    for run, ep, note in SYNTH_SHOWCASE:
        r = copy_report(run, ep)
        if r:
            synth_cards.append((ep, note, *r))

    def cards(items):
        out = []
        for ep, note, rel, verdict in items:
            out.append(f'<a class="card" href="{rel}" target="_blank"><span class="v v-{verdict.lower()}">{verdict}</span>'
                       f'<b>{esc(ep)}</b><span>{esc(note)}</span></a>')
        return "".join(out)

    per_fault = "".join(
        f"<tr><td>{esc(f['episode'])}</td><td>{esc(f['fault'])}</td>"
        f"<td class='{'miss' if f['result'].startswith('MISSED') else 'hit'}'>{esc(f['result'])}</td></tr>"
        for f in real["per_fault"])
    fa = "".join(f"<li><b>{x['count']}×</b> {esc(x['what'])} <span class='muted'>({esc(x['severity'])})</span></li>"
                 for x in real["false_alarm_breakdown"])
    prog = "".join(f"<tr><td>{esc(x['step'])}</td><td>{esc(x['faces_in_frames'])}</td><td>{x['false_alarms']}</td>"
                   f"<td>{esc(x['lip_faults_caught'])}</td></tr>" for x in real["progression"])
    syn_rows = "".join(
        f"<tr><td>{esc(k.replace('_', ' '))}</td><td>{v['faults_caught']}</td><td>{v['faults_caught_major_plus']}</td>"
        f"<td>{v['false_alarms_total']}</td></tr>" for k, v in ev.items())
    links = []
    if os.path.exists(os.path.join(a.out, "solution_doc.pdf")):
        links.append('<a class="btn" href="solution_doc.pdf" target="_blank">Solution doc (PDF)</a>')
    if a.github:
        links.append(f'<a class="btn" href="{esc(a.github)}" target="_blank">GitHub repo</a>')
    if a.demo:
        links.append(f'<a class="btn primary" href="{esc(a.demo)}" target="_blank">Try the live demo</a>')
    real_reports = (f'<div class="grid">{cards(real_cards)}</div>' if real_cards else
                    '<p class="muted">Real-clip reports appear here after running the real set and rebuilding the site.</p>')

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pre-promo QA gate for AI microdramas</title>
<style>
:root{{--ink:#14171f;--muted:#5b6474;--line:#e3e6ec;--bg:#f7f8fa;--acc:#3b4cca;--red:#c62828;--amb:#d9730d;--grn:#2e7d32}}
*{{box-sizing:border-box}}body{{margin:0;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:#fff}}
.wrap{{max-width:1040px;margin:0 auto;padding:0 24px}}
header{{background:var(--ink);color:#fff;padding:64px 0 48px}}
header h1{{font-size:40px;line-height:1.15;margin:0 0 12px;letter-spacing:-.5px}}
header p{{color:#c9cfdb;max-width:720px;font-size:18px;margin:0 0 24px}}
.btn{{display:inline-block;padding:10px 16px;border:1px solid #4a5163;border-radius:8px;color:#fff;text-decoration:none;margin:0 8px 8px 0}}
.btn.primary{{background:#fff;color:var(--ink);font-weight:600}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:32px}}
.stat{{border-top:2px solid #4a5163;padding-top:10px}}.stat b{{display:block;font-size:30px}}.stat span{{color:#c9cfdb;font-size:14px}}
section{{padding:48px 0;border-bottom:1px solid var(--line)}}h2{{font-size:26px;margin:0 0 8px}}
.lead{{color:var(--muted);max-width:760px;margin:0 0 24px}}
.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}
.step{{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:14px;font-size:14px}}
.step b{{display:block;margin-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{color:var(--muted);font-weight:600}}.hit{{color:var(--grn)}}.miss{{color:var(--red);font-weight:600}}.muted{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
.card{{display:flex;flex-direction:column;gap:4px;border:1px solid var(--line);border-radius:10px;padding:14px;text-decoration:none;color:var(--ink);font-size:14px}}
.card:hover{{border-color:var(--acc)}}.card span:last-child{{color:var(--muted)}}
.v{{align-self:flex-start;font-size:11px;font-weight:700;color:#fff;border-radius:4px;padding:2px 6px}}
.v-fail{{background:var(--red)}}.v-review{{background:var(--amb)}}.v-pass{{background:var(--grn)}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:32px}}ul{{padding-left:18px}}
footer{{padding:32px 0;color:var(--muted);font-size:13px}}
@media(max-width:800px){{.stats,.flow,.two{{grid-template-columns:1fr 1fr}}header h1{{font-size:30px}}}}
</style></head><body>
<header><div class="wrap">
<h1>Catch broken AI episodes before they become broken promos</h1>
<p>A QA gate that runs on each AI-generated microdrama episode and its screenplay, just before promo cutting. It flags black frames,
frozen picture, unexpected silence, distorted audio, outfit and background changes inside a scene, and broken lip sync, with
timestamps and frames, then gives a pass / review / fail verdict.</p>
{''.join(links)}
<div class="stats">
<div class="stat"><b>{real['faults_flagged_any']}/{real['faults_total']}</b><span>faults flagged on 10 real AI-video episodes</span></div>
<div class="stat"><b>2</b><span>real lip-sync faults in Veo 3.1 output found by the tool, confirmed by eye</span></div>
<div class="stat"><b>{real['false_alarms']}</b><span>false alarms on real clips (down from 26 during tuning)</span></div>
<div class="stat"><b>$0</b><span>runs locally: ffmpeg + OpenCV + MediaPipe, no paid API</span></div>
</div></div></header>

<section><div class="wrap">
<h2>The problem</h2>
<p class="lead">AI pipelines generate each shot separately, then stitch picture, voice and lip sync together. Those seams produce
exactly these defects. When one reaches a promo, it burns paid reach at the moment a new viewer decides whether to watch.
Today an editor has to watch every episode to catch them.</p>
<div class="flow">
<div class="step"><b>1 · Input</b>Episode video + timestamped screenplay</div>
<div class="step"><b>2 · Technical</b>ffmpeg: black, freeze, silence, clipping, A/V start offset</div>
<div class="step"><b>3 · Continuity</b>Shots → background colour (people masked) and clothing under the main face, per scripted scene</div>
<div class="step"><b>4 · Lip sync</b>YuNet finds faces → MediaPipe mouth opening on a face crop ↔ speech loudness, ±0.5 s</div>
<div class="step"><b>5 · Verdict</b>Blocker → FAIL · major → REVIEW · else PASS. Report with frames</div>
</div>
<p class="muted" style="margin-top:16px">The screenplay says what's <i>intended</i>: a scripted fade isn't an error, silence over a
scripted line is a blocker, a scene is one place with a fixed cast. An optional vision LLM (Gemini or any OpenAI-compatible model,
swapped with an environment variable) can give a second opinion on continuity flags. It can only downgrade a flag.</p>
</div></section>

<section><div class="wrap">
<h2>Results on real AI video</h2>
<p class="lead">10 episodes built from {esc(real['source'])}. Technical faults were injected at known times. Continuity breaks are
real: two models' renders of the same prompt cut into one scene. Every flag was then checked by watching the clips.</p>
<table><tr><th>Episode</th><th>Fault</th><th>Result</th></tr>{per_fault}</table>
<div class="two" style="margin-top:28px">
<div><h3>False alarms ({real['false_alarms']})</h3><ul>{fa}</ul></div>
<div><h3>What testing on real clips changed</h3>
<table><tr><th>Version</th><th>Faces seen</th><th>False alarms</th><th>Lip-sync faults caught</th></tr>{prog}</table></div>
</div>
<h3 style="margin-top:28px">Open the real reports</h3>{real_reports}
</div></section>

<section><div class="wrap">
<h2>Results on the synthetic seeded set</h2>
<p class="lead">Cartoon episodes with exactly timed faults. Thresholds were tuned on the tuning set; the held-out set was built
afterwards with smaller faults. Without the screenplay, continuity detection mostly fails, which is why the script is an input.</p>
<table><tr><th>Run</th><th>Faults flagged</th><th>As major/blocker</th><th>False alarms</th></tr>{syn_rows}</table>
{('<h3 style="margin-top:28px">Sample reports</h3><div class="grid">' + cards(synth_cards) + '</div>') if synth_cards else ''}
</div></section>

<section><div class="wrap two">
<div><h2>Where it fails</h2><ul>
<li>Wrong audio track missed: faces on screen, but the mouth couldn't be measured at those angles.</li>
<li>Lip sync compares loudness with mouth opening. It can't tell if the wrong words are mouthed.</li>
<li>Intent comes from the script. Unscripted stylistic fades and unlisted locations get flagged.</li>
<li>Outfit = colour: a same-colour swap is missed. Clips were 8–12 s; next step is ~20 real episodes labelled by an editor.</li></ul></div>
<div><h2>How a team would know it works</h2><ul>
<li>Share of flags the editor marks real (target &gt;60%)</li>
<li>Blocker defects later found in published promos (target ≈0)</li>
<li>Editor minutes per episode spent on technical review</li>
<li>Regenerations moved before promo cutting, not after</li></ul></div>
</div></section>
<footer><div class="wrap">Kuku AI Generalist assignment · Problem 3 · Real clips: microsoft/AVGen-Bench (MIT license).
Synthetic episodes generated for testing. No Kuku input-pack material is published here.</div></footer>
</body></html>"""
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"site built: {a.out}  (real reports: {len(real_cards)}, synthetic reports: {len(synth_cards)})")


if __name__ == "__main__":
    main()
