"""Build docs/solution_doc.pdf (2 pages) from the evaluation files and real_results/summary.json."""
import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ev = {d: json.load(open(os.path.join(ROOT, "outputs", d, "evaluation.json"), encoding="utf-8"))
      for d in ["tuning_with_script", "heldout_with_script", "tuning_no_script", "heldout_no_script"]}
real = json.load(open(os.path.join(ROOT, "real_results", "summary.json"), encoding="utf-8"))

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.2, leading=10.2, spaceAfter=2.2)
H = ParagraphStyle("h", parent=B, fontName="Helvetica-Bold", fontSize=9.8, leading=11.5, spaceBefore=3.5, spaceAfter=1.5,
                   textColor=colors.HexColor("#1a237e"))
T = ParagraphStyle("t", parent=B, fontName="Helvetica-Bold", fontSize=13.5, leading=16, spaceAfter=1)
S = ParagraphStyle("s", parent=B, fontSize=7.4, leading=8.9)
SB = ParagraphStyle("sb", parent=S, fontName="Helvetica-Bold")


def p(x, st=B):
    return Paragraph(x, st)


def bl(items):
    return [p("• " + i) for i in items]


def table(rows, widths):
    t = Table([[p(c, SB if r == 0 else S) for c in row] for r, row in enumerate(rows)], colWidths=widths)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eaf6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


syn = lambda d: f"{ev[d]['faults_caught']} ({ev[d]['faults_caught_major_plus']} major+), FA {ev[d]['false_alarms_total']}"  # noqa: E731

story = [
    p("Pre-promo QA gate for AI microdramas", T),
    p("Kuku AI Generalist assignment · Problem 3 · <font face='Courier'>python qa.py episode.mp4 --script episode.txt</font> "
      "→ pass / review / fail + timestamped flags with frames", S),
    Spacer(1, 2),

    p("1. Problem and why it matters", H),
    p("<b>Where this came from:</b> as a growth intern at Seekho I worked closely with editors and other teams, and saw how "
      "often AI-generated video needs round after round of iterations, and how some defects still slip through unflagged. A strong "
      "promo pulls viewers in, but if the episode itself is broken, retention suffers, the acquisition spend is wasted, and the "
      "company's LTV/CAC ratio takes the hit. So I placed the check <b>just before promo</b>, the last cheap point to catch a defect. "
      "<b>Who / output:</b> the editor or QA person who signs off new AI episodes every week gets a pass / review / fail verdict with "
      "timestamped flags for black frames, frozen picture, unexpected silence, distorted audio, outfit or background changes inside a "
      "scene, and broken lip sync. Nothing else. "
      "<b>It's real:</b> AI pipelines generate each shot separately and stitch picture, voice and lip sync together, and those seams "
      "are where these defects appear. Trade coverage lists lip-sync mismatches as a common complaint about AI micro-dramas (Pandaily), "
      "and a 2026 short-drama generation paper names production QC that needs extensive manual review as unsolved (arXiv 2605.22144). "
      "My own test showed it too: <b>2 of the 10 real Veo 3.1 clips had lip-sync faults nobody planted</b>, and the tool found both."),

    p("2. Assumptions", H),
    *bl(["Inputs are the final episode file and the timestamped screenplay used to make it (scene headings with time ranges; "
         "<font face='Courier'>[mm:ss] NAME: line</font>). One parser function maps Kuku's real format.",
         "The screenplay defines <i>intent</i>. A scripted FADE or pause is not a defect. A scene is one location with a fixed cast "
         "and outfits unless an action line says otherwise. Without a script the tool still runs but loses most continuity accuracy.",
         "A missed blocker costs ad spend, while a false alarm costs seconds of editor time, so thresholds lean towards flagging and "
         "every flag is graded blocker / major / minor / info."]),

    p("3. How it works: code decides, AI models measure, the editor confirms", H),
    table([
        ["Check", "Method", "Role of the script"],
        ["Black / frozen", "ffmpeg blackdetect, freezedetect (strict −72 dB: a talking head barely changes pixels)", "scripted fade → info"],
        ["Silence / distortion", "ffmpeg silencedetect; clipped-sample share per 100 ms; loudness spikes; container A/V start offset",
         "silence over a scripted line = blocker"],
        ["Outfit / background", "shot cuts → background colour with people masked; clothing colour under the main face (persisting, "
         "1 flag/scene); optional vision-LLM second opinion (Gemini or any OpenAI-compatible, env-var swap; can only downgrade)",
         "scene bounds, cast size, 'changes into' / 'walks into' lines"],
        ["Lip sync", "YuNet finds faces → MediaPipe Face Landmarker measures mouth opening on a zoomed face crop → cross-correlate with "
         "speech loudness ±0.5 s: consistent lag = offset; no relation = mismatch", "which lines are on-screen (skips V.O.)"],
    ], [26 * mm, 112 * mm, 44 * mm]),
    p("Verdict: any blocker → FAIL (regenerate), any major → REVIEW (the editor jumps to flagged timestamps, with frames in the HTML "
      "report), else PASS. Runs locally on a laptop CPU at about real time (an 8 s clip took ~10 s with face analysis), with no paid API."),

    p("4. How I know it's good", H),
    p(f"<b>A. Real AI video (the main evidence).</b> There's no public AI microdrama with a script and labelled faults, so I built 10 "
      f"episodes from {real['source']}. Each clip's generation prompt is its script. Technical faults were injected at known times, "
      f"because published video has already been QC'd. Continuity breaks are <i>real</i>: two models' renders of the same prompt cut "
      f"into one scene, which is what a regenerated shot looks like. Then I watched every clip to confirm flags."),
    table([["Fault", "Result"]] + [[f["fault"] + f" [{f['episode']}]", f["result"]] for f in real["per_fault"]],
          [118 * mm, 64 * mm]),
    p(f"<b>Totals:</b> {real['faults_flagged_any']}/{real['faults_total']} faults flagged "
      f"({real['faults_flagged_major_plus']} as major/blocker), {real['false_alarms']} false alarms "
      f"({real['false_alarms_major_plus']} major/blocker): 4 background flags on trailer clips whose one prompt spans several places, "
      f"2 deliberate dips to black that the prompt-script doesn't mark, 1 outfit flag, 2 weak lip-sync flags (minor). "
      f"<b>What the real clips changed:</b> MediaPipe alone found faces in 0–37% of frames (it's built for selfie-distance faces), so "
      f"lip sync was never judged. YuNet + landmarker-on-crop raised that to 14–100% and caught the offset, but outfit false alarms "
      f"jumped to 17. Judging clothing on the main face only, requiring persistence and one flag per scene cut total false alarms 26 → 11."),
    p(f"<b>B. Synthetic seeded set</b> (cartoon episodes; exact fault timing; thresholds tuned on the tuning set, then a held-out set "
      f"built afterwards with smaller faults). With script: tuning {syn('tuning_with_script')}; held-out {syn('heldout_with_script')}. "
      f"Without script: tuning {syn('tuning_no_script')}; held-out {syn('heldout_no_script')}. The script is what makes continuity work."),

    p("5. Where it fails", H),
    *bl(["<b>Wrong audio track (r07) missed:</b> faces were on screen, but the landmarker couldn't measure the mouth at those angles, so "
         "lip sync was skipped, not failed. Mouth measurement on profile or small faces is the weakest link.",
         "<b>Lip sync</b> correlates loudness with mouth opening. It can't catch the wrong words being mouthed, and it's a weak signal on "
         "real footage, so mismatches are minor unless there's no relation at all.",
         "<b>Intent comes from the script.</b> Unscripted stylistic fades get flagged, and multi-location scenes not marked in the "
         "script get background flags. A real screenplay fixes most of this; a prompt doesn't.",
         "<b>Outfit = colour.</b> A same-colour swap or a missing accessory is missed. Clips are 8–12 s, not 1–2 min. 10 real episodes "
         "is a small sample: next step is ~20 real Kuku episodes labelled by an editor."]),

    p("6. Scale, cost, and how a real team would know it worked", H),
    p("All pass/fail logic is local ffmpeg / OpenCV / MediaPipe: roughly 2–3 CPU-minutes per 2-min episode (measured ~1.2× real time), run in parallel per episode "
      "on any worker queue at near-zero cost. The optional LLM sees only flagged frame pairs (0–3 calls per episode). <b>Pilot "
      "metrics:</b> share of flags the editor marks real (target >60%); blocker-level defects later found in published promos (target ≈0); "
      "editor minutes per episode on technical review; regenerations moved before promo cutting. Every editor decision is kept as a "
      "label to re-tune thresholds per show style."),

    p("7. How I built it, and where AI got it wrong", H),
    p("Problem first, then the solution. From my research background in ML and LLMs I knew that spotting these flaws needs "
      "pretrained, CNN-based vision models rather than an LLM reading frames: here YuNet finds faces, MediaPipe's face landmarker "
      "measures mouth movement, and ffmpeg handles the technical checks. I took publicly available AI-generated videos and injected "
      "known flaws into them to build a labelled test set. Then I built the backend (the detection pipeline), the frontend (a live demo "
      "where you upload an episode and watch the checks run) and the supporting infra with an AI coding assistant (Claude), testing "
      "each piece on the prepared inputs and reviewing every threshold myself. <b>Where the AI got it wrong</b>, each caught by "
      "running the tool and reading its output: ffmpeg's default freeze threshold called a clean talking head 'frozen' 15 times; it "
      "first analysed frames too small for faces in landscape clips; it assumed MediaPipe alone would find faces in cinematic shots, "
      "which the real clips disproved (faces in 0–37% of frames until I switched to YuNet); and its first outfit check raised 17 false "
      "alarms on real footage until it was limited to the main face and to changes that persist."),
]

out = os.path.join(ROOT, "docs", "solution_doc.pdf")
SimpleDocTemplate(out, pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm, topMargin=10 * mm, bottomMargin=9 * mm,
                  title="Pre-promo QA gate for AI microdramas").build(story)
print(out)
