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
    p("<b>Who:</b> the promo editor, and the QA person before them, who cut clips from new AI-generated episodes every week. "
      "<b>Step:</b> just before an episode goes to promo cutting. <b>Output:</b> a verdict with timestamped flags for black frames, "
      "frozen picture, unexpected silence, distorted audio, outfit or background changes inside a scene, and broken lip sync. Nothing else. "
      "<b>Why it's real:</b> AI pipelines generate each shot separately and then stitch picture, voice and lip sync together, and those "
      "seams are where these defects appear. Trade coverage lists lip-sync mismatches as a common complaint about AI micro-dramas (Pandaily), "
      "and a 2026 short-drama generation paper names production QC that needs extensive manual review as unsolved (arXiv 2605.22144). "
      "My own test proved it: <b>2 of the 10 real Veo 3.1 clips I tested had lip-sync faults that nobody planted</b>, and the tool found "
      "both. <b>Why this over P1/P2:</b> a promo is paid reach. A frozen frame or out-of-sync line inside an ad burns spend at the moment "
      "a new viewer decides. This gate sits upstream of Problem 1, so every promo tool after it benefits."),

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

    p("7. AI tools used, and where they got it wrong", H),
    p("Built with Claude (planning, code, test sets, this doc). I reviewed every decision and threshold and ran everything on my "
      "machine. Where the AI got it wrong, each found by running the tool and reading its output: ffmpeg's default freeze "
      "threshold called a clean talking head 'frozen' 15 times; its first loudness-spike detector flagged almost every syllable; it "
      "wrote MediaPipe code for an API that no longer ships; its scoring script counted catches as false alarms; a Windows "
      "file-encoding 'fix' broke the test generator; it analysed frames at 360 px wide, too small for faces in landscape clips; and it "
      "assumed MediaPipe would find cinematic faces, which the real clips disproved."),
]

out = os.path.join(ROOT, "docs", "solution_doc.pdf")
SimpleDocTemplate(out, pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm, topMargin=10 * mm, bottomMargin=9 * mm,
                  title="Pre-promo QA gate for AI microdramas").build(story)
print(out)
