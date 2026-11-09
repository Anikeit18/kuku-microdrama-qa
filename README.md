# Microdrama pre-promo QA

**Results on 10 real AI-video episodes (Veo 3.1 / Sora 2 / LTX-2):** 9/10 faults flagged, including 2 real lip-sync
faults in Veo's output that nobody planted (confirmed by watching); 9 false alarms. Details: `real_results/`,
`docs/solution_doc.pdf`. Deploying the demo + presentation site: `DEPLOY.md`.

A QA gate that runs on an AI-generated microdrama episode **before it goes to promo cutting**.
Input: the episode **video** + its **timestamped screenplay**. Output: a pass / review / fail verdict and
a timestamped list of problems, with frames, in `report.html` + `report.json`.

It flags only these things:

| Check | How it is detected | Where the script helps |
|---|---|---|
| Black frames | ffmpeg `blackdetect` | scripted `FADE TO BLACK` → info, not an error |
| Frozen picture | ffmpeg `freezedetect` (strict −72 dB) | notes when dialogue plays over the freeze |
| Unexpected silence | ffmpeg `silencedetect` (−50 dB, ≥1 s) | silence over a scripted line = blocker; scripted pause = info |
| Distorted audio | numpy: clipped samples per 100 ms, loudness spikes vs the episode's own loud level | – |
| Outfit change | per shot: torso colour histogram under each detected face; clusters per scene vs number of characters | scene boundaries, number of characters, "changes into…" lines |
| Background change | per shot: colour histogram with people masked out; one-off shot or mid-scene switch that never returns | scene boundaries, "walks into…" lines |
| Lip sync | YuNet finds faces → MediaPipe measures mouth opening on a zoomed face crop → correlated with speech loudness over ±0.5 s | which windows have on-screen dialogue (skips V.O./O.S.) |

## Live demo (for presenting)

```powershell
python tests/build_real_inputs.py --out real_inputs   # once: downloads the 10 real AI clips (skip if real_inputs/ exists)
python demo/server.py                                 # opens http://localhost:8000
```
Upload any episode (drag & drop a video, plus its screenplay if you have one) or pick one of the real AI clips →
the diagnosis runs live: checks light up, findings appear, and the video stops on the first problem. Clicking a
finding or a timeline mark jumps the video to that moment.

## Run

Needs Python 3.10+ and `ffmpeg`/`ffprobe` on PATH.

```bash
pip install -r requirements.txt
python qa.py episode.mp4 --script episode.txt --out outputs/ep01      # one episode
python qa.py videos/ --scripts scripts/ --out outputs/                   # batch, matched by file name
streamlit run app.py                                                     # upload UI
python run_all.py                                                        # same as run_all.sh, works on Windows
python run_all.py                                                        # regenerate the synthetic test set + scores
```

On first run two small face models are downloaded to `~/.cache/kuku-qa/`: YuNet (OpenCV, finds faces, including
small and distant ones) and MediaPipe Face Landmarker (measures the mouth on a crop of each face). If they can't load,
the tool falls back to MediaPipe alone, then to a skin-colour finder (`--faces skin`, only good for the cartoon test
episodes). The report says which one ran and in what share of frames it found a face.

## Screenplay format

```
SCENE 1 | INT. CAFE - DAY | 00:00 - 00:12
[00:00.5] ACTION: Maya waits by the window.
[00:01.0 - 00:03.5] MAYA: I told myself I would never come back here.
[00:07] RYAN (V.O.): You were never supposed to see this.
[00:37.2] ACTION: FADE TO BLACK.
```
Line end times are optional (estimated from word count). A JSON form is also accepted (see
`qa/script_parser.py`). If Kuku's screenplays use another layout, only `parse_script()` needs changing.
The script is optional, but without it several checks lose most of their accuracy (see results).

## Swapping the model

The checks that decide pass/fail are plain code. One optional vision-LLM step gives a second opinion on
outfit/background flags (2 frames per flag). It uses only the standard library, and the provider is picked
by an env var:

```bash
export QA_LLM_PROVIDER=gemini GEMINI_API_KEY=...                       # free tier
export QA_LLM_PROVIDER=openai_compatible OPENAI_API_KEY=... QA_LLM_BASE_URL=https://api.groq.com/openai/v1 QA_LLM_MODEL=...
python qa.py ep.mp4 --script ep.txt --llm gemini
```
If the model calls a flag "consistent", it is downgraded to info and kept in the report with the reason.

All thresholds live in `qa/config.py`; override them with `--config my.json`.

## Evaluation

`tests/make_testset.py` builds synthetic episodes (38 s, 3 scenes, 2 characters, dialogue, a scripted
pause and a scripted fade) and injects known faults. There are two sets:
* `tuning`: 11 clips. Thresholds were tuned on these.
* `heldout`: 11 clips built after tuning, with a different seed, different positions, and smaller faults
  near the thresholds.

`tests/evaluate.py` scores each run against the ground truth. All raw reports are in `outputs/`.

## Real test set (10 public AI-generated episodes)

`tests/build_real_inputs.py` downloads real AI video from [microsoft/AVGen-Bench](https://huggingface.co/datasets/microsoft/AVGen-Bench)
(MIT license: Veo 3.1, Sora 2, Kling, LTX-2 and other outputs with native speech) and builds 10 episodes. Each clip's
generation prompt becomes its script, with line timings taken from where speech is detected in the audio.

| Episode | Fault | Real or injected |
|---|---|---|
| r01_clean_3_scenes | none (3 prompts = 3 scenes; scene changes must not be flagged) | real |
| r02_black_frames | 0.5 s black mid-speech | injected |
| r03_frozen_picture | 1.5 s freeze | injected |
| r04_silence_over_dialogue | 1.5 s mute over speech | injected |
| r05_clipped_audio | ×12 gain, hard clip, 2 s | injected |
| r06_av_offset_300ms | audio 300 ms late | injected |
| r07_wrong_audio_track | picture of one clip + speech of another | injected (real mux error) |
| r08_regenerated_second_half | same prompt, model A then model B in one scene: outfit/background break | real renders, spliced |
| r09_regenerated_insert_shot | 2 s of model B inside model A's scene | real renders, spliced |
| r10_weaker_model_untouched | whatever a weaker model gets wrong | real, label by watching |

```powershell
pip install huggingface_hub pandas pyarrow
python tests/build_real_inputs.py --out real_inputs        # ~20 clips downloaded, a few hundred MB
python qa.py real_inputs --scripts real_inputs --out outputs/real
python tests/evaluate.py --samples real_inputs --reports outputs/real --label real
```
Watch the 10 episodes and add any natural faults you see to `real_inputs/natural_labels.json`, then re-run
`evaluate.py`. The labels from my viewing are in `real_results/natural_labels.json`. `SOURCES.json` records the exact model, prompt and file used for every clip. These are AI clips
of 8–12 s, shorter than a real 1–2 min episode.

## Layout

```
qa.py              CLI           app.py          Streamlit UI
qa/technical.py    black/freeze/silence/distortion
qa/visual.py       frame pass, shots, outfit + background continuity
qa/lipsync.py      mouth-vs-audio correlation
qa/faces.py        MediaPipe Face Landmarker + fallback
qa/script_parser.py, qa/llm.py, qa/report.py, qa/pipeline.py, qa/config.py
tests/             test-set generator + scorer
outputs/           raw reports from every run       outputs_archive/  blind held-out score before fixes
real_results/      real-clip summary, run log, natural labels
tools/build_site.py  builds the static presentation site (site/)   real_inputs/  the 10 real AI-clip episodes + scripts
tests/face_debug.py  contact sheet comparing face detectors
```

Note: Kuku's input pack is confidential. Don't commit it to a public repo. The samples here are synthetic.
