"""Streamlit front end:  streamlit run app.py
(Deploys as-is on Hugging Face Spaces: see DEPLOY.md.)"""
import glob
import json
import os
import shutil
import tempfile

import streamlit as st
import streamlit.components.v1 as components

from qa import pipeline, report
from qa.config import load_config

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "real_inputs")

st.set_page_config(page_title="Microdrama pre-promo QA", layout="wide")
st.title("Microdrama pre-promo QA")
st.caption("Flags black frames, frozen picture, silence, distorted audio, outfit/background changes "
           "and lip-sync problems before an episode goes to promo cutting.")

samples = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(SAMPLES, "*.mp4")))
source = st.radio("Input", (["Use a sample episode"] if samples else []) + ["Upload my own"], horizontal=True)

vpath = spath = None
tmp = tempfile.mkdtemp()
if source == "Use a sample episode":
    name = st.selectbox("Sample episode", samples)
    vpath = os.path.join(SAMPLES, name + ".mp4")
    sp = os.path.join(SAMPLES, name + ".txt")
    spath = sp if os.path.exists(sp) else None
    st.video(vpath)
    if spath:
        with st.expander("Screenplay"):
            st.code(open(spath, encoding="utf-8").read())
else:
    c1, c2 = st.columns(2)
    video = c1.file_uploader("Episode video", type=["mp4", "mov", "mkv", "webm"])
    script = c2.file_uploader("Screenplay with timestamps (optional, strongly recommended)", type=["txt", "json"])
    if video:
        vpath = os.path.join(tmp, video.name)
        with open(vpath, "wb") as f:
            f.write(video.getbuffer())
    if script:
        spath = os.path.join(tmp, script.name)
        with open(spath, "wb") as f:
            f.write(script.getbuffer())

with st.expander("Settings"):
    faces = st.selectbox("Face detector", ["auto", "yunet", "mediapipe", "skin"],
                         help="auto = YuNet + MediaPipe, falling back if a model can't load. "
                              "'skin' is a weak fallback.")
    llm = st.selectbox("Vision-LLM verifier for continuity flags", ["off", "gemini", "openai_compatible"],
                       help="Needs GEMINI_API_KEY or OPENAI_API_KEY (+QA_LLM_BASE_URL) set as Space secrets.")
    model = st.text_input("Model name (blank = provider default)")

if vpath and st.button("Run QA", type="primary"):
    cfg = load_config(overrides={"llm_provider": None if llm == "off" else llm, "llm_model": model or None})
    logs, lines = st.empty(), []

    def log(m):
        lines.append(m)
        logs.code("\n".join(lines))

    with st.spinner("Analysing… (about real time on a free CPU)"):
        rep = pipeline.run(vpath, spath, cfg, face_provider=faces, log=log)
        html_path = report.write(rep, os.path.join(tmp, "out"))
    st.subheader(f"Verdict: {rep['verdict']}")
    with open(html_path, encoding="utf-8") as f:
        page = f.read()
    components.html(page, height=900, scrolling=True)
    st.download_button("Download report.json", json.dumps(rep, indent=2, default=float), "report.json")
    st.download_button("Download report.html", page, "report.html")
    shutil.rmtree(os.path.join(tmp, "out"), ignore_errors=True)
