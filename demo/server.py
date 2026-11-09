#!/usr/bin/env python3
"""Live demo UI for presenting / recording: pick or upload an episode, watch the checks run,
click any finding to jump the video to it.

    python demo/server.py            # opens http://localhost:8000
    python demo/server.py --port 8080 --no-browser

Standard library only (plus the qa package). Episodes are read from real_inputs/ (the real AI-clip test set).
Uploads go to demo_uploads/.
"""
import argparse
import glob
import json
import mimetypes
import os
import re
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qa import pipeline  # noqa: E402
from qa.config import load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(ROOT, "demo_uploads")
# (folder, group label, default face detector). Every video used in testing shows up here.
SOURCES = [("demo_uploads", "Your uploads", "auto"),
           ("real_inputs", "Real AI clips (Veo 3.1 / Sora 2 / LTX-2, AVGen-Bench test set)", "auto")]
JOBS = {}
LOCK = threading.Lock()


def episodes():
    out = []
    for folder, kind, faces in SOURCES:
        vids = glob.glob(os.path.join(ROOT, folder, "*.mp4"))
        if folder == "demo_uploads":  # newest upload first
            vids.sort(key=os.path.getmtime, reverse=True)
        else:
            vids.sort()
        for v in vids:
            stem = os.path.splitext(os.path.basename(v))[0]
            scr = next((p for p in (os.path.join(ROOT, folder, stem + e) for e in (".txt", ".json")) if os.path.exists(p)), None)
            out.append({"id": f"{folder}/{stem}", "name": stem, "kind": kind, "faces": faces,
                        "video": v, "script": scr,
                        "script_text": open(scr, encoding="utf-8").read() if scr and scr.endswith(".txt") else ""})
    return out


def find(ep_id):
    return next((e for e in episodes() if e["id"] == ep_id), None)


def run_job(job_id, ep, faces):
    def event(kind, **data):
        with LOCK:
            JOBS[job_id]["events"].append({"kind": kind, "t": round(time.time() - JOBS[job_id]["t0"], 2), **data})

    try:
        cfg = load_config()
        rep = pipeline.run(ep["video"], ep["script"], cfg, face_provider=faces, log=lambda m: event("log", text=m), event=event)
        rep = json.loads(json.dumps(rep, default=float))
        with LOCK:
            JOBS[job_id]["report"] = rep
    except Exception as e:  # noqa: BLE001  show the error in the UI instead of hanging
        event("error", text=f"{type(e).__name__}: {e}")
    finally:
        with LOCK:
            JOBS[job_id]["done"] = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype=None):
        size = os.path.getsize(path)
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:  # video seeking needs byte ranges
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = size - int(m.group(2))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            page = os.path.join(HERE, "index.html")
            if os.path.exists(page):  # an edited demo/index.html wins; otherwise use the built-in copy
                return self._file(page, "text/html; charset=utf-8")
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if u.path == "/api/episodes":
            return self._json([{k: v for k, v in e.items() if k not in ("video", "script")} | {"has_script": bool(e["script"])}
                               for e in episodes()])
        if u.path.startswith("/media/"):
            ep = find(unquote(u.path[len("/media/"):]))
            return self._file(ep["video"], "video/mp4") if ep else self._json({"error": "not found"}, 404)
        if u.path == "/api/frame":
            ep = find(q.get("ep", [""])[0])
            if not ep:
                return self._json({"error": "not found"}, 404)
            jpg = pipeline.grab_frame_fn(ep["video"], width=320)(float(q.get("t", ["0"])[0]))
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            return self.wfile.write(jpg)
        if u.path.startswith("/api/job/"):
            jid = u.path.rsplit("/", 1)[-1]
            since = int(q.get("since", ["0"])[0])
            with LOCK:
                j = JOBS.get(jid)
                if not j:
                    return self._json({"error": "no job"}, 404)
                return self._json({"events": j["events"][since:], "next": len(j["events"]),
                                   "done": j["done"], "report": j["report"] if j["done"] else None})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        n = int(self.headers.get("Content-Length", 0))
        if u.path == "/api/run":
            body = json.loads(self.rfile.read(n) or b"{}")
            ep = find(body.get("id", ""))
            if not ep:
                return self._json({"error": "unknown episode"}, 404)
            jid = uuid.uuid4().hex[:8]
            with LOCK:
                JOBS[jid] = {"events": [], "done": False, "report": None, "t0": time.time()}
            threading.Thread(target=run_job, args=(jid, ep, body.get("faces") or ep["faces"]), daemon=True).start()
            return self._json({"job": jid})
        if u.path == "/api/upload":
            name = os.path.basename(q.get("name", ["upload.mp4"])[0])
            name = re.sub(r"[^\w.\- ]", "_", name)
            if not name.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".txt", ".json")):
                return self._json({"error": "upload a video (.mp4/.mov/.mkv/.webm) and optionally its .txt/.json script"}, 400)
            os.makedirs(UPLOADS, exist_ok=True)
            with open(os.path.join(UPLOADS, name), "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 16, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            stem, ext = os.path.splitext(name)
            if ext.lower() in (".mov", ".mkv", ".webm"):  # browsers + our pipeline are happiest with mp4
                src, dst = os.path.join(UPLOADS, name), os.path.join(UPLOADS, stem + ".mp4")
                import subprocess
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-c:v", "libx264", "-preset", "veryfast",
                                "-c:a", "aac", dst], check=False)
                os.remove(src)
            return self._json({"ok": True, "id": "demo_uploads/" + stem})
        self._json({"error": "not found"}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    n = len(episodes())
    print(f"Demo running at {url}  (Ctrl+C to stop). Episodes found: {n}")
    if not n:
        print("  No real clips yet: put them in real_inputs/ (python tests/build_real_inputs.py --out real_inputs) or upload in the page.")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass



# The demo page, built in so server.py works on its own (demo/index.html overrides it if present).
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pre-promo QA · live demo</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#262b36;--ink:#e8ebf1;--muted:#8b93a5;--acc:#7c8cff;
--blocker:#ef4444;--major:#f59e0b;--minor:#a3a635;--info:#64748b;--ok:#22c55e}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:16px;padding:14px 22px;border-bottom:1px solid var(--line)}
header h1{font-size:17px;margin:0;font-weight:650;letter-spacing:.2px}header .sub{color:var(--muted);font-size:13px}
#verdict{margin-left:auto;font-weight:800;font-size:15px;letter-spacing:1px;padding:7px 16px;border-radius:8px;
background:var(--line);color:var(--muted);transition:all .3s}
#verdict.FAIL{background:var(--blocker);color:#fff}#verdict.REVIEW{background:var(--major);color:#1b1300}
#verdict.PASS{background:var(--ok);color:#03250f}#verdict.RUNNING{background:#1e2440;color:var(--acc)}
main{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(360px,1fr);gap:18px;padding:18px 22px;height:calc(100% - 62px)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px}
#left{display:flex;flex-direction:column;gap:12px;min-height:0}
#player{background:#000;border-radius:12px;overflow:hidden;display:flex;justify-content:center;position:relative}
#now{position:absolute;left:12px;top:10px;z-index:2;background:#000b;border:1px solid #ffffff22;border-radius:8px;padding:5px 10px;
font-size:13px;pointer-events:none;max-width:70%}#now b{color:#fff}#now span{color:var(--muted)}
video{width:100%;max-height:50vh;background:#000;display:block}
#tlwrap{padding:10px 14px 12px}
#tl{position:relative;height:30px;background:#0c0e12;border-radius:6px;cursor:pointer;overflow:hidden}
.mk{position:absolute;top:4px;bottom:4px;border-radius:3px;min-width:4px;opacity:.9}
.shot{position:absolute;top:0;bottom:0;width:1px;background:#2c3242}
#ph{position:absolute;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 6px #fff8}
.tllab{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:4px}
#script{flex:1;min-height:0;overflow:auto;padding:10px 14px;font:12.5px/1.6 ui-monospace,Consolas,monospace;color:#aeb5c4}
#script .ln{padding:1px 6px;border-radius:4px;white-space:pre-wrap}
#script .ln.ts{cursor:pointer}#script .ln.now{background:#2a3150;color:#fff}
#script .scene{color:var(--acc);font-weight:700;margin-top:6px}
#right{display:flex;flex-direction:column;gap:12px;min-height:0}
.ctrl{padding:14px;display:flex;flex-direction:column;gap:10px}
.row{display:flex;gap:8px;align-items:center}
select,button{font:inherit;color:var(--ink);background:#0f1218;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
select{flex:1;min-width:0}
button.run{background:var(--acc);border-color:var(--acc);color:#0b0e1f;font-weight:700;padding:9px 18px;cursor:pointer}
button.run:disabled{opacity:.5;cursor:default}
label.up{color:var(--muted);font-size:12.5px;cursor:pointer;text-decoration:underline}
.meta{color:var(--muted);font-size:12.5px}
#checks{padding:6px 14px}
.ck{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--line)}
.ck:last-child{border-bottom:0}
.ic{width:18px;height:18px;border-radius:50%;border:2px solid var(--line);flex:none;display:grid;place-items:center;font-size:11px;font-weight:800}
.ck.run .ic{border-color:var(--acc);border-top-color:transparent;animation:spin .8s linear infinite}
.ck.ok .ic{background:var(--ok);border-color:var(--ok);color:#03250f}
.ck.bad .ic{background:var(--blocker);border-color:var(--blocker);color:#fff}
.ck.warn .ic{background:var(--major);border-color:var(--major);color:#1b1300}
.ck .nm{flex:1}.ck .ct{color:var(--muted);font-size:12.5px}
.bar{height:3px;background:var(--line);border-radius:2px;margin-top:4px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
@keyframes spin{to{transform:rotate(360deg)}}
#findings{flex:1;min-height:0;overflow:auto;padding:10px 14px;display:flex;flex-direction:column;gap:8px}
.f{border:1px solid var(--line);border-left:4px solid var(--info);border-radius:8px;padding:9px 11px;cursor:pointer;background:#12151c;
animation:pop .35s ease-out}
.f:hover{border-color:#3a4256}.f.sel{outline:2px solid var(--acc)}
.f.blocker{border-left-color:var(--blocker)}.f.major{border-left-color:var(--major)}.f.minor{border-left-color:var(--minor)}
.f.info{opacity:.65}
.f .top{display:flex;gap:8px;align-items:center;margin-bottom:3px}
.sev{font-size:10.5px;font-weight:800;text-transform:uppercase;letter-spacing:.6px;padding:1px 6px;border-radius:4px;color:#fff;background:var(--info)}
.sev.blocker{background:var(--blocker)}.sev.major{background:var(--major);color:#1b1300}.sev.minor{background:var(--minor);color:#141500}
.f .ttl{font-weight:650}.f .tm{margin-left:auto;color:var(--muted);font:12px ui-monospace,Consolas,monospace}
.f .dt{color:#c3c9d6;font-size:13px}.f .pair{display:flex;gap:6px;margin-top:7px}
.f .pair figure{margin:0;flex:1}.f .pair img{width:100%;border-radius:5px;display:block;max-height:120px;object-fit:cover}
.f .pair figcaption{font-size:11px;color:var(--muted)}
.empty{color:var(--muted);text-align:center;padding:30px 10px}
h3{margin:0;font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);padding:12px 14px 0}
#err{color:var(--blocker);font-size:13px}
#drop{border:2px dashed #3a4364;border-radius:12px;padding:18px 14px;text-align:center;cursor:pointer;transition:all .15s;background:#121628}
#drop:hover,#drop.over{border-color:var(--acc);background:#182046}
#drop b{display:block;font-size:15px;margin-bottom:3px}#drop span{color:var(--muted);font-size:12.5px}
#upbar{height:4px;background:var(--line);border-radius:2px;overflow:hidden;display:none;margin-top:10px}#upbar i{display:block;height:100%;width:0;background:var(--acc)}
.or{color:var(--muted);font-size:12px;text-align:center;margin:2px 0 -2px}
</style></head><body>
<header><div><h1>Pre-promo QA gate for AI microdramas</h1>
<div class="sub">Black frames · frozen picture · silence · distorted audio · outfit &amp; background changes · lip sync</div></div>
<div id="verdict">READY</div></header>
<main>
<section id="left">
  <div id="player"><div id="now"></div><video id="vid" controls playsinline></video></div>
  <div class="panel" id="tlwrap"><div id="tl"><div id="ph"></div></div>
    <div class="tllab"><span>00:00</span><span id="tlinfo"></span><span id="dur">--:--</span></div></div>
  <div class="panel" style="display:flex;flex-direction:column;min-height:0;flex:1">
    <h3>Screenplay</h3><div id="script"><div class="empty">Pick an episode</div></div></div>
</section>
<section id="right">
  <div class="panel ctrl">
    <div id="drop"><b>⬆ Upload an episode to diagnose</b>
      <span>Drag &amp; drop a video here, or click to choose one from your computer (.mp4 / .mov / .mkv / .webm).<br>
      Add its screenplay (.txt) in the same selection for script-aware checks. Diagnosis starts automatically.</span>
      <div id="upbar"><i></i></div>
      <input id="file" type="file" accept=".mp4,.mov,.mkv,.webm,.txt,.json" multiple hidden></div>
    <div class="or">or pick any episode used in testing</div>
    <div class="row"><select id="ep"></select><button class="run" id="run">Run QA</button></div>
    <div class="row"><span class="meta">Face detector</span>
      <select id="faces" style="flex:0 0 auto"><option value="">auto for this episode</option>
      <option value="auto">YuNet + MediaPipe</option><option value="skin">skin colour (fallback)</option></select></div>
    <div class="meta" id="epmeta"></div><div id="err"></div>
  </div>
  <div class="panel"><h3>Checks</h3><div id="checks"></div></div>
  <div class="panel" style="display:flex;flex-direction:column;min-height:0;flex:1">
    <h3>Findings <span id="fcount"></span></h3><div id="findings"><div class="empty">Run QA to see findings appear here</div></div></div>
</section>
</main>
<script>
const CHECKS=[["black_frames","Black frames"],["frozen_picture","Frozen picture"],["silence","Unexpected silence"],
 ["distorted_audio","Distorted audio"],["continuity","Outfit & background continuity"],["lip_sync","Lip sync"]];
const LABEL={black_frames:"Black frames",frozen_picture:"Frozen picture",silence:"Silence",distorted_audio:"Distorted audio",
 outfit_change:"Outfit change",background_change:"Background change",lip_sync:"Lip sync",script_mismatch:"Script / video mismatch"};
const SEVC={blocker:"var(--blocker)",major:"var(--major)",minor:"var(--minor)",info:"var(--info)"};
const $=s=>document.querySelector(s), vid=$("#vid");
let EPS=[],cur=null,dur=0,scriptLines=[],job=null,since=0,live=[];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=t=>{t=Math.max(0,t||0);return String(Math.floor(t/60)).padStart(2,"0")+":"+(t%60).toFixed(1).padStart(4,"0")};
const tsec=s=>s.split(":").map(Number).reduce((a,b)=>a*60+b,0);

async function loadEps(select){
  EPS=await (await fetch("/api/episodes")).json();
  if(!EPS.length){$("#ep").innerHTML='<option>No episodes yet: upload one above</option>';
    $("#epmeta").textContent="Real AI clips appear here once real_inputs/ exists (python tests/build_real_inputs.py --out real_inputs).";return}
  const groups={};EPS.forEach(e=>(groups[e.kind]=groups[e.kind]||[]).push(e));
  $("#ep").innerHTML=Object.entries(groups).map(([g,list])=>`<optgroup label="${esc(g)} (${list.length})">`+
    list.map(e=>`<option value="${esc(e.id)}">${esc(e.name)}${e.has_script?"":"  · no script"}</option>`).join("")+"</optgroup>").join("");
  if(select&&EPS.some(e=>e.id===select))$("#ep").value=select;
  pick();
}
function pick(){
  cur=EPS.find(e=>e.id===$("#ep").value); if(!cur)return;
  vid.src="/media/"+encodeURIComponent(cur.id);
  $("#now").innerHTML=`Now playing: <b>${esc(cur.name)}</b> <span>· ${esc(cur.kind)}</span>`;
  $("#epmeta").textContent=`${cur.kind} · ${cur.has_script?"with screenplay":"no screenplay (continuity checks are weaker)"}`;
  renderScript(cur.script_text); resetRun();
}
function renderScript(txt){
  scriptLines=[];const box=$("#script");
  if(!txt){box.innerHTML='<div class="empty">No screenplay for this episode</div>';return}
  box.innerHTML=txt.split("\n").filter(l=>l.trim()&&!/^PROMPT/.test(l)).map((l,i)=>{
    const m=l.match(/^\s*\[(\d{1,2}(?::\d{1,2}){0,2}(?:\.\d+)?)(?:\s*-\s*(\d{1,2}(?::\d{1,2}){0,2}(?:\.\d+)?))?\]/);
    const sc=/^\s*SCENE\b/i.test(l);
    if(m)scriptLines.push({i,t:tsec(m[1]),e:m[2]?tsec(m[2]):null});
    return `<div class="ln${m?" ts":""}${sc?" scene":""}" data-i="${i}" ${m?`data-t="${tsec(m[1])}"`:""}>${esc(l)}</div>`}).join("");
  box.querySelectorAll(".ln.ts").forEach(el=>el.onclick=()=>seek(+el.dataset.t));
}
function resetRun(){
  $("#verdict").className="";$("#verdict").textContent="READY";$("#err").textContent="";live=[];
  $("#checks").innerHTML=CHECKS.map(([k,n])=>`<div class="ck" id="ck-${k}"><div class="ic"></div><div class="nm">${n}
   ${k==="continuity"||k==="lip_sync"?'<div class="bar"><i></i></div>':""}</div><div class="ct"></div></div>`).join("");
  $("#findings").innerHTML='<div class="empty">Run QA to see findings appear here</div>';$("#fcount").textContent="";
  drawTimeline([]);
}
function seek(t){vid.currentTime=Math.max(0,t-0.4);vid.play()}
function drawTimeline(issues,shots){
  const tl=$("#tl");tl.querySelectorAll(".mk,.shot").forEach(e=>e.remove());
  if(!dur)return;
  (shots||[]).slice(1).forEach(s=>{const d=document.createElement("div");d.className="shot";d.style.left=(100*s.start/dur)+"%";tl.appendChild(d)});
  issues.forEach(i=>{const d=document.createElement("div");d.className="mk";d.title=`${LABEL[i.check]||i.check} ${fmt(i.start)}`;
    d.style.left=(100*i.start/dur)+"%";d.style.width=Math.max(0.6,100*(i.end-i.start)/dur)+"%";d.style.background=SEVC[i.severity];
    d.onclick=e=>{e.stopPropagation();seek(i.start)};tl.appendChild(d)});
}
$("#tl").onclick=e=>{const r=e.currentTarget.getBoundingClientRect();if(dur)seek((e.clientX-r.left)/r.width*dur+0.4)};
vid.onloadedmetadata=()=>{dur=vid.duration;$("#dur").textContent=fmt(dur);drawTimeline(live)};
(function tick(){
  if(dur){$("#ph").style.left=(100*vid.currentTime/dur)+"%";
    const t=vid.currentTime;let now=null;
    for(const s of scriptLines){if(s.t<=t+0.05&&(s.e===null||t<=s.e+0.3))now=s.i}
    document.querySelectorAll("#script .ln").forEach(el=>{const on=+el.dataset.i===now;
      if(on&&!el.classList.contains("now"))el.scrollIntoView({block:"nearest",behavior:"smooth"});el.classList.toggle("now",on)});}
  requestAnimationFrame(tick)})();

function card(i){
  const pair=i.ref_time!==undefined?`<div class="pair"><figure><img src="/api/frame?ep=${encodeURIComponent(cur.id)}&t=${i.ref_time}">
    <figcaption>earlier · ${fmt(i.ref_time)}</figcaption></figure><figure><img src="/api/frame?ep=${encodeURIComponent(cur.id)}&t=${i.frame_time}">
    <figcaption>flagged · ${fmt(i.frame_time)}</figcaption></figure></div>`:"";
  const el=document.createElement("div");el.className="f "+i.severity;
  el.innerHTML=`<div class="top"><span class="sev ${i.severity}">${i.severity}</span><span class="ttl">${esc(LABEL[i.check]||i.check)}</span>
   <span class="tm">${fmt(i.start)} → ${fmt(i.end)}</span></div><div class="dt">${esc(i.detail)}</div>${pair}`;
  el.onclick=()=>{document.querySelectorAll(".f.sel").forEach(x=>x.classList.remove("sel"));el.classList.add("sel");seek(i.start)};
  return el;
}
function addFindings(list){
  const box=$("#findings");if(box.querySelector(".empty"))box.innerHTML="";
  list.forEach(i=>{live.push(i);box.appendChild(card(i))});
  $("#fcount").textContent=live.filter(i=>i.severity!=="info").length?`(${live.filter(i=>i.severity!=="info").length})`:"";
  drawTimeline(live);
}
function markCheck(k,state,n){
  const el=$("#ck-"+k);if(!el)return;el.className="ck "+state;
  const ic=el.querySelector(".ic");ic.textContent=state==="ok"?"✓":state==="bad"||state==="warn"?"!":"";
  el.querySelector(".ct").textContent=state==="run"?"running…":state==="skip"?"skipped":(n?`${n} found`:"clean");
  const b=el.querySelector(".bar i");if(b&&state!=="run")b.style.width="100%";
}
function stateFor(issues){const real=issues.filter(i=>i.severity!=="info");
  return real.some(i=>i.severity==="blocker")?"bad":real.some(i=>i.severity==="major")?"warn":"ok"}

$("#run").onclick=async()=>{
  if(!cur)return;resetRun();$("#run").disabled=true;$("#verdict").className="RUNNING";$("#verdict").textContent="ANALYSING…";
  const r=await (await fetch("/api/run",{method:"POST",body:JSON.stringify({id:cur.id,faces:$("#faces").value})})).json();
  job=r.job;since=0;poll();
};
async function poll(){
  const r=await (await fetch(`/api/job/${job}?since=${since}`)).json();since=r.next;
  for(const e of r.events){
    if(e.kind==="check_start")markCheck(e.check,"run");
    if(e.kind==="check_done"){const real=e.issues.filter(i=>i.severity!=="info");markCheck(e.check,stateFor(e.issues),real.length);addFindings(e.issues)}
    if(e.kind==="frames")document.querySelectorAll("#checks .bar i").forEach(b=>b.style.width=(100*e.fraction)+"%");
    if(e.kind==="check_done"&&e.check==="continuity")$("#tlinfo").textContent=`${e.shots} shots · face in ${Math.round(100*e.face_share)}% of frames`;
    if(e.kind==="error")$("#err").textContent=e.text;
  }
  if(!r.done)return setTimeout(poll,250);
  $("#run").disabled=false;
  if(!r.report){$("#verdict").className="FAIL";$("#verdict").textContent="ERROR";return}
  finish(r.report);
}
function finish(rep){
  CHECKS.forEach(([k])=>{const el=$("#ck-"+k);if(el.classList.contains("run")||el.className==="ck")markCheck(k,"skip")});
  const box=$("#findings");box.innerHTML="";live=[];
  if(!rep.issues.length)box.innerHTML='<div class="empty">No issues found</div>';else addFindings(rep.issues);
  drawTimeline(rep.issues,rep.shots);
  // show the problem straight away: jump to the first real finding and pause on it
  const first=rep.issues.filter(i=>i.severity!=="info").sort((a,b)=>a.start-b.start)[0];
  if(first){vid.pause();vid.currentTime=Math.max(0,(first.frame_time??first.start)+0.05);
    const el=[...document.querySelectorAll("#findings .f")][rep.issues.indexOf(first)];if(el){el.classList.add("sel");el.scrollIntoView({block:"nearest"})}}
  const c=rep.counts;$("#verdict").className=rep.verdict;
  $("#verdict").textContent=`${rep.verdict} · ${c.blocker} blocker · ${c.major} major · ${c.minor} minor`;
}
$("#ep").onchange=pick;
const VID=/\.(mp4|mov|mkv|webm)$/i, SCR=/\.(txt|json)$/i;
function put(name,file){return new Promise(res=>{const x=new XMLHttpRequest();x.open("POST","/api/upload?name="+encodeURIComponent(name));
  x.upload.onprogress=e=>{if(e.lengthComputable)$("#upbar i").style.width=(100*e.loaded/e.total)+"%"};
  x.onload=()=>{try{res(JSON.parse(x.responseText))}catch{res({error:"upload failed"})}};x.onerror=()=>res({error:"upload failed"});x.send(file)})}
async function upload(files){
  files=[...files];$("#err").textContent="";
  const vids=files.filter(f=>VID.test(f.name)),scrs=files.filter(f=>SCR.test(f.name));
  if(!vids.length){$("#err").textContent="Please include a video file (.mp4 / .mov / .mkv / .webm).";return}
  $("#upbar").style.display="block";$("#drop b").textContent="Uploading…";
  let first=null;
  for(const v of vids){
    const stem=v.name.replace(/\.[^.]+$/,"");
    $("#upbar i").style.width="0";const r=await put(v.name,v);if(r.error){$("#err").textContent=r.error;continue}
    first=first||r.id;
    // a script with the same name, or the only script when one video is uploaded, belongs to this video
    const s=scrs.find(x=>x.name.replace(/\.[^.]+$/,"")===stem)||(vids.length===1&&scrs.length===1?scrs[0]:null);
    if(s)await put(stem+s.name.slice(s.name.lastIndexOf(".")),s);
  }
  $("#upbar").style.display="none";$("#drop b").textContent="⬆ Upload an episode to diagnose";$("#file").value="";
  if(first){await loadEps(first);$("#run").click()}
}
const drop=$("#drop");
drop.onclick=()=>$("#file").click();
$("#file").onchange=e=>upload(e.target.files);
["dragenter","dragover"].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add("over")}));
["dragleave","drop"].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove("over")}));
drop.addEventListener("drop",e=>upload(e.dataTransfer.files));
document.addEventListener("dragover",e=>e.preventDefault());document.addEventListener("drop",e=>e.preventDefault());
const P=new URLSearchParams(location.search);
loadEps(P.get("ep")).then(()=>{if(P.get("auto"))$("#run").click()});
</script></body></html>
'''


if __name__ == "__main__":
    main()
