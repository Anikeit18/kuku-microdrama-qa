"""Run every check on one episode and build the report dict."""
import time

import cv2

from . import lipsync, llm, media, technical, visual
from .faces import make_provider
from .script_parser import parse_script

SEV_ORDER = {"blocker": 0, "major": 1, "minor": 2, "info": 3}


def grab_frame_fn(path, width=480):
    def grab(t):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return b""
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(h * width / w)))
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    return grab


def run(video_path, script_path=None, cfg=None, face_provider="auto", log=print, event=None):
    """event(kind, **data) streams progress to the live demo UI; ignored by the CLI."""
    def ev(kind, **data):
        if event:
            event(kind, **data)

    t_start = time.time()
    info = media.probe(video_path)
    script = parse_script(script_path) if script_path else None
    log(f"[qa] {video_path}: {info['duration']:.1f}s, {info['fps'] or 0:.1f} fps, "
        f"script={'yes, %d scenes' % len(script['scenes']) if script else 'no'}")

    ev("start", media=info, scenes=len(script["scenes"]) if script else 0)
    issues = technical.run_all(video_path, info, script, cfg,
                               on_check=lambda name, new: ev("check_start", check=name) if new is None
                               else ev("check_done", check=name, issues=new))
    log(f"[qa] technical checks: {len(issues)} findings")

    lip_detail, provider_name, shots = {}, None, []
    if info["has_video"]:
        provider = make_provider(face_provider, cfg["min_face_frac"])
        provider_name = provider.name
        for name in ("continuity", "lip_sync"):
            ev("check_start", check=name)
        fp = visual.frame_pass(video_path, provider, cfg, progress=lambda f: ev("frames", fraction=f))
        shots = visual.detect_shots(fp, cfg)
        shots = visual.shot_descriptors(video_path, fp, shots, cfg)
        cont = visual.continuity_checks(shots, script, cfg)
        cont = llm.verify(cont, grab_frame_fn(video_path), cfg)
        issues += cont
        ev("check_done", check="continuity", issues=cont, shots=len(shots), face_share=fp["face_frame_share"])
        log(f"[qa] {len(shots)} shots, continuity: {len(cont)} findings (faces: {provider_name}, "
            f"a face found in {100 * fp['face_frame_share']:.0f}% of frames at {fp['size'][1]}x{fp['size'][0]})")
        if info["has_audio"]:
            # windows already explained by silence or a frozen picture are not judged again
            silences = [(i["start"], i["end"]) for i in issues if i["check"] in ("silence", "frozen_picture")]
            lip_issues, lip_detail = lipsync.check_lipsync(video_path, fp, script, cfg, silences,
                                                           info.get("av_start_offset", 0.0))
            issues += lip_issues
            ev("check_done", check="lip_sync", issues=lip_issues, summary=lip_detail.get("summary"))
            log(f"[qa] lip sync: {len(lip_issues)} findings, {lip_detail.get('summary')}")

    issues.sort(key=lambda i: (SEV_ORDER[i["severity"]], i["start"]))
    for n, it in enumerate(issues, 1):
        it["id"] = n
    counts = {s: sum(i["severity"] == s for i in issues) for s in SEV_ORDER}
    verdict = "FAIL" if counts["blocker"] else ("REVIEW" if counts["major"] else "PASS")
    return {
        "video": video_path, "script": script_path, "verdict": verdict, "counts": counts,
        "media": info, "face_detector": provider_name,
        "face_frame_share": fp["face_frame_share"] if info["has_video"] else None,
        "shots": [{"start": round(s["start"], 2), "end": round(s["end"], 2)} for s in shots],
        "issues": issues, "lip_sync_windows": lip_detail.get("windows", []),
        "runtime_s": round(time.time() - t_start, 1), "config": cfg,
    }
