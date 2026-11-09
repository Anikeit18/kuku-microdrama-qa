"""One pass over the frames -> shots, faces, mouth signal; then continuity checks
(outfit and background changes inside a scene)."""
import cv2
import numpy as np

from .script_parser import MOVE_WORDS, WARDROBE_WORDS, scene_at, scene_mentions


def analysis_size(w, h, cfg):
    """Scale so the shorter side is cfg['analysis_short_side'] (never upscale)."""
    s = min(1.0, cfg["analysis_short_side"] / min(w, h))
    return max(2, int(w * s)), max(2, int(h * s))


def frame_pass(path, provider, cfg, progress=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    hists, faces, idx = [], [], 0
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(frame, analysis_size(w, h, cfg), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist)
        faces.append(provider.detect(small, idx / fps))
        idx += 1
        if progress and idx % 10 == 0:
            progress(min(1.0, idx / total))
    cap.release()
    with_face = sum(1 for f in faces if f)
    return {"fps": fps, "n": idx, "hists": hists, "faces": faces,
            "size": small.shape[:2] if idx else (0, 0),
            "face_frame_share": round(with_face / idx, 3) if idx else 0.0}


def detect_shots(fp, cfg):
    fps, hists = fp["fps"], fp["hists"]
    cuts = [0]
    for i in range(1, len(hists)):
        d = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        if d > cfg["cut_threshold"] and (i - cuts[-1]) / fps >= cfg["min_shot_dur"]:
            cuts.append(i)
    cuts.append(len(hists))
    return [{"i0": a, "i1": b, "start": a / fps, "end": b / fps} for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def _torso_box(box, H, W):
    x, y, w, h = box
    x0, x1 = int(x - 0.6 * w), int(x + 1.6 * w)
    y0, y1 = int(y + 1.1 * h), int(y + 3.0 * h)
    cx0, cx1, cy0, cy1 = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    if (cx1 - cx0) * (cy1 - cy0) < 0.4 * (x1 - x0) * (y1 - y0):
        return None  # torso mostly out of frame (tight close-up)
    return cx0, cy0, cx1, cy1


def shot_descriptors(path, fp, shots, cfg):
    cap = cv2.VideoCapture(path)
    for sh in shots:
        span = sh["i1"] - sh["i0"]
        # 3 samples per shot, plus one every ~2 s in long shots (an outfit can change without a cut)
        k = max(3, int((sh["end"] - sh["start"]) / 2.0))
        samples = sorted({sh["i0"] + int(span * (j + 0.5) / k) for j in range(k)})
        bgs, torsos = [], []
        for fi in samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, analysis_size(w, h, cfg), interpolation=cv2.INTER_AREA)
            H, W = small.shape[:2]
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            mask = np.full((H, W), 255, np.uint8)
            frame_faces = [f for f in fp["faces"][fi] if f["box"][2] * f["box"][3] >= cfg["min_face_frac"] * H * W]
            main = max(frame_faces, key=lambda f: f["box"][2] * f["box"][3], default=None)
            for f in frame_faces:
                x, y, bw, bh = f["box"]
                cv2.rectangle(mask, (int(x - 0.3 * bw), int(y - 0.3 * bh)), (int(x + 1.3 * bw), int(y + 1.2 * bh)), 0, -1)
                tb = _torso_box(f["box"], H, W)
                cv2.rectangle(mask, (int(x - 0.7 * bw), int(y + bh)), (int(x + 1.7 * bw), H), 0, -1)
                # clothing is judged only under the main face, and only when it is big enough to trust
                if tb and f is main and bh >= cfg["torso_min_face_h"] * H:
                    x0, y0, x1, y1 = tb
                    th = cv2.calcHist([hsv[y0:y1, x0:x1]], [0, 1], None, [16, 4], [0, 180, 0, 256])
                    cv2.normalize(th, th)
                    torsos.append({"hist": th, "cx": x + bw / 2, "t": fi / fp["fps"]})
            if hsv[..., 2].mean() < cfg["dark_shot_v"]:
                continue  # black/fade frames say nothing about the set
            if mask.mean() > 255 * 0.15:
                bh_ = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 4, 4], [0, 180, 0, 256, 0, 256])
                cv2.normalize(bh_, bh_)
                bgs.append(bh_)
        sh["bg"] = np.mean(bgs, axis=0).astype(np.float32) if bgs else None
        sh["torsos"] = torsos
        sh["n_faces"] = max((len(fp["faces"][fi]) for fi in samples), default=0)
        sh["mid"] = (sh["start"] + sh["end"]) / 2
    cap.release()
    return shots


def _dist(a, b):
    return cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


def _cluster(items, key, th):
    reps, labels = [], []
    for it in items:
        for ci, r in enumerate(reps):
            if _dist(r, key(it)) < th:
                labels.append(ci)
                break
        else:
            reps.append(key(it))
            labels.append(len(reps) - 1)
    return labels


def _group_scenes(shots, script, cfg):
    if script:
        groups = {}
        for sh in shots:
            sc = scene_at(script, sh["mid"])
            groups.setdefault(sc["id"] if sc else None, []).append(sh)
        return [(scene_at(script, g[0]["mid"]), g) for k, g in groups.items() if k is not None]
    # no script: consecutive shots with similar background form a pseudo-scene
    groups, cur = [], []
    for sh in shots:
        if cur and sh["bg"] is not None and all(
                s["bg"] is None or _dist(s["bg"], sh["bg"]) > cfg["bg_cluster_dist"] for s in cur):
            groups.append(cur)
            cur = []
        cur.append(sh)
    if cur:
        groups.append(cur)
    return [(None, g) for g in groups]


def continuity_checks(shots, script, cfg):
    issues = []
    for scene, group in _group_scenes(shots, script, cfg):
        label = f"scene {scene['id']} ({scene['heading']})" if scene else "this stretch"
        bg_shots = [s for s in group if s["bg"] is not None]

        # ---- background ----
        if len(bg_shots) >= 2:
            labs = _cluster(bg_shots, lambda s: s["bg"], cfg["bg_cluster_dist"])
            moves = scene_mentions(scene, MOVE_WORDS)
            # (a) one short shot whose location matches neither neighbour, while neighbours match each other
            for i in range(1, len(bg_shots) - 1):
                if labs.count(labs[i]) == 1 and labs[i - 1] == labs[i + 1] != labs[i] \
                        and bg_shots[i]["end"] - bg_shots[i]["start"] <= cfg["glitch_max_dur"]:
                    issues.append(dict(check="background_change", start=round(bg_shots[i]["start"], 2),
                                       end=round(bg_shots[i]["end"], 2), severity="major", confidence="medium",
                                       detail=f"one shot in {label} shows a different background from the shots around it",
                                       ref_time=round(bg_shots[i - 1]["mid"], 2), frame_time=round(bg_shots[i]["mid"], 2)))
            # (b) location switches mid-scene and never comes back (only judged when a script defines the scene)
            if scene and not moves:
                for j in range(1, len(bg_shots)):
                    before, after = set(labs[:j]), set(labs[j:])
                    pre = bg_shots[j - 1]["end"] - bg_shots[0]["start"]
                    post = bg_shots[-1]["end"] - bg_shots[j]["start"]
                    if not (before & after) and pre >= 2.0 and post >= 2.0:
                        issues.append(dict(check="background_change", start=round(bg_shots[j]["start"], 2),
                                           end=round(bg_shots[-1]["end"], 2), severity="major", confidence="medium",
                                           detail=f"background changes mid-{label} and never returns; the script has no move to a new location",
                                           ref_time=round(bg_shots[j - 1]["mid"], 2), frame_time=round(bg_shots[j]["mid"], 2)))
                        break

        # ---- outfit ----
        torsos = [dict(t, shot=s) for s in group for t in s["torsos"]]
        if len(torsos) >= 2 and not scene_mentions(scene, WARDROBE_WORDS):
            labs = _cluster(torsos, lambda t: t["hist"], cfg["torso_cluster_dist"])
            expected = len(scene["characters"]) if scene and scene["characters"] else \
                max((s["n_faces"] for s in group), default=1)
            expected = max(1, expected)
            counts = {l: labs.count(l) for l in set(labs)}
            order = []
            for l in labs:  # outfits in order of first appearance; one-off samples are ignored as noise
                if l not in order and counts[l] >= cfg["torso_min_samples"]:
                    order.append(l)
            for extra in order[expected:expected + 1]:  # at most one outfit flag per scene
                k = labs.index(extra)
                # reference = the most similar earlier outfit (most likely the same person before the change)
                earlier = [t for t, l in zip(torsos[:k], labs[:k]) if l != extra]
                ref_t = min(earlier, key=lambda t: _dist(t["hist"], torsos[k]["hist"]))["t"] if earlier \
                    else torsos[labs.index(order[0])]["t"]
                sh, t_flag = torsos[k]["shot"], torsos[k]["t"]
                issues.append(dict(check="outfit_change", start=round(max(sh["start"], t_flag - 1.0), 2), end=round(sh["end"], 2),
                                   severity="major", confidence="medium",
                                   detail=f"{len(order)} distinct outfits seen in {label} but only {expected} "
                                          f"character(s) expected; clothing likely changed between shots",
                                   ref_time=round(ref_t, 2), frame_time=round(t_flag, 2)))
    return issues
