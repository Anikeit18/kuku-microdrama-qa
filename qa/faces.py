"""Face + mouth-opening providers.

1. MediaPipe Face Landmarker (preferred). Needs `face_landmarker.task`; it is
   downloaded on first run to ~/.cache/kuku-qa/ (or set QA_FACE_MODEL).
2. Skin-colour fallback (no model needed). Much weaker on real footage: only
   used when MediaPipe is unavailable, and the report says which one ran.

Each provider returns, per frame, a list of faces:
    {"box": (x, y, w, h) in pixels, "mouth": float}   # mouth ~ how open, 0..1
"""
import os
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def _cached(url, name, env=None):
    p = os.environ.get(env) if env else None
    if p and os.path.exists(p):
        return p
    cache = os.path.join(os.path.expanduser("~"), ".cache", "kuku-qa", name)
    if os.path.exists(cache):
        return cache
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        urllib.request.urlretrieve(url, cache)
        return cache
    except Exception:
        return None


def _model_path():
    return _cached(MODEL_URL, "face_landmarker.task", "QA_FACE_MODEL")


def _mouth_from_result(res, i, face_h_px=None):
    """Mouth opening 0..1 from a FaceLandmarker result: jawOpen blendshape + inner-lip gap."""
    lms = res.face_landmarks[i]
    face_h = max(1e-6, lms[152].y - lms[10].y)
    lip_gap = abs(lms[14].y - lms[13].y) / face_h
    mouth = min(1.0, lip_gap * 5)
    if res.face_blendshapes and i < len(res.face_blendshapes):
        jaw = {c.category_name: c.score for c in res.face_blendshapes[i]}.get("jawOpen")
        if jaw is not None:
            mouth = 0.5 * jaw + 0.5 * mouth
    return float(mouth)


class YuNetMediaPipeFaces:
    """YuNet finds faces (works on small / distant faces in cinematic shots); MediaPipe then
    measures the mouth on an enlarged crop of each face, where it is reliable.
    On real AI trailer clips YuNet found faces in 22/60 sampled frames vs 7/60 for MediaPipe alone."""
    name = "yunet+mediapipe_crop"

    def __init__(self, max_faces=3, score=0.6):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        yp, lp = _cached(YUNET_URL, "face_detection_yunet_2023mar.onnx", "QA_YUNET_MODEL"), _model_path()
        if not yp or not lp:
            raise RuntimeError("YuNet or face_landmarker model not available")
        self._mp, self.max_faces = mp, max_faces
        self._yn = cv2.FaceDetectorYN.create(yp, "", (320, 320), score, 0.3, 5000)
        self._lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=lp), running_mode=vision.RunningMode.IMAGE,
            num_faces=1, output_face_blendshapes=True,
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

    def detect(self, bgr, t_sec):
        h, w = bgr.shape[:2]
        self._yn.setInputSize((w, h))
        _, dets = self._yn.detect(bgr)
        faces = []
        if dets is None:
            return faces
        for d in sorted(dets, key=lambda d: -d[2] * d[3])[: self.max_faces]:
            x, y, bw, bh = (int(v) for v in d[:4])
            # square crop around the face, 1.8x its size, enlarged to 256 px for the landmarker
            cx, cy, side = x + bw / 2, y + bh / 2, 1.8 * max(bw, bh)
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            mouth = None
            if x1 - x0 > 8 and y1 - y0 > 8:
                crop = cv2.resize(bgr[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_CUBIC)
                res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                                     data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if res.face_landmarks:
                    mouth = _mouth_from_result(res, 0)
            faces.append({"box": (max(0, x), max(0, y), bw, bh), "mouth": mouth})
        return faces


class MediaPipeFaces:
    name = "mediapipe_face_landmarker"

    def __init__(self, max_faces=3):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        path = _model_path()
        if not path:
            raise RuntimeError("face_landmarker.task not available")
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO, num_faces=max_faces,
            output_face_blendshapes=True)
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, bgr, t_sec):
        h, w = bgr.shape[:2]
        ts = max(self._last_ts + 1, int(t_sec * 1000))
        self._last_ts = ts
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(img, ts)
        faces = []
        for i, lms in enumerate(res.face_landmarks or []):
            xs = np.array([p.x for p in lms]) * w
            ys = np.array([p.y for p in lms]) * h
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            faces.append({"box": (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                          "mouth": _mouth_from_result(res, i)})
        return faces


class SkinFaces:
    """Fallback: largest skin-coloured blobs; mouth = dark-pixel share in lower-middle of the blob."""
    name = "skin_colour_fallback"

    def __init__(self, max_faces=3, min_frac=0.004):
        self.max_faces, self.min_frac = max_faces, min_frac

    def detect(self, bgr, t_sec):
        h, w = bgr.shape[:2]
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycc, (40, 138, 80), (240, 175, 125))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # drop thin edge halos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        cands = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.min_frac * w * h:
                continue
            if not (0.5 < bh / max(bw, 1) < 2.2):
                continue
            cands.append((area, (int(x), int(y), int(bw), int(bh))))
        cands.sort(reverse=True)
        faces = []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for _, (x, y, bw, bh) in cands[: self.max_faces]:
            roi = gray[y + int(0.62 * bh): y + int(0.92 * bh), x + int(0.28 * bw): x + int(0.72 * bw)]
            ref = np.median(gray[y:y + bh, x:x + bw])
            mouth = float((roi < 0.45 * ref).mean()) if roi.size else 0.0
            faces.append({"box": (x, y, bw, bh), "mouth": mouth})
        return faces


def make_provider(prefer="auto", min_frac=0.004):
    order = {"auto": ["yunet", "mediapipe"], "yunet": ["yunet"], "mediapipe": ["mediapipe"], "skin": []}[prefer]
    for kind in order:
        try:
            return YuNetMediaPipeFaces() if kind == "yunet" else MediaPipeFaces()
        except Exception as e:  # noqa: BLE001
            if prefer != "auto":
                raise
            print(f"[faces] {kind} unavailable ({e}); trying next")
    if prefer == "auto":
        print("[faces] using skin-colour fallback (weak on real footage)")
    return SkinFaces(min_frac=min_frac)
