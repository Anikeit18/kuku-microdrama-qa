#!/usr/bin/env python3
"""Diagnostic: are faces really missing, or is the detector missing them?

For every video in a folder, samples 6 frames and draws:
  green  = MediaPipe Face Landmarker on the full frame (what qa.py uses now)
  blue   = OpenCV YuNet face detector (handles small / distant faces)
Writes one contact sheet: <out>/face_debug.jpg  (rows = videos, columns = frames)

    python tests/face_debug.py real_inputs --out outputs/face_debug
"""
import argparse
import glob
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def yunet():
    p = os.path.join(os.path.expanduser("~"), ".cache", "kuku-qa", "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, p)
    return cv2.FaceDetectorYN.create(p, "", (320, 320), 0.6, 0.3, 5000)


def mediapipe_image_mode():
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    from qa.faces import _model_path
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=_model_path()), running_mode=vision.RunningMode.IMAGE,
        num_faces=3, min_face_detection_confidence=0.3, min_face_presence_confidence=0.3))

    def detect(bgr):
        h, w = bgr.shape[:2]
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        out = []
        for lms in r.face_landmarks or []:
            xs = [p.x * w for p in lms]
            ys = [p.y * h for p in lms]
            out.append((int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))))
        return out
    return detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="outputs/face_debug")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mp_detect = mediapipe_image_mode()
    yn = yunet()
    rows, stats = [], []
    for v in sorted(glob.glob(os.path.join(a.folder, "*.mp4"))):
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tiles, cm, cy = [], 0, 0
        for k in range(6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / 6))
            ok, f = cap.read()
            if not ok:
                continue
            h, w = f.shape[:2]
            s = 480 / min(h, w)
            f = cv2.resize(f, (int(w * s), int(h * s)))
            m = mp_detect(f)
            yn.setInputSize((f.shape[1], f.shape[0]))
            _, faces = yn.detect(f)
            y = [] if faces is None else [tuple(int(x) for x in r[:4]) for r in faces]
            cm += bool(m)
            cy += bool(y)
            for (x, yy, bw, bh) in m:
                cv2.rectangle(f, (x, yy), (x + bw, yy + bh), (0, 255, 0), 3)
            for (x, yy, bw, bh) in y:
                cv2.rectangle(f, (x - 3, yy - 3), (x + bw + 3, yy + bh + 3), (255, 120, 0), 2)
            tiles.append(cv2.resize(f, (320, int(320 * f.shape[0] / f.shape[1]))))
        cap.release()
        row = np.hstack(tiles)
        label = f"{os.path.basename(v)}   mediapipe {cm}/6   yunet {cy}/6"
        cv2.rectangle(row, (0, 0), (row.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(row, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        rows.append(row)
        stats.append(label)
        print(label)
    width = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows])
    out = os.path.join(a.out, "face_debug.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("contact sheet:", out)


if __name__ == "__main__":
    main()
