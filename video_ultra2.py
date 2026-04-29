#!/usr/bin/env python3
import os

os.environ['ORT_LOGGING_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import argparse
import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)

THRESHOLD = 0.35
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]
CACHE_FILE = "template_cache.npz"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cv2_put_chinese(img, text, pos):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 24
        )
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def load_templates(app, template_dir):
    if os.path.exists(CACHE_FILE):
        d = np.load(CACHE_FILE, allow_pickle=True)
        return d["names"], d["embeddings"]

    names, embs = [], []
    for f in os.listdir(template_dir):
        if not f.lower().endswith((".jpg", ".png")):
            continue
        img = cv2.imread(os.path.join(template_dir, f))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        names.append(os.path.splitext(f)[0])
        embs.append(faces[0].normed_embedding)

    if not embs:
        return [], np.empty((0, 512))

    np.savez(CACHE_FILE, names=np.array(names),
             embeddings=np.stack(embs))
    return names, embs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template_dir")
    ap.add_argument("video_path")
    args = ap.parse_args()

    app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
    app.prepare(ctx_id=0)

    names, embs = load_templates(app, args.template_dir)

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, os.path.basename(args.video_path)),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    pbar = tqdm(total=total, desc="Processing video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            q = face.normed_embedding
            if embs.shape[0] == 0:
                name = "UNKNOWN"
            else:
                scores = np.dot(embs, q)
                idx = int(np.argmax(scores))
                name = names[idx] if scores[idx] >= THRESHOLD else "UNKNOWN"
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2_put_chinese(frame, name, (x1, y1 - 35))

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()


if __name__ == "__main__":
    main()
