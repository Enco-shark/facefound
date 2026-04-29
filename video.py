#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
import joblib
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

THRESHOLD = 0.35
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]
CACHE_FILE = "template_cache.npz"
OUTPUT_DIR = "output"
BATCH_SIZE = 30   # ✅ 关键

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cosine(a, b):
    return np.dot(a, b)


def cv2_put_chinese(img, text, pos):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 24
    )
    draw.text(pos, text, font=font, fill=(0,255,0))
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
        faces = app.get(img)
        if not faces:
            continue
        names.append(os.path.splitext(f)[0])
        embs.append(faces[0].normed_embedding)

    np.savez(CACHE_FILE, names=np.array(names),
             embeddings=np.stack(embs))
    return names, embs


def process_frame(args):
    frame, app, names, embs = args
    results = []
    for face in app.get(frame):
        q = face.normed_embedding
        scores = np.dot(embs, q)
        idx = int(np.argmax(scores))
        name = names[idx] if scores[idx] >= THRESHOLD else "UNKNOWN"
        results.append((face.bbox.astype(int), name))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template_dir")
    ap.add_argument("video_path")
    args = ap.parse_args()

    app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
    app.prepare(ctx_id=0)

    names, embs = load_templates(app, args.template_dir)

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(5)
    w = int(cap.get(3))
    h = int(cap.get(4))

    out = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, os.path.basename(args.video_path)),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)

        if len(batch) == BATCH_SIZE:
            args = [(f, app, names, embs) for f in batch]
            res = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(process_frame)(a)
                for a in tqdm(args, desc="Batch")
            )
            for f, r in zip(batch, res):
                for (x1,y1,x2,y2), name in r:
                    cv2.rectangle(f,(x1,y1),(x2,y2),(0,255,0),2)
                    f = cv2_put_chinese(f, name, (x1,y1-30))
                out.write(f)
            batch.clear()

    cap.release()
    out.release()
    print("[INFO] Video saved")


if __name__ == "__main__":
    main()
