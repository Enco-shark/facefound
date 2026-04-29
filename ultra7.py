#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

# ===============================
# ✅ 唯一可修改参数
# ===============================
THRESHOLD = 0.35
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]

TEMPLATE_DIR = "data"
OUTPUT_DIR = "output"
CACHE_FILE = "template_cache.npz"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cosine(a, b):
    return np.dot(a, b)


# ===============================
# ✅ 中文绘制
# ===============================
def cv2_put_chinese(img, text, pos, color=(0,255,0), fontsize=24):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        fontsize
    )

    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ===============================
# ✅ 加载模板（关键：和 cache 完全一致）
# ===============================
def load_templates(app, template_dir):
    if os.path.exists(CACHE_FILE):
        data = np.load(CACHE_FILE, allow_pickle=True)
        print("[INFO] Load template cache")
        return data["names"], data["embeddings"]

    names, embs = [], []
    for f in os.listdir(template_dir):
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img = cv2.imread(os.path.join(template_dir, f))
        faces = app.get(img)
        if len(faces) == 0:
            print(f"[WARN] No face in template: {f}")
            continue

        names.append(os.path.splitext(f)[0])
        embs.append(faces[0].normed_embedding)

    embs = np.stack(embs)
    np.savez(CACHE_FILE, names=np.array(names), embeddings=embs)
    print(f"[INFO] Cache saved ({len(names)})")
    return names, embs


# ===============================
# ✅ 识别
# ===============================
def recognize(app, names, embs, img):
    faces = app.get(img)
    print(f"[DEBUG] Detected {len(faces)} faces")

    results = []
    for face in faces:
        q = face.normed_embedding
        scores = np.dot(embs, q)
        idx = int(np.argmax(scores))
        score = float(scores[idx])

        name = names[idx] if score >= THRESHOLD else "UNKNOWN"
        print(f"  -> {name} ({score:.3f})")

        results.append((face.bbox.astype(int), name))

    return results


# ===============================
# ✅ 绘制
# ===============================
def draw(img, results):
    for (x1, y1, x2, y2), name in results:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        img = cv2_put_chinese(img, name, (x1, y1 - 30))
    return img


# ===============================
# ✅ 图片处理
# ===============================
def process_image(app, names, embs, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("[ERROR] Image not found:", img_path)
        return

    results = recognize(app, names, embs, img)
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, draw(img, results))
    print("[INFO] Saved:", out_path)


# ===============================
# ✅ 视频处理
# ===============================
def process_video(app, names, embs, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Video open failed")
        return

    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = cap.get(5)

    out = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, os.path.basename(video_path)),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = recognize(app, names, embs, frame)
        out.write(draw(frame, results))

    cap.release()
    out.release()
    print("[INFO] Video saved")


# ===============================
# ✅ 主入口
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template_dir")
    ap.add_argument("input_path")
    args = ap.parse_args()

    app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
    app.prepare(ctx_id=0)

    names, embs = load_templates(app, args.template_dir)

    if os.path.isdir(args.input_path):
        for f in os.listdir(args.input_path):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                process_image(
                    app, names, embs,
                    os.path.join(args.input_path, f)
                )

    elif args.input_path.endswith((".mp4", ".avi", ".mov")):
        process_video(app, names, embs, args.input_path)

    else:
        process_image(app, names, embs, args.input_path)


if __name__ == "__main__":
    main()
