#!/usr/bin/env python3
import os

# 1. 必须在导入 cv2/insightface 之前设置环境变量以屏蔽 ONNX 警告
os.environ['ORT_LOGGING_LEVEL'] = '3' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import argparse
import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis

# 2. 忽略 NumPy 和 Scikit-image 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

THRESHOLD = 0.35
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]
CACHE_FILE = "template_cache.npz"
OUTPUT_DIR = "output"
BATCH_SIZE = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局模型实例，只初始化一次
_app = None

def get_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
        _app.prepare(ctx_id=0)
    return _app


def cv2_put_chinese(img, text, pos):
    """在图像上绘制中文文字，自动处理坐标越界"""
    x, y = pos
    h, w = img.shape[:2]
    if y < 0:
        y = 5
    if x < 0:
        x = 5
    if x > w or y > h:
        return img

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, 24)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def load_templates(template_dir):
    """加载人脸模板，支持缓存。返回 (names_list, embeddings_2d_array)"""
    app = get_app()

    if os.path.exists(CACHE_FILE):
        d = np.load(CACHE_FILE, allow_pickle=True)
        names = d["names"].tolist()
        embs = d["embeddings"]
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return names, embs

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
    return names, np.stack(embs)


def process_frame(frame, names, embs):
    """处理单帧：检测人脸并与模板比对"""
    app = get_app()
    results = []
    faces = app.get(frame)
    for face in faces:
        q = face.normed_embedding
        if embs.shape[0] == 0:
            name = "UNKNOWN"
        else:
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

    # 1. 加载模板
    names, embs = load_templates(args.template_dir)

    # 2. 视频处理
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, os.path.basename(args.video_path)),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    pbar = tqdm(total=total_frames, desc="Processing video")

    batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)

        if len(batch) == BATCH_SIZE:
            for f in batch:
                r = process_frame(f, names, embs)
                for (x1, y1, x2, y2), name in r:
                    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    f = cv2_put_chinese(f, name, (x1, y1 - 35))
                out.write(f)

            pbar.update(len(batch))
            batch.clear()

    if batch:
        for f in batch:
            r = process_frame(f, names, embs)
            for (x1, y1, x2, y2), name in r:
                cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                f = cv2_put_chinese(f, name, (x1, y1 - 35))
            out.write(f)
        pbar.update(len(batch))

    pbar.close()
    cap.release()
    out.release()


if __name__ == "__main__":
    main()
