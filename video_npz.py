#!/usr/bin/env python3
"""
video_npz.py — 纯 npz 缓存版本，无需模板图片目录
直接加载 template_cache.npz 进行人脸识别
"""
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
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

THRESHOLD = 0.35
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]
OUTPUT_DIR = "output"
BATCH_SIZE = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局共享模型（线程安全）
_app = None
_names = None
_embs = None


def init_worker():
    global _app
    if _app is None:
        _app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
        _app.prepare(ctx_id=0)


def cv2_put_chinese(img, text, pos):
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
    draw.text(pos, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def load_templates_from_npz(npz_path):
    """直接从 npz 缓存文件加载模板，无需图片目录"""
    if not os.path.exists(npz_path):
        print(f"[ERROR] 找不到缓存文件: {npz_path}")
        print("[INFO] 请先用其他版本（如 video_ultra3.py）生成 template_cache.npz")
        exit(1)

    d = np.load(npz_path, allow_pickle=True)
    names = d["names"].tolist()
    embs = d["embeddings"]
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    print(f"[INFO] 从 {npz_path} 加载了 {len(names)} 个人脸模板")
    return names, embs


def process_frame(frame):
    """单帧人脸检测+识别（使用全局模型和模板）"""
    global _app, _names, _embs
    init_worker()
    faces = _app.get(frame)
    for face in faces:
        q = face.normed_embedding
        if _embs.shape[0] == 0:
            name = "UNKNOWN"
        else:
            scores = np.dot(_embs, q)
            idx = int(np.argmax(scores))
            name = _names[idx] if scores[idx] >= THRESHOLD else "UNKNOWN"
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame = cv2_put_chinese(frame, name, (x1, y1 - 35))
    return frame


def main():
    global _names, _embs

    ap = argparse.ArgumentParser(
        description="人脸识别视频处理（纯 npz 缓存版，无需模板图片目录）"
    )
    ap.add_argument("video_path", help="输入视频文件路径")
    ap.add_argument(
        "-c", "--cache", default="template_cache.npz",
        help="template_cache.npz 文件路径（默认: template_cache.npz）"
    )
    ap.add_argument(
        "-o", "--output", default=None,
        help="输出视频路径（默认: output/ 目录下）"
    )
    args = ap.parse_args()

    # 直接从 npz 加载模板
    _names, _embs = load_templates_from_npz(args.cache)

    # 初始化模型
    init_worker()

    # 打开视频
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出路径
    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(args.video_path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    pbar = tqdm(total=total, desc="Processing video")

    # 多线程处理 batch
    with ThreadPoolExecutor(max_workers=4) as executor:
        batch = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)

            if len(batch) == BATCH_SIZE:
                futures = {
                    executor.submit(process_frame, f): i
                    for i, f in enumerate(batch)
                }
                results = [None] * len(batch)
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                for f in results:
                    out.write(f)
                pbar.update(len(batch))
                batch.clear()

        # 最后一批
        if batch:
            futures = {
                executor.submit(process_frame, f): i
                for i, f in enumerate(batch)
            }
            results = [None] * len(batch)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            for f in results:
                out.write(f)
            pbar.update(len(batch))

    pbar.close()
    cap.release()
    out.release()
    print(f"[INFO] 处理完成，输出: {out_path}")


if __name__ == "__main__":
    main()
