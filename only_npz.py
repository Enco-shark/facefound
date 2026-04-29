#!/usr/bin/env python3
"""
only_npz.py — 纯 npz 缓存版本，带错误日志
直接加载 template_cache.npz 进行人脸识别
"""
import os
import sys
import traceback

# 把错误输出到文件
log_file = "error_log.txt"
sys.stderr = open(log_file, "w", encoding="utf-8")
sys.stdout = open(log_file, "a", encoding="utf-8")

try:
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

    _app = None
    _names = None
    _embs = None

    def init_worker():
        global _app
        if _app is None:
            print("[INFO] 正在初始化人脸识别模型...")
            _app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
            _app.prepare(ctx_id=0)
            print("[INFO] 模型初始化完成")

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
        if not os.path.exists(npz_path):
            print(f"[ERROR] 找不到缓存文件: {npz_path}")
            print("[INFO] 请先用其他版本生成 template_cache.npz")
            sys.exit(1)
        d = np.load(npz_path, allow_pickle=True)
        names = d["names"].tolist()
        embs = d["embeddings"]
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        print(f"[INFO] 从 {npz_path} 加载了 {len(names)} 个人脸模板")
        return names, embs

    def process_frame(frame):
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
            description="人脸识别视频处理（纯 npz 缓存版）"
        )
        ap.add_argument("video_path", help="输入视频文件路径")
        ap.add_argument("-c", "--cache", default="template_cache.npz",
                        help="template_cache.npz 文件路径")
        ap.add_argument("-o", "--output", default=None,
                        help="输出视频路径")
        args = ap.parse_args()

        print(f"[INFO] 视频文件: {args.video_path}")
        print(f"[INFO] 缓存文件: {args.cache}")

        _names, _embs = load_templates_from_npz(args.cache)
        init_worker()

        print("[INFO] 正在打开视频...")
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开视频文件: {args.video_path}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] 视频: {w}x{h}, {fps}fps, {total}帧")

        if args.output:
            out_path = args.output
        else:
            out_path = os.path.join(OUTPUT_DIR, os.path.basename(args.video_path))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not out.isOpened():
            print(f"[ERROR] 无法创建输出视频: {out_path}")
            sys.exit(1)

        print("[INFO] 开始处理视频...")
        pbar = tqdm(total=total, desc="Processing")

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
        print(f"[INFO] 处理完成! 输出: {out_path}")

    if __name__ == "__main__":
        main()

except Exception as e:
    print("=" * 50)
    print(f"[FATAL ERROR] {str(e)}")
    print("=" * 50)
    traceback.print_exc()
    sys.exit(1)
