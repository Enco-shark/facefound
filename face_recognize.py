#!/usr/bin/env python3
import os, sys
import cv2
import hashlib
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

CACHE_FILE = "template_cache.npz" 

THRESHOLD = 0.4# 阈值

def cosine(a, b):
    return float(np.dot(a, b))

def load_templates(app, dirpath, use_cache=True):
    """
    加载模板并缓存 embedding
    app: InsightFace FaceAnalysis 对象
    dirpath: 模板目录
    use_cache: 是否启用缓存
    """
    if use_cache and os.path.exists(CACHE_FILE):
        # 直接加载缓存
        data = np.load(CACHE_FILE, allow_pickle=True)
        names = data["names"].tolist()
        embeddings = data["embeddings"]
        db = list(zip(names, embeddings))
        tqdm.write(f"[INFO] Loaded {len(db)} templates from cache")
        return db

    # 没有缓存或者禁用缓存，重新生成
    db = []
    files = [f for f in os.listdir(dirpath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    tqdm.write(f"[INFO] Loading {len(files)} templates from images...")

    for fn in tqdm(files, desc="Templates", ncols=80):
        path = os.path.join(dirpath, fn)
        name = os.path.splitext(fn)[0]
        img = cv2.imread(path)
        faces = app.get(img)
        if not faces:
            tqdm.write(f"[!] No face detected in template {fn}")
            continue
        emb = faces[0].normed_embedding
        db.append((name, emb))

    # 保存缓存
    names, embeddings = zip(*db)
    embeddings = np.stack(embeddings)
    np.savez(CACHE_FILE, names=names, embeddings=embeddings)
    tqdm.write(f"[INFO] Cached {len(db)} templates to {CACHE_FILE}")

    return db

def recognize_image(app, db, img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    faces = app.get(img)
    if not faces:
        return None, None
    q = faces[0].normed_embedding
    best_name = None
    best_score = -1
    for name, emb in db:
        score = cosine(q, emb)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= THRESHOLD:
        return best_name, best_score
    else:
        return "UNKNOWN", best_score

def main():
    if len(sys.argv) != 3:
        tqdm.write("Usage: face-recognize <template_dir> <query_path_or_dir>")
        sys.exit(1)

    template_dir = sys.argv[1]
    query_path = sys.argv[2]

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0)

    db = load_templates(app, template_dir)
    if not db:
        tqdm.write("No valid templates found.")
        sys.exit(1)

    # 处理单个文件或目录
    query_files = []
    if os.path.isdir(query_path):
        query_files = [os.path.join(query_path, f) for f in os.listdir(query_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    elif os.path.isfile(query_path):
        query_files = [query_path]
    else:
        tqdm.write(f"[!] Query path {query_path} not found.")
        sys.exit(1)

    tqdm.write(f"[INFO] Recognizing {len(query_files)} images...")
    for img_path in tqdm(query_files, desc="Recognizing", ncols=80):
        name, score = recognize_image(app, db, img_path)
        fname = os.path.basename(img_path)
        if name is None:
            tqdm.write(f"{fname}: No face detected")
        else:
            tqdm.write(f"{fname}: match={name}, score={score:.4f}")

if __name__ == "__main__":
    main()
