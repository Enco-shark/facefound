#!/usr/bin/env python3
"""
make_cache_npz.py — 扫描整个文件夹的人物照片，生成 cache.npz

用法:
    python make_cache_npz.py                          # 扫描当前目录所有图片
    python make_cache_npz.py D:\照片\人物              # 扫描指定文件夹
    python make_cache_npz.py . -o my_cache.npz         # 指定输出文件名
    python make_cache_npz.py . --recursive             # 递归扫描子文件夹

输出:
    cache.npz  — 包含 names (人名列表) 和 embeddings (512维特征向量)
"""

import os
import sys
import argparse
import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['ORT_LOGGING_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ============ 配置 ============
MODEL_NAME = "buffalo_l"
PROVIDERS = ["CPUExecutionProvider"]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

_app = None


def init_model():
    """初始化人脸识别模型"""
    global _app
    if _app is None:
        print("[INFO] 正在初始化人脸识别模型 (buffalo_l)...")
        _app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
        _app.prepare(ctx_id=0)
        print("[INFO] 模型初始化完成")


def get_image_files(root_dir, recursive=False):
    """获取目录下所有支持的图片文件"""
    image_files = []
    pattern = "**/*" if recursive else "*"

    for ext in SUPPORTED_EXTS:
        found = glob.glob(os.path.join(root_dir, pattern + ext),
                          recursive=recursive)
        image_files.extend(found)

    # 去重并排序
    image_files = sorted(set(image_files))
    return image_files


def extract_face_embedding(img_path):
    """
    从图片中提取人脸特征向量
    返回: (person_name, embedding) 或 (person_name, None)
    """
    global _app

    # 用文件名（不含扩展名）作为人名
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ 无法读取图片: {img_path}")
            return base_name, None

        faces = _app.get(img)
        if len(faces) == 0:
            print(f"  ⚠ 未检测到人脸: {img_path}")
            return base_name, None

        # 取第一张人脸的特征
        embedding = faces[0].normed_embedding
        return base_name, embedding

    except Exception as e:
        print(f"  ❌ 处理失败: {img_path} — {e}")
        return base_name, None


def main():
    parser = argparse.ArgumentParser(
        description="扫描文件夹中的人物照片，生成 cache.npz"
    )
    parser.add_argument(
        "input_dir", nargs="?", default=".",
        help="要扫描的文件夹路径（默认: 当前目录）"
    )
    parser.add_argument(
        "-o", "--output", default="cache.npz",
        help="输出 npz 文件路径（默认: cache.npz）"
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="递归扫描子文件夹"
    )
    parser.add_argument(
        "--min-faces", type=int, default=1,
        help="每张图片最少检测到的人脸数才纳入（默认: 1）"
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 路径不存在或不是文件夹: {input_dir}")
        sys.exit(1)

    print(f"[INFO] 扫描文件夹: {input_dir}")
    print(f"[INFO] 递归扫描: {'是' if args.recursive else '否'}")
    print(f"[INFO] 输出文件: {os.path.abspath(args.output)}")

    # 1. 获取所有图片文件
    image_files = get_image_files(input_dir, recursive=args.recursive)
    if not image_files:
        print(f"[ERROR] 在 {input_dir} 中未找到任何图片文件")
        print(f"[INFO] 支持的格式: {', '.join(sorted(SUPPORTED_EXTS))}")
        sys.exit(1)

    print(f"[INFO] 找到 {len(image_files)} 张图片")

    # 2. 初始化模型
    init_model()

    # 3. 逐张提取人脸特征
    names = []
    embeddings = []
    skipped = 0

    print("[INFO] 开始提取人脸特征...")
    for img_path in tqdm(image_files, desc="Processing", unit="img"):
        person_name, emb = extract_face_embedding(img_path)
        if emb is not None:
            names.append(person_name)
            embeddings.append(emb)
        else:
            skipped += 1

    # 4. 保存为 npz
    if len(names) == 0:
        print("[ERROR] 没有成功提取到任何人脸特征，无法生成 cache.npz")
        sys.exit(1)

    embeddings_arr = np.array(embeddings, dtype=np.float32)
    names_arr = np.array(names, dtype=object)

    np.savez_compressed(args.output, names=names_arr, embeddings=embeddings_arr)

    print("=" * 50)
    print(f"✅ 处理完成!")
    print(f"   成功提取: {len(names)} 个人脸")
    print(f"   跳过:     {skipped} 张图片")
    print(f"   特征维度: {embeddings_arr.shape}")
    print(f"   输出文件: {os.path.abspath(args.output)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
