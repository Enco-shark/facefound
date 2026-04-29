# 人脸识别工具 (Face Recognition Tool)

基于 [InsightFace](https://github.com/deepinsight/insightface) 的人脸识别一键工具，支持图片和视频分析，提供图形界面和命令行两种使用方式。

## 功能特点

- ✅ **图片人脸识别** — 自动检测并标注人脸姓名
- ✅ **视频人脸识别** — 逐帧分析视频，输出带标注的结果视频
- ✅ **图形界面** — 简单易用的 Tkinter GUI
- ✅ **命令行支持** — 适合批量处理和自动化脚本
- ✅ **GPU 加速** — 自动尝试 DirectML 加速，回退到 CPU
- ✅ **中文标注** — 支持在图片上绘制中文姓名

## 项目文件说明

| 文件 | 说明 |
|------|------|
| `face_tool.py` | **主程序** — 图形界面版，一键图片/视频分析 |
| `make_cache_npz.py` | **模板生成工具** — 扫描照片文件夹，生成人脸特征缓存 |
| `only_npz.py` | **命令行版** — 纯命令行视频处理，支持多线程加速 |
| `人脸识别工具.spec` | PyInstaller 打包配置文件 |
| `template_cache.npz` | 人脸特征缓存文件（由 make_cache_npz.py 生成） |
| `buffalo_l/` | InsightFace 模型文件夹（需自行下载） |
| `insightface_pkg/` | 自定义 InsightFace 包（修复了部分兼容性问题） |
| `data/` | 示例人物照片文件夹 |
| `output/` | 处理结果输出目录 |

## 环境要求

- Python 3.8+
- Windows / Linux / macOS

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `insightface` — 人脸检测与识别
- `onnxruntime` — ONNX 模型推理引擎
- `opencv-python` — 图像/视频处理
- `numpy` — 数值计算
- `Pillow` — 图片绘制（中文支持）
- `tqdm` — 进度条

## 快速开始

### 1. 下载模型

从 InsightFace 官方 Release 下载 `buffalo_l` 模型：

```bash
# 方式一：自动下载（首次运行会自动下载）
python face_tool.py

# 方式二：手动下载
# 下载地址: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
# 解压后放在程序同目录下的 models/buffalo_l/ 文件夹
```

模型目录结构：
```
models/
└── buffalo_l/
    ├── det_10g.onnx        # 人脸检测模型
    ├── w600k_r50.onnx      # 人脸识别模型
    ├── 1k3d68.onnx         # 3D 关键点模型
    ├── 2d106det.onnx       # 2D 关键点模型
    └── genderage.onnx      # 性别年龄模型
```

### 2. 生成人脸特征缓存

准备一批人物照片（每张照片以人物姓名命名），然后运行：

```bash
# 扫描当前目录所有图片，生成 cache.npz
python make_cache_npz.py

# 扫描指定文件夹
python make_cache_npz.py D:\照片\人物

# 递归扫描子文件夹
python make_cache_npz.py . --recursive

# 指定输出文件名
python make_cache_npz.py . -o template_cache.npz
```

照片命名规则：
- `张三.jpg` → 识别结果为 "张三"
- `李四.png` → 识别结果为 "李四"
- 支持格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### 3. 运行人脸识别

#### 图形界面版（推荐）

```bash
python face_tool.py
```

操作步骤：
1. 点击 "浏览..." 选择 `template_cache.npz` 文件
2. 点击 "选择文件..." 选择要分析的图片或视频
3. 点击 "▶ 开始分析" 开始处理
4. 处理结果保存在 `output/` 目录

#### 命令行版

```bash
# 处理视频
python only_npz.py 视频文件.mp4 -c template_cache.npz

# 指定输出路径
python only_npz.py 视频文件.mp4 -c template_cache.npz -o result.mp4
```

## 打包为独立 EXE

使用 PyInstaller 打包为 Windows 可执行文件，无需 Python 环境即可运行。

### 安装 PyInstaller

```bash
pip install pyinstaller
```

### 打包命令

```bash
pyinstaller 人脸识别工具.spec --clean --noconfirm
```

打包完成后，exe 文件在 `dist/人脸识别工具.exe`。

### 分发说明

将以下文件一起分发：

```
dist/
├── 人脸识别工具.exe          # 主程序
└── models/
    └── buffalo_l/            # 模型文件夹（需自行下载）
        ├── det_10g.onnx
        ├── w600k_r50.onnx
        ├── 1k3d68.onnx
        ├── 2d106det.onnx
        └── genderage.onnx
```

用户只需将 `template_cache.npz` 放在 exe 同目录下即可使用。

### 打包注意事项

1. **模型文件不打包进 exe** — 模型文件较大（~200MB），需单独分发
2. **insightface 自定义包** — 项目使用 `insightface_pkg/` 中的自定义版本，已修复部分兼容性问题
3. **meanshape_68.pkl** — 打包时自动包含此数据文件，用于 3D 姿态估计
4. **首次启动较慢** — 模型加载需要几秒钟，请耐心等待

## 配置说明

在 `face_tool.py` 顶部可修改以下配置：

```python
THRESHOLD = 0.35      # 人脸识别相似度阈值（越低越严格）
MODEL_NAME = "buffalo_l"  # 模型名称
OUTPUT_DIR = "output"     # 输出目录
```

## 常见问题

### Q: 启动后提示 "找不到模型目录"
A: 确保 `models/buffalo_l/` 文件夹存在且包含 `.onnx` 模型文件。

### Q: 识别准确率不高
A: 可以调低 `THRESHOLD` 值（如 0.3），或确保模板照片清晰、正面、光照良好。

### Q: 视频处理很慢
A: 默认使用 CPU 推理。如果有 NVIDIA GPU，可以安装 `onnxruntime-gpu` 获得加速。

### Q: 打包后 exe 体积很大
A: 这是正常的，因为包含了 Python 运行时和所有依赖库（约 140MB）。

## 技术架构

```
用户输入 (图片/视频)
    ↓
OpenCV 读取帧
    ↓
InsightFace (buffalo_l) 人脸检测
    ↓
特征提取 (512维向量)
    ↓
与模板库 (npz) 进行余弦相似度匹配
    ↓
OpenCV 绘制标注框和姓名
    ↓
输出结果 (图片/视频)
```

## 开源协议

本项目基于 MIT 协议开源。

## 致谢

- [InsightFace](https://github.com/deepinsight/insightface) — 提供人脸检测与识别模型
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 模型推理引擎
- [PyInstaller](https://github.com/pyinstaller/pyinstaller) — Python 打包工具

---

**Powered by Enco**
