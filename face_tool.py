#!/usr/bin/env python3
"""
face_tool.py — 人脸识别一键工具（图形界面版）
支持图片/视频分析，自动绘制输出
模型放在 exe 同目录下的 buffalo_l/ 文件夹
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import traceback
import glob

# 错误日志
LOG_FILE = "face_tool_error.txt"
sys.stderr = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = open(LOG_FILE, "a", encoding="utf-8")

try:
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    from insightface.app import FaceAnalysis
    from tqdm import tqdm
except Exception as e:
    with open(LOG_FILE, "w") as f:
        f.write(f"Import error: {e}\n")
        traceback.print_exc(file=f)

# ============ 配置 ============
THRESHOLD = 0.35
MODEL_NAME = "buffalo_l"
OUTPUT_DIR = "output"

# 模型目录：exe 同目录下的 models/buffalo_l/ 文件夹
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# insightface 的 root 参数需要指向包含 models/buffalo_l 目录的父目录
# 所以如果 buffalo_l 在 exe 同目录下的 models/ 里，root 就是 BASE_DIR
# 优先检查 BASE_DIR/models/buffalo_l，再检查 BASE_DIR/../dist/models/buffalo_l（开发时），最后检查 BASE_DIR/buffalo_l
_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, 'models', MODEL_NAME),       # exe 同目录下的 models/buffalo_l
    os.path.join(BASE_DIR, MODEL_NAME),                  # exe 同目录下的 buffalo_l
    os.path.join(os.path.dirname(BASE_DIR), 'dist', 'models', MODEL_NAME),  # 开发时：py/dist/models/buffalo_l
]
MODEL_DIR = None
for _p in _MODEL_CANDIDATES:
    if os.path.isdir(_p):
        MODEL_DIR = _p
        break
if MODEL_DIR is None:
    MODEL_DIR = _MODEL_CANDIDATES[0]  # 默认第一个

# root 需要是 models/ 的父目录
# 如果 MODEL_DIR 是 BASE_DIR/models/buffalo_l，则 root = BASE_DIR
# 如果 MODEL_DIR 是 BASE_DIR/../dist/models/buffalo_l，则 root = BASE_DIR/../dist
if 'dist' in MODEL_DIR and 'dist' not in BASE_DIR:
    MODEL_ROOT = os.path.join(os.path.dirname(BASE_DIR), 'dist')
else:
    MODEL_ROOT = BASE_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

_app = None
_names = None
_embs = None


def check_model():
    """检查模型是否存在"""
    if not os.path.exists(MODEL_DIR):
        return False, f"找不到模型目录: {MODEL_DIR}\n请将 buffalo_l 文件夹放在程序同目录下"

    onnx_files = glob.glob(os.path.join(MODEL_DIR, "*.onnx"))
    if not onnx_files:
        return False, f"模型目录为空: {MODEL_DIR}"

    required = ["det_10g.onnx", "w600k_r50.onnx"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return False, f"缺少模型文件: {', '.join(missing)}"

    return True, f"模型就绪 ({len(onnx_files)} 个文件)"


def init_model():
    global _app
    if _app is None:
        # 尝试 DirectML 提供程序（GPU加速），失败则回退到 CPU
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        try:
            _app = FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT, providers=providers)
        except Exception:
            try:
                _app = FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT, providers=["CPUExecutionProvider"])
            except TypeError:
                _app = FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT)
        _app.prepare(ctx_id=0)


def load_npz(npz_path):
    global _names, _embs
    if not os.path.exists(npz_path):
        return False, f"找不到缓存文件: {npz_path}"
    try:
        d = np.load(npz_path, allow_pickle=True)
        _names = d["names"].tolist()
        _embs = d["embeddings"]
        if _embs.ndim == 1:
            _embs = _embs.reshape(1, -1)
        return True, f"已加载 {len(_names)} 个人脸模板"
    except Exception as e:
        return False, f"加载失败: {e}"


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


def process_image(img_path, output_path, callback=None):
    global _app, _names, _embs
    init_model()
    img = cv2.imread(img_path)
    if img is None:
        return False, f"无法读取图片: {img_path}"

    faces = _app.get(img)
    if callback:
        callback(f"检测到 {len(faces)} 张人脸")

    for face in faces:
        q = face.normed_embedding
        if _embs.shape[0] == 0:
            name = "UNKNOWN"
        else:
            scores = np.dot(_embs, q)
            idx = int(np.argmax(scores))
            name = _names[idx] if scores[idx] >= THRESHOLD else "UNKNOWN"
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2_put_chinese(img, name, (x1, y1 - 35))

    cv2.imwrite(output_path, img)
    return True, f"已保存: {output_path}"


def process_video(video_path, output_path, callback=None):
    global _app, _names, _embs
    init_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"无法打开视频: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False, f"无法创建输出视频: {output_path}"

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        out.write(frame)
        frame_count += 1
        if callback and frame_count % 30 == 0:
            callback(f"处理中... {frame_count}/{total} 帧 ({frame_count*100//total}%)")

    cap.release()
    out.release()
    return True, f"视频处理完成! 共 {frame_count} 帧\n输出: {output_path}"


# ============ 图形界面 ============
class FaceToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸识别工具 v1.0")
        self.root.geometry("700x600")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("vista")

        self.npz_path = tk.StringVar(value="template_cache.npz")
        self.input_path = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="就绪")
        self.is_processing = False

        self._build_ui()
        self._auto_load_npz()
        self._check_model()

    def _build_ui(self):
        title = tk.Label(self.root, text="人脸识别工具", font=("微软雅黑", 18, "bold"), fg="#2c3e50")
        title.pack(pady=15)

        # ===== 模型状态 =====
        model_frame = tk.LabelFrame(self.root, text="模型状态", font=("微软雅黑", 10), padx=10, pady=10)
        model_frame.pack(fill="x", padx=20, pady=5)

        self.model_status = tk.Label(model_frame, text="检查中...", font=("微软雅黑", 9), anchor="w", justify="left")
        self.model_status.pack(fill="x")

        # ===== 模板文件选择 =====
        frame1 = tk.LabelFrame(self.root, text="① 选择模板文件 (template_cache.npz)", font=("微软雅黑", 10), padx=10, pady=10)
        frame1.pack(fill="x", padx=20, pady=5)

        row1 = tk.Frame(frame1)
        row1.pack(fill="x")
        tk.Entry(row1, textvariable=self.npz_path, font=("微软雅黑", 9)).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(row1, text="浏览...", command=self._browse_npz, font=("微软雅黑", 9)).pack(side="right")

        self.npz_status = tk.Label(frame1, text="未加载", fg="red", font=("微软雅黑", 9))
        self.npz_status.pack(anchor="w", pady=(5, 0))

        # ===== 输入文件选择 =====
        frame2 = tk.LabelFrame(self.root, text="② 选择输入文件 (图片或视频)", font=("微软雅黑", 10), padx=10, pady=10)
        frame2.pack(fill="x", padx=20, pady=5)

        row2 = tk.Frame(frame2)
        row2.pack(fill="x")
        tk.Entry(row2, textvariable=self.input_path, font=("微软雅黑", 9)).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(row2, text="选择文件...", command=self._browse_input, font=("微软雅黑", 9)).pack(side="right")

        # ===== 操作按钮 =====
        frame3 = tk.Frame(self.root)
        frame3.pack(pady=15)

        self.btn_start = tk.Button(
            frame3, text="▶ 开始分析", command=self._start_processing,
            font=("微软雅黑", 12, "bold"), bg="#27ae60", fg="white",
            padx=30, pady=8, cursor="hand2"
        )
        self.btn_start.pack()

        # ===== 进度条 =====
        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=500)
        self.progress.pack(pady=5)

        # ===== 状态/日志 =====
        frame4 = tk.LabelFrame(self.root, text="运行日志", font=("微软雅黑", 10), padx=10, pady=10)
        frame4.pack(fill="both", expand=True, padx=20, pady=5)

        self.log_text = tk.Text(frame4, height=10, font=("Consolas", 9), wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(self.log_text)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.log_text.yview)

        status_bar = tk.Label(self.root, textvariable=self.status_text, bd=1, relief="sunken", anchor="w", font=("微软雅黑", 9))
        status_bar.pack(side="bottom", fill="x", expand=True)

        # 右下角 Powered by Enco
        powered = tk.Label(self.root, text="Powered by Enco", font=("微软雅黑", 8), fg="#999999", anchor="e")
        powered.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-5)

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.root.update()

    def _check_model(self):
        ok, msg = check_model()
        lines = [
            f"📁 模型目录: {MODEL_DIR}",
            f"{'✅' if ok else '❌'} {msg}",
        ]
        if not ok:
            lines.append(f"💡 请将 buffalo_l 文件夹放在 exe 同目录下")
        self.model_status.config(text="\n".join(lines))

    def _browse_npz(self):
        path = filedialog.askopenfilename(
            title="选择 template_cache.npz 文件",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if path:
            self.npz_path.set(path)
            self._auto_load_npz()

    def _auto_load_npz(self):
        path = self.npz_path.get()
        if os.path.exists(path):
            ok, msg = load_npz(path)
            if ok:
                self.npz_status.config(text=f"✅ {msg}", fg="green")
            else:
                self.npz_status.config(text=f"❌ {msg}", fg="red")
        else:
            self.npz_status.config(text="⚠ 文件不存在，请选择", fg="orange")

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="选择图片或视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        if path:
            self.input_path.set(path)

    def _start_processing(self):
        if self.is_processing:
            return

        # 检查模型
        model_ok, model_msg = check_model()
        if not model_ok:
            messagebox.showerror("错误", f"模型未就绪!\n{model_msg}\n\n请将 buffalo_l 文件夹放在 exe 同目录下")
            return

        if not os.path.exists(self.npz_path.get()):
            messagebox.showerror("错误", "请先选择有效的 template_cache.npz 文件")
            return

        input_path = self.input_path.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("错误", "请选择有效的输入文件")
            return

        ok, msg = load_npz(self.npz_path.get())
        if not ok:
            messagebox.showerror("错误", msg)
            return

        self.is_processing = True
        self.btn_start.config(state="disabled", text="⏳ 处理中...")
        self.progress.start()
        self._log("=" * 50)
        self._log(f"开始处理: {os.path.basename(input_path)}")

        thread = threading.Thread(target=self._process_thread, args=(input_path,))
        thread.daemon = True
        thread.start()

    def _process_thread(self, input_path):
        try:
            ext = os.path.splitext(input_path)[1].lower()
            video_exts = [".mp4", ".avi", ".mov", ".mkv"]
            image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_output.mp4")

            def callback(msg):
                self.root.after(0, lambda: self._log(f"  {msg}"))

            if ext in video_exts:
                self.root.after(0, lambda: self._log("  [视频模式] 正在分析每一帧..."))
                ok, msg = process_video(input_path, output_path, callback)
            elif ext in image_exts:
                output_path = os.path.join(OUTPUT_DIR, f"{base_name}_output.jpg")
                self.root.after(0, lambda: self._log("  [图片模式] 正在分析..."))
                ok, msg = process_image(input_path, output_path, callback)
            else:
                ok, msg = False, f"不支持的文件格式: {ext}"

            self.root.after(0, lambda: self._log(f"  {'✅' if ok else '❌'} {msg}"))
            self.root.after(0, lambda: self._log("=" * 50))

            if ok:
                self.root.after(0, lambda: messagebox.showinfo("完成", msg))

        except Exception as e:
            self.root.after(0, lambda: self._log(f"  ❌ 错误: {e}"))
            traceback.print_exc()
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.btn_start.config(state="normal", text="▶ 开始分析"))
            self.root.after(0, lambda: self.progress.stop())


def main():
    root = tk.Tk()
    app = FaceToolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
