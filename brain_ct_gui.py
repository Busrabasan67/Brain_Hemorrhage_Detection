import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import matplotlib
import json

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Import models and transforms from the new 'src' architecture
from src.dataset import get_transforms
from src.models.custom_cnn_v6 import get_custom_cnn_v6
from src.models.resnet18_se import get_resnet18_se

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Pastel Color Palette ---
PASTEL = {
    "bg": "#F5F0EB",
    "sidebar": "#FFFFFF",
    "card": "#FFFFFF",
    "card_alt": "#F9F5F0",
    "accent": "#6C63FF",
    "accent_light": "#EEEDFF",
    "text": "#2D2D3A",
    "subtext": "#8E8EA0",
    "healthy": "#4CAF8B",
    "healthy_bg": "#E8F8F0",
    "warning": "#E06469",
    "warning_bg": "#FDE8E8",
    "border": "#E8E4DF",
    "btn_hover": "#5A52D5",
}

# English Model Performance Data & Descriptions
MODEL_STATS = {
    "Custom CNN V6 (Lightweight & Fast)": {
        "accuracy": "93.3%",
        "precision": "93.3%",
        "recall": "93.3%",
        "f1": "93.3%",
        "params": "~1.2M Parameters",
        "description": "100% original architecture trained from scratch. Optimized for \nsmall medical datasets using Spatial Dropout for robustness."
    },
    "ResNet18 + SE-Block (Hybrid TL)": {
        "accuracy": "100%",
        "precision": "100%",
        "recall": "100%",
        "f1": "100%",
        "params": "~11.2M Parameters",
        "description": "Hybrid architecture using ResNet18 and Squeeze-and-Excitation \nBlocks. Recalibrates channel-wise features for precise bleeding detection."
    }
}


class ModernBrainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroScan AI ─ Brain CT Analysis Dashboard")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 700)
        self.root.configure(bg=PASTEL["bg"])

        self.current_image_path = None
        self.models = {}

        self._build_ui()
        threading.Thread(target=self._load_models, daemon=True).start()

    def _sync_metrics_from_files(self):
        # Update RESNET
        resnet_json = os.path.join(_SCRIPT_DIR, 'outputs', 'resnet18_se', 'test_metrics.json')
        if os.path.exists(resnet_json):
            try:
                with open(resnet_json, 'r') as f:
                    data = json.load(f)
                MODEL_STATS["ResNet18 + SE-Block (Hybrid TL)"].update(data)
            except Exception:
                pass

        # Update V6
        v6_json = os.path.join(_SCRIPT_DIR, 'outputs', 'custom_cnn_v6', 'test_metrics.json')
        if os.path.exists(v6_json):
            try:
                with open(v6_json, 'r') as f:
                    data = json.load(f)
                MODEL_STATS["Custom CNN V6 (Lightweight & Fast)"].update(data)
            except Exception:
                pass

    def _load_models(self):
        self._set_status("⏳ Loading AI models, please wait...")
        self._sync_metrics_from_files()  # Modelleri yüklerken JSON'dan metrikleri çek
        
        try:
            # Load ResNet18-SE
            resnet = get_resnet18_se().to(DEVICE)
            resnet.load_state_dict(torch.load(os.path.join(_SCRIPT_DIR, 'outputs', 'resnet18_se', 'en_iyi_model.pth'), map_location=DEVICE))
            resnet.eval()
            self.models["ResNet18 + SE-Block (Hybrid TL)"] = resnet

            # Load Custom CNN V6
            v6 = get_custom_cnn_v6().to(DEVICE)
            v6.load_state_dict(torch.load(os.path.join(_SCRIPT_DIR, 'outputs', 'custom_cnn_v6', 'en_iyi_model.pth'), map_location=DEVICE))
            v6.eval()
            self.models["Custom CNN V6 (Lightweight & Fast)"] = v6

            self._set_status("✅ System Ready — Select a model and image to start.")
        except Exception as e:
            self._set_status(f"❌ Model Load Error: {e}")

    def _set_status(self, text):
        self.status_var.set(text)

    def _build_ui(self):
        # --- LEFT SIDEBAR ---
        sidebar = tk.Frame(self.root, bg=PASTEL["sidebar"], width=360)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="🧠", font=("Segoe UI Emoji", 22), bg=PASTEL["sidebar"]).pack(pady=(20, 0))
        tk.Label(sidebar, text="NeuroScan AI", font=("Segoe UI", 17, "bold"), bg=PASTEL["sidebar"],
                 fg=PASTEL["accent"]).pack(pady=(2, 0))
        tk.Label(sidebar, text="Brain CT Diagnostic System", font=("Segoe UI", 9), bg=PASTEL["sidebar"],
                 fg=PASTEL["subtext"]).pack(pady=(2, 16))

        tk.Frame(sidebar, bg=PASTEL["border"], height=1).pack(fill="x", padx=24)

        # Model Selection
        tk.Label(sidebar, text="Model Selection", font=("Segoe UI", 10, "bold"), bg=PASTEL["sidebar"],
                 fg=PASTEL["text"], anchor="w").pack(fill="x", padx=24, pady=(14, 6))

        style = ttk.Style()
        style.theme_use('clam')
        self.model_var = tk.StringVar()
        combo = ttk.Combobox(sidebar, textvariable=self.model_var, values=list(MODEL_STATS.keys()), state="readonly",
                             font=("Segoe UI", 10))
        combo.pack(fill="x", padx=24, ipady=6)
        combo.current(0)
        combo.bind("<<ComboboxSelected>>", self._on_model_change)

        # Upload Button
        tk.Label(sidebar, text="Image Management", font=("Segoe UI", 10, "bold"), bg=PASTEL["sidebar"],
                 fg=PASTEL["text"], anchor="w").pack(fill="x", padx=24, pady=(16, 6))
        self.btn_upload = tk.Button(sidebar, text="📂 Select Image & Analyze", font=("Segoe UI", 11, "bold"),
                                    bg=PASTEL["accent"], fg="#FFFFFF", relief="flat", cursor="hand2", bd=0,
                                    command=self._upload_image)
        self.btn_upload.pack(fill="x", padx=24, pady=(0, 16), ipady=12)

        # Preview Area
        prev_outer = tk.Frame(sidebar, bg=PASTEL["border"], bd=0)
        prev_outer.pack(padx=24, fill="both", expand=True, pady=(0, 8))
        self.preview_inner = tk.Frame(prev_outer, bg=PASTEL["card_alt"])
        self.preview_inner.pack(fill="both", expand=True, padx=1, pady=1)
        self.preview_lbl = tk.Label(self.preview_inner, text="No image selected", bg=PASTEL["card_alt"],
                                    fg=PASTEL["subtext"], font=("Segoe UI", 9))
        self.preview_lbl.pack(fill="both", expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        tk.Label(sidebar, textvariable=self.status_var, bg=PASTEL["sidebar"], fg=PASTEL["subtext"],
                 font=("Segoe UI", 9), wraplength=270, justify="left").pack(side="bottom", anchor="w", padx=24, pady=18)

        # --- RIGHT MAIN PANEL ---
        right = tk.Frame(self.root, bg=PASTEL["bg"])
        right.pack(side="right", fill="both", expand=True, padx=(10, 24), pady=24)

        # Model Info Card
        info_card = tk.Frame(right, bg=PASTEL["card"], highlightbackground=PASTEL["border"], highlightthickness=1)
        info_card.pack(fill="x", pady=(0, 16))
        self.lbl_model_name = tk.Label(info_card, text="", font=("Segoe UI", 16, "bold"), bg=PASTEL["card"],
                                       fg=PASTEL["accent"])
        self.lbl_model_name.pack(anchor="w", padx=24, pady=(18, 2))
        self.lbl_model_desc = tk.Label(info_card, text="", font=("Segoe UI", 10), bg=PASTEL["card"],
                                       fg=PASTEL["subtext"], justify="left")
        self.lbl_model_desc.pack(anchor="w", padx=24, pady=(0, 14))

        # Metrics Row
        metrics_row = tk.Frame(right, bg=PASTEL["bg"])
        metrics_row.pack(fill="x", pady=(0, 16))
        metric_configs = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                          ("f1", "F1 Score")]
        self.metric_labels = {}
        colors = ["#EDE7FF", "#E3F6ED", "#FFF3E0", "#FCE4EC"]
        accent_colors = [PASTEL["accent"], PASTEL["healthy"], "#E09F3E", PASTEL["warning"]]

        for i, (key, title) in enumerate(metric_configs):
            cell = tk.Frame(metrics_row, bg=colors[i], highlightbackground=PASTEL["border"], highlightthickness=1)
            cell.grid(row=0, column=i, padx=6, sticky="nsew")
            metrics_row.grid_columnconfigure(i, weight=1)
            tk.Label(cell, text=title, font=("Segoe UI", 9), bg=colors[i], fg=PASTEL["subtext"]).pack(pady=(14, 0))
            val_lbl = tk.Label(cell, text="–", font=("Segoe UI", 20, "bold"), bg=colors[i], fg=accent_colors[i])
            val_lbl.pack(pady=(2, 14))
            self.metric_labels[key] = val_lbl

        # Bottom Panel (Result & Graph)
        bottom = tk.Frame(right, bg=PASTEL["bg"])
        bottom.pack(fill="both", expand=True)
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)

        # Prediction Card
        self.result_card = tk.Frame(bottom, bg=PASTEL["card"], highlightbackground=PASTEL["border"],
                                    highlightthickness=1)
        self.result_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        tk.Label(self.result_card, text="Live Analysis Result", font=("Segoe UI", 11, "bold"), bg=PASTEL["card"],
                 fg=PASTEL["subtext"]).pack(pady=(24, 8))
        self.lbl_prediction = tk.Label(self.result_card, text="Awaiting Input", font=("Segoe UI", 22, "bold"),
                                       bg=PASTEL["card"], fg=PASTEL["text"])
        self.lbl_prediction.pack(pady=(16, 6))
        self.lbl_confidence = tk.Label(self.result_card, text="Confidence: –", font=("Segoe UI", 12), bg=PASTEL["card"],
                                       fg=PASTEL["subtext"])
        self.lbl_confidence.pack(pady=(0, 8))
        self.lbl_icon = tk.Label(self.result_card, text="🔬", font=("Segoe UI Emoji", 38), bg=PASTEL["card"])
        self.lbl_icon.pack(pady=(8, 24))

        # Probability Graph Card
        graph_card = tk.Frame(bottom, bg=PASTEL["card"], highlightbackground=PASTEL["border"], highlightthickness=1)
        graph_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        tk.Label(graph_card, text="Probability Distribution", font=("Segoe UI", 11, "bold"), bg=PASTEL["card"],
                 fg=PASTEL["subtext"]).pack(pady=(16, 0))
        self.fig, self.ax = plt.subplots(figsize=(5.2, 3.4))
        self.fig.patch.set_facecolor(PASTEL["card"])
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_card)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=14, pady=(4, 14))
        self._draw_empty_graph()

        self._on_model_change()

    def _on_model_change(self, event=None):
        self._sync_metrics_from_files() # JSON'lardan güncel verileri tekrar oku (belki arkada eğitim bitmiştir)
        name = self.model_var.get()
        stats = MODEL_STATS.get(name, {})
        self.lbl_model_name.config(text=name)
        self.lbl_model_desc.config(text=stats.get("description", ""))
        for key in self.metric_labels:
            self.metric_labels[key].config(text=stats.get(key, "–"))
        if self.current_image_path:
            self._analyze()

    def _upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path: return
        self.current_image_path = path
        img = Image.open(path)
        img.thumbnail((320, 320))
        img_tk = ImageTk.PhotoImage(img)
        self.preview_lbl.configure(image=img_tk, text="")
        self.preview_lbl.image = img_tk
        self._analyze()

    def _draw_empty_graph(self):
        self.ax.clear()
        self.ax.set_facecolor(PASTEL["card"])
        self.ax.text(0.5, 0.5, "Waiting for scan...", color=PASTEL["subtext"], ha='center', va='center', fontsize=11)
        for s in self.ax.spines.values(): s.set_visible(False)
        self.ax.set_xticks([]);
        self.ax.set_yticks([])
        self.canvas.draw()

    def _draw_prob_graph(self, prob):
        self.ax.clear()
        self.ax.set_facecolor(PASTEL["card"])
        labels = ['Healthy', 'Hemorrhage']
        values = [1.0 - prob, prob]
        bars = self.ax.barh(labels, values, color=[PASTEL["healthy"], PASTEL["warning"]], height=0.45)
        self.ax.set_xlim(0, 1.15)
        self.ax.invert_yaxis()
        self.ax.tick_params(axis='y', colors=PASTEL["text"], labelsize=11)
        for spine in ['top', 'right', 'left']: self.ax.spines[spine].set_visible(False)
        self.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        for bar in bars:
            w = bar.get_width()
            self.ax.text(w + 0.03, bar.get_y() + bar.get_height() / 2, f"{w * 100:.1f}%", va='center',
                         fontweight='bold')
        self.fig.subplots_adjust(left=0.25)
        self.canvas.draw()

    def _analyze(self):
        name = self.model_var.get()
        if name not in self.models:
            self._set_status("⏳ AI Model not ready, please wait...")
            return
        model = self.models[name]
        try:
            image = cv2.imdecode(np.fromfile(self.current_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
            image_pil = Image.fromarray(cv2.merge([processed] * 3))
            
            _, val_test_transforms = get_transforms(is_v6=False)
            tensor = val_test_transforms(image_pil).unsqueeze(0).to(DEVICE)
        except Exception as e:
            self._set_status(f"❌ Image Processing Error: {e}")
            return

        self._set_status("🔍 Analyzing CT Scan...")
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()

        is_hem = prob > 0.5
        conf = prob if is_hem else (1 - prob)

        # Result Card Styling
        res_text = "Hemorrhage Detected" if is_hem else "Healthy (Normal Scan)"
        res_color = PASTEL["warning"] if is_hem else PASTEL["healthy"]
        res_bg = PASTEL["warning_bg"] if is_hem else PASTEL["healthy_bg"]
        res_icon = "⚠️" if is_hem else "✅"

        self.lbl_prediction.config(text=res_text, fg=res_color)
        self.lbl_confidence.config(text=f"Confidence Score: {conf * 100:.2f}%")
        self.lbl_icon.config(text=res_icon)
        self.result_card.config(bg=res_bg, highlightbackground=res_color, highlightthickness=2)
        for child in self.result_card.winfo_children():
            try:
                child.config(bg=res_bg)
            except:
                pass

        self._draw_prob_graph(prob)
        self._set_status(f"✅ Analysis Complete — Model: {name}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ModernBrainGUI(root)
    root.mainloop()