"""
Traffic Sign AI Recognition Studio
Upgraded Desktop Application powered by CustomTkinter & TensorFlow
"""
import os
import sys
import threading
from typing import Optional, List, Dict, Any
from PIL import Image

import customtkinter as ctk
from tkinter import filedialog, messagebox

from core.model_engine import TrafficSignModel
from core.sign_metadata import SIGN_CLASSES, get_sign_info
from core.batch_processor import BatchProcessor
from core.webcam_engine import WebcamEngine

# Set default CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TrafficSignApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Traffic Sign AI Recognition Studio")
        self.geometry("1100x750")
        self.minsize(980, 680)

        # Base directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.samples_dir = os.path.join(self.base_dir, "samples")
        
        # Determine model path (.keras or .h5)
        keras_path = os.path.join(self.base_dir, "traffic_classifier.keras")
        h5_path = os.path.join(self.base_dir, "traffic_classifier.h5")
        self.model_path = keras_path if os.path.exists(keras_path) else h5_path

        # Core Engines
        self.model_engine = TrafficSignModel()
        self.batch_processor = BatchProcessor(self.model_engine)
        self.webcam_engine = WebcamEngine(self.model_engine)

        # State variables
        self.current_image_path: Optional[str] = None
        self.batch_results: List[Dict[str, Any]] = []

        # Build UI
        self._build_header()
        self._build_tabs()
        self._build_statusbar()

        # Load model asynchronously to prevent UI freeze
        self.after(100, self._async_load_model)

    def _async_load_model(self):
        def _load():
            try:
                self.model_engine.load(self.model_path)
                self.after(0, lambda: self.status_label.configure(
                    text=f"Model: {os.path.basename(self.model_path)} (Loaded & Ready)",
                    text_color="#4EBA6F"
                ))
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(
                    text=f"Model Error: {str(e)}",
                    text_color="#E63946"
                ))
        threading.Thread(target=_load, daemon=True).start()

    def _build_header(self):
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, height=65, fg_color=("gray90", "gray14"))
        self.header_frame.pack(side="top", fill="x")

        title_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=20, pady=12)

        title = ctk.CTkLabel(
            title_box,
            text="Traffic Sign AI Studio",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            title_box,
            text="Deep Learning Vision Classifier & Real-Time Highway Assistant",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60")
        )
        subtitle.pack(anchor="w")

        # Right side controls: Theme switch
        controls_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        controls_box.pack(side="right", padx=20)

        theme_label = ctk.CTkLabel(controls_box, text="Theme:", font=ctk.CTkFont(size=12))
        theme_label.pack(side="left", padx=(0, 6))

        self.theme_option = ctk.CTkOptionMenu(
            controls_box,
            values=["Dark", "Light", "System"],
            width=90,
            command=self._change_theme
        )
        self.theme_option.set("Dark")
        self.theme_option.pack(side="left")

    def _change_theme(self, theme: str):
        ctk.set_appearance_mode(theme)

    def _build_statusbar(self):
        self.statusbar = ctk.CTkFrame(self, height=28, corner_radius=0, fg_color=("gray85", "gray12"))
        self.statusbar.pack(side="bottom", fill="x")

        self.status_label = ctk.CTkLabel(
            self.statusbar,
            text="Loading Model Weights...",
            font=ctk.CTkFont(size=11),
            text_color="#F4A261"
        )
        self.status_label.pack(side="left", padx=15)

        self.version_label = ctk.CTkLabel(
            self.statusbar,
            text="GTSRB 43-Class CNN | v2.0 Upgraded",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60")
        )
        self.version_label.pack(side="right", padx=15)

    def _build_tabs(self):
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(10, 5))

        self.tab_single = self.tabview.add("Single Image Analysis")
        self.tab_webcam = self.tabview.add("Live Webcam Recognition")
        self.tab_batch = self.tabview.add("Batch Analysis & Export")

        self._build_single_image_tab()
        self._build_webcam_tab()
        self._build_batch_tab()

    # -------------------------------------------------------------
    # TAB 1: Single Image Analysis
    # -------------------------------------------------------------
    def _build_single_image_tab(self):
        self.tab_single.grid_columnconfigure(0, weight=4)
        self.tab_single.grid_columnconfigure(1, weight=6)
        self.tab_single.grid_rowconfigure(0, weight=1)

        # Left Card: Image Input & Preview
        left_card = ctk.CTkFrame(self.tab_single, corner_radius=12)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        left_card.grid_rowconfigure(2, weight=1)
        left_card.grid_columnconfigure(0, weight=1)

        btn_bar = ctk.CTkFrame(left_card, fg_color="transparent")
        btn_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 8))

        upload_btn = ctk.CTkButton(
            btn_bar,
            text="Upload Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=36,
            command=self._upload_image
        )
        upload_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Sample dropdown
        sample_files = []
        if os.path.exists(self.samples_dir):
            sample_files = [f for f in os.listdir(self.samples_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        
        self.sample_menu = ctk.CTkOptionMenu(
            btn_bar,
            values=["Load Sample Sign..."] + sample_files if sample_files else ["No samples found"],
            command=self._load_selected_sample,
            height=36
        )
        self.sample_menu.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Image display container
        self.image_display_frame = ctk.CTkFrame(left_card, fg_color=("gray85", "gray17"), corner_radius=10)
        self.image_display_frame.grid(row=2, column=0, sticky="nsew", padx=15, pady=10)

        self.image_preview_label = ctk.CTkLabel(
            self.image_display_frame,
            text="No Image Loaded\n\nClick 'Upload Image' or choose a sample",
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray60")
        )
        self.image_preview_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.image_meta_label = ctk.CTkLabel(
            left_card,
            text="Resolution: - | Format: -",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60")
        )
        self.image_meta_label.grid(row=3, column=0, pady=(0, 10))

        # Right Card: Prediction & Details
        right_card = ctk.CTkFrame(self.tab_single, corner_radius=12)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)
        right_card.grid_columnconfigure(0, weight=1)

        # Hero Banner
        self.hero_frame = ctk.CTkFrame(right_card, corner_radius=10, fg_color=("gray85", "gray18"))
        self.hero_frame.pack(fill="x", padx=15, pady=15)

        hero_top = ctk.CTkFrame(self.hero_frame, fg_color="transparent")
        hero_top.pack(fill="x", padx=15, pady=(12, 4))

        self.category_badge = ctk.CTkLabel(
            hero_top,
            text="READY",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#3B82F6",
            corner_radius=6,
            text_color="white",
            padx=8,
            pady=2
        )
        self.category_badge.pack(side="left")

        self.latency_label = ctk.CTkLabel(
            hero_top,
            text="Latency: -- ms",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60")
        )
        self.latency_label.pack(side="right")

        self.predicted_sign_label = ctk.CTkLabel(
            self.hero_frame,
            text="Awaiting Image Input",
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w"
        )
        self.predicted_sign_label.pack(fill="x", padx=15, pady=(4, 2))

        self.confidence_hero_label = ctk.CTkLabel(
            self.hero_frame,
            text="Confidence: --%",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4EBA6F",
            anchor="w"
        )
        self.confidence_hero_label.pack(fill="x", padx=15, pady=(0, 6))

        # Action Instruction Box
        self.instruction_frame = ctk.CTkFrame(right_card, corner_radius=8, fg_color=("gray90", "gray16"))
        self.instruction_frame.pack(fill="x", padx=15, pady=(0, 15))

        inst_title = ctk.CTkLabel(
            self.instruction_frame,
            text="Traffic Rule & Driver Instruction:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray30", "gray70"),
            anchor="w"
        )
        inst_title.pack(fill="x", padx=12, pady=(8, 2))

        self.instruction_text = ctk.CTkLabel(
            self.instruction_frame,
            text="Upload or select a traffic sign to display corresponding rules and safety precautions.",
            font=ctk.CTkFont(size=12),
            wraplength=480,
            justify="left",
            anchor="w"
        )
        self.instruction_text.pack(fill="x", padx=12, pady=(0, 10))

        # Top 5 Confidence Bars Section
        dist_title = ctk.CTkLabel(
            right_card,
            text="Top-5 Prediction Probability Breakdown",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        dist_title.pack(fill="x", padx=15, pady=(5, 8))

        self.top5_container = ctk.CTkFrame(right_card, fg_color="transparent")
        self.top5_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.top5_bars = []
        for i in range(5):
            row = ctk.CTkFrame(self.top5_container, fg_color="transparent")
            row.pack(fill="x", pady=3)

            lbl = ctk.CTkLabel(row, text=f"#{i+1} --", width=220, anchor="w", font=ctk.CTkFont(size=11))
            lbl.pack(side="left")

            bar = ctk.CTkProgressBar(row, height=12)
            bar.set(0.0)
            bar.pack(side="left", fill="x", expand=True, padx=8)

            pct = ctk.CTkLabel(row, text="0.0%", width=45, anchor="e", font=ctk.CTkFont(size=11, weight="bold"))
            pct.pack(side="right")

            self.top5_bars.append((lbl, bar, pct))

    def _load_selected_sample(self, choice: str):
        if choice in ["Load Sample Sign...", "No samples found"]:
            return
        sample_path = os.path.join(self.samples_dir, choice)
        if os.path.exists(sample_path):
            self._process_single_image(sample_path)

    def _upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.ppm"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self._process_single_image(file_path)

    def _process_single_image(self, file_path: str):
        self.current_image_path = file_path
        try:
            # Display preview
            pil_img = Image.open(file_path)
            orig_w, orig_h = pil_img.size
            self.image_meta_label.configure(text=f"File: {os.path.basename(file_path)} | Size: {orig_w}x{orig_h} | Mode: {pil_img.mode}")

            # Fit into preview box
            preview_img = pil_img.copy()
            preview_img.thumbnail((320, 320), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=preview_img, dark_image=preview_img, size=preview_img.size)

            self.image_preview_label.configure(image=ctk_img, text="")
            self.image_preview_label.image = ctk_img

            # Run prediction
            if not self.model_engine.is_loaded():
                self.predicted_sign_label.configure(text="Model Still Initializing...")
                return

            result = self.model_engine.predict(pil_img, top_k=5)
            best = result["best"]

            # Update Hero
            self.predicted_sign_label.configure(text=best["name"])
            self.confidence_hero_label.configure(text=f"Confidence: {best['confidence_str']}")
            self.category_badge.configure(
                text=best["category"].upper(),
                fg_color=best["badge_color"]
            )
            self.instruction_text.configure(text=best["action_instruction"])
            self.latency_label.configure(text=f"Latency: {result['latency_ms']} ms")

            # Update Top 5
            for i, pred in enumerate(result["top_k"]):
                if i < len(self.top5_bars):
                    lbl, bar, pct = self.top5_bars[i]
                    lbl.configure(text=f"#{i+1} {pred['name']}")
                    bar.set(pred["confidence"] / 100.0)
                    pct.configure(text=pred["confidence_str"])

        except Exception as e:
            messagebox.showerror("Inference Error", f"Failed to analyze image:\n{str(e)}")

    # -------------------------------------------------------------
    # TAB 2: Live Webcam Recognition
    # -------------------------------------------------------------
    def _build_webcam_tab(self):
        self.tab_webcam.grid_columnconfigure(0, weight=7)
        self.tab_webcam.grid_columnconfigure(1, weight=3)
        self.tab_webcam.grid_rowconfigure(0, weight=1)

        # Left Video Stream Box
        left_video = ctk.CTkFrame(self.tab_webcam, corner_radius=12)
        left_video.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        left_video.grid_rowconfigure(1, weight=1)
        left_video.grid_columnconfigure(0, weight=1)

        # Video controls bar
        vid_controls = ctk.CTkFrame(left_video, fg_color="transparent")
        vid_controls.grid(row=0, column=0, sticky="ew", padx=15, pady=10)

        self.webcam_btn = ctk.CTkButton(
            vid_controls,
            text="Start Webcam",
            fg_color="#2A9D8F",
            hover_color="#21867A",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._toggle_webcam
        )
        self.webcam_btn.pack(side="left", padx=(0, 10))

        cam_lbl = ctk.CTkLabel(vid_controls, text="Camera Index:")
        cam_lbl.pack(side="left", padx=(5, 5))

        self.cam_index_menu = ctk.CTkOptionMenu(
            vid_controls,
            values=["0 (Default)", "1", "2"],
            width=110
        )
        self.cam_index_menu.pack(side="left")

        self.fps_label = ctk.CTkLabel(
            vid_controls,
            text="FPS: 0.0",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#F4A261"
        )
        self.fps_label.pack(side="right", padx=10)

        # Live Canvas / Label
        self.video_label = ctk.CTkLabel(
            left_video,
            text="Webcam feed is inactive.\n\nPosition a traffic sign inside the green box once active.",
            font=ctk.CTkFont(size=14),
            text_color=("gray40", "gray60")
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Right Live HUD
        right_hud = ctk.CTkFrame(self.tab_webcam, corner_radius=12)
        right_hud.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)

        hud_title = ctk.CTkLabel(
            right_hud,
            text="Live Recognition HUD",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        hud_title.pack(padx=15, pady=(15, 10), anchor="w")

        self.webcam_sign_card = ctk.CTkFrame(right_hud, corner_radius=10, fg_color=("gray85", "gray18"))
        self.webcam_sign_card.pack(fill="x", padx=15, pady=5)

        self.cam_cat_badge = ctk.CTkLabel(
            self.webcam_sign_card,
            text="STANDBY",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#6C757D",
            corner_radius=5,
            text_color="white",
            padx=6,
            pady=1
        )
        self.cam_cat_badge.pack(padx=10, pady=(10, 2), anchor="w")

        self.cam_sign_name = ctk.CTkLabel(
            self.webcam_sign_card,
            text="No Sign Detected",
            font=ctk.CTkFont(size=16, weight="bold"),
            wraplength=220,
            justify="left",
            anchor="w"
        )
        self.cam_sign_name.pack(fill="x", padx=10, pady=2)

        self.cam_confidence = ctk.CTkLabel(
            self.webcam_sign_card,
            text="Confidence: --%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#4EBA6F",
            anchor="w"
        )
        self.cam_confidence.pack(fill="x", padx=10, pady=(0, 10))

        # Advice box
        self.cam_advice_frame = ctk.CTkFrame(right_hud, corner_radius=8, fg_color=("gray90", "gray16"))
        self.cam_advice_frame.pack(fill="both", expand=True, padx=15, pady=10)

        advice_hdr = ctk.CTkLabel(
            self.cam_advice_frame,
            text="Safety Instruction:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("gray30", "gray70"),
            anchor="w"
        )
        advice_hdr.pack(fill="x", padx=10, pady=(10, 2))

        self.cam_advice_txt = ctk.CTkLabel(
            self.cam_advice_frame,
            text="Ensure proper lighting and center the sign within the green bounding guide.",
            font=ctk.CTkFont(size=12),
            wraplength=220,
            justify="left",
            anchor="nw"
        )
        self.cam_advice_txt.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _toggle_webcam(self):
        if self.webcam_engine.is_running:
            self.webcam_engine.stop()
            self.webcam_btn.configure(text="Start Webcam", fg_color="#2A9D8F", hover_color="#21867A")
            self.video_label.configure(image=None, text="Webcam stopped.")
            self.fps_label.configure(text="FPS: 0.0")
        else:
            cam_str = self.cam_index_menu.get().split()[0]
            try:
                cam_idx = int(cam_str)
            except ValueError:
                cam_idx = 0

            try:
                self.webcam_engine.start(camera_index=cam_idx, on_frame_callback=self._on_webcam_frame)
                self.webcam_btn.configure(text="Stop Webcam", fg_color="#E63946", hover_color="#C12836")
            except Exception as e:
                messagebox.showerror("Webcam Error", f"Unable to access camera: {str(e)}")

    def _on_webcam_frame(self, pil_frame: Image.Image, result: Optional[Dict[str, Any]], fps: float):
        def _update():
            display_frame = pil_frame.copy()
            display_frame.thumbnail((560, 420), Image.Resampling.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=display_frame, dark_image=display_frame, size=display_frame.size)
            self.video_label.configure(image=ctk_img, text="")
            self.video_label.image = ctk_img

            self.fps_label.configure(text=f"FPS: {fps:.1f}")

            if result and result.get("best"):
                best = result["best"]
                self.cam_sign_name.configure(text=best["name"])
                self.cam_confidence.configure(text=f"Confidence: {best['confidence_str']}")
                self.cam_cat_badge.configure(text=best["category"].upper(), fg_color=best["badge_color"])
                self.cam_advice_txt.configure(text=best["action_instruction"])

        self.after(0, _update)

    # -------------------------------------------------------------
    # TAB 3: Batch Analysis & Export
    # -------------------------------------------------------------
    def _build_batch_tab(self):
        self.tab_batch.grid_columnconfigure(0, weight=1)
        self.tab_batch.grid_rowconfigure(2, weight=1)

        # Batch Controls
        ctrl_bar = ctk.CTkFrame(self.tab_batch, corner_radius=10, fg_color="transparent")
        ctrl_bar.grid(row=0, column=0, sticky="ew", pady=(5, 10))

        add_files_btn = ctk.CTkButton(
            ctrl_bar,
            text="Select Image Files",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._batch_select_files
        )
        add_files_btn.pack(side="left", padx=(0, 10))

        add_folder_btn = ctk.CTkButton(
            ctrl_bar,
            text="Select Folder",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self._batch_select_folder
        )
        add_folder_btn.pack(side="left", padx=(0, 10))

        self.export_csv_btn = ctk.CTkButton(
            ctrl_bar,
            text="Export CSV Report",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3B82F6",
            command=self._export_batch_csv,
            state="disabled"
        )
        self.export_csv_btn.pack(side="right")

        # Stats summary frame
        self.batch_stats_frame = ctk.CTkFrame(self.tab_batch, corner_radius=10, fg_color=("gray85", "gray18"))
        self.batch_stats_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.batch_stats_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.stat_total = ctk.CTkLabel(self.batch_stats_frame, text="Total Signs: 0", font=ctk.CTkFont(size=12, weight="bold"))
        self.stat_total.grid(row=0, column=0, pady=10)

        self.stat_avg_conf = ctk.CTkLabel(self.batch_stats_frame, text="Average Confidence: 0.0%", font=ctk.CTkFont(size=12, weight="bold"))
        self.stat_avg_conf.grid(row=0, column=1, pady=10)

        self.batch_progress = ctk.CTkProgressBar(self.batch_stats_frame, height=10)
        self.batch_progress.set(0.0)
        self.batch_progress.grid(row=0, column=2, padx=20, pady=10, sticky="ew")

        # Scrollable table / results list
        self.batch_results_scroll = ctk.CTkScrollableFrame(
            self.tab_batch,
            corner_radius=10,
            label_text="Batch Recognition Results"
        )
        self.batch_results_scroll.grid(row=2, column=0, sticky="nsew")

    def _batch_select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Traffic Sign Images",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.ppm")]
        )
        if files:
            self._run_batch_processing(list(files))

    def _batch_select_folder(self):
        folder = filedialog.askdirectory(title="Select Folder Containing Sign Images")
        if folder:
            valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm")
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
            if not files:
                messagebox.showwarning("No Images Found", f"No supported image files found in {folder}")
                return
            self._run_batch_processing(files)

    def _run_batch_processing(self, file_paths: List[str]):
        if not self.model_engine.is_loaded():
            messagebox.showwarning("Model Not Ready", "Model is still loading. Please wait a moment.")
            return

        for widget in self.batch_results_scroll.winfo_children():
            widget.destroy()

        self.batch_progress.set(0.0)
        self.export_csv_btn.configure(state="disabled")

        def _progress(done: int, total: int):
            self.after(0, lambda: self.batch_progress.set(done / total))

        def _worker():
            results = self.batch_processor.process_files(file_paths, progress_callback=_progress)
            self.batch_results = results
            self.after(0, self._render_batch_results)

        threading.Thread(target=_worker, daemon=True).start()

    def _render_batch_results(self):
        total = len(self.batch_results)
        if total == 0:
            return

        avg_conf = sum(r["Confidence (%)"] for r in self.batch_results) / total
        self.stat_total.configure(text=f"Total Signs: {total}")
        self.stat_avg_conf.configure(text=f"Average Confidence: {avg_conf:.1f}%")
        self.export_csv_btn.configure(state="normal")

        # Table Header
        header = ctk.CTkFrame(self.batch_results_scroll, fg_color=("gray75", "gray22"), height=30)
        header.pack(fill="x", pady=2)
        ctk.CTkLabel(header, text="File Name", width=180, anchor="w", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Predicted Sign", width=220, anchor="w", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Category", width=110, anchor="center", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Confidence", width=90, anchor="center", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)
        ctk.CTkLabel(header, text="Latency", width=80, anchor="center", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=10)

        for res in self.batch_results:
            row = ctk.CTkFrame(self.batch_results_scroll, fg_color=("gray90", "gray17"))
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(row, text=res["Filename"], width=180, anchor="w", font=ctk.CTkFont(size=12)).pack(side="left", padx=10)
            ctk.CTkLabel(row, text=res["Predicted Sign"], width=220, anchor="w", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)

            info = get_sign_info(res["Class ID"]) if res["Class ID"] >= 0 else None
            badge_color = info.badge_color if info else "#6C757D"
            badge = ctk.CTkLabel(row, text=res["Category"], width=110, fg_color=badge_color, text_color="white", corner_radius=6, font=ctk.CTkFont(size=10, weight="bold"))
            badge.pack(side="left", padx=10, pady=4)

            conf_color = "#4EBA6F" if res["Confidence (%)"] > 70 else "#F4A261"
            ctk.CTkLabel(row, text=f"{res['Confidence (%)']:.1f}%", width=90, text_color=conf_color, anchor="center", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=10)
            ctk.CTkLabel(row, text=f"{res['Latency (ms)']} ms", width=80, anchor="center", font=ctk.CTkFont(size=11), text_color=("gray40", "gray60")).pack(side="left", padx=10)

    def _export_batch_csv(self):
        if not self.batch_results:
            return
        out_path = filedialog.asksaveasfilename(
            title="Save Batch Report as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if out_path:
            self.batch_processor.export_csv(self.batch_results, out_path)
            messagebox.showinfo("Export Successful", f"Saved batch report to:\n{out_path}")

    def on_closing(self):
        if self.webcam_engine.is_running:
            self.webcam_engine.stop()
        self.destroy()

def main():
    app = TrafficSignApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
