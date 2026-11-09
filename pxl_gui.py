#!/usr/bin/env python3
"""PXL Studio — polished Tkinter desktop application."""
from __future__ import annotations

import multiprocessing
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk, ImageOps

from pxl import compress, decompress, inspect, cuda_status
from pxl7_hdr import probe_high_precision
from codec_compare import compare_codecs, export_csv
from pxl_batch import execute_queued_job
from pxl_native import backend_info


BG = "#0b1020"
SURFACE = "#121a2b"
SURFACE_2 = "#182338"
BORDER = "#293750"
TEXT = "#e7eefb"
MUTED = "#91a2bb"
ACCENT = "#55c5e9"
ACCENT_DARK = "#123c54"
GREEN = "#58d6a0"
ORANGE = "#f6b96b"
RED = "#f17e83"


class PXLStudio:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PXL Studio 8.0.2 | Adaptive Lossless + HDR Image Workbench")
        self.root.geometry("1180x850")
        self.root.minsize(980, 720)
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.preview_photo = None
        self.compare_results: list[dict] = []

        self.mode = tk.StringVar(value="lossless")
        self.quality = tk.IntVar(value=88)
        self.tile_size = tk.StringVar(value="256")
        self.adaptive_tiles = tk.BooleanVar(value=True)
        self.checkpoint_interval = tk.StringVar(value="8")
        self.effort = tk.StringVar(value="balanced")
        self.backend_choice = tk.StringVar(value="Auto")
        self.worker_choice = tk.StringVar(value="Auto")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.status = tk.StringVar(value="Ready · choose an image to begin")
        self.compare_mode = tk.StringVar(value="lossless")
        self.compare_quality = tk.IntVar(value=88)
        self.compare_effort = tk.StringVar(value="balanced")
        self.compare_source = tk.StringVar()

        # Multi-file job queue. Each queued file runs in its own spawned process;
        # the queue limits simultaneous processes to avoid nested-pool overload.
        self.jobs: dict[str, dict] = {}
        self.job_order: list[str] = []
        self.job_rows: dict[str, dict] = {}
        self.selected_job: str | None = None
        self.job_sequence = 0
        self.queue_executor = None
        self.queue_manager = None
        self.queue_progress_queue = None
        self.queue_futures: dict[str, object] = {}
        self.queue_running = False
        self.queue_paused = False
        self.queue_slots = 1
        self.queue_concurrency = tk.StringVar(value=str(min(2, max(1, os.cpu_count() or 1))))
        self.queue_output_dir = tk.StringVar()
        self.queue_overwrite = tk.BooleanVar(value=False)
        self.queue_summary = tk.StringVar(value="No files in queue")
        self.queue_total_label = tk.StringVar(value="0% · 0 of 0 jobs complete")

        self._style()
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(90, self._poll_events)

    # ---------- styling and shell ----------
    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg=BG)
        style.configure("TFrame", background=BG)
        style.configure("Surface.TFrame", background=SURFACE)
        style.configure("Surface2.TFrame", background=SURFACE_2)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Surface.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("SurfaceMuted.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=BG, foreground="#f5f8ff", font=("Segoe UI Semibold", 24))
        style.configure("Section.TLabel", background=SURFACE, foreground="#d9e9fa", font=("Segoe UI Semibold", 10))
        style.configure("Eyebrow.TLabel", background=SURFACE, foreground=ACCENT, font=("Segoe UI Semibold", 8))
        style.configure("TButton", font=("Segoe UI Semibold", 9), padding=(13, 9), background=SURFACE_2, foreground=TEXT, borderwidth=0)
        style.map("TButton", background=[("active", "#263750"), ("disabled", "#192337")], foreground=[("disabled", "#61718a")])
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=(16, 10), background=ACCENT, foreground="#071521")
        style.map("Accent.TButton", background=[("active", "#8be0f5"), ("disabled", "#37596a")], foreground=[("disabled", "#9aaaba")])
        style.configure("TEntry", padding=8, fieldbackground="#0d1627", foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER)
        style.configure("TCombobox", padding=6, fieldbackground="#0d1627", background=SURFACE_2, foreground=TEXT, arrowcolor=ACCENT)
        style.map("TCombobox", fieldbackground=[("readonly", "#0d1627")], foreground=[("readonly", TEXT)])
        style.configure("TRadiobutton", background=SURFACE, foreground=TEXT, font=("Segoe UI", 9), padding=(2, 5))
        style.map("TRadiobutton", background=[("active", SURFACE)], foreground=[("active", "#ffffff")])
        style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, padding=(18, 11), font=("Segoe UI Semibold", 9), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", SURFACE_2), ("active", "#202e45")], foreground=[("selected", TEXT), ("active", TEXT)])
        style.configure("Treeview", background="#0d1627", fieldbackground="#0d1627", foreground=TEXT,
                        rowheight=29, borderwidth=0, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=SURFACE_2, foreground="#b9cce4", relief="flat",
                        font=("Segoe UI Semibold", 9), padding=(7, 8))
        style.map("Treeview", background=[("selected", "#214b66")], foreground=[("selected", "#ffffff")])
        style.configure("Horizontal.TProgressbar", troughcolor="#26344b", background=ACCENT,
                        bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)
        style.configure("TScale", background=SURFACE, troughcolor="#2b3b52")

    def _build(self):
        shell = ttk.Frame(self.root, padding=(24, 18, 24, 14))
        shell.pack(fill="both", expand=True)

        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 17))
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left", anchor="w")
        logo = tk.Canvas(brand, width=45, height=45, bg=BG, highlightthickness=0)
        logo.pack(side="left", padx=(0, 12))
        logo.create_rectangle(2, 2, 43, 43, outline=ACCENT, width=2, fill="#10263b")
        logo.create_text(22, 22, text="P", fill=ACCENT, font=("Segoe UI Semibold", 25))
        titlebox = tk.Frame(brand, bg=BG)
        titlebox.pack(side="left")
        tk.Label(titlebox, text="PXL Studio", bg=BG, fg="#f5f8ff", font=("Segoe UI Semibold", 23)).pack(anchor="w")
        tk.Label(titlebox, text="IMAGE COMPRESSION WORKBENCH  /  PXL V4 · BATCH + MULTICORE + CUDA", bg=BG, fg=MUTED,
                 font=("Segoe UI Semibold", 8)).pack(anchor="w", pady=(0, 2))
        right = tk.Frame(header, bg=BG)
        right.pack(side="right", anchor="e", pady=(7, 0))
        tk.Label(right, text="PYTHON EDITION", bg=BG, fg=ACCENT, font=("Segoe UI Semibold", 9)).pack(anchor="e")
        tk.Label(right, text="Encode · Decode · Compare · Accelerate", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="e", pady=(3, 0))

        self.notebook = ttk.Notebook(shell)
        self.notebook.pack(fill="both", expand=True)
        self.encode_tab = ttk.Frame(self.notebook, padding=16)
        self.compare_tab = ttk.Frame(self.notebook, padding=16)
        self.queue_tab = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.encode_tab, text="  ENCODE / DECODE  ")
        self.notebook.add(self.queue_tab, text="  BATCH QUEUE  ")
        self.notebook.add(self.compare_tab, text="  CODEC LAB  ")
        self._build_encode_tab()
        self._build_queue_tab()
        self._build_compare_tab()

        footer = ttk.Frame(shell)
        footer.pack(fill="x", pady=(12, 0))
        self.footer_status = ttk.Label(footer, textvariable=self.status, style="Muted.TLabel")
        self.footer_status.pack(side="left")
        ttk.Label(footer, text="PXL is experimental · inspect results on your own image set", style="Muted.TLabel").pack(side="right")

    # ---------- reusable cards ----------
    def _card(self, parent, eyebrow: str, title: str | None = None):
        frame = ttk.Frame(parent, style="Surface.TFrame", padding=15)
        head = ttk.Frame(frame, style="Surface.TFrame")
        head.pack(fill="x", pady=(0, 12))
        ttk.Label(head, text=eyebrow.upper(), style="Eyebrow.TLabel").pack(anchor="w")
        if title:
            ttk.Label(head, text=title, style="Section.TLabel").pack(anchor="w", pady=(3, 0))
        return frame

    # ---------- encode/decode tab ----------
    def _build_encode_tab(self):
        top = ttk.Frame(self.encode_tab)
        top.pack(fill="both", expand=True)
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 9))
        right = ttk.Frame(top)
        right.pack(side="left", fill="both", expand=True, padx=(9, 0))

        files = self._card(left, "01 · Source and destination", "Choose what to process")
        files.pack(fill="x", pady=(0, 12))
        ttk.Label(files, text="SOURCE IMAGE / PXL FILE", style="SurfaceMuted.TLabel").pack(anchor="w")
        source_row = ttk.Frame(files, style="Surface.TFrame")
        source_row.pack(fill="x", pady=(6, 12))
        ttk.Entry(source_row, textvariable=self.input_path).pack(side="left", fill="x", expand=True)
        ttk.Button(source_row, text="Browse", command=self._choose_input).pack(side="left", padx=(8, 0))
        ttk.Label(files, text="OUTPUT FILE", style="SurfaceMuted.TLabel").pack(anchor="w")
        output_row = ttk.Frame(files, style="Surface.TFrame")
        output_row.pack(fill="x", pady=(6, 0))
        ttk.Entry(output_row, textvariable=self.output_path).pack(side="left", fill="x", expand=True)
        ttk.Button(output_row, text="Save as", command=self._choose_output).pack(side="left", padx=(8, 0))

        settings = self._card(left, "02 · Compression setup", "Select a profile")
        settings.pack(fill="both", expand=True, pady=(0, 12))
        profiles = ttk.Frame(settings, style="Surface2.TFrame", padding=4)
        profiles.pack(fill="x", pady=(0, 12))
        ttk.Radiobutton(profiles, text="Lossless  ·  exact pixels", variable=self.mode, value="lossless",
                        command=self._mode_changed).pack(side="left", padx=(8, 10), pady=7)
        ttk.Radiobutton(profiles, text="Lossy  ·  smaller files", variable=self.mode, value="lossy",
                        command=self._mode_changed).pack(side="left", padx=10, pady=7)

        qrow = ttk.Frame(settings, style="Surface.TFrame")
        qrow.pack(fill="x", pady=(2, 10))
        self.quality_label = ttk.Label(qrow, text="QUALITY · not used", style="Surface.TLabel")
        self.quality_label.pack(side="left")
        self.quality_value = ttk.Label(qrow, text="88", style="Surface.TLabel")
        self.quality_value.pack(side="right")
        self.quality_scale = ttk.Scale(settings, from_=1, to=100, orient="horizontal", command=self._quality_changed)
        self.quality_scale.set(88)
        self.quality_scale.pack(fill="x", pady=(0, 13))

        options = ttk.Frame(settings, style="Surface.TFrame")
        options.pack(fill="x")
        tilebox = ttk.Frame(options, style="Surface.TFrame")
        tilebox.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Label(tilebox, text="TILE SIZE", style="SurfaceMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.tile_combo = ttk.Combobox(tilebox, textvariable=self.tile_size,
                                       values=("64", "128", "256", "512", "1024"), state="readonly", width=12)
        self.tile_combo.pack(anchor="w")
        effortbox = ttk.Frame(options, style="Surface.TFrame")
        effortbox.pack(side="left", fill="x", expand=True)
        ttk.Label(effortbox, text="COMPRESSION EFFORT", style="SurfaceMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.effort_combo = ttk.Combobox(effortbox, textvariable=self.effort,
                                         values=("fast", "balanced", "maximum"), state="readonly", width=14)
        self.effort_combo.pack(anchor="w")

        acceleration = ttk.Frame(settings, style="Surface.TFrame")
        acceleration.pack(fill="x", pady=(14, 0))
        backend_box = ttk.Frame(acceleration, style="Surface.TFrame")
        backend_box.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Label(backend_box, text="PROCESSING BACKEND", style="SurfaceMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.backend_combo = ttk.Combobox(
            backend_box, textvariable=self.backend_choice,
            values=("Auto", "CPU multi-core", "CPU single-core", "CUDA GPU"),
            state="readonly", width=20)
        self.backend_combo.pack(anchor="w")
        self.backend_combo.bind("<<ComboboxSelected>>", self._backend_changed)
        worker_box = ttk.Frame(acceleration, style="Surface.TFrame")
        worker_box.pack(side="left", fill="x", expand=True)
        ttk.Label(worker_box, text="CPU WORKERS", style="SurfaceMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.worker_combo = ttk.Combobox(
            worker_box, textvariable=self.worker_choice,
            values=("Auto", "2", "4", "6", "8", "12", "16", "All"),
            state="readonly", width=12)
        self.worker_combo.pack(anchor="w")

        layout_options = ttk.Frame(settings, style="Surface.TFrame")
        layout_options.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(layout_options, text="Adaptive tile boundaries", variable=self.adaptive_tiles).pack(
            side="left", anchor="w")
        checkpoint_box = ttk.Frame(layout_options, style="Surface.TFrame")
        checkpoint_box.pack(side="right")
        ttk.Label(checkpoint_box, text="DECODE GROUP", style="SurfaceMuted.TLabel").pack(side="left", padx=(0, 6))
        self.checkpoint_combo = ttk.Combobox(
            checkpoint_box, textvariable=self.checkpoint_interval,
            values=("1", "2", "4", "8", "16", "32"), state="readonly", width=5)
        self.checkpoint_combo.pack(side="left")
        acceleration_status = ttk.Frame(settings, style="Surface.TFrame")
        acceleration_status.pack(fill="x", pady=(10, 0))
        self.cuda_label = ttk.Label(acceleration_status, text="Checking CUDA support…",
                                    style="SurfaceMuted.TLabel", wraplength=410)
        self.cuda_label.pack(side="left", fill="x", expand=True, anchor="w")
        self.cuda_refresh_button = ttk.Button(
            acceleration_status, text="Recheck", width=9,
            command=lambda: self._refresh_cuda_status(refresh=True))
        self.cuda_refresh_button.pack(side="right", padx=(8, 0))
        self.native_label = ttk.Label(settings, text="Checking native C++ backend…",
                                      style="SurfaceMuted.TLabel", wraplength=480)
        self.native_label.pack(anchor="w", pady=(5, 0))
        self._refresh_cuda_status()

        ttk.Label(settings, text="Higher lossy quality retains more detail and usually creates larger files.",
                  style="SurfaceMuted.TLabel", wraplength=480).pack(anchor="w", pady=(13, 0))
        self._mode_changed()

        actions = ttk.Frame(left)
        actions.pack(fill="x")
        self.compress_btn = ttk.Button(actions, text="Compress to PXL  →", style="Accent.TButton", command=self._start_compress)
        self.compress_btn.pack(side="left")
        self.decompress_btn = ttk.Button(actions, text="Decode PXL", command=self._start_decompress)
        self.decompress_btn.pack(side="left", padx=(8, 0))
        self.info_btn = ttk.Button(actions, text="Inspect", command=self._inspect)
        self.info_btn.pack(side="left", padx=(8, 0))

        preview = self._card(right, "Preview", "Source image")
        preview.pack(fill="both", expand=True, pady=(0, 12))
        self.preview_canvas = tk.Canvas(preview, height=270, bg="#0a1220", highlightthickness=1,
                                        highlightbackground=BORDER)
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.create_text(220, 130, text="Choose an image to preview", fill="#71849f",
                                        font=("Segoe UI", 11), tags="placeholder")
        self.image_meta = tk.StringVar(value="No image selected")
        ttk.Label(preview, textvariable=self.image_meta, style="SurfaceMuted.TLabel", wraplength=480).pack(anchor="w", pady=(10, 0))

        activity = self._card(right, "Process monitor", "Live operation status")
        activity.pack(fill="both", expand=True)
        self.progress = ttk.Progressbar(activity, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 8))
        self.progress_label = ttk.Label(activity, text="Ready", style="SurfaceMuted.TLabel")
        self.progress_label.pack(anchor="w", pady=(0, 9))
        self.log = tk.Text(activity, height=8, bg="#09111e", fg="#cbd8e9", insertbackground=TEXT,
                           selectbackground="#244e6b", relief="flat", wrap="word", font=("Cascadia Code", 9),
                           padx=11, pady=10, borderwidth=0)
        self.log.pack(fill="both", expand=True)
        native = backend_info()
        if native["available"]:
            engine_line = f"Native engine: {native['implementation']} ({native['library']})."
        else:
            engine_line = "Native C++ engine not built; using compatible Python/Numba fallback. Run: python build_native.py"
        simd_line = f" SIMD: {native.get('simd', 'scalar/fallback')}." if native["available"] else ""
        self.log.insert("end", f"PXL Studio 8.0.2 ready. {engine_line}{simd_line} Compression profiles: fast / balanced / maximum.\n")
        self.log.configure(state="disabled")

    # ---------- batch queue tab ----------
    def _build_queue_tab(self):
        intro = ttk.Frame(self.queue_tab)
        intro.pack(fill="x", pady=(0, 12))
        ttk.Label(intro, text="Batch Queue", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            intro,
            text="Queue several images and PXL archives. Each file gets its own progress bar; CPU jobs run in separate processes.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        controls = self._card(self.queue_tab, "01 · Batch setup", "Add files and choose how many run together")
        controls.pack(fill="x", pady=(0, 12))
        button_row = ttk.Frame(controls, style="Surface.TFrame")
        button_row.pack(fill="x", pady=(0, 10))
        ttk.Button(button_row, text="Add images / PXL files…", command=self._add_queue_files).pack(side="left")
        ttk.Button(button_row, text="Add current selection", command=self._add_current_file_to_queue).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Remove selected", command=self._remove_selected_job).pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="Clear finished", command=self._clear_finished_jobs).pack(side="left", padx=(8, 0))

        output_row = ttk.Frame(controls, style="Surface.TFrame")
        output_row.pack(fill="x", pady=(0, 9))
        ttk.Label(output_row, text="OUTPUT FOLDER", style="SurfaceMuted.TLabel", width=17).pack(side="left")
        ttk.Entry(output_row, textvariable=self.queue_output_dir).pack(side="left", fill="x", expand=True)
        ttk.Button(output_row, text="Browse…", command=self._choose_queue_output).pack(side="left", padx=(8, 0))

        option_row = ttk.Frame(controls, style="Surface.TFrame")
        option_row.pack(fill="x")
        ttk.Label(option_row, text="SIMULTANEOUS FILES", style="SurfaceMuted.TLabel").pack(side="left")
        max_slots = max(1, min(8, os.cpu_count() or 1))
        ttk.Spinbox(option_row, from_=1, to=max_slots, textvariable=self.queue_concurrency,
                    width=5, state="readonly").pack(side="left", padx=(8, 18))
        ttk.Checkbutton(option_row, text="Overwrite existing outputs", variable=self.queue_overwrite).pack(side="left")
        ttk.Label(option_row, text=f"Detected logical CPUs: {os.cpu_count() or 1}",
                  style="SurfaceMuted.TLabel").pack(side="right")

        monitor = self._card(self.queue_tab, "02 · Queue monitor", "Overall progress")
        monitor.pack(fill="x", pady=(0, 12))
        summary_row = ttk.Frame(monitor, style="Surface.TFrame")
        summary_row.pack(fill="x", pady=(0, 7))
        ttk.Label(summary_row, textvariable=self.queue_summary, style="Surface.TLabel").pack(side="left")
        ttk.Label(summary_row, textvariable=self.queue_total_label, style="SurfaceMuted.TLabel").pack(side="right")
        self.queue_overall_progress = ttk.Progressbar(monitor, mode="determinate", maximum=100,
                                                      style="Horizontal.TProgressbar")
        self.queue_overall_progress.pack(fill="x")
        self.queue_plan_label = ttk.Label(
            monitor,
            text="CPU batch mode uses one process per file. CUDA batch mode is serialized to protect GPU memory.",
            style="SurfaceMuted.TLabel", wraplength=1000,
        )
        self.queue_plan_label.pack(anchor="w", pady=(7, 0))

        list_card = self._card(self.queue_tab, "03 · Tasks", "Individual file progress")
        list_card.pack(fill="both", expand=True)
        list_shell = ttk.Frame(list_card, style="Surface.TFrame")
        list_shell.pack(fill="both", expand=True)
        self.jobs_canvas = tk.Canvas(list_shell, bg=SURFACE, highlightthickness=0, borderwidth=0)
        jobs_scroll = ttk.Scrollbar(list_shell, orient="vertical", command=self.jobs_canvas.yview)
        self.jobs_canvas.configure(yscrollcommand=jobs_scroll.set)
        jobs_scroll.pack(side="right", fill="y")
        self.jobs_canvas.pack(side="left", fill="both", expand=True)
        self.jobs_inner = tk.Frame(self.jobs_canvas, bg=SURFACE)
        self.jobs_window = self.jobs_canvas.create_window((0, 0), window=self.jobs_inner, anchor="nw")
        self.jobs_inner.bind("<Configure>", lambda _e: self.jobs_canvas.configure(
            scrollregion=self.jobs_canvas.bbox("all")))
        self.jobs_canvas.bind("<Configure>", lambda e: self.jobs_canvas.itemconfigure(
            self.jobs_window, width=e.width))
        self.jobs_canvas.bind_all("<MouseWheel>", self._queue_mousewheel)

        footer = ttk.Frame(self.queue_tab)
        footer.pack(fill="x", pady=(11, 0))
        self.queue_start_button = ttk.Button(footer, text="Start queue  →", style="Accent.TButton",
                                             command=self._start_queue)
        self.queue_start_button.pack(side="left")
        self.queue_pause_button = ttk.Button(footer, text="Pause queue", command=self._pause_queue,
                                             state="disabled")
        self.queue_pause_button.pack(side="left", padx=(8, 0))
        self.queue_footer_label = ttk.Label(footer, text="Queued tasks are not started until you press Start.",
                                           style="Muted.TLabel")
        self.queue_footer_label.pack(side="right")

    def _queue_mousewheel(self, event):
        if self.notebook.select() == str(self.queue_tab):
            self.jobs_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ---------- codec comparison tab ----------
    def _build_compare_tab(self):
        intro = ttk.Frame(self.compare_tab)
        intro.pack(fill="x", pady=(0, 13))
        ttk.Label(intro, text="Codec Lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(intro, text="Compare 8-bit PXL, PNG, WebP, TIFF, QOI, AVIF and JPEG XL. PXL encoding/decoding separately supports exact uint16 and floating-point TIFF/HDR samples.",
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        setup = self._card(self.compare_tab, "Benchmark input", "Use the same image for every codec")
        setup.pack(fill="x", pady=(0, 12))
        source_line = ttk.Frame(setup, style="Surface.TFrame")
        source_line.pack(fill="x", pady=(0, 11))
        ttk.Entry(source_line, textvariable=self.compare_source).pack(side="left", fill="x", expand=True)
        ttk.Button(source_line, text="Choose image…", command=self._choose_compare_source).pack(side="left", padx=(8, 0))

        controls = ttk.Frame(setup, style="Surface.TFrame")
        controls.pack(fill="x")
        modes = ttk.Frame(controls, style="Surface.TFrame")
        modes.pack(side="left", padx=(0, 20))
        ttk.Label(modes, text="PROFILE", style="SurfaceMuted.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Radiobutton(modes, text="Lossless", variable=self.compare_mode, value="lossless",
                        command=self._compare_mode_changed).pack(anchor="w")
        ttk.Radiobutton(modes, text="Lossy", variable=self.compare_mode, value="lossy",
                        command=self._compare_mode_changed).pack(anchor="w")

        quality = ttk.Frame(controls, style="Surface.TFrame")
        quality.pack(side="left", fill="x", expand=True, padx=(0, 20))
        qline = ttk.Frame(quality, style="Surface.TFrame")
        qline.pack(fill="x")
        ttk.Label(qline, text="QUALITY", style="SurfaceMuted.TLabel").pack(side="left")
        self.compare_quality_label = ttk.Label(qline, text="88", style="Surface.TLabel")
        self.compare_quality_label.pack(side="right")
        self.compare_quality_scale = ttk.Scale(quality, from_=1, to=100, orient="horizontal",
                                               command=self._compare_quality_changed)
        self.compare_quality_scale.set(88)
        self.compare_quality_scale.pack(fill="x", pady=(7, 0))

        eff = ttk.Frame(controls, style="Surface.TFrame")
        eff.pack(side="left", padx=(0, 16))
        ttk.Label(eff, text="EFFORT", style="SurfaceMuted.TLabel").pack(anchor="w", pady=(0, 5))
        self.compare_effort_combo = ttk.Combobox(eff, textvariable=self.compare_effort,
                                                 values=("fast", "balanced", "maximum"), state="readonly", width=13)
        self.compare_effort_combo.pack(anchor="w")
        self.compare_btn = ttk.Button(controls, text="Run comparison  →", style="Accent.TButton", command=self._start_compare)
        self.compare_btn.pack(side="right", anchor="e")
        self._compare_mode_changed()

        compare_progress = ttk.Frame(self.compare_tab, style="Surface.TFrame", padding=(12, 8))
        compare_progress.pack(fill="x", pady=(0, 12))
        self.compare_progress = ttk.Progressbar(compare_progress, mode="determinate", maximum=100)
        self.compare_progress.pack(fill="x")
        self.compare_progress_label = ttk.Label(compare_progress, text="Ready to compare", style="SurfaceMuted.TLabel")
        self.compare_progress_label.pack(anchor="w", pady=(5, 0))

        results_card = self._card(self.compare_tab, "Results", "Size, speed and fidelity")
        results_card.pack(fill="both", expand=True)
        summary = ttk.Frame(results_card, style="Surface.TFrame")
        summary.pack(fill="x", pady=(0, 10))
        self.compare_summary = ttk.Label(summary, text="No benchmark run yet", style="SurfaceMuted.TLabel")
        self.compare_summary.pack(side="left", anchor="w")
        self.export_btn = ttk.Button(summary, text="Export CSV", command=self._export_comparison, state="disabled")
        self.export_btn.pack(side="right")

        columns = ("codec", "status", "size", "raw_ratio", "encode", "decode", "exact", "psnr", "ssim")
        self.results_tree = ttk.Treeview(results_card, columns=columns, show="headings", height=8)
        headings = {"codec": "CODEC", "status": "STATUS", "size": "FILE SIZE", "raw_ratio": "VS RAW",
                    "encode": "ENCODE", "decode": "DECODE", "exact": "EXACT", "psnr": "PSNR dB", "ssim": "SSIM"}
        widths = {"codec": 145, "status": 95, "size": 105, "raw_ratio": 85, "encode": 95,
                  "decode": 95, "exact": 70, "psnr": 82, "ssim": 75}
        for col in columns:
            self.results_tree.heading(col, text=headings[col])
            self.results_tree.column(col, width=widths[col], anchor="w" if col == "codec" else "center", stretch=True)
        self.results_tree.pack(fill="x", expand=False)
        self.results_tree.tag_configure("best", foreground=GREEN)
        self.results_tree.tag_configure("check", foreground=ORANGE)
        self.results_tree.tag_configure("unavailable", foreground="#72829a")
        self.results_tree.bind("<<TreeviewSelect>>", self._comparison_row_selected)

        bottom = ttk.Frame(results_card, style="Surface.TFrame")
        bottom.pack(fill="both", expand=True, pady=(12, 0))
        chart_frame = ttk.Frame(bottom, style="Surface2.TFrame", padding=10)
        chart_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(chart_frame, text="RELATIVE FILE SIZE  ·  SMALLER IS BETTER", style="SurfaceMuted.TLabel").pack(anchor="w")
        self.chart = tk.Canvas(chart_frame, height=145, bg=SURFACE_2, highlightthickness=0)
        self.chart.pack(fill="both", expand=True, pady=(7, 0))
        note_frame = ttk.Frame(bottom, style="Surface2.TFrame", padding=10)
        note_frame.pack(side="left", fill="both", expand=True, padx=(8, 0))
        ttk.Label(note_frame, text="SELECTED CODEC NOTES", style="SurfaceMuted.TLabel").pack(anchor="w")
        self.codec_note = tk.StringVar(value="Run a benchmark, then select a row to view encoder details.")
        ttk.Label(note_frame, textvariable=self.codec_note, style="Surface.TLabel", wraplength=350,
                  justify="left").pack(anchor="nw", fill="both", expand=True, pady=(8, 0))
        ttk.Label(self.compare_tab,
                  text="Comparisons use the same normalized 8-bit pixels. Quality numbers are codec-specific, not perceptually calibrated. Timings include encoding/decoding and local temporary-file I/O; results vary by machine and Pillow build.",
                  style="Muted.TLabel", wraplength=1080, justify="left").pack(anchor="w", pady=(10, 0))

    # ---------- general UI helpers ----------
    def _write_log(self, message: str):
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _mode_changed(self):
        lossy = self.mode.get() == "lossy"
        self.quality_scale.configure(state="normal" if lossy else "disabled")
        self.quality_label.configure(text="QUALITY · lossy profile" if lossy else "QUALITY · not used")
        self.quality_value.configure(text=str(self.quality.get()) if lossy else "—")

    def _quality_changed(self, value):
        self.quality.set(int(float(value) + 0.5))
        self.quality_value.configure(text=str(self.quality.get()))

    def _compare_mode_changed(self):
        lossy = self.compare_mode.get() == "lossy"
        self.compare_quality_scale.configure(state="normal" if lossy else "disabled")
        self.compare_quality_label.configure(text=str(self.compare_quality.get()) if lossy else "—")

    def _compare_quality_changed(self, value):
        self.compare_quality.set(int(float(value) + 0.5))
        self.compare_quality_label.configure(text=str(self.compare_quality.get()))

    def _show_preview(self, path: str):
        self.preview_canvas.delete("all")
        self.preview_photo = None
        try:
            with Image.open(path) as opened:
                opened.seek(0)
                image = ImageOps.exif_transpose(opened).copy()
                meta = f"{image.width:,} × {image.height:,} px   ·   {image.mode}   ·   {Path(path).stat().st_size:,} bytes"
                image.thumbnail((520, 275), Image.Resampling.LANCZOS)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGB")
                self.preview_photo = ImageTk.PhotoImage(image)
                self.preview_canvas.create_image(0, 0, image=self.preview_photo, anchor="center", tags="preview")
                self.preview_canvas.update_idletasks()
                self.preview_canvas.coords("preview", self.preview_canvas.winfo_width()//2,
                                           self.preview_canvas.winfo_height()//2)
                self.image_meta.set(meta)
        except Exception as exc:
            self.image_meta.set(f"Preview unavailable: {exc}")
            self.preview_canvas.create_text(220, 130, text="Preview unavailable", fill=MUTED,
                                            font=("Segoe UI", 11))

    # ---------- file dialogs ----------
    def _choose_input(self):
        path = filedialog.askopenfilename(title="Select an image or PXL file", filetypes=[
            ("Images and PXL", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif *.avif *.qoi *.exr *.pxl"),
            ("All files", "*.*")])
        if not path:
            return
        self.input_path.set(path)
        src = Path(path)
        is_pxl = src.suffix.lower() == ".pxl"
        if is_pxl:
            try:
                meta = inspect(src)
                output_suffix = ".tif" if meta.get("sample_format") in ("uint16", "float16", "float32") else ".png"
            except Exception:
                output_suffix = ".png"
            self.output_path.set(str(src.with_suffix(output_suffix)))
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(220, 130, text="PXL archive selected\nDecode to preview", fill=MUTED,
                                            justify="center", font=("Segoe UI", 11))
            self.image_meta.set(f"PXL file · {src.stat().st_size:,} bytes")
        else:
            self.output_path.set(str(src.with_suffix(".pxl")))
            self._show_preview(path)
            self.compare_source.set(path)

    def _choose_output(self):
        is_pxl = self.input_path.get().lower().endswith(".pxl")
        if is_pxl:
            try:
                meta = inspect(self.input_path.get())
                high_precision = meta.get("sample_format") in ("uint16", "float16", "float32")
            except Exception:
                high_precision = False
            if high_precision:
                filetypes = [("TIFF high-precision image", "*.tif *.tiff")]
                if meta.get("sample_format") in ("float16", "float32"):
                    filetypes.insert(1, ("OpenEXR floating-point image", "*.exr"))
                else:
                    filetypes.insert(1, ("16-bit PNG", "*.png"))
                filetypes.append(("All files", "*.*"))
                path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=filetypes)
            else:
                path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image", "*.png"), ("All files", "*.*")])
        else:
            path = filedialog.asksaveasfilename(defaultextension=".pxl", filetypes=[("PXL image", "*.pxl"), ("All files", "*.*")])
        if path:
            self.output_path.set(path)

    def _choose_compare_source(self):
        path = filedialog.askopenfilename(title="Choose an image to benchmark", filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif *.avif *.qoi"), ("All files", "*.*")])
        if path:
            self.compare_source.set(path)

    # ---------- acceleration ----------
    def _refresh_cuda_status(self, refresh=False):
        try:
            status = cuda_status(refresh=refresh)
            native = backend_info(refresh=refresh)
            if status.get("available"):
                self.backend_combo.configure(values=("Auto", "CPU multi-core", "CPU single-core", "CUDA GPU"))
                self.cuda_label.configure(text=(
                    f"CUDA READY · {status.get('device', 'NVIDIA GPU')} · "
                    f"CC {status.get('compute_capability', '?')} · "
                    f"driver {status.get('driver_version', '?')} / runtime {status.get('runtime_version', '?')} · "
                    f"CuPy {status.get('cupy_version', '?')} · smoke test PASS · predictor analysis only; entropy/dictionaries run on CPU"))
            else:
                self.backend_combo.configure(values=("Auto", "CPU multi-core", "CPU single-core"))
                reason = status.get("reason", "unknown CUDA initialization error")
                self.cuda_label.configure(text=f"CUDA unavailable · {reason}")
                if self.backend_choice.get() == "CUDA GPU":
                    self.backend_choice.set("Auto")
            if native.get("available"):
                self.native_label.configure(text=(
                    f"Native C++ backend READY · ABI {native.get('abi')} · "
                    f"{native.get('simd')} · {native.get('library')}"))
            else:
                detail = native.get("error") or "DLL/SO not found"
                self.native_label.configure(text=(
                    "Native C++ backend unavailable; PXL remains usable via Python/Numba. "
                    f"Build with `py build_native.py`. Detail: {detail}"))
        except Exception as exc:
            self.cuda_label.configure(text=f"Acceleration check failed · {type(exc).__name__}: {exc}")

    def _backend_changed(self, _event=None):
        if self.backend_choice.get() == "CPU single-core":
            self.worker_combo.configure(state="disabled")
        elif self.backend_choice.get() == "CUDA GPU":
            self.worker_combo.configure(state="disabled")
        else:
            self.worker_combo.configure(state="readonly")

    def _selected_backend(self):
        choice = self.backend_choice.get()
        worker_text = self.worker_choice.get()
        workers = None if worker_text == "Auto" else (-1 if worker_text == "All" else int(worker_text))
        if choice == "CPU single-core":
            return "cpu", 1
        if choice == "CPU multi-core":
            return "cpu", workers or 0
        if choice == "CUDA GPU":
            return "cuda", None
        return "auto", workers

    # ---------- operations ----------
    def _validate_paths(self):
        src, dst = self.input_path.get().strip(), self.output_path.get().strip()
        if not src or not Path(src).is_file():
            messagebox.showerror("Missing input", "Choose an existing input file first.")
            return None
        if not dst:
            messagebox.showerror("Missing output", "Choose an output file path first.")
            return None
        if Path(src).resolve() == Path(dst).resolve():
            messagebox.showerror("Invalid output", "Input and output paths must be different.")
            return None
        return src, dst

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (self.compress_btn, self.decompress_btn, self.info_btn, self.compare_btn):
            widget.configure(state=state)
        self.export_btn.configure(state="disabled" if busy or not self.compare_results else "normal")
        if hasattr(self, "queue_start_button"):
            self.queue_start_button.configure(state="disabled" if busy else "normal")

    def _start_compress(self):
        if self.queue_executor is not None:
            messagebox.showinfo("Batch queue active", "Finish or clear the batch queue before starting a separate single-file operation.")
            return
        paths = self._validate_paths()
        if not paths:
            return
        src, dst = paths
        if src.lower().endswith(".pxl"):
            messagebox.showerror("Invalid source", "Choose a regular image for compression. The selected file is already a PXL archive.")
            return
        mode, quality, tile, effort = self.mode.get(), self.quality.get(), int(self.tile_size.get()), self.effort.get()
        backend, workers = self._selected_backend()
        adaptive = bool(self.adaptive_tiles.get())
        checkpoint_interval = int(self.checkpoint_interval.get())
        self._run_worker("Compressing", lambda cb: compress(
            src, dst, mode, quality, tile, cb, effort, backend, workers,
            adaptive_tiles=adaptive, checkpoint_interval=checkpoint_interval), "encode")

    def _start_decompress(self):
        if self.queue_executor is not None:
            messagebox.showinfo("Batch queue active", "Finish or clear the batch queue before starting a separate single-file operation.")
            return
        paths = self._validate_paths()
        if not paths:
            return
        src, dst = paths
        if not src.lower().endswith(".pxl"):
            if not messagebox.askyesno("Input extension", "The input does not end in .pxl. Try decoding it anyway?"):
                return
        backend, workers = self._selected_backend()
        try:
            metadata = inspect(src)
            if metadata.get("profile", "").startswith("lossless") and backend == "cuda":
                backend = "cpu"
                self._write_log("Lossless archives decode on CPU; using the selected CPU worker count.")
        except Exception:
            pass  # The decoder worker will display the precise format error.
        self._run_worker("Decoding PXL", lambda cb: decompress(src, dst, cb, backend, workers), "decode")

    def _start_compare(self):
        if self.queue_executor is not None:
            messagebox.showinfo("Batch queue active", "Finish or clear the batch queue before starting a codec comparison.")
            return
        src = self.compare_source.get().strip()
        if not src or not Path(src).is_file():
            messagebox.showerror("Missing source", "Choose an existing image for the codec comparison.")
            return
        if src.lower().endswith(".pxl"):
            messagebox.showerror("Unsupported source", "Codec Lab needs a regular image, not a PXL archive.")
            return
        try:
            if probe_high_precision(src) is not None:
                messagebox.showinfo("High-precision comparison", "PXL can encode this uint16/HDR image losslessly, but Codec Lab's comparison table currently supports normalized 8-bit images only.")
                return
        except Exception as exc:
            messagebox.showerror("Cannot inspect image", str(exc))
            return
        mode, quality, effort = self.compare_mode.get(), self.compare_quality.get(), self.compare_effort.get()
        backend, workers = self._selected_backend()
        self._run_worker("Comparing codecs", lambda cb: compare_codecs(
            src, mode, quality, effort, cb, backend, workers), "compare")

    # ---------- batch queue operations ----------
    def _choose_queue_output(self):
        path = filedialog.askdirectory(title="Choose a common output folder")
        if path:
            self.queue_output_dir.set(path)

    def _add_current_file_to_queue(self):
        src = self.input_path.get().strip()
        if not src or not Path(src).is_file():
            messagebox.showerror("Missing input", "Choose an image or PXL file first.")
            return
        output = self.output_path.get().strip() or None
        try:
            self._add_queue_path(src, output_override=output)
            self.notebook.select(self.queue_tab)
        except Exception as exc:
            messagebox.showerror("Cannot add file", str(exc))

    def _add_queue_files(self):
        paths = filedialog.askopenfilenames(
            title="Add images and PXL archives to the batch queue",
            filetypes=[
                ("Supported images and PXL", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif *.avif *.qoi *.exr *.pxl"),
                ("All files", "*.*"),
            ],
        )
        added, rejected = 0, []
        for path in paths:
            try:
                self._add_queue_path(path)
                added += 1
            except Exception as exc:
                rejected.append(f"{Path(path).name}: {exc}")
        if added:
            self.notebook.select(self.queue_tab)
            self.status.set(f"Added {added} file(s) to batch queue")
        if rejected:
            messagebox.showwarning("Some files were not added", "\n".join(rejected[:12]))

    def _add_queue_path(self, path: str, output_override: str | None = None):
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"File not found: {src}")
        is_pxl = src.suffix.lower() == ".pxl"
        if is_pxl:
            meta = inspect(src)
            kind = "decompress"
            width = meta.get("width")
            height = meta.get("height")
            if width is None or height is None:
                dims = str(meta.get("dimensions", "1 x 1")).lower().split("x")
                width, height = int(dims[0].strip()), int(dims[1].strip())
            output_suffix = ".tif" if meta.get("sample_format") in ("uint16", "float16", "float32") else ".png"
            default_name = src.with_suffix(output_suffix).name
        else:
            high_precision = probe_high_precision(src)
            if high_precision is not None:
                width, height = high_precision["width"], high_precision["height"]
            else:
                with Image.open(src) as image:
                    image.verify()
                with Image.open(src) as image:
                    width, height = image.size
            kind = "compress"
            default_name = src.with_suffix(".pxl").name

        folder_text = self.queue_output_dir.get().strip()
        out_dir = Path(folder_text).expanduser().resolve() if folder_text else src.parent
        dst = Path(output_override).expanduser().resolve() if output_override else out_dir / default_name
        if src == dst:
            raise ValueError("Input and output paths must be different")

        # Ensure files with identical basenames from different folders never
        # silently target the same destination inside one batch.
        existing_targets = {Path(self.jobs[jid]["dst"]).resolve() for jid in self.job_order}
        if dst in existing_targets:
            stem, suffix = dst.stem, dst.suffix
            index = 2
            while dst in existing_targets:
                dst = dst.with_name(f"{stem}_{index}{suffix}")
                index += 1

        self.job_sequence += 1
        job_id = f"pxl-job-{self.job_sequence:05d}"
        backend, _workers = self._selected_backend()
        job = {
            "id": job_id,
            "src": str(src),
            "dst": str(dst),
            "kind": kind,
            "status": "queued",
            "progress": 0,
            "message": "Waiting for queue",
            "weight": max(1, int(width) * int(height)),
            "start_time": None,
            "mode": self.mode.get(),
            "quality": self.quality.get(),
            "tile_size": int(self.tile_size.get()),
            "adaptive_tiles": bool(self.adaptive_tiles.get()),
            "checkpoint_interval": int(self.checkpoint_interval.get()),
            "effort": self.effort.get(),
            "backend_choice": backend,
            "overwrite": self.queue_overwrite.get(),
        }
        self.jobs[job_id] = job
        self.job_order.append(job_id)
        self._create_job_row(job_id)
        self._render_job_row(job_id)
        self._update_queue_dashboard()
        return job_id

    def _create_job_row(self, job_id: str):
        row = tk.Frame(self.jobs_inner, bg=SURFACE, highlightthickness=1,
                       highlightbackground=BORDER, padx=12, pady=9)
        row.pack(fill="x", pady=(0, 7), padx=(0, 3))
        top = tk.Frame(row, bg=SURFACE)
        top.pack(fill="x")
        name = tk.Label(top, text="", bg=SURFACE, fg=TEXT,
                        font=("Segoe UI Semibold", 10), anchor="w")
        name.pack(side="left", fill="x", expand=True)
        status = tk.Label(top, text="Queued", bg=SURFACE, fg=MUTED,
                          font=("Segoe UI Semibold", 9), anchor="e")
        status.pack(side="right", padx=(10, 0))
        path = tk.Label(row, text="", bg=SURFACE, fg=MUTED,
                        font=("Segoe UI", 8), anchor="w")
        path.pack(fill="x", pady=(2, 7))
        progress_row = tk.Frame(row, bg=SURFACE)
        progress_row.pack(fill="x")
        bar = ttk.Progressbar(progress_row, mode="determinate", maximum=100, length=280)
        bar.pack(side="left", fill="x", expand=True)
        percent = tk.Label(progress_row, text="0%", bg=SURFACE, fg=TEXT,
                           font=("Segoe UI Semibold", 9), width=7, anchor="e")
        percent.pack(side="left", padx=(10, 0))
        detail = tk.Label(row, text="", bg=SURFACE, fg=MUTED,
                          font=("Segoe UI", 8), anchor="w")
        detail.pack(fill="x", pady=(5, 0))
        widgets = {"frame": row, "name": name, "status": status, "path": path,
                   "bar": bar, "percent": percent, "detail": detail}
        self.job_rows[job_id] = widgets
        for widget in (row, top, name, status, path, progress_row, percent, detail):
            widget.bind("<Button-1>", lambda _e, jid=job_id: self._select_job(jid))

    def _select_job(self, job_id: str):
        self.selected_job = job_id
        for jid, widgets in self.job_rows.items():
            color = ACCENT if jid == job_id else BORDER
            widgets["frame"].configure(highlightbackground=color)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        if seconds >= 3600:
            return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
        if seconds >= 60:
            return f"{seconds // 60}m {seconds % 60:02d}s"
        return f"{seconds}s"

    def _render_job_row(self, job_id: str):
        job = self.jobs.get(job_id)
        widgets = self.job_rows.get(job_id)
        if not job or not widgets:
            return
        src, dst = Path(job["src"]), Path(job["dst"])
        operation = "COMPRESS → PXL" if job["kind"] == "compress" else "DECODE → IMAGE"
        widgets["name"].configure(text=f"{src.name}   ·   {operation}")
        widgets["path"].configure(text=f"{src}   →   {dst}")
        status_text = {
            "queued": "QUEUED", "running": "RUNNING", "done": "COMPLETE",
            "failed": "FAILED", "skipped": "SKIPPED",
        }.get(job["status"], job["status"].upper())
        status_color = {"queued": MUTED, "running": ACCENT, "done": GREEN,
                        "failed": RED, "skipped": ORANGE}.get(job["status"], TEXT)
        widgets["status"].configure(text=status_text, fg=status_color)
        pct = int(job.get("progress", 0))
        widgets["bar"].configure(value=pct)
        widgets["percent"].configure(text=f"{pct}%")
        detail = job.get("message", "")
        if job["status"] == "running" and job.get("start_time") is not None:
            elapsed = time.monotonic() - job["start_time"]
            if 0 < pct < 100:
                eta = elapsed * (100 - pct) / max(1, pct)
                detail = f"{detail}   ·   elapsed {self._format_duration(elapsed)}   ·   ETA {self._format_duration(eta)}"
            else:
                detail = f"{detail}   ·   elapsed {self._format_duration(elapsed)}"
        widgets["detail"].configure(text=detail)
        widgets["frame"].configure(highlightbackground=ACCENT if self.selected_job == job_id else BORDER)

    def _remove_selected_job(self):
        if self.selected_job:
            self._remove_job(self.selected_job)

    def _remove_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if not job or job["status"] != "queued":
            return
        widgets = self.job_rows.pop(job_id, None)
        if widgets:
            widgets["frame"].destroy()
        self.jobs.pop(job_id, None)
        self.job_order = [jid for jid in self.job_order if jid != job_id]
        if self.selected_job == job_id:
            self.selected_job = None
        self._update_queue_dashboard()

    def _clear_finished_jobs(self):
        finished = {"done", "failed", "skipped"}
        for job_id in list(self.job_order):
            if self.jobs.get(job_id, {}).get("status") in finished:
                widgets = self.job_rows.pop(job_id, None)
                if widgets:
                    widgets["frame"].destroy()
                self.jobs.pop(job_id, None)
                self.job_order.remove(job_id)
                if self.selected_job == job_id:
                    self.selected_job = None
        self._update_queue_dashboard()

    def _queue_backend_for(self, job: dict) -> str:
        choice = job.get("backend_choice", "auto")
        if choice == "auto":
            return "cuda" if cuda_status().get("available") else "cpu"
        return choice

    def _queue_cuda_needed(self) -> bool:
        pending = [self.jobs[jid] for jid in self.job_order
                   if jid in self.jobs and self.jobs[jid]["status"] == "queued"]
        return any(self._queue_backend_for(job) == "cuda" for job in pending)

    def _start_queue(self):
        if self.busy:
            messagebox.showinfo("Operation active", "Finish the current single-file operation before starting the batch queue.")
            return
        if self.queue_executor is not None:
            self.queue_running = True
            self.queue_paused = False
            self.queue_pause_button.configure(text="Pause queue", state="normal")
            self._schedule_queue_jobs()
            return
        pending = [jid for jid in self.job_order if self.jobs[jid]["status"] == "queued"]
        if not pending:
            messagebox.showinfo("Queue is empty", "Add one or more images or PXL archives first.")
            return

        wants_cuda = self._queue_cuda_needed()
        try:
            requested = max(1, int(self.queue_concurrency.get()))
        except ValueError:
            requested = 1
        if wants_cuda:
            self.queue_slots = 1
            self.queue_plan_label.configure(
                text="CUDA is selected for at least one queued task. GPU work is serialized; choose CPU multi-core for parallel files."
            )
            # Explicit CUDA requests fail early with a useful diagnostic.
            if not cuda_status().get("available") and any(
                self.jobs[jid].get("backend_choice") == "cuda" for jid in pending
            ):
                messagebox.showerror("CUDA unavailable", "CUDA GPU was selected, but CuPy or a usable NVIDIA device was not detected.")
                return
        else:
            self.queue_slots = min(requested, max(1, os.cpu_count() or 1), 8)
            self.queue_plan_label.configure(
                text=f"CPU batch mode: up to {self.queue_slots} independent file processes, one PXL worker per file (no nested pools)."
            )
        try:
            spawn_context = multiprocessing.get_context("spawn")
            self.queue_manager = spawn_context.Manager()
            self.queue_progress_queue = self.queue_manager.Queue()
            self.queue_executor = ProcessPoolExecutor(
                max_workers=self.queue_slots,
                mp_context=spawn_context,
            )
        except Exception as exc:
            self._dispose_queue_runtime()
            messagebox.showerror("Cannot start batch workers", str(exc))
            return

        self.queue_running = True
        self.queue_paused = False
        self.queue_start_button.configure(state="disabled")
        self.queue_pause_button.configure(text="Pause queue", state="normal")
        self.queue_footer_label.configure(text="Starting worker processes…")
        self._schedule_queue_jobs()

    def _pause_queue(self):
        if self.queue_running:
            self.queue_running = False
            self.queue_paused = True
            self.queue_pause_button.configure(text="Resume queue", state="normal")
            self.queue_footer_label.configure(text="Paused · active files will finish; queued files are held")
            self.queue_start_button.configure(state="normal")
        elif self.queue_paused and self.queue_executor is not None:
            self.queue_running = True
            self.queue_paused = False
            self.queue_pause_button.configure(text="Pause queue", state="normal")
            self.queue_start_button.configure(state="disabled")
            self.queue_footer_label.configure(text="Queue resumed")
            self._schedule_queue_jobs()

    def _schedule_queue_jobs(self):
        if not self.queue_running or self.queue_executor is None:
            self._finish_queue_if_idle()
            return
        while self.queue_running and len(self.queue_futures) < self.queue_slots:
            next_id = next((jid for jid in self.job_order
                            if self.jobs.get(jid, {}).get("status") == "queued"), None)
            if next_id is None:
                break
            job = self.jobs[next_id]
            dst = Path(job["dst"])
            overwrite = self.queue_overwrite.get()
            if dst.exists() and not overwrite:
                job["status"] = "skipped"
                job["progress"] = 100
                job["message"] = "Output already exists · enable overwrite to replace it"
                self._render_job_row(next_id)
                continue
            task = dict(job)
            task["overwrite"] = overwrite
            task["backend"] = self._queue_backend_for(job)
            job["status"] = "running"
            job["progress"] = 0
            job["message"] = "Starting worker process"
            job["start_time"] = time.monotonic()
            job["effective_backend"] = task["backend"]
            self._render_job_row(next_id)
            try:
                future = self.queue_executor.submit(execute_queued_job, task, self.queue_progress_queue)
                self.queue_futures[next_id] = future
                future.add_done_callback(lambda f, jid=next_id: self._queue_future_done(jid, f))
            except Exception as exc:
                self.events.put(("queue_done", next_id, False, f"{type(exc).__name__}: {exc}"))
                break
        self._update_queue_dashboard()
        self._finish_queue_if_idle()

    def _queue_future_done(self, job_id, future):
        try:
            result = future.result()
            self.events.put(("queue_done", job_id, True, result))
        except BaseException as exc:
            self.events.put(("queue_done", job_id, False, f"{type(exc).__name__}: {exc}"))

    def _drain_queue_progress(self):
        if self.queue_progress_queue is not None:
            while True:
                try:
                    job_id, pct, message = self.queue_progress_queue.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError, BrokenPipeError):
                    break
                job = self.jobs.get(job_id)
                if not job or job["status"] != "running":
                    continue
                job["progress"] = max(0, min(100, int(pct)))
                job["message"] = str(message)
                self._render_job_row(job_id)
        self._update_queue_dashboard()

    def _handle_queue_done(self, job_id, success, payload):
        self.queue_futures.pop(job_id, None)
        job = self.jobs.get(job_id)
        if job:
            if success:
                job["status"] = "done"
                job["progress"] = 100
                output_bytes = payload.get("output_bytes") if isinstance(payload, dict) else None
                job["message"] = f"Complete · {output_bytes:,} output bytes" if output_bytes is not None else "Complete"
                self._write_log(f"BATCH COMPLETE: {Path(job['src']).name} → {job['dst']}")
            else:
                job["status"] = "failed"
                job["message"] = str(payload)
                self._write_log(f"BATCH FAILED: {Path(job['src']).name} · {payload}")
            self._render_job_row(job_id)
        self._schedule_queue_jobs()

    def _update_queue_dashboard(self):
        states = [self.jobs[jid]["status"] for jid in self.job_order if jid in self.jobs]
        total = len(states)
        queued = states.count("queued")
        running = states.count("running")
        done = states.count("done")
        failed = states.count("failed")
        skipped = states.count("skipped")
        terminal = done + failed + skipped
        total_weight = sum(self.jobs[jid].get("weight", 1) for jid in self.job_order if jid in self.jobs)
        completed_weight = 0.0
        for jid in self.job_order:
            job = self.jobs.get(jid)
            if not job:
                continue
            pct = 100 if job["status"] in ("done", "skipped") else job.get("progress", 0)
            completed_weight += job.get("weight", 1) * pct / 100.0
        overall = (completed_weight / total_weight * 100.0) if total_weight else 0.0
        self.queue_overall_progress.configure(value=overall)
        self.queue_summary.set(f"{queued} queued  ·  {running} running  ·  {done} complete  ·  {failed} failed  ·  {skipped} skipped")
        self.queue_total_label.set(f"{overall:.1f}% · {terminal} of {total} finished")
        self.queue_start_button.configure(state="disabled" if self.busy or self.queue_running else "normal")
        pause_available = self.queue_running or (self.queue_paused and self.queue_executor is not None)
        self.queue_pause_button.configure(state="normal" if pause_available and not self.busy else "disabled")
        if not total:
            self.queue_footer_label.configure(text="Add files to build a batch")
        elif self.queue_running:
            self.queue_footer_label.configure(text=f"Processing {running} file(s) · {self.queue_slots} worker slot(s) available")
        elif self.queue_paused:
            self.queue_footer_label.configure(text="Paused · resume to process remaining files")
        elif terminal == total:
            self.queue_footer_label.configure(text="All queued tasks have finished")

    def _finish_queue_if_idle(self):
        if self.queue_futures:
            return
        has_queued = any(self.jobs.get(jid, {}).get("status") == "queued" for jid in self.job_order)
        if has_queued:
            return
        if self.queue_executor is not None:
            self._dispose_queue_runtime()
        self.queue_running = False
        self.queue_paused = False
        self.queue_start_button.configure(state="normal")
        self.queue_pause_button.configure(state="disabled", text="Pause queue")
        self._update_queue_dashboard()

    def _dispose_queue_runtime(self):
        executor, manager = self.queue_executor, self.queue_manager
        self.queue_executor = None
        self.queue_manager = None
        self.queue_progress_queue = None
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

    def _on_close(self):
        active = bool(self.queue_futures) or self.busy or self.queue_executor is not None
        if active and not messagebox.askyesno(
            "Operations are active", "Some operations are still running. Close PXL Studio anyway? Queued-but-not-started tasks will be cancelled; active worker processes may continue until their current files finish."
        ):
            return
        self.queue_running = False
        self._dispose_queue_runtime()
        self.root.destroy()

    def _run_worker(self, label, operation, kind):
        if self.busy:
            return
        self._set_busy(True)
        self.progress.configure(value=0)
        self.compare_progress.configure(value=0)
        self.compare_progress_label.configure(text=label + "…")
        self.progress_label.configure(text=label + "…")
        self.status.set(label + "…")
        if kind != "compare":
            self._write_log(label + " started")

        def report(pct, message):
            self.events.put(("progress", pct, message, kind))

        def worker():
            try:
                result = operation(report)
                self.events.put(("done", result, kind))
            except Exception as exc:
                self.events.put(("error", f"{type(exc).__name__}: {exc}", kind))

        threading.Thread(target=worker, daemon=True, name=f"pxl-{kind}").start()

    def _poll_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "progress":
                    _, pct, msg, kind = event
                    self.progress.configure(value=pct)
                    self.compare_progress.configure(value=pct)
                    self.compare_progress_label.configure(text=msg)
                    self.progress_label.configure(text=msg)
                    self.status.set(msg)
                elif event[0] == "done":
                    _, result, kind = event
                    self._set_busy(False)
                    self.progress.configure(value=100)
                    self.compare_progress.configure(value=100)
                    self.compare_progress_label.configure(text="Complete")
                    self.progress_label.configure(text="Complete")
                    self.status.set("Complete · " + kind)
                    if kind == "compare":
                        self.compare_results = result
                        self._display_comparison(result)
                    else:
                        self._write_log("Completed successfully")
                        for key, value in result.items():
                            if key in ("input_bytes", "output_bytes"):
                                value = f"{value:,} bytes"
                            self._write_log(f"  {key.replace('_', ' ').title()}: {value}")
                        if "input_bytes" in result and "output_bytes" in result:
                            saved = 1 - result["output_bytes"] / max(1, result["input_bytes"])
                            self._write_log(f"  Size change: {saved*100:+.1f}% smaller (negative means larger)")
                        if kind == "decode":
                            self._show_preview(self.output_path.get())
                elif event[0] == "queue_done":
                    _, job_id, success, payload = event
                    self._handle_queue_done(job_id, success, payload)
                elif event[0] == "error":
                    _, error, kind = event
                    self._set_busy(False)
                    self.progress_label.configure(text="Failed")
                    self.compare_progress_label.configure(text="Failed")
                    self.status.set("Failed · " + kind)
                    if kind == "compare":
                        self.compare_summary.configure(text="Comparison failed")
                    else:
                        self._write_log("ERROR: " + error)
                    messagebox.showerror("PXL operation failed", error)
        except queue.Empty:
            pass
        self._drain_queue_progress()
        self._schedule_queue_jobs_if_needed()
        self.root.after(90, self._poll_events)

    def _schedule_queue_jobs_if_needed(self):
        if self.queue_running:
            self._schedule_queue_jobs()
        else:
            self._finish_queue_if_idle()

    # ---------- inspection, comparison table and chart ----------
    def _inspect(self):
        src = self.input_path.get().strip()
        if not src or not Path(src).is_file():
            messagebox.showerror("Missing input", "Choose an existing PXL file first.")
            return
        try:
            info = inspect(src)
            self._write_log("PXL file information:")
            for key, value in info.items():
                self._write_log(f"  {key.replace('_', ' ').title()}: {value}")
            self.status.set("PXL metadata loaded")
        except Exception as exc:
            messagebox.showerror("Cannot inspect file", str(exc))

    @staticmethod
    def _fmt_bytes(value: int) -> str:
        if value >= 1024 * 1024:
            return f"{value / (1024*1024):.2f} MB"
        if value >= 1024:
            return f"{value / 1024:.1f} KB"
        return f"{value} B"

    def _display_comparison(self, results: list[dict]):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        valid = [r for r in results if r["size_bytes"] > 0 and r["psnr_db"] is not None]
        eligible = ([r for r in valid if r["exact"]]
                    if self.compare_mode.get() == "lossless" else valid)
        smallest = min(eligible or valid, key=lambda r: r["size_bytes"]) if valid else None
        for row in results:
            exact = "YES" if row.get("exact") else "No"
            psnr = "∞" if row.get("psnr_db") == float("inf") else (
                f"{row['psnr_db']:.2f}" if row.get("psnr_db") is not None else "—")
            ssim = f"{row['ssim']:.4f}" if row.get("ssim") is not None else "—"
            values = (row["codec"], row["status"], self._fmt_bytes(row["size_bytes"]) if row["size_bytes"] else "—",
                      f"{row['ratio_raw']:.3f}×" if row["size_bytes"] else "—",
                      f"{row['encode_ms']:.1f} ms" if row["size_bytes"] else "—",
                      f"{row['decode_ms']:.1f} ms" if row["size_bytes"] else "—",
                      exact if row["size_bytes"] else "—", psnr, ssim)
            tag = "best" if smallest is row else ("unavailable" if row["status"] == "Unavailable" else ("check" if row["status"] == "CHECK" else ""))
            self.results_tree.insert("", "end", iid=row["codec"], values=values, tags=(tag,))
        if smallest:
            exact_count = sum(1 for r in valid if r["exact"])
            label = "smallest exact file" if self.compare_mode.get() == "lossless" else "smallest file"
            self.compare_summary.configure(
                text=f"{len(valid)} codecs ran · {label}: {smallest['codec']} ({self._fmt_bytes(smallest['size_bytes'])}) · {exact_count} exact result(s)")
            self.export_btn.configure(state="normal")
        else:
            self.compare_summary.configure(text="No compatible codec completed successfully")
            self.export_btn.configure(state="disabled")
        self._draw_size_chart(valid, smallest)
        if self.compare_mode.get() == "lossy":
            self.codec_note.set("Lossy quality numbers are not standardized across codecs. Compare both file size and PSNR/SSIM, then inspect images visually. JPEG does not preserve alpha.")
        else:
            self.codec_note.set("Lossless candidates should report EXACT = YES. BMP is an uncompressed baseline; PNG, TIFF, WebP and QOI use different compression strategies.")

    def _draw_size_chart(self, results: list[dict], smallest: dict | None):
        self.chart.delete("all")
        self.chart.update_idletasks()
        width = max(450, self.chart.winfo_width())
        if not results:
            self.chart.create_text(width//2, 65, text="No completed results to chart", fill=MUTED,
                                   font=("Segoe UI", 10))
            return
        rows = sorted(results, key=lambda r: r["size_bytes"])[:7]
        max_size = max(r["size_bytes"] for r in rows) or 1
        y = 10
        for row in rows:
            self.chart.create_text(4, y+7, text=row["codec"][:17], anchor="w", fill=TEXT,
                                   font=("Segoe UI", 8))
            bar_left, bar_right = 145, width-70
            bar_width = (bar_right-bar_left) * row["size_bytes"] / max_size
            color = GREEN if row is smallest else ACCENT
            self.chart.create_rectangle(bar_left, y+1, bar_left+bar_width, y+14, fill=color, outline="")
            self.chart.create_text(width-4, y+7, text=self._fmt_bytes(row["size_bytes"]), anchor="e",
                                   fill=MUTED, font=("Segoe UI", 8))
            y += 19

    def _comparison_row_selected(self, _event=None):
        selected = self.results_tree.selection()
        if not selected:
            return
        name = selected[0]
        row = next((r for r in self.compare_results if r["codec"] == name), None)
        if row:
            self.codec_note.set(f"{row['codec']}\nStatus: {row['status']}\nMode compared: {row.get('target_mode', '—')}\n{row['note']}")

    def _export_comparison(self):
        if not self.compare_results:
            return
        path = filedialog.asksaveasfilename(title="Export benchmark results", defaultextension=".csv",
                                            filetypes=[("CSV spreadsheet", "*.csv")])
        if not path:
            return
        try:
            export_csv(self.compare_results, path)
            self.status.set("Benchmark exported · " + Path(path).name)
            messagebox.showinfo("Export complete", f"Saved benchmark results to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


def main():
    multiprocessing.freeze_support()
    root = tk.Tk()
    PXLStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
