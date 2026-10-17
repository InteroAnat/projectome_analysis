"""
Visual_toolkit_gui.py - GUI for NeuroVis Visual Toolkit

Version: 1.9.0 (local/unreleased; package version)
Last updated: 2026-10-17
Change history: CHANGELOG.md (this folder)
Status: candidate (visual review pending)

DESCRIPTION:
    A complete Tkinter-based GUI for the Visual_toolkit. Provides an intuitive
    interface for acquiring and visualizing macaque brain data at both high 
    (0.65µm) and low (5.0µm) resolutions. Features auto soma coordinate loading,
    threaded processing, and progress indicators.

KEY FEATURES:
    - Two-column layout: settings on the left, a live figure preview on the right
    - Plot lane: tick Soma / Context / Sagittal, then "Render selected" (F5)
    - Preview: one panel at a time in the right pane; the combined TrioPlot is an
      optional extra tick, not the default view
    - Save lane: tick which PNGs (and the Soma NIfTI) to write, plus the folder;
      no ticks means preview only, and Context NIfTI is not exposed here
    - Auto-Load on Startup: Automatically loads default neuron (003.swc) when GUI opens
    - Auto-Fill Soma: Automatically extract soma coordinates from neuron SWC files
    - Threaded Processing: Non-blocking downloads with progress indicators

USAGE NOTES:
    1. Launch the GUI (package-relative imports):
        cd main_scripts
        python -m Visual_toolkit.gui

    2. Auto-Load Feature:
        - The GUI automatically loads the default neuron (003.swc) on startup
        - Soma coordinates are auto-filled after ~0.5 seconds
        - Status bar shows loading progress
        - No popup on successful auto-load (silent operation)

    3. Basic Workflow:
        - Sample ID (e.g., '251637') and Neuron ID (e.g., '003.swc') are pre-filled
        - If auto-load succeeded, coordinates are already populated
        - Click 'Load & Auto-Fill Soma' to manually reload or load different neuron
        - Adjust the Soma Plot / Context Plot settings if needed
        - Tick the panels in the Plot lane and press 'Render selected' (F5), or
          'Render all' (F6) for every panel; tick the review-figure option only when
          you want the combined trio as well
        - The rendered figure appears in the right-hand preview pane

    4. Output Directory:
        - Default: <repo>/work/visual_gui/<sample_id>_<YYYYMMDD>
        - Use 'Browse' to select custom directory
        - Note: Changing Sample ID auto-updates path unless manually set

    5. Processing:
        - Soma: downloads the soma-centred high-resolution window and renders the MIP
        - Context: downloads the 5 µm sections and renders the MIP with SWC overlay
        - Each ticked panel renders independently: a missing source is reported for
          that panel and the others still render and preview

CONFIGURATION:
    - Window size: 1280x980 pixels, resizable; the minimum is the built layout size
    - Default values: Sample='251637', Neuron='003.swc'; the FOV boxes
      take their initial values from REVIEW_SETTINGS['context'] (8000x8000x30 µm)
      and the Soma window from REVIEW_SETTINGS['soma_window']
    - Progress bar shows during active processing
    - Status bar provides step-by-step feedback

UPDATE NOTES (v1.5.0):
    - Added auto-load feature: default neuron loads automatically on startup
    - Added auto_load_default_neuron() method for startup initialization
    - Modified load_and_autofill() with silent_success parameter for auto-load
    - Auto-load runs after 500ms delay to ensure GUI is fully rendered

UNRELEASED LOCAL CHANGE (2026-10-16):
    - Capture immutable processing requests on the UI thread and use per-job toolkit/client instances.
    - Clear auto-filled coordinates after sample/neuron edits; discard late fills for a different identity.
    - Preserve fractional raw soma coordinates and marshal worker UI updates through Tk's event queue.
    - Prevent overlapping operations and clarify grid-radius-to-cube-count behavior.

UNRELEASED LOCAL CHANGE (2026-10-17):
    - Soma actions now request a soma-centred one-cube native window
      (``get_high_res_soma_window``) so the marker sits at the panel centre; a
      missing neighbour cube falls back to the soma's own cube and is recorded.
    - New "Sagittal Context Plot" action (registered processed-DS panel) and
      "All Plots + Trio" action (Soma + Context + Sagittal plus the one-figure
      tri-plot). Labels follow the plot titles: Soma Plot / Context Plot.
    - Soma actions use the larger soma-centred window from
      ``REVIEW_SETTINGS['soma_window']`` (1.5 cubes by default), so the marker sits
      at the panel centre with context around it.
    - Context/widefield defaults are read from ``REVIEW_SETTINGS['context']``
      instead of repeating the numbers here.
    - Layout check against the legacy v1.5.0 window found the status bar clipped by
      the fixed 700x700 frame once the two review actions were added. The status
      frame is now packed first, the default window is 760x840, resizing is allowed
      and the minimum size is derived from the built layout.
    - v1.7.0 redesign after a side-by-side check with the legacy GUI: Binavi-style
      cards on a pale page with a navy header, a real "Plot" lane (Soma / Context /
      Sagittal ticks) and a "Save" lane (per-artifact PNG ticks, Soma NIfTI, output
      folder), one "Render selected" action plus "Render all + trio", and a right-hand
      figure preview that shows the newest panel or review figure.
    - The default output folder no longer inherits the legacy
      ``resource/segmented_cubes/<sample>`` path or the cwd-derived project root; it
      is ``<repo>/work/visual_gui/<sample>_<YYYYMMDD>``, resolved from this module.
    - Panel failures are isolated: the Sagittal panel's missing raw-SWC source is
      reported in the status bar (and pre-warned in the Plot lane) while Soma and
      Context still render and preview. A modal appears only when nothing rendered.
    - ``plot_soma_block``/``plot_widefield_context`` now return ``output_png`` so the
      preview pane can resolve every panel type; F5/F6 run the lanes.
    - The preview defaults to a single panel (Soma / Context / Sagittal tabs); the
      combined trio is an opt-in tick under the Plot lane, and the second action is
      named "Render all" rather than "Render all + trio".

UPDATE NOTES (v1.4.0):
    - Synchronized output directory logic with Visual_toolkit.py
    - Added default path generation matching core toolkit behavior
    - Added 'Browse' button for custom output directory selection
    - Manual path changes persist across Sample ID updates
    - Improved import error handling with warning instead of exit
    - Thread-safe status updates and progress bar management

DEPENDENCIES:
    - Visual_toolkit module (sibling import)
    - IONData module from neuron-vis package
    - tkinter, threading, standard library

See CHANGELOG.md for detailed version history.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import sys
import os
import math
import tempfile
import time
import hashlib
import json
import queue
import contextlib
import shutil
from pathlib import Path
from dataclasses import dataclass

try:  # preview pane only; the rest of the GUI works without Pillow
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - environment without Pillow
    Image = ImageTk = None


class _LogStream:
    """Redirect the toolkit's prints into the GUI process log."""

    def __init__(self, sink):
        self._sink = sink

    def write(self, text):
        for line in str(text).splitlines():
            line = line.strip()
            if line:
                self._sink.put(line)

    def flush(self):
        pass


@dataclass(frozen=True)
class ProcessingRequest:
    """Values captured on the UI thread for one immutable processing run."""
    sample_id: str
    neuron_id: str
    soma_xyz: tuple[float, float, float]
    grid_radius: int
    width_um: int
    height_um: int
    depth_um: int
    output_dir: str
    save_outputs: bool = True
    save_soma_nifti: bool = True
    soma_cubes: float = 1.5
    plot_soma: bool = True
    plot_context: bool = True
    plot_sagittal: bool = True
    save_soma_png: bool = True
    save_context_png: bool = True
    save_sagittal_png: bool = True
    save_trio_png: bool = True
    make_review_figure: bool = False


def make_processing_request(sample_id, neuron_id, xyz, grid_radius,
                            width_um, height_um, depth_um, output_dir,
                            save_outputs=True, save_soma_nifti=True, soma_cubes=None,
                            plot_soma=True, plot_context=True, plot_sagittal=True,
                            save_soma_png=True, save_context_png=True,
                            save_sagittal_png=True, save_trio_png=True,
                            make_review_figure=False):
    """Parse UI values once so worker threads never follow changing widgets."""
    sample_id, neuron_id = str(sample_id).strip(), str(neuron_id).strip()
    if not sample_id or not neuron_id:
        raise ValueError("sample and neuron identity are required")
    coords = tuple(float(value) for value in xyz)
    if len(coords) != 3:
        raise ValueError("soma coordinates must contain exactly three values")
    if not all(math.isfinite(value) for value in coords):
        raise ValueError("soma coordinates must be finite")
    radius, width, height, depth = int(grid_radius), int(width_um), int(height_um), int(depth_um)
    if radius < 1 or min(width, height, depth) <= 0:
        raise ValueError("grid radius and field dimensions must be positive")
    cubes = float(REVIEW_SETTINGS['soma_window']['cubes_xy'] if soma_cubes is None else soma_cubes)
    if not math.isfinite(cubes) or cubes <= 0:
        raise ValueError("soma window size must be a positive number of cubes")
    return ProcessingRequest(
        sample_id, neuron_id, coords, radius, width, height, depth,
        str(output_dir).strip(), bool(save_outputs), bool(save_soma_nifti), cubes,
        bool(plot_soma), bool(plot_context), bool(plot_sagittal),
        bool(save_soma_png), bool(save_context_png), bool(save_sagittal_png),
        bool(save_trio_png), bool(make_review_figure),
    )


def identity_matches(request_identity, current_sample, current_neuron):
    return tuple(request_identity) == (str(current_sample).strip(), str(current_neuron).strip())

# Import the core toolkit
try:
    from .toolkit import Visual_toolkit
    from .toolkit import REVIEW_SETTINGS
    from .toolkit import BLOCK_SIZE_PIXELS
    import IONData as IT
except ImportError as e:
    print(f"WARNING: Could not import required modules: {e}") 

class NeuroVisGUI:
    """Complete GUI for NeuroVis Visual Toolkit."""

    # Palette taken from the Binavi interactive viewer (white cards on a pale page,
    # navy header strip, teal accent, muted slate labels, 1-px pale borders).
    COLORS = {
        'page': '#f6f8fa', 'card': '#ffffff', 'border': '#d9e3e9',
        'header': '#263746', 'header_text': '#f4f7f9', 'header_muted': '#a9c0cc',
        'text': '#173747', 'muted': '#526b79', 'accent': '#168a91',
        'accent_dark': '#126b78', 'accent_soft': '#edf6f7', 'strip': '#eef3f6',
        'preview_bg': '#0f1a24', 'field': '#fbfdfe', 'disabled': '#9aa9b3',
    }
    
    def __init__(self, root):
        """Initialize GUI window and widgets."""
        self.root = root
        self.root.title("Visual_Toolkit_gui v1.7.2 (2026-10-17)")
        self.root.geometry("1280x980")
        self.root.configure(bg=self.COLORS['page'])
        # Resizable, with the minimum taken from the built layout below, so a future
        # extra control cannot trap the status bar outside a fixed window.
        self.root.resizable(True, True)
        
        # === 1. Processing Variables ===
        self.sample_id = tk.StringVar(value="251637")
        self.neuron_id = tk.StringVar(value="003.swc")
        self.x_coord = tk.StringVar(value="0")
        self.y_coord = tk.StringVar(value="0")
        self.z_coord = tk.StringVar(value="0")
        self.grid_radius = tk.IntVar(value=1)
        # Context/widefield defaults come from the shared review settings so the GUI
        # and the bulk runners cannot disagree.
        context = REVIEW_SETTINGS['context']
        self.width_um = tk.StringVar(value=str(context['width_um']))
        self.height_um = tk.StringVar(value=str(context['height_um']))
        self.depth_um = tk.StringVar(value=str(context['depth_um']))
        # Soma panel window in cubes; the toolkit takes the largest covered centred step.
        self.soma_cubes = tk.StringVar(
            value=f"{REVIEW_SETTINGS['soma_window']['cubes_xy']:g}")
        self.soma_source = tk.StringVar(value="")
        self.context_source = tk.StringVar(value="")
        # Plot lane: which panels to render.
        self.plot_soma = tk.BooleanVar(value=True)
        self.plot_context = tk.BooleanVar(value=True)
        self.plot_sagittal = tk.BooleanVar(value=True)
        self.force_rerender = tk.BooleanVar(value=False)
        # Save lane: which rendered artifacts are written to the output folder.
        self.save_soma_png = tk.BooleanVar(value=True)
        self.save_context_png = tk.BooleanVar(value=True)
        self.save_sagittal_png = tk.BooleanVar(value=True)
        self.save_trio_png = tk.BooleanVar(value=True)
        self.save_soma_nifti = tk.BooleanVar(value=True)
        # The combined review figure is optional; the preview shows single panels.
        self.make_review_figure = tk.BooleanVar(value=False)
        # Master switch kept for API compatibility; the GUI decides per artifact.
        self.save_outputs = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Ready")

        # Preview pane state (populated by _register_outputs after each run).
        self.preview_files = {}
        self.preview_buttons = {}
        self.current_preview = None
        self._preview_photo = None
        self._preview_after = None
        self._ephemeral_images = {}
        self._log_queue = queue.Queue()
        self._log_stream = _LogStream(self._log_queue)
        
        # === 2. Default Output Directory Logic ===
        # The legacy default wrote into resource/segmented_cubes and derived the
        # project root from the current working directory. Both are retired: the
        # default is a per-sample staging folder under work/, resolved from this
        # module's location so the launcher's cwd no longer matters.
        self.project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..'))
        self.output_root = self._configured_output_root()
        default_path = self._generate_default_path(self.sample_id.get())
        self.output_dir = tk.StringVar(value=default_path)

        # Flag to track if user manually changed the path
        self.manual_path_set = False

        # Add listener: Update output path when Sample ID changes (if not manually set)
        self.sample_id.trace_add("write", self._on_sample_id_change)
        self.sample_id.trace_add("write", self._on_identity_change)
        self.neuron_id.trace_add("write", self._on_identity_change)
        self._selected_identity = (self.sample_id.get(), self.neuron_id.get())
        
        # Toolkit instances
        self.toolkit = None
        self.ion = None
        self._operation_running = False
        self._autofill_identity = None
        
        self.build_full_gui()
        
        # Auto-load the default neuron after GUI is fully initialized
        self.root.after(500, self.auto_load_default_neuron)
    
    def _generate_default_path(self, sample_id):
        """Per-sample GUI staging folder under work/, dated for the run day."""
        sample = str(sample_id).strip() or 'sample'
        return os.path.join(self.output_root, f"{sample}_{time.strftime('%Y%m%d')}")

    def _configured_output_root(self):
        """Output/reload root from visual_toolkit_config.yaml, else <repo>/work/visual_gui."""
        try:
            from .dataset_config import load
            root = (load().get('visual_toolkit') or {}).get('output_root')
            if root:
                return str(root)
        except Exception:
            pass
        return os.path.join(self.project_root, 'work', 'visual_gui')

    def _on_sample_id_change(self, *args):
        """Updates output directory automatically when Sample ID changes."""
        if not self.manual_path_set:
            new_path = self._generate_default_path(self.sample_id.get())
            self.output_dir.set(new_path)

    def _on_identity_change(self, *args):
        """Clear coordinates when identity changes, preventing silent stale reuse."""
        identity = (self.sample_id.get().strip(), self.neuron_id.get().strip())
        if identity != self._selected_identity:
            self._selected_identity = identity
            self._autofill_identity = None
            self.x_coord.set("")
            self.y_coord.set("")
            self.z_coord.set("")

    def build_full_gui(self):
        """Construct the Binavi-style layout: settings left, figure preview right."""
        C = self.COLORS
        self.root.configure(bg=C['page'])
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.sample_id.trace_add('write', lambda *_: self._refresh_dataset_sources())
        self._refresh_dataset_sources()

        # Header strip
        header = tk.Frame(self.root, bg=C['header'], height=54)
        header.grid(row=0, column=0, columnspan=2, sticky='ew')
        header.grid_propagate(False)
        tk.Label(header, text="NeuroVis Visual Toolkit", bg=C['header'],
                 fg=C['header_text'], font=('Segoe UI', 14, 'bold')).pack(
                     side='left', padx=(18, 8))
        tk.Label(header, text="DEB lab · fMOST neuron reviewer", bg=C['header'],
                 fg=C['header_muted'], font=('Segoe UI', 9)).pack(side='left')
        self._flat_button(header, "Config YAML", self._open_config_yaml,
                          small=True).pack(side='right', padx=16)

        # Left column: scrollable setting cards
        left = tk.Frame(self.root, bg=C['page'])
        left.grid(row=1, column=0, sticky='nsew', padx=(14, 8), pady=12)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        canvas = tk.Canvas(left, bg=C['page'], highlightthickness=0, width=430)
        scrollbar = ttk.Scrollbar(left, orient='vertical', command=canvas.yview)
        inner = tk.Frame(canvas, bg=C['page'])
        self._settings_inner = inner
        window_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        inner.bind('<Configure>',
                   lambda event: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda event: canvas.itemconfigure(window_id, width=event.width))
        for widget in (canvas, inner):
            widget.bind('<MouseWheel>',
                        lambda event: canvas.yview_scroll(int(-event.delta / 120), 'units'))

        body = self._card(inner, "Sample")
        self._field(body, "Sample ID", self.sample_id, 0, width=16)
        self._field(body, "Neuron ID", self.neuron_id, 1, width=16)
        self._flat_button(body, "Load & Auto-Fill Soma", self.load_and_autofill,
                          primary=True).grid(
            row=2, column=0, columnspan=3, sticky='ew', pady=(8, 2))
        stage_frame = tk.Frame(body, bg=C['field'], highlightthickness=1,
                               highlightbackground='#c4d2dc')
        stage_frame.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(4, 2))
        self.stage_guide = tk.Text(
            stage_frame, height=6, wrap='word', relief='flat', bd=0,
            bg=C['field'], fg=C['muted'], font=('Segoe UI', 8),
            padx=6, pady=5, cursor='arrow', insertwidth=0)
        stage_scroll = ttk.Scrollbar(stage_frame, orient='vertical',
                                     command=self.stage_guide.yview)
        self.stage_guide.configure(yscrollcommand=stage_scroll.set)
        self.stage_guide.pack(side='left', fill='both', expand=True)
        stage_scroll.pack(side='right', fill='y')
        self.stage_guide.insert(
            '1.0',
            "1  LOAD      Load & Auto-Fill Soma\n"
            "   Portal raw SWC → resource/swc_raw/<sample>/\n"
            "2  TARGET    Target soma coordinate (µm)\n"
            "   SWC root fills X/Y/Z; edit if needed.\n"
            "3  ACQUIRE   Soma Plot cubes\n"
            "   Reuse 0.65 µm cubes in resource/cubes/<sample>/high_res_http.\n"
            "4  FALLBACK  Missing cube\n"
            "   Fetch online and cache before drawing.\n"
            "5  REVIEW    Render, save and Update figures\n"
            "   Changed saves create new timestamped PNGs and keep older PNGs; the "
            "receipt tracks the newest set.")
        self.stage_guide.configure(state='disabled')
        for widget in (stage_frame, self.stage_guide):
            widget.bind(
                '<MouseWheel>',
                lambda event: (self.stage_guide.yview_scroll(
                    int(-event.delta / 120), 'units'), 'break')[1])

        body = self._card(inner, "Target soma (µm)")
        for column, (label, variable) in enumerate((
            ("X", self.x_coord), ("Y", self.y_coord), ("Z", self.z_coord))):
            tk.Label(body, text=label, bg=C['card'], fg=C['muted'],
                     font=('Segoe UI', 9)).grid(row=0, column=column * 2, sticky='w')
            entry = tk.Entry(body, textvariable=variable, width=11, relief='flat',
                             bg=C['field'], fg=C['text'], highlightthickness=1,
                             highlightbackground='#8fa3b2', highlightcolor=C['accent'],
                             font=('Segoe UI', 9), insertbackground=C['text'])
            entry.grid(row=0, column=column * 2 + 1, sticky='ew', padx=(6, 10),
                       pady=3, ipady=3)
            body.grid_columnconfigure(column * 2 + 1, weight=1)

        body = self._card(inner, "Soma Plot")
        self._field(body, "Window diameter (blocks)", self.soma_cubes, 0, width=8,
                    hint="1 block = 234 µm · default "
                         f"{REVIEW_SETTINGS['soma_window']['cubes_xy']:g}")
        tk.Label(body,
                 text="Block 0.234×0.234×0.27 mm.",
                 bg=C['card'], fg=C['muted'], font=('Segoe UI', 8),
                 wraplength=340, justify='left').grid(
                     row=1, column=0, columnspan=3, sticky='w', pady=(2, 0))
        tk.Label(body, textvariable=self.soma_source, bg=C['card'], fg=C['muted'],
                 font=('Segoe UI', 8), wraplength=380, justify='left').grid(
                     row=2, column=0, columnspan=3, sticky='w', pady=(2, 0))

        body = self._card(inner, "Context Plot")
        self._field(body, "Width (µm)", self.width_um, 0, width=8)
        self._field(body, "Height (µm)", self.height_um, 1, width=8)
        self._field(body, "Depth (µm)", self.depth_um, 2, width=8,
                    hint="3 µm per section")
        tk.Label(body, textvariable=self.context_source, bg=C['card'], fg=C['muted'],
                 font=('Segoe UI', 8), wraplength=380, justify='left').grid(
                     row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))

        # Plot lane: which panels to render
        body = self._card(inner, "Plotting")
        for column, (role, variable) in enumerate((
            ("Soma", self.plot_soma), ("Context", self.plot_context),
            ("Sagittal", self.plot_sagittal), ("Trio", self.make_review_figure))):
            check = tk.Checkbutton(body, text=role, variable=variable,
                                   bg=C['card'], fg=C['text'],
                                   activebackground=C['card'],
                                   selectcolor=C['field'], font=('Segoe UI', 9),
                                   command=self._refresh_plot_state)
            check.grid(row=0, column=column, sticky='w', padx=(0, 10))
        self.sagittal_hint = tk.Label(body, text="", bg=C['card'], fg=C['muted'],
                                      font=('Segoe UI', 8), wraplength=260,
                                      justify='left')
        self.sagittal_hint.grid(row=1, column=0, columnspan=3, sticky='w', pady=(4, 0))
        tk.Checkbutton(body, text="Force re-render", variable=self.force_rerender,
                       bg=C['card'], fg=C['text'], activebackground=C['card'],
                       selectcolor=C['field'], font=('Segoe UI', 8)).grid(
                           row=1, column=3, sticky='e', pady=(4, 0))
        for variable in (self.sample_id, self.neuron_id, self.plot_sagittal):
            variable.trace_add('write', lambda *_: self._refresh_sagittal_hint())
        self._refresh_sagittal_hint()

        # Save lane: which artifacts to write, and where
        body = self._card(inner, "Saving")
        for column, (label, variable) in enumerate((
            ("Soma PNG", self.save_soma_png), ("Context PNG", self.save_context_png),
            ("Sagittal PNG", self.save_sagittal_png))):
            tk.Checkbutton(body, text=label, variable=variable, bg=C['card'],
                           fg=C['text'], activebackground=C['card'],
                           selectcolor=C['field'], font=('Segoe UI', 9)).grid(
                               row=0, column=column, sticky='w', padx=(0, 10))
        self.soma_nifti_check = tk.Checkbutton(
            body, text="Soma NIfTI", variable=self.save_soma_nifti,
            bg=C['card'], fg=C['text'], activebackground=C['card'],
            selectcolor=C['field'], font=('Segoe UI', 9))
        self.soma_nifti_check.grid(row=1, column=0, sticky='w', pady=(4, 0))
        self.trio_check = tk.Checkbutton(
            body, text="Trio PNG", variable=self.save_trio_png,
            bg=C['card'], fg=C['text'], activebackground=C['card'],
            selectcolor=C['field'], font=('Segoe UI', 9))
        self.trio_check.grid(row=1, column=1, columnspan=2, sticky='w', pady=(4, 0))
        tk.Label(body,
                 text="Save selected writes ticked artifacts; otherwise preview-only "
                      "(temp removed). PNG runs are timestamped and retained; Soma "
                      "NIfTI uses a fixed name and is overwritten. Receipt: "
                      "visual_gui_receipt.json.",
                 bg=C['card'], fg=C['muted'], font=('Segoe UI', 8),
                 wraplength=360, justify='left').grid(
                     row=3, column=0, columnspan=4, sticky='w', pady=(6, 0))
        tk.Label(body, text="Save to", bg=C['card'], fg=C['muted'],
                 font=('Segoe UI', 9)).grid(row=2, column=0, sticky='w', pady=(8, 3))
        out_entry = tk.Entry(body, textvariable=self.output_dir, relief='flat',
                             bg=C['field'], fg=C['text'], highlightthickness=1,
                             highlightbackground='#8fa3b2', highlightcolor=C['accent'],
                             font=('Segoe UI', 9), insertbackground=C['text'])
        out_entry.grid(row=2, column=1, columnspan=2, sticky='ew', padx=(8, 6),
                       pady=(8, 3), ipady=3)
        out_entry.bind("<Key>", lambda event: setattr(self, 'manual_path_set', True))
        body.grid_columnconfigure(1, weight=1)
        self._flat_button(body, "Browse…", self.browse_folder, small=True).grid(
            row=2, column=3, sticky='w', pady=(8, 3))

        # Right column: figure preview
        right = tk.Frame(self.root, bg=C['page'])
        right.grid(row=1, column=1, sticky='nsew', padx=(8, 14), pady=12)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        preview_card = tk.Frame(right, bg=C['card'], highlightbackground=C['border'],
                                highlightthickness=1)
        preview_card.grid(row=0, column=0, sticky='nsew')
        head = tk.Frame(preview_card, bg='#6b4fa0')
        head.pack(fill='x', pady=(0, 6))
        tk.Label(head, text="Figure preview", bg='#6b4fa0', fg='#ffffff',
                 font=('Segoe UI', 11, 'bold')).pack(side='left', padx=14, pady=6)
        self.preview_caption = tk.StringVar(value="Run a plot to see it here")
        tk.Label(head, textvariable=self.preview_caption, bg='#6b4fa0', fg='#e7dff5',
                 font=('Segoe UI', 9)).pack(side='right', padx=14, pady=(10, 2))
        self.preview_identity = tk.StringVar(value="")
        tk.Label(head, textvariable=self.preview_identity, bg='#6b4fa0',
                 fg='#ffffff', font=('Segoe UI', 9, 'bold')).pack(
                     side='right', padx=(0, 10), pady=(10, 2))
        for variable in (self.sample_id, self.neuron_id):
            variable.trace_add('write', lambda *_: self._refresh_preview_identity())
        self._refresh_preview_identity()
        self.preview_holder = tk.Frame(preview_card, bg=C['preview_bg'],
                                       highlightbackground='#0b141c', highlightthickness=1)
        self.preview_holder.pack(fill='both', expand=True, padx=14, pady=(0, 8))
        self.preview_label = tk.Label(
            self.preview_holder, bg=C['preview_bg'], fg='#c9d6de', justify='center',
            font=('Segoe UI', 11),
            text="No figure yet\n\nSoma Plot · Context Plot · Sagittal · TrioPlot")
        self.preview_label.pack(fill='both', expand=True)
        self.preview_holder.bind('<Configure>', self._on_preview_resize)
        tabs = tk.Frame(preview_card, bg=C['card'])
        tabs.pack(fill='x', padx=14, pady=(0, 12))
        # Segmented grid: equal cells separated by 1-px border lines.
        tab_grid = tk.Frame(tabs, bg=C['border'], highlightthickness=1,
                            highlightbackground=C['border'])
        tab_grid.pack(side='left')
        for index, role in enumerate(('Soma', 'Context', 'Sagittal', 'TrioPlot')):
            button = tk.Button(tab_grid, text=role,
                               command=lambda r=role: self._show_role(r),
                               relief='flat', bd=0, bg=C['card'], fg=C['text'],
                               activebackground='#e7eef3', activeforeground=C['text'],
                               disabledforeground=C['disabled'],
                               font=('Segoe UI', 9, 'bold'), padx=16, pady=4,
                               cursor='hand2', highlightthickness=0)
            button.grid(row=0, column=index, sticky='nsew',
                        padx=(0 if index == 0 else 1, 0))
            tab_grid.grid_columnconfigure(index, weight=1, uniform='tabs')
            self.preview_buttons[role] = button
        reload_outline = tk.Frame(tabs, bg='#8da1b2')
        reload_outline.pack(side='right')
        tk.Button(reload_outline, text="Update figures", command=self._reload_existing,
                  relief='flat', bd=0, bg='#f7fafc', fg=C['text'],
                  activebackground='#e7eef3', activeforeground=C['text'],
                  font=('Segoe UI', 9, 'bold'), padx=12, pady=4,
                  cursor='hand2', highlightthickness=0).pack(padx=1, pady=1)

        # Process log (the toolkit's fetch/download/render lines, like the legacy status)
        log_frame = tk.Frame(preview_card, bg=C['card'])
        log_frame.pack(fill='x', padx=14, pady=(0, 10))
        tk.Label(log_frame, text="Process log", bg=C['card'], fg=C['muted'],
                 font=('Segoe UI', 8, 'bold')).pack(anchor='w')
        self.log_widget = tk.Text(log_frame, height=5, bg='#f7fafb', fg='#334955',
                                  relief='flat', highlightthickness=1,
                                  highlightbackground='#d9e3e9',
                                  font=('Consolas', 8), wrap='none', state='disabled')
        self.log_widget.pack(fill='x', pady=(2, 0))
        self._append_log("Ready.")

        # Bottom action bar
        actions = tk.Frame(self.root, bg=C['card'], highlightbackground=C['border'],
                           highlightthickness=1)
        actions.grid(row=2, column=0, columnspan=2, sticky='ew')
        action_row = tk.Frame(actions, bg=C['card'])
        action_row.pack(fill='x', padx=14, pady=10)
        for text, command, primary, colour in (
            ("Render selected", self.run_selected, True, None),
            ("Render all", self.run_all_panels, True, ('#6b4fa0', '#563d85')),
        ):
            button = self._flat_button(
                action_row, text, command, primary=primary,
                color=colour[0] if colour else None,
                active=colour[1] if colour else None)
            button.pack(side='left', padx=(0, 8))
        tk.Checkbutton(action_row, text="Save selected", variable=self.save_outputs,
                       bg=C['card'], fg=C['text'], activebackground=C['card'],
                       selectcolor=C['field'], font=('Segoe UI', 10, 'bold')).pack(
                           side='left', padx=(10, 10))
        self._flat_button(action_row, "Open output folder", self._open_output_folder,
                          small=True).pack(side='left')

        # Status strip
        strip = tk.Frame(self.root, bg=C['strip'], highlightbackground=C['border'],
                         highlightthickness=1)
        strip.grid(row=3, column=0, columnspan=2, sticky='ew')
        tk.Label(strip, textvariable=self.status_text, bg=C['strip'], fg=C['text'],
                 font=('Segoe UI', 9), anchor='w').pack(side='left', padx=14, pady=5)
        self.progress = ttk.Progressbar(strip, mode='indeterminate', length=240)

        self._refresh_preview_tabs()
        self._refresh_plot_state()
        # Keyboard access to the two actions (also handy for scripted checks).
        self.root.bind('<F5>', lambda event: self.run_selected())
        self.root.bind('<F6>', lambda event: self.run_all_panels())
        self.root.update_idletasks()
        self.root.minsize(max(1120, self.root.winfo_reqwidth()),
                          max(720, self.root.winfo_reqheight()))
        # Quick reload of already-rendered figures (also available as a button).
        self.root.after(1500, lambda: self._reload_existing(quiet=True))
        self.root.after(150, self._drain_log)

    # ---------- styling and preview helpers ----------
    def _card(self, parent, title):
        C = self.COLORS
        card = tk.Frame(parent, bg=C['card'], highlightbackground=C['border'],
                        highlightthickness=1)
        card.pack(fill='x', padx=2, pady=(0, 1))
        tk.Label(card, text=title, bg=C['card'], fg=C['header'],
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=12, pady=(4, 2))
        body = tk.Frame(card, bg=C['card'])
        body.pack(fill='x', padx=12, pady=(4, 5))
        return body

    def _field(self, parent, label, variable, row, column=0, width=12, hint=None):
        C = self.COLORS
        tk.Label(parent, text=label, bg=C['card'], fg=C['muted'],
                 font=('Segoe UI', 9)).grid(row=row, column=column, sticky='w', pady=1)
        entry = tk.Entry(parent, textvariable=variable, width=width, relief='flat',
                         bg=C['field'], fg=C['text'], highlightthickness=1,
                         highlightbackground='#8fa3b2', highlightcolor=C['accent'],
                         font=('Segoe UI', 9), insertbackground=C['text'])
        entry.grid(row=row, column=column + 1, sticky='ew', padx=(8, 6), pady=1, ipady=2)
        parent.grid_columnconfigure(column + 1, weight=1)
        if hint:
            tk.Label(parent, text=hint, bg=C['card'], fg=C['muted'],
                     font=('Segoe UI', 8)).grid(row=row, column=column + 2,
                                                sticky='w', padx=(0, 4), pady=1)
        return entry

    def _flat_button(self, parent, text, command, primary=False, small=False,
                     color=None, active=None):
        C = self.COLORS
        fill = color or (C['accent'] if primary else C['card'])
        hover = active or (C['accent_dark'] if primary else '#e7eef3')
        return tk.Button(
            parent, text=text, command=command, relief='flat', bd=0,
            bg=fill,
            fg='#ffffff' if primary else C['text'],
            activebackground=hover,
            activeforeground='#ffffff' if primary else C['text'],
            highlightthickness=1,
            highlightbackground=fill if primary else '#8da1b2',
            font=('Segoe UI', 9 if small else 10, 'bold'),
            padx=10 if small else 24, pady=3 if small else 9, cursor='hand2')

    def _refresh_preview_identity(self):
        if getattr(self, 'preview_identity', None) is not None:
            sample = self.sample_id.get().strip() or '—'
            neuron = self.neuron_id.get().strip() or '—'
            self.preview_identity.set(f"{sample} · {neuron}")

    def _refresh_dataset_sources(self):
        sample = self.sample_id.get().strip() or 'sample'
        self.soma_source.set(
            f"0.65 µm CH1 cubes · cached or fetched online · "
            f"resource/cubes/{sample}/high_res_http")
        try:
            from .dataset_config import sample as sample_entry
            entry = sample_entry(sample) or {}
            resample = entry.get('resample_5um')
        except Exception:
            resample = None
        if resample:
            tail = "\\".join(Path(resample).parts[-2:])
            self.context_source.set(f"5 µm CH1 resample · …\\{tail}")
        else:
            self.context_source.set(
                "5 µm CH1 resample · not configured in visual_toolkit_config.yaml")

    def _refresh_plot_state(self):
        """The soma NIfTI needs the Soma volume, so it follows the Soma plot tick."""
        enabled = bool(self.plot_soma.get())
        check = getattr(self, 'soma_nifti_check', None)
        if check is not None:
            check.configure(state='normal' if enabled else 'disabled',
                            fg=self.COLORS['text'] if enabled else self.COLORS['disabled'])

    def _preview_target_dir(self, request):
        """Where this run writes PNGs: the chosen folder, or a preview-only temp folder."""
        if request.save_outputs:
            return request.output_dir
        safe = lambda value: ''.join(ch if ch.isalnum() or ch in '-_.' else '_'
                                     for ch in value)
        target = os.path.join(tempfile.gettempdir(), 'visual_toolkit_preview',
                              f"{safe(request.sample_id)}_{safe(request.neuron_id)}")
        os.makedirs(target, exist_ok=True)
        return target

    @staticmethod
    def _panel_png(panel):
        return panel.get('output_png') if isinstance(panel, dict) else None

    @staticmethod
    def _panel_spec(panel):
        if not isinstance(panel, dict):
            return {}
        return {'fov_um': list(panel.get('fov_um') or []),
                'resolution_um': list(panel.get('resolution_um') or [])}

    @staticmethod
    def _soma_window_px(request):
        side = int(round(float(request.soma_cubes) * BLOCK_SIZE_PIXELS[0]))
        return (side, side)

    def _finish(self, request, paths, message, errors=None):
        kept = [path for path in paths if path]
        self._ui_call(self._register_outputs, kept)
        folder = os.path.dirname(kept[0]) if kept else request.output_dir
        note = "" if request.save_outputs else "  ·  preview only (not saved)"
        if errors:
            summary = " · ".join(errors)
            self.update_status(f"⚠️ {message}; skipped: {summary}")
            if not kept:
                self._ui_call(messagebox.showwarning, "Panel unavailable", summary)
            return
        self.update_status(f"✅ {message}{note}  ·  {folder}")

    @staticmethod
    def _friendly_error(exc):
        """Turn the known missing-source failures into an actionable sentence."""
        text = str(exc)
        if "registered cached raw SWC" in text or "Raw SWC is unavailable" in text:
            return ("no registered raw SWC for this sample yet — stage its SWC cache "
                    "or choose a sample that has one; the other panels still render")
        if "processed-DS" in text or "processed DS" in text:
            return "this sample has no registered processed-DS sagittal source"
        if "central low-resolution section" in text:
            return "no 5 µm section at this soma, so the Context panel is unavailable"
        return text

    def _refresh_sagittal_hint(self):
        """Warn before a run when the Sagittal panel's registered sources are missing."""
        label = getattr(self, 'sagittal_hint', None)
        if label is None:
            return
        sample = self.sample_id.get().strip()
        neuron = self.neuron_id.get().strip().removesuffix('.swc')
        C = self.COLORS
        try:
            from .sagittal import SAMPLE_PATHS, resolve_swc_path
            if sample not in SAMPLE_PATHS:
                ok, text = False, f"Sagittal: no processed-DS source registered for {sample}"
            else:
                swc, source = resolve_swc_path(sample, neuron, SAMPLE_PATHS.get(sample) or {})
                ok = swc is not None
                if source == "registered_cache":
                    text = "Sagittal: raw SWC ready (registered cache)"
                elif ok:
                    text = "Sagittal: raw SWC ready (legacy raw cache)"
                else:
                    repo = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), '..', '..'))
                    cache = os.path.join(repo, 'resource', 'swc_raw', sample)
                    cached = (len([name for name in os.listdir(cache)
                                   if name.endswith('.swc')])
                              if os.path.isdir(cache) else 0)
                    text = (f"Sagittal: no raw SWC for {sample}/{neuron} yet "
                            f"({cached} cached); Render will try the portal")
        except Exception as exc:
            ok, text = False, f"Sagittal: source check failed ({exc})"
        label.configure(text=text, fg=C['muted'] if ok else '#b45309')

    def _register_outputs(self, paths):
        if getattr(self, 'preview_label', None) is None:
            return
        mapping = {}
        for path in paths or []:
            if not path or not os.path.isfile(path):
                continue
            name = os.path.basename(path)
            if name.endswith('_Soma_Plot.png'):
                role = 'Soma'
            elif name.endswith('_Context_Plot.png'):
                role = 'Context'
            elif name.endswith('_Sagittal_Plot.png'):
                role = 'Sagittal'
            elif 'TriPlot' in name or 'PairPlot' in name:
                role = 'TrioPlot'
            else:
                continue
            mapping[role] = os.path.abspath(path)
        if not mapping:
            return
        # Preview-only panels land in the temp root: hold the image in memory and
        # remove the file, while saved panels stay on disk.
        temp_root = os.path.abspath(os.path.join(tempfile.gettempdir(),
                                                 'visual_toolkit_preview'))
        ephemeral = {role: path for role, path in mapping.items()
                     if os.path.abspath(path).startswith(temp_root + os.sep)}
        if ephemeral:
            self._ephemeral_images = getattr(self, '_ephemeral_images', {})
            for role, path in ephemeral.items():
                try:
                    if Image is not None:
                        with Image.open(path) as image:
                            self._ephemeral_images[role] = image.copy()
                except Exception:
                    pass
            for path in ephemeral.values():
                try:
                    os.remove(path)
                except OSError:
                    pass
            for folder in {os.path.dirname(path) for path in ephemeral.values()}:
                try:
                    if os.path.isdir(folder) and not os.listdir(folder):
                        os.rmdir(folder)
                except OSError:
                    pass
        self.preview_files.update(mapping)
        self._refresh_preview_tabs()
        for role in ('Soma', 'Context', 'Sagittal', 'TrioPlot'):
            if role in mapping:
                self._show_role(role)
                break

    def _show_role(self, role):
        path = getattr(self, 'preview_files', {}).get(role)
        if path:
            self.current_role = role
            self.show_preview(path, role=role)

    def show_preview(self, path, role=None):
        if getattr(self, 'preview_label', None) is None:
            return
        if Image is None:
            self.preview_label.configure(image='',
                                         text=f"{os.path.basename(path)}\n"
                                              "(Pillow not available for preview)")
            return
        ephemeral = (getattr(self, '_ephemeral_images', {}).get(role)
                     if role is not None else None)
        try:
            image = ephemeral.copy() if ephemeral is not None else Image.open(path)
        except Exception as exc:
            self.preview_label.configure(image='',
                                         text=f"Cannot display {os.path.basename(path)}\n{exc}")
            return
        width = max(220, self.preview_holder.winfo_width() - 20)
        height = max(220, self.preview_holder.winfo_height() - 20)
        try:
            image.thumbnail((width, height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
        except Exception as exc:
            self.preview_label.configure(image='',
                                         text=f"Cannot display {os.path.basename(path)}\n{exc}")
            return
        self._preview_photo = photo
        self.current_preview = os.path.abspath(path)
        self.current_role = role
        self.preview_label.configure(image=photo, text='')
        suffix = "  ·  preview only" if ephemeral is not None else ""
        self.preview_caption.set(f"{os.path.basename(path)}{suffix}")

    def _on_preview_resize(self, event):
        if not getattr(self, 'current_preview', None):
            return
        if getattr(self, '_preview_after', None):
            try:
                self.root.after_cancel(self._preview_after)
            except Exception:
                pass
        self._preview_after = self.root.after(
            150, lambda: self.show_preview(self.current_preview))

    def _refresh_preview_tabs(self):
        files = getattr(self, 'preview_files', {})
        for role, button in getattr(self, 'preview_buttons', {}).items():
            state = 'normal' if role in files else 'disabled'
            button.configure(state=state,
                             fg=self.COLORS['text'] if state == 'normal'
                             else self.COLORS['disabled'])

    RECEIPT_NAME = 'visual_gui_receipt.json'

    def _fingerprint(self):
        """Current identity + render settings, or None when coordinates are invalid."""
        try:
            xyz = [round(float(self.x_coord.get()), 3), round(float(self.y_coord.get()), 3),
                   round(float(self.z_coord.get()), 3)]
            cubes = float(self.soma_cubes.get() or REVIEW_SETTINGS['soma_window']['cubes_xy'])
            context = [int(float(self.width_um.get())), int(float(self.height_um.get())),
                       int(float(self.depth_um.get()))]
        except Exception:
            return None
        return {
            'sample': self.sample_id.get().strip(),
            'neuron': self.neuron_id.get().strip(),
            'soma_xyz': xyz,
            'soma_cubes': round(cubes, 3),
            'context_um': context,
            'marker': dict(REVIEW_SETTINGS['marker']),
            'widefield': dict(REVIEW_SETTINGS['widefield']),
        }

    @staticmethod
    def _fingerprint_key(fingerprint):
        return hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode('utf-8')).hexdigest()

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()

    def _write_receipt(self, folder, fingerprint, outputs, specs=None, trio_path=None):
        """Record what was rendered, with settings and file hashes, for cache checks."""
        path = os.path.join(folder, self.RECEIPT_NAME)
        try:
            data = json.load(open(path, encoding='utf-8')) if os.path.isfile(path) else {}
        except Exception:
            data = {}
        entries = data.get('entries') or {}
        files = {}
        folder_abs = os.path.abspath(folder)
        specs = specs or {}
        for role, png in dict(outputs or {}).items():
            if (png and os.path.isfile(png)
                    and os.path.abspath(png).startswith(folder_abs + os.sep)):
                item = {'path': os.path.abspath(png),
                        'sha256': self._file_sha256(png),
                        'bytes': os.path.getsize(png)}
                spec = specs.get(role) or {}
                if spec.get('fov_um'):
                    item['fov_um'] = spec['fov_um']
                if spec.get('resolution_um'):
                    item['resolution_um'] = spec['resolution_um']
                files[role] = item
        if (trio_path and os.path.isfile(trio_path)
                and os.path.abspath(trio_path).startswith(folder_abs + os.sep)):
            files['TrioPlot'] = {'path': os.path.abspath(trio_path),
                                 'sha256': self._file_sha256(trio_path),
                                 'bytes': os.path.getsize(trio_path)}
        neuron = fingerprint['neuron'].removesuffix('.swc')
        entries[neuron] = {
            'sample': fingerprint['sample'], 'neuron': neuron,
            'rendered_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'fingerprint': fingerprint,
            'key': self._fingerprint_key(fingerprint),
            'roles': sorted(files),
            'outputs': files,
        }
        os.makedirs(folder, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump({'schema': 1, 'entries': entries}, stream, indent=2, ensure_ascii=False)

    def _read_receipt(self, folder, neuron):
        path = os.path.join(folder, self.RECEIPT_NAME)
        if not os.path.isfile(path):
            return None
        try:
            data = json.load(open(path, encoding='utf-8'))
        except Exception:
            return None
        entries = data.get('entries') or {}
        key = str(neuron).removesuffix('.swc')
        return entries.get(key) or entries.get(f"{key}.swc")

    def _evaluate_cached(self, entry, roles=None):
        """Three-stage cache check: coordinates, settings key, file hashes."""
        current = self._fingerprint()
        if current is None:
            return 'no_coords', ['coordinates are not filled in']
        cached = entry.get('fingerprint') or {}
        notes = []
        same_coords = cached.get('soma_xyz') == current['soma_xyz']
        settings_now = {k: v for k, v in current.items() if k != 'soma_xyz'}
        settings_then = {k: v for k, v in cached.items() if k != 'soma_xyz'}
        same_settings = settings_now == settings_then
        same_key = entry.get('key') == self._fingerprint_key(current)
        outputs = entry.get('outputs') or {}
        if roles is not None:
            outputs = {role: outputs[role] for role in outputs if role in roles}
        same_files = all(
            os.path.isfile(item.get('path', ''))
            and self._file_sha256(item['path']) == item.get('sha256')
            for item in outputs.values())
        if not same_coords:
            notes.append('coordinates changed')
        if not same_settings:
            notes.append('settings changed')
        if not same_files:
            notes.append('cached files changed or missing')
        return ('same' if (same_coords and same_settings and same_key and same_files)
                else 'differs'), notes

    def _find_cached_figures(self, sample, neuron):
        """Newest cached PNG per role for one neuron, plus its receipt entry."""
        folders = []
        current = self.output_dir.get().strip()
        if current:
            folders.append(current)
        root = getattr(self, 'output_root', None) or self._configured_output_root()
        if os.path.isdir(root):
            candidates = [os.path.join(root, name) for name in os.listdir(root)
                          if name.startswith(f"{sample}_")
                          and os.path.isdir(os.path.join(root, name))]
            candidates.sort(key=os.path.getmtime, reverse=True)
            folders.extend(candidates[:8])
        found, receipt_entry = {}, None
        for folder in dict.fromkeys(folders):
            if not os.path.isdir(folder):
                continue
            if receipt_entry is None:
                receipt_entry = self._read_receipt(folder, neuron)
            for name in os.listdir(folder):
                if not (name.startswith(f"{sample}_{neuron}")
                        and name.lower().endswith('.png')):
                    continue
                if name.endswith('_Soma_Plot.png'):
                    role = 'Soma'
                elif name.endswith('_Context_Plot.png'):
                    role = 'Context'
                elif name.endswith('_Sagittal_Plot.png'):
                    role = 'Sagittal'
                elif 'TriPlot' in name or 'PairPlot' in name:
                    role = 'TrioPlot'
                else:
                    continue
                path = os.path.join(folder, name)
                if role not in found or os.path.getmtime(path) > os.path.getmtime(found[role]):
                    found[role] = path
        return folders, found, receipt_entry

    def _reload_existing(self, quiet=False, prompt=True):
        """Load existing figures, checking whether a re-render is needed."""
        sample = self.sample_id.get().strip()
        neuron = self.neuron_id.get().strip().removesuffix('.swc')
        if not sample or not neuron:
            if not quiet:
                self.update_status("Choose a sample and neuron first")
            return
        folders, found, receipt_entry = self._find_cached_figures(sample, neuron)
        root = getattr(self, 'output_root', None) or self._configured_output_root()
        if not found:
            if not quiet:
                self.update_status(f"No existing figures for {sample}/{neuron} under {root}")
            return
        state, notes = (self._evaluate_cached(receipt_entry) if receipt_entry
                        else ('no_receipt', ['no render receipt recorded']))
        ask = (state == 'differs') or (prompt and not quiet and state != 'no_coords')
        if ask:
            detail = ("match the current coordinates and settings" if state == 'same'
                      else "differ (" + "; ".join(notes) + ")")
            question = (f"Cached figures for {sample}/{neuron} {detail}.\n\n"
                        "Re-render now?\nYes = re-render, No = load the cached figures.")
            if messagebox.askyesno("Cached figures", question):
                self.run_selected(force_rerender=True)
                return
        self.preview_files.update(found)
        self._refresh_preview_tabs()
        for role in ('Soma', 'Context', 'Sagittal', 'TrioPlot'):
            if role in found:
                self._show_role(role)
                break
        note = "" if state == 'same' else f" · {state}: " + ", ".join(notes)
        self.update_status(
            f"Updated preview from {len(found)} existing figure(s) for "
            f"{sample}/{neuron}{note}")

    def _open_output_folder(self):
        folder = (os.path.dirname(self.current_preview)
                  if getattr(self, 'current_preview', None) else self.output_dir.get())
        try:
            os.makedirs(folder, exist_ok=True)
            os.startfile(folder)
        except Exception as exc:
            messagebox.showerror("Open folder", f"Could not open {folder}:\n{exc}")

    def _open_config_yaml(self):
        path = os.path.join(os.path.dirname(__file__), 'visual_toolkit_config.yaml')
        window = getattr(self, '_config_window', None)
        if window is not None and window.winfo_exists():
            window.lift()
            window.focus_force()
            return
        window = tk.Toplevel(self.root)
        self._config_window = window
        window.title("VisualToolkit config — visual_toolkit_config.yaml")
        window.geometry("900x640")
        window.configure(bg=self.COLORS['page'])
        header = tk.Frame(window, bg=self.COLORS['header'])
        header.pack(fill='x')
        tk.Label(header, text=path, bg=self.COLORS['header'],
                 fg=self.COLORS['header_text'], font=('Consolas', 9)).pack(
                     side='left', padx=12, pady=6)
        frame = tk.Frame(window)
        frame.pack(fill='both', expand=True, padx=8, pady=8)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        text = tk.Text(frame, wrap='none', font=('Consolas', 9),
                       bg=self.COLORS['field'], fg=self.COLORS['text'],
                       insertbackground=self.COLORS['text'], undo=True)
        scroll_y = ttk.Scrollbar(frame, orient='vertical', command=text.yview)
        scroll_x = ttk.Scrollbar(frame, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        text.grid(row=0, column=0, sticky='nsew')
        scroll_y.grid(row=0, column=1, sticky='ns')
        scroll_x.grid(row=1, column=0, sticky='ew')

        def load():
            try:
                content = Path(path).read_text(encoding='utf-8')
            except Exception as exc:
                content = f"Could not read {path}:\n{exc}"
            text.delete('1.0', 'end')
            text.insert('1.0', content)

        def save():
            content = text.get('1.0', 'end-1c')
            if self._save_config_text(path, content):
                load()
                self._refresh_dataset_sources()
                messagebox.showinfo(
                    "Config YAML",
                    "Saved. The previous file is in config_backups/ and the pristine "
                    "copy is visual_toolkit_config.default.yaml.\n\n"
                    "Acquisition paths apply to new toolkit instances; restart the GUI "
                    "to pick up a changed source root.")

        load()
        buttons = tk.Frame(window)
        buttons.pack(pady=(0, 8))
        ttk.Button(buttons, text="Save", command=save).pack(side='left', padx=4)
        ttk.Button(buttons, text="Reload", command=load).pack(side='left', padx=4)
        tk.Label(window, text="Backups: config_backups/ · default copy: "
                              "visual_toolkit_config.default.yaml",
                 bg=self.COLORS['page'], fg=self.COLORS['muted'],
                 font=('Segoe UI', 8)).pack(pady=(0, 6))

    def _save_config_text(self, path, content):
        """Validate, back up the current file and atomically write the new YAML."""
        import yaml
        from .dataset_config import load as load_config
        try:
            yaml.safe_load(content)
        except Exception as exc:
            messagebox.showerror("Config YAML", f"Invalid YAML:\n{exc}")
            return False
        folder = os.path.dirname(os.path.abspath(path))
        temp = os.path.join(folder, f".visual_toolkit_config_{os.getpid()}.tmp.yaml")
        try:
            with open(temp, 'w', encoding='utf-8') as stream:
                stream.write(content)
            load_config(temp)  # schema validation
        except Exception as exc:
            messagebox.showerror("Config YAML", f"Config rejected:\n{exc}")
            try:
                os.remove(temp)
            except OSError:
                pass
            return False
        backups = os.path.join(folder, 'config_backups')
        os.makedirs(backups, exist_ok=True)
        if os.path.isfile(path):
            stamp = time.strftime('%Y%m%d-%H%M%S')
            shutil.copy2(path, os.path.join(
                backups, f"visual_toolkit_config_{stamp}.yaml"))
            default_copy = os.path.join(folder, 'visual_toolkit_config.default.yaml')
            if not os.path.exists(default_copy):
                shutil.copy2(path, default_copy)
        os.replace(temp, path)
        return True

    def _show_progress(self):
        self.progress.pack(side='right', padx=14, pady=6)
        self.progress.start()

    def _hide_progress(self):
        self.progress.stop()
        self.progress.pack_forget()

    def _append_log(self, line):
        """Append one line to the process log and mirror it in the status strip."""
        widget = getattr(self, 'log_widget', None)
        if widget is None:
            return
        widget.configure(state='normal')
        widget.insert('end', str(line) + '\n')
        total = int(widget.index('end-1c').split('.')[0])
        if total > 400:
            widget.delete('1.0', f'{total - 400}.0')
        widget.see('end')
        widget.configure(state='disabled')
        text = str(line).strip()
        if text and not text.startswith('Plot saved:'):
            self.update_status(text)

    def _drain_log(self):
        try:
            while True:
                self._append_log(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log)
    
    def validate_inputs(self):
        """Validate numeric inputs."""
        try:
            float(self.x_coord.get())
            float(self.y_coord.get())
            float(self.z_coord.get())
            int(self.width_um.get())
            int(self.height_um.get())
            int(self.depth_um.get())
            cubes = float(self.soma_cubes.get())
            limit = float(REVIEW_SETTINGS['soma_window']['max_cubes_xy'])
            if not 0 < cubes <= limit:
                raise ValueError(f"soma window must be between 0 and {limit:g} cubes")
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid numeric value: {e}")
            return False

    def browse_folder(self):
        """Open folder selection dialog."""
        selected_dir = filedialog.askdirectory(initialdir=self.output_dir.get(), title="Select Output Folder")
        if selected_dir:
            self.output_dir.set(selected_dir)
            self.manual_path_set = True
    
    def update_status(self, message):
        if threading.current_thread() is not threading.main_thread():
            self._ui_call(self.update_status, message)
            return
        self.status_text.set(message)
        self.root.update_idletasks()

    def _ui_call(self, callback, *args, **kwargs):
        """Marshal widget calls from acquisition workers onto Tk's UI thread."""
        if threading.current_thread() is threading.main_thread():
            callback(*args, **kwargs)
            return
        try:
            self.root.after(0, lambda: callback(*args, **kwargs))
        except tk.TclError:
            # The window may be closing while an acquisition finishes.
            pass

    def prepare_output_dir(self):
        """Ensure output directory exists and return it."""
        path = self.output_dir.get().strip()
        
        # If empty, fallback to the calculated default
        if not path:
            path = self._generate_default_path(self.sample_id.get())
            self.output_dir.set(path)
        
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                self.update_status(f"📁 Created output directory: {path}")
            except OSError as e:
                messagebox.showerror("Error", f"Could not create output directory:\n{e}")
                return None
        return path

    def auto_load_default_neuron(self):
        """Automatically loads the default neuron (003.swc) on GUI startup.
        
        Called via root.after() to ensure GUI is fully rendered before
        starting the potentially slow network operation.
        """
        self.update_status("🔄 Auto-loading default neuron...")
        self.load_and_autofill(silent_success=True)
    
    def load_and_autofill(self, silent_success=False):
        """Loads the neuron and automatically fills soma coordinates.
        
        Args:
            silent_success: If True, don't show success messagebox (for auto-load)
        """
        sample_id = self.sample_id.get().strip()
        neuron_id = self.neuron_id.get().strip()

        def worker():
            try:
                tree = IT.IONData().getRawNeuronTreeByID(sample_id, neuron_id)
                if tree is None:
                    self.root.after(0, lambda: messagebox.showwarning("Not Found", "Could not load neuron tree."))
                    return
                soma_xyz = tuple(float(value) for value in (tree.root.x, tree.root.y, tree.root.z))

                def apply_if_current():
                    if not identity_matches((sample_id, neuron_id), self.sample_id.get(), self.neuron_id.get()):
                        self.update_status(f"⚠️ Discarded stale auto-fill for {sample_id}/{neuron_id}")
                        return
                    self.x_coord.set(str(soma_xyz[0]))
                    self.y_coord.set(str(soma_xyz[1]))
                    self.z_coord.set(str(soma_xyz[2]))
                    self._autofill_identity = (sample_id, neuron_id)
                    self.update_status(f"✅ Loaded: {soma_xyz[0]:g}, {soma_xyz[1]:g}, {soma_xyz[2]:g}")
                    if not silent_success:
                        messagebox.showinfo("Success", "Neuron Loaded & Coordinates Auto-filled!")
                self.root.after(0, apply_if_current)
            except Exception as e:
                status = f"⚠️ Auto-load failed: {e}" if silent_success else f"⚠️ Load failed: {e}"
                self.root.after(0, lambda: self.update_status(status))

        self._launch_processing(worker)

    def _capture_processing_request(self):
        if not self.validate_inputs():
            return None
        out_path = self.prepare_output_dir()
        if not out_path:
            return None
        try:
            def flag(name, default=True):
                variable = getattr(self, name, None)
                return bool(variable.get()) if variable is not None else default

            return make_processing_request(
                self.sample_id.get(), self.neuron_id.get(),
                (self.x_coord.get(), self.y_coord.get(), self.z_coord.get()),
                self.grid_radius.get(), self.width_um.get(), self.height_um.get(),
                self.depth_um.get(), out_path, save_outputs=flag('save_outputs'),
                save_soma_nifti=flag('save_soma_nifti'),
                soma_cubes=(float(self.soma_cubes.get())
                            if hasattr(self, 'soma_cubes') else None),
                plot_soma=flag('plot_soma'), plot_context=flag('plot_context'),
                plot_sagittal=flag('plot_sagittal'),
                save_soma_png=flag('save_soma_png'),
                save_context_png=flag('save_context_png'),
                save_sagittal_png=flag('save_sagittal_png'),
                save_trio_png=flag('save_trio_png'),
                make_review_figure=flag('make_review_figure', False),
            )
        except (TypeError, ValueError) as error:
            messagebox.showerror("Input Error", f"Invalid processing request: {error}")
            return None

    def _launch_processing(self, process):
        if self._operation_running:
            messagebox.showwarning("Busy", "A VisualToolkit operation is already running.")
            return
        self._operation_running = True

        def guarded():
            try:
                process()
            finally:
                self._ui_call(setattr, self, "_operation_running", False)

        threading.Thread(target=guarded, daemon=True).start()

    # ---------- render helpers (shared by the lane runner and the direct actions) ----------
    def _target_for(self, request, save_flag):
        """Write into the chosen folder only when that artifact's save tick is on."""
        if request.save_outputs and save_flag:
            return request.output_dir
        return self._preview_target_dir(request)

    def _render_soma(self, toolkit, request, target):
        vol, origin, res = toolkit.get_high_res_soma_window(
            request.soma_xyz, window_px=self._soma_window_px(request))
        if request.save_outputs and request.save_soma_nifti:
            toolkit.export_data(vol, origin, res, request.neuron_id,
                                suffix="SomaBlock", output_dir=request.output_dir)
        return toolkit.plot_soma_block(vol, origin, res, request.soma_xyz,
                                       request.neuron_id, output_dir=target)

    def _render_context(self, toolkit, ion, request, target):
        tree = ion.getRawNeuronTreeByID(request.sample_id, request.neuron_id)
        if tree is None:
            raise ValueError(f"Raw neuron {request.sample_id}/{request.neuron_id} was not found")
        vol, origin, res = toolkit.get_low_res_widefield(
            request.soma_xyz, width_um=request.width_um, height_um=request.height_um,
            depth_um=request.depth_um)
        return toolkit.plot_widefield_context(
            vol, origin, res, request.soma_xyz, request.neuron_id,
            swc_tree=tree, output_dir=target)

    def _render_sagittal(self, toolkit, request, target):
        from .sagittal import SAMPLE_PATHS, resolve_swc_path
        neuron = request.neuron_id.removesuffix('.swc')
        swc, _ = resolve_swc_path(request.sample_id, neuron,
                                  SAMPLE_PATHS.get(request.sample_id) or {})
        if swc is None:
            # Trigger the portal download into resource/swc_raw, then let the
            # adapter re-resolve; a network failure leaves the recorded reason.
            try:
                IT.IONData().getRawNeuronTreeByID(request.sample_id, request.neuron_id)
            except Exception:
                pass
        panel = toolkit.render_sagittal_context_from_ds(
            request.neuron_id, request.soma_xyz, output_dir=target)['panel']
        self._ui_call(self._refresh_sagittal_hint)
        return panel

    def _run_worker(self, request, work, message):
        """Shared thread body: request + toolkit lifecycle, errors and progress."""
        def process():
            toolkit = None
            try:
                self._ui_call(self._show_progress)
                self._ui_call(self._append_log,
                              f"--- {request.sample_id}/{request.neuron_id} ---")
                toolkit = Visual_toolkit(request.sample_id)
                stream = getattr(self, '_log_stream', None)
                if stream is not None:
                    # Show the toolkit's fetch/download/render lines in the GUI log.
                    with contextlib.redirect_stdout(stream):
                        paths, errors = work(toolkit)
                else:
                    paths, errors = work(toolkit)
                self._finish(request, paths, message(request), errors)
            except Exception as e:
                self._ui_call(messagebox.showerror, "Processing Error", f"Error: {str(e)}")
            finally:
                self._ui_call(self._hide_progress)
                if toolkit:
                    toolkit.close()
        self._launch_processing(process)

    def run_selected(self, roles=None, force_review_figure=False, force_rerender=False):
        """Render the ticked panels; save only the artifacts ticked in the Save lane."""
        request = self._capture_processing_request()
        if request is None:
            return
        if roles is None:
            roles = [role for role, enabled in (
                ("Soma", request.plot_soma), ("Context", request.plot_context),
                ("Sagittal", request.plot_sagittal)) if enabled]
        roles = list(roles)
        if not roles:
            messagebox.showwarning("Nothing to render",
                                   "Tick at least one panel in the Plot lane.")
            return

        rendered = []
        review_option = getattr(self, 'make_review_figure', None)
        make_review = force_review_figure or (
            bool(review_option.get()) if review_option is not None else False)
        force_option = getattr(self, 'force_rerender', None)
        force = force_rerender or (
            bool(force_option.get()) if force_option is not None else False)
        fingerprint = self._fingerprint()

        if not force:
            try:
                needed = list(roles) + (["TrioPlot"] if make_review else [])
                _folders, found, entry = self._find_cached_figures(
                    request.sample_id, request.neuron_id.removesuffix('.swc'))
                if (entry and all(role in found for role in needed)
                        and getattr(self, 'preview_label', None) is not None):
                    state, _notes = self._evaluate_cached(entry, roles=needed)
                    if state == 'same':
                        self.preview_files.update(
                            {role: found[role] for role in needed})
                        self._refresh_preview_tabs()
                        for role in ('Soma', 'Context', 'Sagittal', 'TrioPlot'):
                            if role in needed:
                                self._show_role(role)
                                break
                        self.update_status(
                            "Cached figures match; loaded without re-rendering. "
                            "Tick Force re-render to redraw.")
                        self._append_log("Cached figures reused; no re-render.")
                        return
            except Exception:
                # A partially built or headless instance renders normally.
                pass

        def work(toolkit):
            ion = IT.IONData() if "Context" in roles else None
            panels, paths, errors = {}, [], []
            for role in roles:
                self.update_status(f"{role} Plot · rendering…")
                try:
                    if role == "Soma":
                        panel = self._render_soma(
                            toolkit, request,
                            self._target_for(request, request.save_soma_png))
                    elif role == "Context":
                        panel = self._render_context(
                            toolkit, ion, request,
                            self._target_for(request, request.save_context_png))
                    else:
                        panel = self._render_sagittal(
                            toolkit, request,
                            self._target_for(request, request.save_sagittal_png))
                except Exception as exc:
                    # One unavailable panel must not discard the others.
                    errors.append(f"{role}: {self._friendly_error(exc)}")
                    continue
                panels[role] = panel
                paths.append(self._panel_png(panel))
                rendered.append(role)
            trio_path = None
            if len(panels) == 3 and make_review:
                self.update_status("Rendering review figure…")
                saving = request.save_outputs and request.save_trio_png
                trio_target = request.output_dir if saving else self._preview_target_dir(request)
                trio_path = toolkit.plot_tri_plot(
                    [panels["Soma"], panels["Context"], panels["Sagittal"]],
                    request.neuron_id, output_dir=trio_target)
                paths.append(trio_path)
            specs = {role: self._panel_spec(panel) for role, panel in panels.items()}
            if request.save_outputs and fingerprint is not None:
                self._write_receipt(
                    request.output_dir, fingerprint,
                    {role: self._panel_png(panel) for role, panel in panels.items()},
                    specs, trio_path=trio_path)
            return paths, errors

        self._run_worker(
            request, work,
            lambda r: f"Rendered {' + '.join(rendered) if rendered else 'nothing'}")

    def run_highres(self):
        """Render the Soma panel with the current ticks (kept as a direct action)."""
        request = self._capture_processing_request()
        if request is None: return

        def work(toolkit):
            self.update_status("Soma Plot · acquiring soma-centred window…")
            panel = self._render_soma(
                toolkit, request, self._target_for(request, request.save_soma_png))
            return [self._panel_png(panel)], []

        self._run_worker(request, work, lambda r: "Soma plot ready")

    def run_lowres(self):
        """Render the Context panel with the current ticks."""
        request = self._capture_processing_request()
        if request is None: return

        def work(toolkit):
            self.update_status("Context Plot · acquiring 5 µm sections…")
            panel = self._render_context(
                toolkit, IT.IONData(), request,
                self._target_for(request, request.save_context_png))
            return [self._panel_png(panel)], []

        self._run_worker(request, work, lambda r: "Context plot ready")

    def run_both(self):
        """Render Soma and Context together (direct action)."""
        request = self._capture_processing_request()
        if request is None or not messagebox.askyesno("Confirm", "Render Soma and Context panels?"):
            return

        def work(toolkit):
            soma = self._render_soma(
                toolkit, request, self._target_for(request, request.save_soma_png))
            context = self._render_context(
                toolkit, IT.IONData(), request,
                self._target_for(request, request.save_context_png))
            return [self._panel_png(soma), self._panel_png(context)], []

        self._run_worker(request, work, lambda r: "Soma + Context ready")

    def run_sagittal(self):
        """Render the registered processed-DS sagittal panel for the current neuron."""
        request = self._capture_processing_request()
        if request is None: return

        def work(toolkit):
            self.update_status("Sagittal Context Plot · resolving processed-DS panel…")
            panel = self._render_sagittal(
                toolkit, request,
                self._target_for(request, request.save_sagittal_png))
            return [self._panel_png(panel)], []

        self._run_worker(request, work, lambda r: "Sagittal context ready")

    def run_all_panels(self):
        """Render all three panels and their TrioPlot."""
        self.run_selected(roles=("Soma", "Context", "Sagittal"),
                          force_review_figure=True)

# ==================== MAIN LAUNCHER ====================
def main():
    root = tk.Tk()
    app = NeuroVisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
