# Visual Toolkit GUI - Learning Guide

## Overview

The Visual Toolkit GUI is a Tkinter-based interface for acquiring and visualizing macaque brain data at both high (0.65µm) and low (5.0µm) resolutions. It provides an intuitive way to download 3D brain blocks, generate plots, and overlay neuron traces (SWC) on anatomical context.

---

## File Structure

```
main_scripts/
├── Visual_toolkit/
│   ├── gui.py                 # Tkinter GUI (this guide)
│   ├── toolkit.py             # Core backend + REVIEW_SETTINGS
│   ├── sagittal.py            # Processed-DS sagittal panel + raw-SWC resolver
│   ├── dataset_config.py      # Dataset-link loader/checker
│   ├── visual_toolkit_config.yaml  # Per-sample DS00 / 5 µm / SWC links
│   ├── derived_context.py     # Per-sample context helpers
│   └── __main__.py            # Headless demo (not the GUI)
└── VISUAL_TOOLKIT_GUIDE.md    # This document
```

---

## Launching the GUI

The GUI uses package-relative imports, so start it as a module with `main_scripts`
on the import path:

```powershell
cd D:\projectome_analysis\main_scripts
C:\Users\laika_yan\miniconda3\envs\projectome\python.exe -m Visual_toolkit.gui
```

or from the repository root:

```powershell
$env:PYTHONPATH = "D:\projectome_analysis\main_scripts"
C:\Users\laika_yan\miniconda3\envs\projectome\python.exe -m Visual_toolkit.gui
```

The window opens on sample `251637` / neuron `003.swc` and auto-loads that soma's
coordinates after about half a second (this step needs portal access; editing the
sample or neuron clears the coordinates until the next load). `python -m Visual_toolkit`
runs the headless demo instead — it is not the GUI.

Layout (v1.7.0, Binavi-style):

- Left column: `Sample` and `Target soma (µm)` cards, then `Soma Plot` (window in
  cubes), `Context Plot` (width/height/depth), the `Plotting` lane and the `Saving` lane.
  Cards scroll if the window is short.
- `Plotting` lane: tick Soma / Context / Sagittal, then **Render selected** (F5). A line
  under the ticks warns when the Sagittal panel's registered raw-SWC source is
  missing for the current sample.
- `Saving` lane: Soma PNG / Context PNG / Sagittal PNG / Soma NIfTI / Trio PNG ticks plus the
  output folder. Context NIfTI is deliberately not exposed in the GUI; the toolkit
  still exports it.
- Bottom bar: **Render selected** (teal), **Render all** (purple), **Save selected**
  (default on) and **Open output folder**. With Save selected off, panels are
  preview-only: their temp PNGs are deleted once the preview has them, and no receipt
  or NIfTI is written.
- Right column: live figure preview showing **one panel at a time**, with Soma /
  Context / Sagittal tabs (and a TrioPlot tab only when the combined figure exists)
  plus **Update figures** (scan existing outputs for this sample/neuron without
  re-rendering) and **Open folder**.
- **Render all** (F6, purple) renders all three panels and always adds the TrioPlot,
  regardless of the Plot ticks. **Render selected** (F5, teal) renders the ticked
  panels and adds the TrioPlot only when "Also make the combined figure (TrioPlot)"
  is ticked.
- Both render actions check the receipt and cached PNGs first: when the coordinates,
  settings and file hashes match the requested panels, the cached figures load and
  nothing is re-rendered. Tick **Force re-render** to redraw anyway. **Update figures**
  remains the explicit way to browse cached sets and choose re-render vs load when they
  differ.
- A **Process log** below the preview mirrors the toolkit's fetch/download/render lines
  (`Fetching Z-Slices…`, `Overlaying SWC Edges…`, `Plot saved…`), and the progress bar
  runs for the whole render.

Default output folder: `<repo>/work/visual_gui/<sample>_<YYYYMMDD>` (the legacy
`resource/segmented_cubes/<sample>` default is retired and the path no longer depends
on the launch directory).

The default root comes from `visual_toolkit.output_root` in
`main_scripts/Visual_toolkit/visual_toolkit_config.yaml`; `visual_toolkit.auto_reload` controls the
startup scan that fills the preview with existing figures. **Update figures** repeats
that scan on demand (`<output_root>/<sample>_*`, newest first), so earlier renders can
be reviewed without re-running acquisition. Each saved run writes
`visual_gui_receipt.json` in its folder (coordinates, settings, role files, hashes);
reload compares coordinates, the settings key and the file hashes and asks whether to
re-render when anything differs. Package change history is in
[CHANGELOG.md](Visual_toolkit/CHANGELOG.md).

## Package layout and dataset links

Everything the review figures need lives in `main_scripts/Visual_toolkit/`:
`toolkit.py` (drawing, `REVIEW_SETTINGS`, soma/context/sagittal renderers), `gui.py`,
`sagittal.py` (processed-DS panel preparation), `dataset_config.py` and
`visual_toolkit_config.yaml` (per-sample DS00 processed volume, 5 µm resample folder and raw-SWC
cache). The bulk wrappers in `group_analysis/scripts/` import
`Visual_toolkit.sagittal`, so there is no tool-specific copy under `tools/` any more.

Check or edit the dataset links with:

```powershell
python main_scripts/Visual_toolkit/dataset_config.py --check
python main_scripts/Visual_toolkit/dataset_config.py --check --sample 252527 --strict
```

A raw SWC resolves from the configured `swc_cache` first and from
`resource/swc_raw/<sample>/<neuron>.swc` (the legacy raw download cache) second; the
panel provenance records `swc.source` as `registered_cache` or `legacy_raw_cache`.
That fallback is what makes 251637/003's Sagittal panel available.

The header **Config YAML** button opens the config in an editable text viewer. **Save**
validates the YAML and schema, copies the previous file to `config_backups/`, keeps a
pristine `visual_toolkit_config.default.yaml`, and writes atomically. Acquisition
settings apply to new toolkit instances, so restart the GUI after changing a source
root. The loader falls back to the legacy `datasets.yaml` name if the new file is
absent.

---

## Essential Components

### 1. **NeuroVisGUI Class** (Lines 75-444)

The main GUI class that manages the entire interface.

#### Key Initialization (`__init__`, Lines 78-114)
```python
def __init__(self, root):
    self.root = root
    self.sample_id = tk.StringVar(value="251637")      # Default sample
    self.neuron_id = tk.StringVar(value="003.swc")     # Default neuron
    self.x_coord = tk.StringVar(value="0")             # Soma X
    self.y_coord = tk.StringVar(value="0")             # Soma Y
    self.z_coord = tk.StringVar(value="0")             # Soma Z
    # ... other variables
```

**Key Variables:**
- `sample_id`: The fMOST sample ID (e.g., '251637')
- `neuron_id`: The neuron SWC file name (e.g., '003.swc')
- `x/y/z_coord`: Soma coordinates in micrometers
- `grid_radius`: For high-res block acquisition (1, 2, or 3)
- `width_um/height_um/depth_um`: Field of view dimensions
- `output_dir`: Where results are saved

---

### 2. **GUI Layout** (`build_full_gui`, Lines 126-225)

The GUI is organized into labeled sections:

```
┌─────────────────────────────────────┐
│  🧠 NeuroVis Visual Toolkit         │  <- Title
├─────────────────────────────────────┤
│  📁 Sample Configuration            │  <- Sample/Neuron IDs
│    - Sample ID: [251637     ]       │
│    - Neuron ID: [003.swc    ]       │
│    - [Load & Auto-Fill Soma]        │
├─────────────────────────────────────┤
│  📍 Soma Coordinates (µm)           │  <- X/Y/Z inputs
│    - X: [0] (horizontal)            │
│    - Y: [0] (vertical)              │
│    - Z: [0] (depth)                 │
├─────────────────────────────────────┤
│  🔍 High-Res Soma Block             │  <- Grid radius selector
│    - Grid Radius: [1 ▼]             │
├─────────────────────────────────────┤
│  🌐 Low-Res Wide Field              │  <- FOV dimensions
│    - Width/Height/Depth (µm)        │
├─────────────────────────────────────┤
│  💾 Output Configuration            │  <- Output directory
│    - Save to: [path...] [Browse]    │
├─────────────────────────────────────┤
│  [Soma Plot] [Wide-field] [Both]    │  <- Action buttons
├─────────────────────────────────────┤
│  Status: Ready              [====]  │  <- Status bar + progress
└─────────────────────────────────────┘
```

---

### 3. **Core Actions**

#### **Load & Auto-Fill Soma** (`load_and_autofill`, Lines 270-302)

This is the key workflow starter:

```python
def load_and_autofill(self):
    """Loads the neuron and automatically fills soma coordinates."""
    def worker():
        # 1. Connect to database
        self.toolkit = Visual_toolkit(self.sample_id.get())
        self.ion = IT.IONData()
        
        # 2. Load neuron tree
        tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
        
        # 3. Extract soma coordinates from root
        soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
        self.x_coord.set(str(int(soma_xyz[0])))
        self.y_coord.set(str(int(soma_xyz[1])))
        self.z_coord.set(str(int(soma_xyz[2])))
```

**Flow:**
1. Creates `Visual_toolkit` instance (connects to data sources)
2. Creates `IONData` instance (database interface)
3. Fetches raw neuron SWC tree
4. Extracts soma coordinates from `tree.root`
5. Auto-fills X/Y/Z coordinate fields

---

#### **Run High-Resolution** (`run_highres`, Lines 304-341)

Downloads high-res blocks around the soma:

```python
def run_highres(self):
    # 1. Initialize toolkit
    self.toolkit = Visual_toolkit(self.sample_id.get())
    soma_xyz = [float(self.x_coord.get()), ...]
    
    # 2. Download blocks
    vol, origin, res = self.toolkit.get_high_res_block(
        soma_xyz, grid_radius=self.grid_radius.get()
    )
    
    # 3. Export and plot
    self.toolkit.export_data(vol, origin, res, self.neuron_id.get(), suffix="SomaBlock")
    self.toolkit.plot_soma_block(vol, origin, res, soma_xyz, self.neuron_id.get())
```

---

#### **Run Low-Resolution** (`run_lowres`, Lines 343-388)

Downloads wide-field low-res slices:

```python
def run_lowres(self):
    # 1. Initialize
    self.toolkit = Visual_toolkit(self.sample_id.get())
    self.ion = IT.IONData()
    tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
    
    # 2. Download slices via SSH
    vol, origin, res = self.toolkit.get_low_res_widefield(
        soma_xyz, width_um=..., height_um=..., depth_um=...
    )
    
    # 3. Export and plot with SWC overlay
    self.toolkit.export_data(vol, origin, res, ..., suffix="WideField")
    self.toolkit.plot_widefield_context(vol, origin, res, soma_xyz, ..., swc_tree=tree)
```

WideField display follows the early legacy order: MIP p1–99.5 normalization,
clip, then background gain. The user selected gain 2.0 / threshold 100 for
all samples after reviewing the cohort comparison; there are no lower-gain
sample defaults. The legacy low-maximum fallback remains, and explicit plot
parameters still override the defaults. These settings affect PNG display
only; they do not alter image voxels or establish anatomical acceptance.
HighRes SomaBlock keeps the existing p0.5–99.5, gamma 0.5 display. See the
[cohort review index](../work/legacy_contrast_20261016/index.html).

---

### 4. **Threading Model**

All long-running operations run in separate threads to keep the GUI responsive:

```python
def load_and_autofill(self):
    def worker():
        # ... actual work ...
        pass
    
    threading.Thread(target=worker, daemon=True).start()
```

**Key Points:**
- `daemon=True` ensures threads die when main program exits
- Progress bar starts before work, stops after
- Status updates happen via `self.update_status()`
- Message boxes are thread-safe (Tkinter handles this)

---

## Places of Customization

### 1. **Default Values** (Lines 87-95)

Change the default sample/neuron that appears on startup:

```python
self.sample_id = tk.StringVar(value="251637")      # <-- Change this
self.neuron_id = tk.StringVar(value="003.swc")     # <-- Change this
self.x_coord = tk.StringVar(value="0")
self.y_coord = tk.StringVar(value="0")
self.z_coord = tk.StringVar(value="0")
self.grid_radius = tk.IntVar(value=1)              # Default grid radius
self.width_um = tk.StringVar(value="8000")         # Default FOV width
self.height_um = tk.StringVar(value="8000")        # Default FOV height
self.depth_um = tk.StringVar(value="30")           # Default depth
```

---

### 2. **Window Appearance** (Lines 81-84)

```python
self.root.title(" Visual_Toolkit_gui v1.4")         # Window title
self.root.geometry("700x700")                       # Window size
self.root.configure(bg='#f0f0f0')                   # Background color
self.root.resizable(False, False)                   # Allow resizing?
```

---

### 3. **Grid Radius Options** (Line 176)

Change available grid radius values:

```python
ttk.Combobox(highres_group, textvariable=self.grid_radius, 
             values=[1, 2, 3],                      # <-- Add more values
             state="readonly", width=10)
```

---

### 4. **Output Directory Logic** (Lines 98-124)

The default output path is generated as:

```python
def _generate_default_path(self, sample_id):
    return os.path.join(self.project_root, 'resource', 'segmented_cubes', sample_id)
```

Change this to customize where files are saved by default.

---

### 5. **Adding New Action Buttons** (Lines 206-219)

Example of adding a new button:

```python
# Add this in build_full_gui()
ttk.Button(button_frame, text="My New Action", 
           command=self.run_my_action,
           style='XLarge.TButton', width=18).pack(side=tk.LEFT, padx=5)

# Then add the handler method:
def run_my_action(self):
    if not self.validate_inputs(): return
    def process():
        # Your custom processing here
        pass
    threading.Thread(target=process, daemon=True).start()
```

---

### 6. **Status Messages** (Line 248-250)

Customize how status updates appear:

```python
def update_status(self, message):
    self.status_text.set(message)
    self.root.update_idletasks()  # Forces immediate UI update
```

---

## Integration with Backend (Visual_toolkit.py)

### Key Backend Methods Used:

| GUI Action | Backend Method | Purpose |
|------------|----------------|---------|
| Load Neuron | `IONData.getRawNeuronTreeByID()` | Fetch SWC from server |
| High-Res | `Visual_toolkit.get_high_res_block()` | Download HTTP blocks |
| Low-Res | `Visual_toolkit.get_low_res_widefield()` | Download SSH slices |
| Export | `Visual_toolkit.export_data()` | Save as NIfTI/TIFF |
| Plot | `Visual_toolkit.plot_soma_block()` | Generate visualization |
| Plot | `Visual_toolkit.plot_widefield_context()` | Generate with SWC overlay |
| Plot | `Visual_toolkit.plot_sagittal_context()` | Draw a prepared processed-DS sagittal panel in the same style |
| Plot | `Visual_toolkit.plot_tri_plot()` | One review figure: Soma, Context and Sagittal panels side by side |
| Plot | `Visual_toolkit.render_sagittal_context_from_ds()` | Resolve the registered processed-DS panel for a neuron and draw it |

---

## Review panel style and titles (2026-10-17, local/unreleased)

Every panel uses one drawing path and one target marker, so the single PNGs and the
combined review figure cannot drift apart: black background, grayscale/green layers,
the cyan/turquoise open box plus cross described below, a white scale bar, and a
three-line title.

**Target marker (all three panel types).** Every Soma, Context and Sagittal panel
marks the target with the same marker: a cyan/turquoise (`#00e5ee`) open box with a
cross whose intersection is the exact mapped position. Only the drawn size is
enlarged for visibility, so the marker is not to scale. `Visual_toolkit._draw_target_marker`
is the single implementation; `tests/test_visual_toolkit_panel_render.py` checks the
rendered pixels for the box and the cross in each panel type.

**Review settings (single source).** `Visual_toolkit.toolkit.REVIEW_SETTINGS` holds the
values the GUI, the bulk runners and the sagittal adapter share, so they cannot
disagree: `marker` (colour, minimum box size, cross arms, label), `soma_window`
(target window in cubes plus the descending centred alternatives), `context`
(8,000 × 8,000 × 30 µm crop), `widefield` (gain 2.0 / threshold 100),
`scale_bar_um` (Soma 30 µm, Context 500 µm, Sagittal automatic) and `figure`
(panel and review-figure size/DPI). Explicit call or CLI parameters still override.

**Soma centring and blocks.** One downloaded block is 360×360×90 voxels at
0.65×0.65×3 µm = 234×234×270 µm. The Soma panel is a soma-centred native window with
**diameter** `REVIEW_SETTINGS['soma_window']['cubes_xy']` blocks (1.5 blocks =
540 px ≈ 351 µm) and one block deep (270 µm); only the blocks that window touches are
downloaded (typically 4–6). The legacy `grid radius` is a different parameter: radius
r gives diameter 2r−1 blocks and downloads (2r−1)³ blocks (r = 1 → 1, 2 → 3³ = 27,
3 → 5³ = 125).
larger than the old one-cube crop. The runner takes the largest window step whose
cubes it can all load, so the soma stays centred with only sub-pixel rounding
(`window_centre_offset_px`), and it records `window_centered`, `window_cubes_xy` and
the per-step `soma_window_attempts` in the receipt for every neuron.

Title format: `sample | neuron | <panel> | src_dataset: <source dataset>`

| Panel | Title name | Source dataset (`src_dataset:`) | Display |
|-------|-----------|-------------------------|---------|
| SomaBlock PNG | `Soma Plot` | `0.65µm_raw` | High-resolution cube MIP, gamma 0.5, 30 µm bar |
| WideField PNG | `Context Plot` | `5µm_localserver` | 8,000 × 8,000 × 30 µm local-share crop, gain 2.0 / threshold 100, 500 µm bar |
| Sagittal PNG | `Sagittal Context Plot` | `DS00 N4` | Whole constant-i processed-DS plane, p1–p99.5 gray, auto 1/2/5 mm bar |
| Tri-plot PNG | panel titles as above | — | The three panels side by side; `legend=False` by default, `legend=True` adds one shared legend row. Pass `suffix="PairPlot"` for a two-panel figure when one source is unavailable |

Written filenames carry a per-figure timestamp before the role suffix, e.g.
`<sample>_<neuron>_<region>_<YYYYMMDD-HHMMSS>_Soma_Plot.png` (same for
`_Context_Plot`, `_Sagittal_Plot`, `_TriPlot` / `_PairPlot`). The role suffix stays
last, so suffix readers still match; the internal call-site suffixes stay
`SomaBlock`/`WideField`. Receipts record the exact timestamped paths.

Target marking in the sagittal panel: the same cyan box **and** cross are both centred
on the exact mapped position; the panel is the whole constant-i plane (the largest view
available, since a whole-plane view cannot centre the target), and only the drawn size
is enlarged for visibility at whole-hemisphere zoom. The dashed red/blue traces are the
two unresolved raw-SWC A/B candidate mappings and the box marks candidate A.

---

## Operator entry points and quick parameter tuning

The toolkit is wrapped by two operator scripts and the GUI; each accepts `--set
PATH=VALUE` (repeatable) so a run can be retuned without editing code. The resolved
settings are written into the run manifest, so every artifact can be tied to the values
that produced it.

| Task | Command |
|------|---------|
| One sample, one cohort | `python group_analysis/scripts/run_bulk_visual_sample.py --sample 252527 --cohort Atlas_neurons --context-sagittal --cached-only --output-root work/<run>` |
| Selected samples, both cohorts | `python group_analysis/scripts/bulk_visual_multi_monkey.py --selected-cohorts --samples 252527 --cohorts Atlas_neurons,Candidate_neurons --context-sagittal --cached-only --output-root work/<run>` |
| Interactive single neuron | `python -m Visual_toolkit.gui` (sample, neuron, FOV, output dir, plot actions) |

Add `--dry-run` to either script to print the resolved source, parameters and output
paths before anything is written. Examples:

```powershell
python group_analysis/scripts/run_bulk_visual_sample.py --sample 252527 --cohort Atlas_neurons `
  --context-sagittal --cached-only --dry-run --set widefield.gain=1.6 --set "marker.crosshair_px=80"
```

| `--set` path | Default | Effect |
|--------------|---------|--------|
| `marker.color` | `#00e5ee` | box + cross colour in all three panels |
| `marker.min_px` | `8` | smallest drawn box side (display px) |
| `marker.crosshair_px` | `60` | cross arm length from the target point |
| `marker.label_fontsize` / `marker.label_offset_px` | `12` / `22` | "Target Soma" label size and offset |
| `soma_window.cubes_xy` | `1.5` | Soma window **diameter** in blocks (≈351 µm); the legacy `grid radius` is a different parameter (its diameter = 2r−1 blocks) |
| `soma_window.max_cubes_xy` | `3` | cap for the soma window |
| `context.width_um` / `context.height_um` / `context.depth_um` | `8000` / `8000` / `30` | Context display crop |
| `widefield.gain` / `widefield.threshold` | `2.0` / `100` | Context display gain and green-signal threshold |
| `scale_bar_um.soma` / `scale_bar_um.context` | `30` / `500` | fixed scale bars |

Only existing scalar leaves can be overridden; an unknown path, a non-scalar leaf or an
invalid value stops the run with a clear error. Non-scalar settings (`figure` sizes,
`soma_window.cubes_xy_steps`, the scale-bar candidate list) are edited in
`REVIEW_SETTINGS` directly.

---

## Data Flow Diagram

```
User Input (Sample/Neuron ID)
         │
         ▼
┌─────────────────────┐
│  Load & Auto-Fill   │ ───► IONData.getRawNeuronTreeByID()
│      Soma           │      └──► tree.root.x/y/z
└─────────────────────┘              │
         │                           │
         ▼                           ▼
┌─────────────────────┐     ┌─────────────────┐
│  Soma Coordinates   │◄────┘  Auto-filled    │
│    (X/Y/Z fields)   │       into GUI fields │
└─────────────────────┘
         │
         ├──► Soma Plot ───────► get_high_res_block()
         │                            └──► HTTP download
         │
         ├──► Wide-field Plot ─► get_low_res_widefield()
         │                            └──► SSH download
         │
         └──► Plot Both ───────► Sequential execution
```

---

## Common Customization Scenarios

### Scenario 1: Auto-load a different default neuron

Change lines 87-88:
```python
self.sample_id = tk.StringVar(value="192106")      # Different sample
self.neuron_id = tk.StringVar(value="001.swc")     # Different neuron
```

### Scenario 2: Change default FOV dimensions

Change lines 93-95:
```python
self.width_um = tk.StringVar(value="10000")        # Wider FOV
self.height_um = tk.StringVar(value="10000")       # Taller FOV
self.depth_um = tk.StringVar(value="60")           # More slices
```

### Scenario 3: Add keyboard shortcuts

Add to `__init__` after `build_full_gui()`:
```python
self.root.bind('<Return>', lambda e: self.load_and_autofill())
self.root.bind('<F5>', lambda e: self.run_highres())
```

### Scenario 4: Change progress bar style

Modify line 225:
```python
self.progress = ttk.Progressbar(status_frame, mode='indeterminate', 
                                length=400, style='Custom.Horizontal.TProgressbar')
```

---

## Troubleshooting Tips

1. **"Could not import required modules"**
   - Check that `Visual_toolkit.py` and `IONData` are in the Python path
   - Verify `neurovis_path` in `Visual_toolkit.py` points to correct location

2. **SSH Connection Failures**
   - Check SSH credentials in `Visual_toolkit.py` (lines 95-99)
   - Verify network access to `SSH_HOST:SSH_PORT`

3. **HTTP Download Failures**
   - Check `HTTP_HOST` and `HTTP_PATH` in `Visual_toolkit.py` (lines 89-90)
   - Verify sample ID exists on server

4. **GUI Freezes**
   - All long operations should use the `threading.Thread` pattern
   - Don't run blocking code in the main GUI thread

---

## Summary

The Visual Toolkit GUI follows a clean separation:
- **GUI Layer** (`Visual_toolkit_gui.py`): Handles user interaction, threading, status updates
- **Backend Layer** (`Visual_toolkit.py`): Handles data acquisition, processing, export

Key customization points:
1. Default values in `__init__()`
2. Layout in `build_full_gui()`
3. Action handlers (`load_and_autofill`, `run_highres`, etc.)
4. Output directory logic in `_generate_default_path()`
