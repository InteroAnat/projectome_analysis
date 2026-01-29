# Visual Toolkit GUI - Learning Guide

## Overview

The Visual Toolkit GUI is a Tkinter-based interface for acquiring and visualizing macaque brain data at both high (0.65µm) and low (5.0µm) resolutions. It provides an intuitive way to download 3D brain blocks, generate plots, and overlay neuron traces (SWC) on anatomical context.

---

## File Structure

```
main_scripts/
├── Visual_toolkit_gui.py      # Main GUI application (THIS GUIDE)
├── Visual_toolkit.py          # Core backend logic
└── subsidary_functions/
    └── utilities.py           # Helper functions
```

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
