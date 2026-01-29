"""
Visual_toolkit_gui.py - GUI for NeuroVis Visual Toolkit

Version: 1.5.0 (Auto-Load Default Neuron)

DESCRIPTION:
    A complete Tkinter-based GUI for the Visual_toolkit. Provides an intuitive
    interface for acquiring and visualizing macaque brain data at both high 
    (0.65µm) and low (5.0µm) resolutions. Features auto soma coordinate loading,
    threaded processing, and progress indicators.

KEY FEATURES:
    - Auto-Load on Startup: Automatically loads default neuron (003.swc) when GUI opens
    - Auto-Fill Soma: Automatically extract soma coordinates from neuron SWC files
    - Interactive Parameters: Adjust grid radius, FOV dimensions via GUI
    - Threaded Processing: Non-blocking downloads with progress indicators
    - Flexible Output: Default path matches Visual_toolkit, or choose custom directory
    - Three Actions: Soma Plot (high-res), Wide-field Plot (low-res), or Both

USAGE NOTES:
    1. Launch the GUI:
        python Visual_toolkit_gui.py

    2. Auto-Load Feature:
        - The GUI automatically loads the default neuron (003.swc) on startup
        - Soma coordinates are auto-filled after ~0.5 seconds
        - Status bar shows loading progress
        - No popup on successful auto-load (silent operation)

    3. Basic Workflow:
        - Sample ID (e.g., '251637') and Neuron ID (e.g., '003.swc') are pre-filled
        - If auto-load succeeded, coordinates are already populated
        - Click 'Load & Auto-Fill Soma' to manually reload or load different neuron
        - Adjust parameters if needed (Grid Radius, Width/Height/Depth)
        - Click action button: 'Soma Plot', 'Wide-field Plot', or 'Plot Both'

    4. Output Directory:
        - Default: project_root/resource/segmented_cubes/sample_id
        - Use 'Browse' to select custom directory
        - Note: Changing Sample ID auto-updates path unless manually set

    5. Processing:
        - High-res: Downloads 3D blocks around soma, generates middle slice plot
        - Low-res: Downloads wide field slices, generates MIP with SWC overlay
        - Both: Sequential processing, closes/reopens connection between phases

CONFIGURATION:
    - Window size: 700x700 pixels, non-resizable
    - Default values: Sample='251637', Neuron='003.swc', Grid Radius=1, FOV=8000x8000x30 µm
    - Progress bar shows during active processing
    - Status bar provides step-by-step feedback

UPDATE NOTES (v1.5.0):
    - Added auto-load feature: default neuron loads automatically on startup
    - Added auto_load_default_neuron() method for startup initialization
    - Modified load_and_autofill() with silent_success parameter for auto-load
    - Auto-load runs after 500ms delay to ensure GUI is fully rendered

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

# Import the core toolkit
try:
    from Visual_toolkit import Visual_toolkit
    import IONData as IT
except ImportError as e:
    print(f"WARNING: Could not import required modules: {e}") 

class NeuroVisGUI:
    """Complete GUI for NeuroVis Visual Toolkit."""
    
    def __init__(self, root):
        """Initialize GUI window and widgets."""
        self.root = root
        self.root.title(" Visual_Toolkit_gui v1.4")
        self.root.geometry("700x700")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(False, False)
        
        # === 1. Processing Variables ===
        self.sample_id = tk.StringVar(value="251637")
        self.neuron_id = tk.StringVar(value="003.swc")
        self.x_coord = tk.StringVar(value="0")
        self.y_coord = tk.StringVar(value="0")
        self.z_coord = tk.StringVar(value="0")
        self.grid_radius = tk.IntVar(value=1)
        self.width_um = tk.StringVar(value="8000")
        self.height_um = tk.StringVar(value="8000")
        self.depth_um = tk.StringVar(value="30")
        self.status_text = tk.StringVar(value="Ready")
        
        # === 2. Default Output Directory Logic ===
        # Matches Visual_toolkit.py: project_root/resource/segmented_cubes/sample_id
        self.project_root = os.path.dirname(os.getcwd()) 
        default_path = self._generate_default_path(self.sample_id.get())
        self.output_dir = tk.StringVar(value=default_path)

        # Flag to track if user manually changed the path
        self.manual_path_set = False

        # Add listener: Update output path when Sample ID changes (if not manually set)
        self.sample_id.trace_add("write", self._on_sample_id_change)
        
        # Toolkit instances
        self.toolkit = None
        self.ion = None
        
        self.build_full_gui()
        
        # Auto-load the default neuron after GUI is fully initialized
        self.root.after(500, self.auto_load_default_neuron)
    
    def _generate_default_path(self, sample_id):
        """Generates the default path string based on Sample ID."""
        return os.path.join(self.project_root, 'resource', 'segmented_cubes', sample_id)

    def _on_sample_id_change(self, *args):
        """Updates output directory automatically when Sample ID changes."""
        if not self.manual_path_set:
            new_path = self._generate_default_path(self.sample_id.get())
            self.output_dir.set(new_path)

    def build_full_gui(self):
        """Construct all GUI elements."""
        # Title
        title = tk.Label(self.root, text="🧠 NeuroVis Visual Toolkit", 
                        font=('Helvetica', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title.pack(pady=15)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Sample Configuration ===
        sample_group = ttk.LabelFrame(main_frame, text="📁 Sample Configuration", padding="10")
        sample_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(sample_group, text="Sample ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(sample_group, textvariable=self.sample_id, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(sample_group, text="Neuron ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(sample_group, textvariable=self.neuron_id, width=20).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Integrated Button
        action_btn_style = ttk.Style()
        action_btn_style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        
        ttk.Button(sample_group, text="Load & Auto-Fill Soma", 
                  command=self.load_and_autofill, style='Action.TButton', width=25).grid(
                      row=0, column=2, rowspan=2, sticky=tk.NS, padx=15, pady=5)
        
        # === Soma Coordinates ===
        coord_group = ttk.LabelFrame(main_frame, text="📍 Soma Coordinates (µm)", padding="10")
        coord_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(coord_group, text="X:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(coord_group, textvariable=self.x_coord, width=15).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(coord_group, text="(horizontal)").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(coord_group, text="Y:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(coord_group, textvariable=self.y_coord, width=15).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(coord_group, text="(vertical)").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        ttk.Label(coord_group, text="Z:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(coord_group, textvariable=self.z_coord, width=15).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Label(coord_group, text="(depth)").grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # === High-Resolution ===
        highres_group = ttk.LabelFrame(main_frame, text="🔍 High-Res Soma Block", padding="10")
        highres_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(highres_group, text="Grid Radius:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Combobox(highres_group, textvariable=self.grid_radius, values=[1, 2, 3], 
                    state="readonly", width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # === Low-Resolution ===
        lowres_group = ttk.LabelFrame(main_frame, text="🌐 Low-Res Wide Field", padding="10")
        lowres_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(lowres_group, text="Width (µm):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(lowres_group, textvariable=self.width_um, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(lowres_group, text="Height (µm):").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Entry(lowres_group, textvariable=self.height_um, width=10).grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(lowres_group, text="Depth (µm) (3µm per slice):").grid(row=0, column=4, sticky=tk.W, padx=5)
        ttk.Entry(lowres_group, textvariable=self.depth_um, width=10).grid(row=0, column=5, sticky=tk.W, padx=5)

        # === Output Configuration ===
        output_group = ttk.LabelFrame(main_frame, text="💾 Output Configuration", padding="10")
        output_group.pack(fill=tk.X, pady=5)

        ttk.Label(output_group, text="Save to:").pack(side=tk.LEFT, padx=5)
        
        # Bind key press to detect manual edit
        out_entry = ttk.Entry(output_group, textvariable=self.output_dir, font=('Arial', 9))
        out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        out_entry.bind("<Key>", lambda e: setattr(self, 'manual_path_set', True))
        
        ttk.Button(output_group, text="Browse...", command=self.browse_folder).pack(side=tk.LEFT, padx=5)
        
        # === Action Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=15)

        style = ttk.Style()
        style.configure('XLarge.TButton', font=('Arial', 11, 'bold'), padding=(15, 15))

        ttk.Button(button_frame, text="Soma Plot", command=self.run_highres,
                style='XLarge.TButton', width=18).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Wide-field Plot", command=self.run_lowres,
                style='XLarge.TButton', width=18).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Plot Both", command=self.run_both,
                style='XLarge.TButton', width=18).pack(side=tk.LEFT, padx=5)

        # === Status ===
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(status_frame, textvariable=self.status_text, font=('Helvetica', 10), fg='#2c3e50').pack(pady=5)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=400)
    
    def validate_inputs(self):
        """Validate numeric inputs."""
        try:
            float(self.x_coord.get())
            float(self.y_coord.get())
            float(self.z_coord.get())
            int(self.width_um.get())
            int(self.height_um.get())
            int(self.depth_um.get())
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
        self.status_text.set(message)
        self.root.update_idletasks()

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
        def worker():
            try:
                self.progress.pack(pady=5)
                self.progress.start()
                self.update_status("🗂️ Connecting to database...")
                self.toolkit = Visual_toolkit(self.sample_id.get())
                self.ion = IT.IONData()
                
                self.update_status(f"🌲 Loading neuron {self.neuron_id.get()}...")
                tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
                
                if not tree:
                    messagebox.showwarning("Not Found", "Could not load neuron tree.")
                    return

                soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
                self.x_coord.set(str(int(soma_xyz[0])))
                self.y_coord.set(str(int(soma_xyz[1])))
                self.z_coord.set(str(int(soma_xyz[2])))
                
                self.update_status(f"✅ Loaded: {int(soma_xyz[0])}, {int(soma_xyz[1])}, {int(soma_xyz[2])}")
                if not silent_success:
                    messagebox.showinfo("Success", "Neuron Loaded & Coordinates Auto-filled!")
                
            except Exception as e:
                if silent_success:
                    self.update_status(f"⚠️ Auto-load failed: {e}")
                else:
                    messagebox.showerror("Error", f"An error occurred: {e}")
            finally:
                if self.toolkit: self.toolkit.close()
                self.progress.stop()
                self.progress.pack_forget()
        
        threading.Thread(target=worker, daemon=True).start()

    def run_highres(self):
        if not self.validate_inputs(): return
        out_path = self.prepare_output_dir()
        if not out_path: return
        
        def process():
            try:
                self.progress.pack(pady=5)
                self.progress.start()
                
                self.update_status("🔧 Initializing toolkit...")
                self.toolkit = Visual_toolkit(self.sample_id.get())
                soma_xyz = [float(self.x_coord.get()), float(self.y_coord.get()), float(self.z_coord.get())]
                
                self.update_status("📥 Downloading high-res blocks...")
                vol, origin, res = self.toolkit.get_high_res_block(
                    soma_xyz, grid_radius=self.grid_radius.get()
                )
                
                self.update_status("💾 Exporting high-res data...")
                self.toolkit.export_data(vol, origin, res, self.neuron_id.get(), 
                                       suffix="SomaBlock", output_dir=out_path)
                
                self.update_status("📊 Generating plot...")
                self.toolkit.plot_soma_block(vol, origin, res, soma_xyz, 
                                           self.neuron_id.get(), output_dir=out_path)
                
                self.update_status("✅ High-res processing complete!")
                messagebox.showinfo("Success", f"Files saved to:\n{out_path}")
            
            except Exception as e:
                messagebox.showerror("Processing Error", f"Error: {str(e)}")
            finally:
                self.progress.stop()
                self.progress.pack_forget()
                if self.toolkit: self.toolkit.close()
        
        threading.Thread(target=process, daemon=True).start()
    
    def run_lowres(self):
        if not self.validate_inputs(): return
        out_path = self.prepare_output_dir()
        if not out_path: return
        
        def process():
            try:
                self.progress.pack(pady=5)
                self.progress.start()
                
                self.update_status("🔧 Initializing toolkit...")
                self.toolkit = Visual_toolkit(self.sample_id.get())
                self.ion = IT.IONData()
                tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
                
                soma_xyz = [float(self.x_coord.get()), float(self.y_coord.get()), float(self.z_coord.get())]
                
                self.update_status("📥 Downloading low-res slices...")
                vol, origin, res = self.toolkit.get_low_res_widefield(
                    soma_xyz,
                    width_um=int(self.width_um.get()),
                    height_um=int(self.height_um.get()),
                    depth_um=int(self.depth_um.get())
                )
                
                self.update_status("💾 Exporting low-res data...")
                self.toolkit.export_data(vol, origin, res, self.neuron_id.get(), 
                                       suffix="WideField", output_dir=out_path)
                
                self.update_status("📊 Generating plot...")
                self.toolkit.plot_widefield_context(
                    vol, origin, res, soma_xyz, self.neuron_id.get(), 
                    bg_intensity=2.0, swc_tree=tree, output_dir=out_path
                )
                
                self.update_status("✅ Low-res processing complete!")
                messagebox.showinfo("Success", f"Files saved to:\n{out_path}")
            
            except Exception as e:
                messagebox.showerror("Processing Error", f"Error: {str(e)}")
            finally:
                self.progress.stop()
                self.progress.pack_forget()
                if self.toolkit: self.toolkit.close()
        
        threading.Thread(target=process, daemon=True).start()
    
    def run_both(self):
        if not self.validate_inputs(): return
        out_path = self.prepare_output_dir()
        if not out_path: return
        
        if messagebox.askyesno("Confirm", "Process both resolutions?"):
            def process():
                try:
                    self.progress.pack(pady=5)
                    self.progress.start()
                    
                    soma_xyz = [float(self.x_coord.get()), float(self.y_coord.get()), float(self.z_coord.get())]
                    
                    # Phase 1: High-res
                    self.update_status("Processing High-res...")
                    self.toolkit = Visual_toolkit(self.sample_id.get())
                    vol_h, origin_h, res_h = self.toolkit.get_high_res_block(
                        soma_xyz, grid_radius=self.grid_radius.get()
                    )
                    self.toolkit.export_data(vol_h, origin_h, res_h, self.neuron_id.get(), 
                                           suffix="SomaBlock", output_dir=out_path)
                    self.toolkit.plot_soma_block(vol_h, origin_h, res_h, soma_xyz, 
                                               self.neuron_id.get(), output_dir=out_path)
                    self.toolkit.close()
                    
                    # Phase 2: Low-res
                    self.update_status("Processing Low-res...")
                    self.ion = IT.IONData()
                    tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
                    self.toolkit = Visual_toolkit(self.sample_id.get())
                    
                    vol_l, origin_l, res_l = self.toolkit.get_low_res_widefield(
                        soma_xyz,
                        width_um=int(self.width_um.get()),
                        height_um=int(self.height_um.get()),
                        depth_um=int(self.depth_um.get())
                    )
                    self.toolkit.export_data(vol_l, origin_l, res_l, self.neuron_id.get(), 
                                           suffix="WideField", output_dir=out_path)
                    self.toolkit.plot_widefield_context(
                        vol_l, origin_l, res_l, soma_xyz, self.neuron_id.get(), 
                        bg_intensity=2.0, swc_tree=tree, output_dir=out_path
                    )
                    
                    self.update_status("✅ All processing complete!")
                    messagebox.showinfo("Success", f"All results saved to:\n{out_path}")
                
                except Exception as e:
                    messagebox.showerror("Processing Error", f"Error: {str(e)}")
                finally:
                    self.progress.stop()
                    self.progress.pack_forget()
                    if self.toolkit: self.toolkit.close()
            
            threading.Thread(target=process, daemon=True).start()

# ==================== MAIN LAUNCHER ====================
def main():
    root = tk.Tk()
    app = NeuroVisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()