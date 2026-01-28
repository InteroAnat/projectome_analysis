"""
_toolkit_gui.py - GUI for NeuroVis Visual Toolkit

Version: 1.1.0 (Integrated Loading)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Import the core toolkit
try:
    from Visual_toolkit import Visual_toolkit
    import IONData as IT
except ImportError as e:
    # Fallback for UI testing if modules aren't present
    print(f"WARNING: Could not import required modules: {e}") 
    # sys.exit(1) # Commented out to allow GUI preview

class NeuroVisGUI:
    """Complete GUI for NeuroVis Visual Toolkit with integrated loading."""
    
    def __init__(self, root):
        """Initialize GUI window and widgets."""
        self.root = root
        self.root.title("NeuroVis Visual Toolkit v1.1")
        self.root.geometry("700x750")
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(False, False)
        
        # Processing variables
        self.sample_id = tk.StringVar(value="251637")
        self.neuron_id = tk.StringVar(value="003.swc")
        self.x_coord = tk.StringVar(value="18000")
        self.y_coord = tk.StringVar(value="18000")
        self.z_coord = tk.StringVar(value="1000")
        self.grid_radius = tk.IntVar(value=2)
        self.width_um = tk.StringVar(value="8000")
        self.height_um = tk.StringVar(value="8000")
        self.depth_um = tk.StringVar(value="30")
        self.status_text = tk.StringVar(value="Ready - Load neuron to auto-fill soma coordinates")
        
        # Toolkit instances
        self.toolkit = None
        self.ion = None
        
        self.build_full_gui()
    
    def build_full_gui(self):
        """Construct all GUI elements including action buttons."""
        # Title
        title = tk.Label(self.root, text="🧠 NeuroVis Visual Toolkit", 
                        font=('Helvetica', 18, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title.pack(pady=15)
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Sample Configuration Section (UPDATED) ===
        sample_group = ttk.LabelFrame(main_frame, text="📁 Sample Configuration", padding="10")
        sample_group.pack(fill=tk.X, pady=8)
        
        # Column 0 & 1: Inputs
        ttk.Label(sample_group, text="Sample ID:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        sid_entry = ttk.Entry(sample_group, textvariable=self.sample_id, width=20, font=('Arial', 10))
        sid_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(sample_group, text="Neuron ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        nid_entry = ttk.Entry(sample_group, textvariable=self.neuron_id, width=20, font=('Arial', 10))
        nid_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Column 2: Integrated Button (Spans 2 rows)
        # We create a specific style for this action button to make it distinct
        action_btn_style = ttk.Style()
        action_btn_style.configure('Action.TButton', font=('Arial', 10, 'bold'))
        
        load_btn = ttk.Button(sample_group, text="Load & Auto-Fill 📍", 
                             command=self.load_and_autofill, style='Action.TButton', width=20)
        load_btn.grid(row=0, column=2, rowspan=2, sticky=tk.NS, padx=15, pady=5)
        
        # === Soma Coordinates Section ===
        coord_group = ttk.LabelFrame(main_frame, text="📍 Soma Coordinates (µm)", padding="10")
        coord_group.pack(fill=tk.X, pady=8)
        
        # X coordinate
        ttk.Label(coord_group, text="X:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(coord_group, textvariable=self.x_coord, width=15, font=('Arial', 10)).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(coord_group, text="(horizontal)").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Y coordinate
        ttk.Label(coord_group, text="Y:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(coord_group, textvariable=self.y_coord, width=15, font=('Arial', 10)).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(coord_group, text="(vertical)").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Z coordinate
        ttk.Label(coord_group, text="Z:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(coord_group, textvariable=self.z_coord, width=15, font=('Arial', 10)).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(coord_group, text="(depth)").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # === High-Resolution Parameters ===
        highres_group = ttk.LabelFrame(main_frame, text="🔍 High-Res Soma Block (0.65 µm/pixel)", padding="10")
        highres_group.pack(fill=tk.X, pady=8)
        
        ttk.Label(highres_group, text="Grid Radius:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(highres_group, textvariable=self.grid_radius, values=[1, 2, 3], 
                    state="readonly", width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(highres_group, text="  blocks (1=single, 2=3x3x3, 3=5x5x5)").grid(
            row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # === Low-Resolution Parameters ===
        lowres_group = ttk.LabelFrame(main_frame, text="🌐 Low-Res Wide Field (5.0 µm/pixel)", padding="10")
        lowres_group.pack(fill=tk.X, pady=8)
        
        # Width
        ttk.Label(lowres_group, text="Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(lowres_group, textvariable=self.width_um, width=12, font=('Arial', 10)).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(lowres_group, text="µm").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Height
        ttk.Label(lowres_group, text="Height:").grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(lowres_group, textvariable=self.height_um, width=12, font=('Arial', 10)).grid(
            row=0, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(lowres_group, text="µm").grid(row=0, column=5, sticky=tk.W, padx=5, pady=5)
        
        # Depth
        ttk.Label(lowres_group, text="Depth:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(lowres_group, textvariable=self.depth_um, width=12, font=('Arial', 10)).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(lowres_group, text="µm (Z-thickness)").grid(row=1, column=2, columnspan=4, sticky=tk.W, padx=5, pady=5)
        
        # === ACTION BUTTONS SECTION ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)

        # Create extra-large button style
        style = ttk.Style()
        style.configure('XLarge.TButton', 
                        font=('Arial', 11, 'bold'),
                        padding=(15, 15),
                        foreground='#2c3e50',
                        background='#e0e0e0')

        ttk.Button(button_frame, text="Soma Plot", command=self.run_highres,
                style='XLarge.TButton', width=20).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Wide-field Plot", command=self.run_lowres,
                style='XLarge.TButton', width=20).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Plot Both", command=self.run_both,
                style='XLarge.TButton', width=20).pack(side=tk.LEFT, padx=5)

        # === STATUS BAR ===
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_label = tk.Label(status_frame, textvariable=self.status_text, font=('Helvetica', 10), fg='#2c3e50')
        status_label.pack(pady=5)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=400)
        
        info_label = tk.Label(status_frame, text="Ready - Enter data and click process to begin", 
                             font=('Helvetica', 8), fg='gray')
        info_label.pack(pady=2)
    
    def validate_inputs(self):
        """Validate all user inputs before processing."""
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
    
    def update_status(self, message):
        """Thread-safe status bar update."""
        self.status_text.set(message)
        self.root.update_idletasks()

    # === NEW INTEGRATED METHOD ===
    def load_and_autofill(self):
        """Loads the neuron and automatically fills soma coordinates in one action."""
        def worker():
            try:
                self.progress.pack(pady=5)
                self.progress.start()
                
                # Step 1: Initialize
                self.update_status("🗂️ Connecting to database...")
                self.toolkit = Visual_toolkit(self.sample_id.get())
                self.ion = IT.IONData()
                
                # Step 2: Load Tree
                self.update_status(f"🌲 Loading neuron {self.neuron_id.get()}...")
                tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
                
                if not tree:
                    self.update_status("❌ Failed to find neuron.")
                    messagebox.showwarning("Not Found", "Could not load neuron tree.\nPlease verify Sample ID and Neuron ID.")
                    return

                # Step 3: Extract Coordinates
                self.update_status("📍 Extracting soma data...")
                try:
                    soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
                    
                    # Step 4: Update GUI
                    self.x_coord.set(str(int(soma_xyz[0])))
                    self.y_coord.set(str(int(soma_xyz[1])))
                    self.z_coord.set(str(int(soma_xyz[2])))
                    
                    self.update_status(f"✅ Loaded & Filled: X={int(soma_xyz[0])}, Y={int(soma_xyz[1])}, Z={int(soma_xyz[2])}")
                    messagebox.showinfo("Success", f"Neuron Loaded Successfully!\n\nSoma Coordinates Auto-filled:\nX: {soma_xyz[0]:.1f}\nY: {soma_xyz[1]:.1f}\nZ: {soma_xyz[2]:.1f}")
                
                except AttributeError:
                    self.update_status("⚠️ Neuron loaded, but soma root not found.")
                    messagebox.showwarning("Structure Error", "Neuron tree loaded but root coordinates could not be extracted.")

            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
                self.update_status(f"❌ Error: {e}")
            finally:
                if self.toolkit:
                    self.toolkit.close()
                self.progress.stop()
                self.progress.pack_forget()
        
        threading.Thread(target=worker, daemon=True).start()

    def run_highres(self):
        """Execute high-resolution processing."""
        if not self.validate_inputs():
            return
        
        def process():
            try:
                self.progress.pack(pady=5)
                self.progress.start()
                
                self.update_status("🔧 Initializing toolkit...")
                self.toolkit = Visual_toolkit(self.sample_id.get())
                
                soma_xyz = [float(self.x_coord.get()), float(self.y_coord.get()), float(self.z_coord.get())]
                
                self.update_status("📥 Downloading high-res blocks...")
                high_res_volume, high_res_origin, high_res_resolution = self.toolkit.get_high_res_block(
                    soma_xyz, grid_radius=self.grid_radius.get()
                )
                
                self.update_status("💾 Exporting high-res data...")
                self.toolkit.export_data(high_res_volume, high_res_origin, high_res_resolution,
                                        self.neuron_id.get(), suffix="SomaBlock")
                
                self.update_status("📊 Generating plot...")
                self.toolkit.plot_soma_block(high_res_volume, high_res_origin, high_res_resolution,
                                            soma_xyz, self.neuron_id.get())
                
                self.update_status("✅ High-res processing complete!")
                messagebox.showinfo("Success", "High-resolution processing completed!\n\nResults saved to output directory.")
            
            except Exception as e:
                messagebox.showerror("Processing Error", f"Error: {str(e)}")
                self.update_status(f"❌ Error: {str(e)}")
            finally:
                self.progress.stop()
                self.progress.pack_forget()
                if self.toolkit:
                    self.toolkit.close()
        
        threading.Thread(target=process, daemon=True).start()
    
    def run_lowres(self):
        """Execute low-resolution processing."""
        if not self.validate_inputs():
            return
        
        def process():
            try:
                self.progress.pack(pady=5)
                self.progress.start()
                
                self.update_status("🔧 Initializing toolkit...")
                self.toolkit = Visual_toolkit(self.sample_id.get())
                self.ion = IT.IONData()
                
                # Load neuron tree for overlay
                self.update_status("🗂️ Loading neuron tree...")
                tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
                if not tree:
                    raise ValueError("Could not load neuron tree")
                
                soma_xyz = [float(self.x_coord.get()), float(self.y_coord.get()), float(self.z_coord.get())]
                
                self.update_status("📥 Downloading low-res slices...")
                low_res_volume, low_res_origin, low_res_resolution = self.toolkit.get_low_res_widefield(
                    soma_xyz,
                    width_um=int(self.width_um.get()),
                    height_um=int(self.height_um.get()),
                    depth_um=int(self.depth_um.get())
                )
                
                self.update_status("💾 Exporting low-res data...")
                self.toolkit.export_data(low_res_volume, low_res_origin, low_res_resolution,
                                        self.neuron_id.get(), suffix="WideField")
                
                self.update_status("📊 Generating composite plot...")
                self.toolkit.plot_widefield_context(
                    low_res_volume, low_res_origin, low_res_resolution,
                    soma_xyz, self.neuron_id.get(), bg_intensity=2.0, swc_tree=tree
                )
                
                self.update_status("✅ Low-res processing complete!")
                messagebox.showinfo("Success", "Low-resolution processing completed!\n\nResults saved to output directory.")
            
            except Exception as e:
                messagebox.showerror("Processing Error", f"Error: {str(e)}")
                self.update_status(f"❌ Error: {str(e)}")
            finally:
                self.progress.stop()
                self.progress.pack_forget()
                if self.toolkit:
                    self.toolkit.close()
        
        threading.Thread(target=process, daemon=True).start()
    
    def run_both(self):
        """Execute both high-res and low-res processing sequentially."""
        if not self.validate_inputs():
            return
        
        if messagebox.askyesno("Confirm", "Process both high and low resolution?\n\nThis may take several minutes."):
            def process():
                try:
                    self.progress.pack(pady=5)
                    self.progress.start()
                    
                    self.update_status("🔄 Starting both processes...")
                    
                    soma_xyz = [float(self.x_coord.get()), float(self.y_coord.get()), float(self.z_coord.get())]
                    
                    # Phase 1: High-res
                    self.toolkit = Visual_toolkit(self.sample_id.get())
                    self.ion = IT.IONData()
                    
                    self.update_status("📥 Phase 1/2: High-res blocks...")
                    high_res_volume, high_res_origin, high_res_resolution = self.toolkit.get_high_res_block(
                        soma_xyz, grid_radius=self.grid_radius.get()
                    )
                    self.toolkit.export_data(high_res_volume, high_res_origin, high_res_resolution,
                                            self.neuron_id.get(), suffix="SomaBlock")
                    self.toolkit.plot_soma_block(high_res_volume, high_res_origin, high_res_resolution,
                                                soma_xyz, self.neuron_id.get())
                    
                    self.toolkit.close()
                    
                    # Phase 2: Low-res
                    tree = self.ion.getRawNeuronTreeByID(self.sample_id.get(), self.neuron_id.get())
                    self.toolkit = Visual_toolkit(self.sample_id.get())
                    
                    self.update_status("📥 Phase 2/2: Low-res slices...")
                    low_res_volume, low_res_origin, low_res_resolution = self.toolkit.get_low_res_widefield(
                        soma_xyz,
                        width_um=int(self.width_um.get()),
                        height_um=int(self.height_um.get()),
                        depth_um=int(self.depth_um.get())
                    )
                    self.toolkit.export_data(low_res_volume, low_res_origin, low_res_resolution,
                                            self.neuron_id.get(), suffix="WideField")
                    self.toolkit.plot_widefield_context(
                        low_res_volume, low_res_origin, low_res_resolution,
                        soma_xyz, self.neuron_id.get(), bg_intensity=2.0, swc_tree=tree
                    )
                    
                    self.update_status("✅ All processing complete!")
                    messagebox.showinfo("Success", "All processing completed!\n\nBoth high-res and low-res results saved.")
                
                except Exception as e:
                    messagebox.showerror("Processing Error", f"Error: {str(e)}")
                    self.update_status(f"❌ Error: {str(e)}")
                finally:
                    self.progress.stop()
                    self.progress.pack_forget()
                    if self.toolkit:
                        self.toolkit.close()
            
            threading.Thread(target=process, daemon=True).start()

# ==================== MAIN LAUNCHER ====================
def main():
    """Launch the GUI application."""
    print("=" * 50)
    print("NeuroVis Visual Toolkit - GUI Mode")
    print("=" * 50)
    print("Initializing GUI...")
    
    root = tk.Tk()
    app = NeuroVisGUI(root)
    
    print("GUI loaded successfully!")
    print("-" * 50)
    print("Instructions:")
    print("1. Enter Sample ID and Neuron ID")
    print("2. Click 'Load & Auto-Fill' to populate soma coordinates")
    print("3. Adjust grid/resolution parameters if needed")
    print("4. Click any Plot button to run")
    print("-" * 50)
    
    root.mainloop()

if __name__ == "__main__":
    main()