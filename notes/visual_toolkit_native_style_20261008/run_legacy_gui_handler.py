"""Run unchanged committed GUI Plot Both handler with nonvisual UI adapters.

No GUI window is controlled. Only variables, status/progress and message boxes
are adapted; acquisition, export and plot calls execute the committed code.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import threading
import types

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'group_analysis/legacy_gui_check_252384_003_20261008'
OUT.mkdir(exist_ok=False)
os.chdir(ROOT / 'main_scripts')
sys.path.insert(0, str(ROOT / 'main_scripts'))
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

def blob(path):
    return subprocess.check_output(['git', '-c', 'safe.directory=D:/projectome_analysis',
                                    'show', 'HEAD:' + path], cwd=ROOT)

def sha(data):
    return hashlib.sha256(data).hexdigest()

backend = blob('main_scripts/Visual_toolkit.py')
gui_source = blob('main_scripts/Visual_toolkit_gui.py')
assert gui_source == (ROOT / 'main_scripts/Visual_toolkit_gui.py').read_bytes()
head = subprocess.check_output(['git', '-c', 'safe.directory=D:/projectome_analysis',
                                'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
module = types.ModuleType('Visual_toolkit')
module.__file__ = str(ROOT / 'main_scripts/Visual_toolkit.py')
sys.modules['Visual_toolkit'] = module
exec(compile(backend, 'committed_legacy/Visual_toolkit.py', 'exec'), module.__dict__)
gui = types.ModuleType('legacy_gui')
gui.__file__ = str(ROOT / 'main_scripts/Visual_toolkit_gui.py')
exec(compile(gui_source, 'committed_legacy/Visual_toolkit_gui.py', 'exec'), gui.__dict__)

class Variable:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value

class Progress:
    def pack(self, **kwargs): pass
    def start(self): pass
    def stop(self): pass
    def pack_forget(self): pass

messages = []
errors = []
gui.messagebox.askyesno = lambda *args, **kwargs: True
gui.messagebox.showinfo = lambda *args, **kwargs: messages.append(list(args))
gui.messagebox.showerror = lambda *args, **kwargs: errors.append(list(args))
gui.messagebox.showwarning = lambda *args, **kwargs: errors.append(list(args))
# Isolate every cache write without changing the legacy class or its methods.
gui.Visual_toolkit = lambda sample: module.Visual_toolkit(sample, cache_dir=str(OUT / 'cache'))

raw = ROOT / 'group_analysis/visual_review_20261002/swc_raw/252384/003.swc'
from swc_validation import parse_swc
nodes = parse_swc(raw.read_text(), source=str(raw))
root = next(row for row in nodes if row[6] == -1)
native = list(root[2:5])
app = gui.NeuroVisGUI.__new__(gui.NeuroVisGUI)
app.sample_id, app.neuron_id = Variable('252384'), Variable('003.swc')
# Preserve the original GUI's int() auto-fill behavior for this comparison.
app.x_coord, app.y_coord, app.z_coord = [Variable(str(int(v))) for v in native]
app.grid_radius = Variable(1)
app.width_um, app.height_um, app.depth_um = [Variable(str(v)) for v in (8000, 8000, 30)]
app.output_dir = Variable(str(OUT / 'plots_and_volumes'))
app.progress, app.toolkit, app.ion = Progress(), None, None
app.update_status = lambda text: print(text, flush=True)
manifest = {'status':'running', 'head':head, 'legacy_backend_sha256':sha(backend),
            'legacy_gui_sha256':sha(gui_source), 'gui_source_unchanged':True,
            'execution':'Unmodified GUI run_both handler, nonvisual variables/progress/messagebox adapters; no GUI window interaction',
            'gui_control_failure':'Windows computer-use helper sandbox setup failed',
            'only_backend_adapter':'Constructor factory supplies separate cache_dir',
            'sample':'252384', 'neuron':'003.swc', 'raw_swc':str(raw),
            'raw_swc_sha256':sha(raw.read_bytes()), 'native_soma_xyz_um':native,
            'gui_autofilled_soma_xyz_um':[int(v) for v in native],
            'grid_radius':1, 'requested_fov_xyz_um':[8000,8000,30],
            'legacy_lowres_expected_sections':list(range(12459,12470)),
            'legacy_nifti_geometry':'Unchanged legacy affine; not an accepted calibrated export',
            'anatomical_acceptance':False, 'original_outputs_modified':False}
report = OUT / 'legacy_gui_execution.json'
report.write_text(json.dumps(manifest, indent=2)+'\n')
before = set(threading.enumerate())
app.run_both()
for worker in set(threading.enumerate()) - before:
    worker.join(timeout=240)
    if worker.is_alive():
        errors.append(['Timeout', 'Legacy GUI worker did not finish within 240 seconds'])
files = []
for path in sorted((OUT / 'plots_and_volumes').glob('*')):
    files.append({'path':str(path.relative_to(OUT)), 'size_bytes':path.stat().st_size,
                  'sha256':sha(path.read_bytes())})
manifest.update(status='finished_pending_visual_inspection' if not errors else 'failed',
                messages=messages, errors=errors, outputs=files,
                raw_swc_unchanged=sha(raw.read_bytes()) == manifest['raw_swc_sha256'])
report.write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({'status':manifest['status'], 'outputs':len(files), 'errors':errors}), flush=True)
raise SystemExit(bool(errors))
