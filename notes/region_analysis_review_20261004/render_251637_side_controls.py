"""Same-sample native control locators anchored by agreeing Henry/atlas labels."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "case_112_114_20261004"
sys.path.insert(0, str(ROOT / "main_scripts"))
from Visual_toolkit import Visual_toolkit, RESOLUTION_LOW
from swc_validation import parse_swc

def main():
    annotations = json.loads((OUT / "annotation_and_identity_readback.json").read_text(encoding="utf-8"))
    # 111 is an immediately neighboring R annotation; 422 is a retained L case.
    selected = [{"NeuronID": "111.swc", "manual_area": "R-IDD5", "side": "R"},
                {"NeuronID": "422.swc", "manual_area": "L-IDD5", "side": "L"}]
    report = {"status": "rendering_same_sample_controls", "started_at": datetime.now().astimezone().isoformat(), "controls": []}
    toolkit = Visual_toolkit("251637", cache_dir=str(OUT / "control_cache"), low_res_slice_cache_limit=1)
    for rec in selected:
        nid = rec["NeuronID"]
        url = f"http://10.10.31.31/swc/newswc/Monkey/251637/swc_raw/{nid}"
        response = requests.get(url, timeout=(5,25)); response.raise_for_status()
        nodes = parse_swc(response.text, source=url)
        root = next(n for n in nodes if n[6] == -1)
        soma = np.array(root[2:5])
        (OUT / ("control_raw_" + nid)).write_bytes(response.content)
        z = int(soma[2]/RESOLUTION_LOW[2])
        source = toolkit._get_low_res_slice_path(z)
        if not source: raise FileNotFoundError("Own control section unavailable")
        full = toolkit._read_cached_section(source)
        small = full[::8,::8]
        vmax = np.percentile(small[small>0],99.5)
        fig,ax = plt.subplots(figsize=(9,7))
        ax.imshow(small,cmap="gray",origin="upper",vmin=0,vmax=vmax)
        ax.scatter(soma[0]/RESOLUTION_LOW[0]/8,soma[1]/RESOLUTION_LOW[1]/8,marker="+",color="cyan",s=150)
        ax.set_title(f"251637/{nid}: comparison control\nHenry {rec['manual_area']} / atlas {rec['side']}; native anatomical orientation unverified")
        ax.set_xlabel("Native X increases to viewer-right")
        fig.tight_layout()
        path=OUT / f"control_{nid[:-4]}_native_locator.png"
        fig.savefig(path,dpi=140);plt.close(fig)
        report["controls"].append({**rec,"native_root_xyz_um":soma.tolist(),"native_section_index":z,
            "source_path":str(source),"source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "raw_sha256":hashlib.sha256(response.content).hexdigest(),"figure":path.relative_to(ROOT).as_posix(),
            "figure_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        del full
    toolkit.close()
    report["status"]="rendered_pending_actual_inspection"
    (OUT / "same_sample_native_controls.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"controls":len(report["controls"])}))

if __name__ == "__main__":
    main()
