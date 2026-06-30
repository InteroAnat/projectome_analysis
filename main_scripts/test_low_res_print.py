"""
Smoke test: low-res widefield NIfTI + plot via UNC share.

Usage (from main_scripts/):
    python test_low_res_print.py
    python test_low_res_print.py 007.swc
    python test_low_res_print.py 006.swc R-IAL
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(r"D:\projectome_analysis\neuron-vis\neuronVis"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Visual_toolkit import Visual_toolkit
import IONData as IT

SAMPLE_ID = "251637"
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    "test_low_res_print",
)


def main():
    parser = argparse.ArgumentParser(description="Low-res widefield print test")
    parser.add_argument("neuron_id", nargs="?", default="007.swc")
    parser.add_argument("soma_region", nargs="?", default="R-IAL")
    parser.add_argument("--width", type=int, default=8000)
    parser.add_argument("--height", type=int, default=8000)
    parser.add_argument("--depth", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ion = IT.IONData()
    toolkit = Visual_toolkit(SAMPLE_ID)

    print(f"[test] sample={SAMPLE_ID} neuron={args.neuron_id} region={args.soma_region}")
    print(f"[test] output -> {OUTPUT_DIR}")

    tree = ion.getRawNeuronTreeByID(SAMPLE_ID, args.neuron_id)
    if not tree:
        raise SystemExit(f"SWC not found: {args.neuron_id}")

    soma_xyz = [float(tree.root.x), float(tree.root.y), float(tree.root.z)]
    print(f"[test] soma XYZ (fMOST µm): {[round(v, 1) for v in soma_xyz]}")

    vol, origin, res = toolkit.get_low_res_widefield(
        soma_xyz,
        width_um=args.width,
        height_um=args.height,
        depth_um=args.depth,
    )

    vmax = float(vol.max())
    nz = int(np.count_nonzero(vol))
    print(f"[test] volume shape={vol.shape} max={vmax:.0f} nonzero={nz} origin={origin}")

    if vmax <= 0:
        toolkit.close()
        raise SystemExit("[FAIL] all-zero volume — check UNC share path / z-slices")

    toolkit.export_data(
        vol, origin, res, args.neuron_id,
        suffix="WideField",
        soma_region=args.soma_region,
        soma_coords=soma_xyz,
        output_dir=OUTPUT_DIR,
    )
    toolkit.plot_widefield_context(
        vol, origin, res, soma_xyz, args.neuron_id,
        bg_intensity=2.0,
        swc_tree=tree,
        soma_region=args.soma_region,
        output_dir=OUTPUT_DIR,
    )

    toolkit.close()
    print("[PASS] low-res widefield test OK")


if __name__ == "__main__":
    main()
