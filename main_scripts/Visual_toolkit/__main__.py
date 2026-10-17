"""Run the original toolkit demo: python -m Visual_toolkit

Version: 1.9.1 (released 2867eef; package version)
Last updated: 2026-10-17
Change history: CHANGELOG.md (this folder)
"""
from .toolkit import Visual_toolkit

toolkit = Visual_toolkit("251637")
import IONData as IT

ion = IT.IONData()
NEURON = "003.swc"
tree = ion.getRawNeuronTreeByID("251637", NEURON)

if tree:
    try:
        soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
    except Exception:
        soma_xyz = tree.root.xyz
    print(f"Target Soma: {soma_xyz}")
    test_region = "DI"
    high_res_volume, high_res_origin, high_res_resolution = toolkit.get_high_res_block(
        soma_xyz, grid_radius=1)
    toolkit.plot_soma_block(
        high_res_volume, high_res_origin, high_res_resolution, soma_xyz, NEURON,
        soma_region=test_region)
    low_res_volume, low_res_origin, low_res_resolution = toolkit.get_low_res_widefield(
        soma_xyz, width_um=8000, height_um=8000, depth_um=30)
    toolkit.plot_widefield_context(
        low_res_volume, low_res_origin, low_res_resolution, soma_xyz, NEURON,
        swc_tree=tree, soma_region=test_region)

toolkit.close()
