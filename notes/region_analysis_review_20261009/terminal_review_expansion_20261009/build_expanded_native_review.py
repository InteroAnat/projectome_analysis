"""Six new local-only native-image review units; no biological acceptance."""
from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import nibabel as nib
import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = HERE.parent / "terminal_field_assessment_20261009"
helper_path = PRIOR / "assess_native_terminal_fields_current_rint.py"
spec = importlib.util.spec_from_file_location("prior_native_review", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
CASES = [
    ("252383::124.swc", 729, "source_ARM0"),
    ("252790::037.swc", 229, "source_ARM0"),
    ("252718::162.swc", 229, "source_ARM0_new_animal"),
    ("252385::056.swc", 549, "source_PrCO_new_animal"),
    ("252527::056.swc", 549, "source_PrCO_new_animal"),
    ("252714::099.swc", 228, "source_INS_new_animal"),
]


def build():
    output = HERE / "review_panels"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    summary = pd.read_csv(helper.SUMMARY, dtype=str, keep_default_na=False)
    coverage_path = PRIOR / "native_cube_pilot_current_rint/selected462_native_image_coverage.csv"
    coverage = pd.read_csv(coverage_path, dtype=str, keep_default_na=False).set_index("NeuronUID")
    if len(summary) != 462 or summary.NeuronUID.duplicated().any():
        raise ValueError("Exact unchanged 462-neuron ledger required")
    summary = summary.set_index("NeuronUID")
    atlas = np.asarray(nib.load(helper.ATLAS).dataobj)[..., 0, 5]
    key = pd.read_csv(helper.KEY, sep="\t", dtype=str, keep_default_na=False)
    names = dict(zip(key.Index.astype(int), key.Full_Name))
    records = []
    for uid, target, purpose in CASES:
        row = summary.loc[uid]
        native_path = helper.RAW / row.SampleID / row.NeuronID
        atlas_path = ROOT / row.SWCPath
        native_hash = helper.sha(native_path)
        atlas_hash = helper.sha(atlas_path)
        if native_hash != coverage.loc[uid, "native_swc_sha256"] or atlas_hash != row.SWCSHA256:
            raise ValueError(f"Source hash mismatch: {uid}")
        native = np.asarray(helper.parse_swc(native_path.read_text(), str(native_path)), float)
        warped = np.asarray(helper.parse_swc(atlas_path.read_text(), str(atlas_path)), float)
        if helper.identity_signature(native) != helper.identity_signature(warped):
            raise ValueError(f"Identity/type/parent mismatch: {uid}")
        by_id, children, leaves = helper.graph_information(native)
        # Independent full-graph endpoint arithmetic, including children of all types.
        parent_ids = set(native[:, 6].astype(int))
        independent_leaves = {
            int(n[0]) for n in native
            if n[1] == 2 and n[6] != -1 and int(n[0]) not in parent_ids
        }
        if leaves != independent_leaves:
            raise ValueError("Full-graph leaf definition disagreement")
        labels = helper.atlas_labels(warped, atlas)
        covered = [n for n in leaves if labels[n] == target
                   and helper.cube_path(row.SampleID, helper.cube_index(by_id[n][2:5])).is_file()]
        counts = Counter(helper.cube_index(by_id[n][2:5]) for n in covered)
        if not counts:
            raise ValueError(f"No target leaf image coverage: {uid}")
        cube = sorted(counts, key=lambda c: (-counts[c], c))[0]
        origin = np.asarray(cube) * helper.BLOCK
        local = [n for n in covered if helper.cube_index(by_id[n][2:5]) == cube]
        def face_margin(n):
            point = by_id[n][2:5]
            return float(np.min(np.r_[point-origin, origin+helper.BLOCK-point]))
        selected = sorted(local, key=lambda n: (-face_margin(n), n))[0]
        point = by_id[selected][2:5]
        component = next(c for c in helper.target_components(native, labels, target) if selected in c)
        node_set = set(component)
        component_leaves = sorted(node_set & leaves)
        branches = sorted(n for n in component if sum(c in node_set for c in children[n]) >= 2)
        edges = [(int(by_id[n][6]), n) for n in component if int(by_id[n][6]) in node_set]
        starts = [n for n in component if int(by_id[n][6]) not in node_set]
        if len(starts) != 1 or len(edges) != len(component)-1:
            raise ValueError("Target component must be one original connected tree")
        exits = [(n,c) for n in component for c in children[n] if c not in node_set]
        image_path = helper.cube_path(row.SampleID, cube)
        image = tifffile.imread(image_path)
        if image.shape != (90,360,360) or image.dtype != np.uint16:
            raise ValueError("Unexpected native source image geometry")
        centre = np.floor((point-origin)/helper.SPACING + 0.5).astype(int)
        planes = list(range(max(0,centre[2]-4), min(90,centre[2]+5)))
        x0,x1 = max(0,centre[0]-77),min(360,centre[0]+78)
        y0,y1 = max(0,centre[1]-77),min(360,centre[1]+78)
        lo,hi = np.percentile(image[planes,y0:y1,x0:x1], [1,99.7])
        hi = max(float(hi),float(lo)+1)
        fig,axes = plt.subplots(4,3,figsize=(15,16),layout="constrained")
        for ax,(x,y) in zip(axes[0],[(0,1),(0,2),(1,2)]):
            segments = [np.asarray([by_id[a][2:5],by_id[b][2:5]])[:,[x,y]] for a,b in edges]
            ax.add_collection(LineCollection(segments,colors="#536b8e",linewidths=.5))
            p = np.asarray([by_id[n][2:5] for n in component])
            ax.scatter(p[:,x],p[:,y],s=.4,color="#536b8e")
            tips = np.asarray([by_id[n][2:5] for n in component_leaves])
            ax.scatter(tips[:,x],tips[:,y],s=7,color="#d94a3b")
            ax.scatter(point[x],point[y],s=70,facecolors="none",edgecolors="#139c82")
            ax.plot([origin[x],origin[x]+helper.BLOCK[x],origin[x]+helper.BLOCK[x],origin[x],origin[x]],
                    [origin[y],origin[y],origin[y]+helper.BLOCK[y],origin[y]+helper.BLOCK[y],origin[y]],color="#139c82")
            ax.autoscale()
            ax.set_aspect("equal",adjustable="datalim")
            ax.set_xlabel(f"Native {'XYZ'[x]} (nominal µm)")
            ax.set_ylabel(f"Native {'XYZ'[y]} (nominal µm)")
            ax.set_title("Connected axon section; original ends red",fontsize=10)
        for ax,z in zip(axes[1:].flat,planes):
            ax.imshow(image[z,y0:y1,x0:x1],origin="lower",cmap="gray",vmin=lo,vmax=hi,
                      extent=[origin[0]+(x0-.5)*.65,origin[0]+(x1-.5)*.65,
                              origin[1]+(y0-.5)*.65,origin[1]+(y1-.5)*.65])
            ax.scatter(point[0],point[1],s=45,facecolors="none",edgecolors="#ff654f",linewidths=.8)
            ax.set_title(f"Actual XY plane {z}; native Z={origin[2]+z*3:.1f} µm",fontsize=10)
            ax.set_xlabel("Native X (nominal µm)")
            ax.set_ylabel("Native Y (nominal µm)")
        for ax in list(axes[1:].flat)[len(planes):]:
            ax.set_axis_off()
        title = f"{uid} | animal {row.AnimalID} | original axon leaf {selected}"
        target_name = names[target]
        fig.suptitle(title + "\n" + textwrap.fill("Declared ARM6 target: " + target_name,90)
                     + f"\nSection: {len(component)} nodes, {len(branches)} branch nodes, {len(component_leaves)} original leaves. "
                     + "One leaf imaged here.\n"
                     + f"Ring is leaf XY only; leaf Z={point[2]:.2f} nominal µm. Native axes are not anatomical directions.",fontsize=12)
        panel_path = output / (uid.replace("::","_").removesuffix(".swc")+"_section_and_optical_planes.png")
        fig.savefig(panel_path,dpi=140,bbox_inches="tight")
        plt.close(fig)
        node_records = [{"NeuronUID":uid,"node_id":n,"parent_id":int(by_id[n][6]),
                         "native_x_um":by_id[n][2],"native_y_um":by_id[n][3],"native_z_um":by_id[n][4],
                         "full_graph_axon_leaf":n in leaves,"declared_ARM6_index":labels[n]} for n in sorted(component)]
        nodes_path = output / (uid.replace("::","_").removesuffix(".swc")+"_section_nodes.csv")
        pd.DataFrame(node_records).to_csv(nodes_path,index=False)
        record = {
            "NeuronUID":uid,"AnimalID":row.AnimalID,"purpose":purpose,
            "source_ARM_index":int(row.ARMIndex),"source_ARM_full_name":row.ARMFullName,
            "source_ARM_status":row.SourceARMStatus,"original_portal_region":row.portal_region,
            "evidence_stratum":row.EvidenceStratum,"source_evidence_group":row.EvidenceSourceGroup,
            "source_localization_acceptance":"not_independently_accepted",
            "declared_target_ARM6_index":target,"declared_target_full_name":target_name,
            "target_lookup_policy":"np.rint(atlas_XYZ/250); existing declared current center-origin, ties-to-even",
            "native_swc":{"path":helper.relative(native_path),"sha256":native_hash},
            "atlas_swc":{"path":row.SWCPath,"sha256":atlas_hash},
            "full_graph_parser_passed":True,"node_identity_type_parent_match":True,
            "independent_full_graph_leaf_set_matches":True,
            "full_graph_candidate_axon_leaves":len(leaves),"target_covered_original_leaves":len(covered),
            "section_root_node_id":starts[0],"section_node_count":len(component),
            "section_branch_ids":branches,"section_original_leaf_ids":component_leaves,
            "section_entry_link":[int(by_id[starts[0]][6]),starts[0]],"section_exit_links":exits,
            "selected_leaf_id":selected,"selected_leaf_parent_id":int(by_id[selected][6]),
            "selected_leaf_native_xyz_um":point.tolist(),
            "selected_leaf_cube_face_distances_um":np.r_[point-origin,origin+helper.BLOCK-point].tolist(),
            "selection_rule":"Fixed purposive UID and target; cached target cube with most original leaves (XYZ tie); leaf with greatest minimum cube-face margin (node-ID tie). Display-selection rule, not acceptance.",
            "image":{"path":helper.relative(image_path),"sha256":helper.sha(image_path),
                     "shape_ZYX":list(image.shape),"dtype":str(image.dtype),"cube_index_XYZ":list(cube),
                     "nominal_spacing_xyz_um":helper.SPACING.tolist(),"nominal_origin_xyz_um":origin.tolist(),
                     "optical_resolution":"unverified","display_planes_Z_index":planes,
                     "display_contrast_percentiles":[1,99.7],
                     "adjacent_6_face_cubes_cached":{f"{'XYZ'[axis]}{direction:+}":helper.cube_path(row.SampleID,tuple(cube[i]+(direction if i==axis else 0) for i in range(3))).is_file() for axis in range(3) for direction in [-1,1]}},
            "panel_path":helper.relative(panel_path),"panel_sha256":helper.sha(panel_path),
            "section_nodes_path":helper.relative(nodes_path),"section_nodes_sha256":helper.sha(nodes_path),
            "image_review_status":"awaiting_actual_panel_inspection",
            "biological_terminal_state":"unassessed","bouton_or_synapse_state":"unassessed",
        }
        records.append(record)
        print(uid,"leaf",selected,"branches",len(branches),"sectionLeaves",len(component_leaves),"margin",round(face_margin(selected),3),flush=True)
    helper.write_json(HERE/"assessment_inputs.json",records)
    bindings = [helper.SUMMARY,coverage_path,helper.ATLAS,helper.KEY,helper_path,Path(__file__),ROOT/"main_scripts/swc_validation.py"]
    helper.write_json(HERE/"generation_provenance.json",{
        "status":"software_generated_pending_actual_panel_review","new_neurons":len(records),
        "new_animals":len(set(r["AnimalID"] for r in records)),"no_downloads":True,"no_source_changes":True,
        "inputs":[{"path":helper.relative(p),"sha256":helper.sha(p)} for p in bindings],
        "outputs":[{"path":helper.relative(p),"sha256":helper.sha(p)} for p in sorted(output.iterdir())]+[
            {"path":helper.relative(HERE/"assessment_inputs.json"),"sha256":helper.sha(HERE/"assessment_inputs.json")}],
    })


if __name__ == "__main__":
    build()
