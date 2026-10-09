"""Persist actual seven-panel review and independently recheck source arithmetic."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
from collections import Counter, defaultdict
import numpy as np
import nibabel as nib
import pandas as pd
import tifffile
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write("\n")

records=json.loads((HERE/"assessment_inputs.json").read_text())
notes={
    "252383::124.swc": {
        "image_observation":"A bright compact signal occupies the leaf ring in planes8–11; surrounding curved shafts and other puncta vary with depth. The selected branch is not uniquely separable throughout the displayed depth sweep.",
        "field_observation":"Small connected section with two original branch nodes and three original leaves; not an extensive dense arbor on morphology alone.",
        "continuation_limit":"Local signal correspondence supported; branch completion and possible neighboring/continuing neurites remain unresolved.",
    },
    "252790::037.swc": {
        "image_observation":"A diagonal fluorescent shaft approaches the leaf ring in planes51–56. A longer upward-running shaft becomes visible through planes54–59 near and beyond this XY location.",
        "field_observation":"One branch node and two original leaves in the selected section; local bifurcation candidate rather than a demonstrated dense field.",
        "continuation_limit":"Potential continuation or crossing cannot be assigned to this neuron from these planes alone; reconstructed leaf retained, completed ending not accepted.",
    },
    "252718::162.swc": {
        "image_observation":"A short bright oblique shaft crosses the leaf ring in planes11–14; separate neighboring spots and shafts are present in a noisy background.",
        "field_observation":"Two branch nodes and two original leaves; the section includes a long shaft, with a regional exit preserved.",
        "continuation_limit":"Local correspondence supported; shaft orientation through depth and nearby signal prevent a complete terminal interpretation.",
    },
    "252385::056.swc": {
        "image_observation":"An angular fluorescent branch reaches the ring in planes22–25. Bright signal can be seen beyond the ring along the same projected direction in some planes, with other spots nearby.",
        "field_observation":"Extensively branched connected morphology:196 branch nodes and214 original leaves. This is a branched-field morphology candidate, not214 image-reviewed endings.",
        "continuation_limit":"Possible continuation versus overlapping neurite requires more tracing; one leaf has local support, but endpoint completion and whole-field completeness are unresolved.",
    },
    "252527::056.swc": {
        "image_observation":"Compact/short fluorescent signal is present at the ring in planes49–52; a rightward curved shaft becomes visible close to it in planes53–55.",
        "field_observation":"Extensively branched morphology:208 branch nodes and203 original leaves, plus18 preserved regional exits.",
        "continuation_limit":"Possible depth continuation or neighboring shaft remains unresolved. A missing cached X-minus neighbor is recorded; selected leaf is158.625 nominal um from that face, so the leaf is not at that missing cube boundary.",
    },
    "252714::099.swc": {
        "image_observation":"A thin fluorescent trajectory approaches the ring from the upper-left in plane61, with a compact signal in plane62; the surrounding field contains additional bright shafts and dense background spots.",
        "field_observation":"Extensively branched morphology:94 branch nodes and77 original leaves, with33 regional exits retained.",
        "continuation_limit":"Local correspondence supported but depth completion uncertain. Missing X-minus/Y-minus cached neighbors limit wider review; selected leaf is183.098/106.203 nominal um from those faces, respectively.",
    },
}
reviewed=[]
for r in records:
    record={"NeuronUID":r["NeuronUID"],"AnimalID":r["AnimalID"],"reviewer":"Codex",
            "review_date":"2026-10-09","selected_leaf_id":r["selected_leaf_id"],
            "actual_panel_viewed":True,"panel_path":r["panel_path"],"panel_sha256":r["panel_sha256"],
            "source_image_correspondence":"local_fluorescent_signal_at_declared_native_leaf_location",
            "review_unit":"one_original_full_graph_axon_leaf_and_local_native_planes; section_morphology_context_only",
            "ending_completeness":"unresolved","biological_terminal_state":"unassessed",
            "bouton_synapse_state":"unassessed","source_or_target_anatomical_acceptance":"not_established",
            "automatic_label_or_cohort_change":False,
            **notes[r["NeuronUID"]]}
    reviewed.append(record)
write(HERE/"codex_image_assessments.json",{
    "status":"six_new_leaf_locations_and_one_internal_passage_actually_reviewed",
    "assessment_inputs_sha256":sha(HERE/"assessment_inputs.json"),"leaf_reviews":reviewed,
    "passage_review":{
        "NeuronUID":"252790::037.swc","selected_internal_node_id":7419,"actual_panel_viewed":True,
        "observation":"Bright local signal at ring across several planes, with a curved trajectory through and beyond the ring in planes60–63. Original graph has an entry and exit and no leaf in this38node section.",
        "classification":"reconstructed_passage_with_local_image_support; entire_section_not_image_proofread",
        "limits":"A bright spot at an internal node does not create a terminal or establish a bouton; section-wide entry/exit correspondence and biological synapses remain unassessed.",
        "passage_inputs_sha256":sha(HERE/"passage_inputs.json"),
    },
    "anatomical_acceptance":False,"biological_terminal_acceptance":False,
})

# Independent arithmetic does not import either generation script.
summary_path=ROOT/"notes/region_analysis_review_20261009/hierarchy_tables_20261009/combined_arm_projection_tables_462/neuron_summary.csv"
summary=pd.read_csv(summary_path,dtype=str,keep_default_na=False).set_index("NeuronUID")
atlas_path=ROOT/"atlas/NMT_v2.1_sym/NMT_v2.1_sym/ARM_in_NMT_v2.1_sym.nii.gz"
atlas=np.asarray(nib.load(atlas_path).dataobj)[...,0,5]
generation=json.loads((HERE/"generation_provenance.json").read_text())
for binding in generation["inputs"]+generation["outputs"]:
    if sha(ROOT/binding["path"])!=binding["sha256"]:
        raise ValueError("Generation binding changed: "+binding["path"])
prior_generation_path=HERE.parent/"terminal_field_assessment_20261009/native_cube_pilot_current_rint/generation_provenance.json"
prior_generation=json.loads(prior_generation_path.read_text())
coverage_relative="notes/region_analysis_review_20261009/terminal_field_assessment_20261009/native_cube_pilot_current_rint/selected462_native_image_coverage.csv"
coverage_binding=next(x for x in prior_generation["outputs"] if x["path"]==coverage_relative)
if sha(ROOT/coverage_relative)!=coverage_binding["sha256"]:
    raise ValueError("Prior all462 coverage changed")
checks=[]
for r in records:
    uid=r["NeuronUID"]
    npath=ROOT/r["native_swc"]["path"]
    apath=ROOT/r["atlas_swc"]["path"]
    n=np.loadtxt(npath,ndmin=2)
    a=np.loadtxt(apath,ndmin=2)
    nb={int(x[0]):x for x in n}
    ab={int(x[0]):x for x in a}
    if len(nb)!=len(n) or set(nb)!=set(ab):
        raise ValueError("Duplicate/colliding node identities")
    if any(tuple(nb[x][[1,6]])!=tuple(ab[x][[1,6]]) for x in nb):
        raise ValueError("Native/atlas graph identity mismatch")
    children=defaultdict(list)
    for x in n:
        children[int(x[6])].append(int(x[0]))
    leaf=r["selected_leaf_id"]
    if nb[leaf][1]!=2 or nb[leaf][6]==-1 or children[leaf]:
        raise ValueError("Selected leaf is not an original nonroot type2 graph leaf")
    labels={}
    for node,x in ab.items():
        index=np.rint(x[2:5]/250).astype(int)
        labels[node]=int(atlas[tuple(index)]) if ((index>=0)&(index<np.array(atlas.shape))).all() else -1
    target=r["declared_target_ARM6_index"]
    # Traverse the undirected connected type2, same-target section from leaf.
    found=set();pending=[leaf]
    while pending:
        node=pending.pop()
        if node in found or nb[node][1]!=2 or labels[node]!=target:
            continue
        found.add(node)
        neighbors=children[node]+([int(nb[node][6])] if int(nb[node][6]) in nb else [])
        pending.extend(neighbors)
    stored=pd.read_csv(ROOT/r["section_nodes_path"])
    if found!=set(stored.node_id) or len(found)!=r["section_node_count"]:
        raise ValueError("Independent target-section membership mismatch")
    actual_leaves=sorted(x for x in found if nb[x][1]==2 and nb[x][6]!=-1 and not children[x])
    actual_branches=sorted(x for x in found if sum(c in found for c in children[x])>=2)
    if actual_leaves!=r["section_original_leaf_ids"] or actual_branches!=r["section_branch_ids"]:
        raise ValueError("Independent section endpoint/branch mismatch")
    if sha(npath)!=r["native_swc"]["sha256"] or sha(apath)!=r["atlas_swc"]["sha256"] or sha(apath)!=summary.loc[uid,"SWCSHA256"]:
        raise ValueError("Source hash changed")
    image_path=ROOT/r["image"]["path"]
    image=tifffile.imread(image_path)
    if sha(image_path)!=r["image"]["sha256"] or list(image.shape)!=r["image"]["shape_ZYX"]:
        raise ValueError("Image hash/shape changed")
    panel=ROOT/r["panel_path"]
    if sha(panel)!=r["panel_sha256"]:
        raise ValueError("Panel changed")
    with Image.open(panel) as im:
        shape=list(im.size)
    checks.append({"NeuronUID":uid,"source_hashes_unchanged":True,"independent_graph_section_and_leaf_counts_match":True,
                   "target_leaf_current_rint":labels[leaf],"panel_size_pixels":shape,
                   "minimum_cube_face_distance_nominal_um":min(r["selected_leaf_cube_face_distances_um"])})
passage=json.loads((HERE/"passage_inputs.json").read_text())
n=np.loadtxt(ROOT/passage["native_swc"]["path"],ndmin=2)
by={int(x[0]):x for x in n};c=Counter(n[:,6].astype(int))
ids=set(passage["node_ids"])
if any(c[x]==0 for x in ids) or int(by[7400][6])!=7399 or int(by[7438][6])!=7437 or c[7419]!=1:
    raise ValueError("Independent original passage check failed")
for s in [passage["native_swc"],passage["atlas_swc"],passage["source_image"],passage["panel"]]:
    if sha(ROOT/s["path"])!=s["sha256"]:
        raise ValueError("Passage source/display hash changed")
prior=json.loads((HERE.parent/"terminal_field_assessment_20261009/codex_terminal_field_assessment.json").read_text())
receipt={
    "status":"software_independent_saved_readback_passed_with_actual_Codex_panel_review",
    "recorded_at":datetime.now().astimezone().isoformat(),"scope":"Software identity, source hash, graph arithmetic and saved display readback; no registration/anatomical/biological terminal acceptance",
    "selected_ledger_count":len(summary),"new_leaf_review_count":6,"new_passage_review_count":1,
    "new_review_neuron_count":6,"new_review_animal_count":6,
    "newly_added_animals_vs_prior":["331","631","900","945"],
    "combined_prior_plus_expansion_individually_reviewed_leaves":11,
    "combined_prior_plus_expansion_passing_locations":2,
    "combined_prior_plus_expansion_neurons":11,"combined_prior_plus_expansion_animals":7,
    "all_selected_neurons_remain_462":True,"all_original_inputs_unchanged":True,
    "case_checks":checks,"independent_passage_checks_passed":True,
    "generation_provenance_sha256":sha(HERE/"generation_provenance.json"),
    "source_summary":{"path":summary_path.relative_to(ROOT).as_posix(),"sha256":sha(summary_path)},
    "artifacts":[{"path":p.relative_to(HERE).as_posix(),"sha256":sha(p)} for p in sorted(HERE.rglob("*")) if p.is_file()],
}
write(HERE/"independent_saved_readback.json",receipt)
print(json.dumps({k:receipt[k] for k in ["status","new_leaf_review_count","new_passage_review_count","combined_prior_plus_expansion_animals"]},indent=2))
