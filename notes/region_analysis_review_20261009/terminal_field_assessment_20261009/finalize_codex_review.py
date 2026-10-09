"""Record the bounded Codex figure review, sources and current-policy checks."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import assess_native_terminal_fields_current_rint as pilot


OBSERVATIONS = {
    "252790::014.swc": {
        "morphology_assessment": "branched_terminal_field_candidate",
        "image_assessment": "selected_ending_and_local_neurites_supported_by_fluorescence",
        "observation": "The target-connected skeleton has many branches and endings, with clustered shorter branches and longer entry/exit trajectories. Local fluorescence follows several displayed trajectories. Leaf 1727 lies at a bright local neurite cap across nearby XY planes, including nominal Z 19029 and 19032 µm. No definite continuation beyond this selected end is resolved in the inspected planes. This is a spot-check of one cube, not validation of all 71 leaves or a closed arbor segmentation.",
        "truncation_assessment": "no_definite_truncation_in_selected_views; weak_or_out_of_plane_continuation_not_excluded",
    },
    "252790::032.swc": {
        "morphology_assessment": "simple_reconstructed_ending_not_branched_arbor",
        "image_assessment": "selected_ending_supported_by_local_fluorescence",
        "observation": "The selected target-connected component is an unbranched trajectory with one original leaf, 2344. Its curved terminal portion and bright end coincide locally with fluorescence in nominal Z 29457–29469 µm planes, especially 29460/29463. The clear local cap supports an image-backed ending candidate, not a terminal-field arbor or synapse. Other endings in the same cube do not belong to this selected connected component.",
        "truncation_assessment": "no_definite_continuation_beyond_selected_end_seen; biological_completion_unassessed",
    },
    "252383::018.swc": {
        "morphology_assessment": "branched_terminal_field_candidate",
        "image_assessment": "selected_ending_supported; whole_field_image_review_incomplete",
        "observation": "The skeleton shows multiple repeatedly branching local trajectories and many endings, supporting a morphology-reviewed field candidate. Leaf 2294 coincides with a short fluorescent ending near nominal Z 30198–30204 µm; nearby signal and other fibers require the plane sweep rather than a projected overlap claim. One cube/ending spot-check does not verify every branch or all 45 leaves.",
        "truncation_assessment": "no_definite_continuation_beyond_selected_end_seen; incomplete_field_coverage_review",
    },
    "252383::121.swc": {
        "morphology_assessment": "simple_reconstructed_ending_not_branched_arbor",
        "image_assessment": "local_ending_candidate_with_noisy_weak_plane_support",
        "observation": "The putamen-labelled component contains a single short unbranched trajectory ending at node 867. A curved fluorescent neurite and terminal bright spot are visible near its XY location. The plane sweep is noisy; the connecting shaft is stronger at Z 30294–30300 µm and the spot remains near the nominal leaf plane 30306 µm. Local support is weaker than the clean 252790::032 example, so this is an image-backed simple-ending candidate with depth uncertainty, not a putamen terminal arbor.",
        "truncation_assessment": "unresolved_due_to_weak_depth_and_background_support; no_confirmed_truncation",
    },
    "252384::047.swc": {
        "morphology_assessment": "simple_reconstructed_ending_not_branched_arbor",
        "image_assessment": "ambiguous_ending_correspondence_near_uncached_cube_boundary",
        "observation": "Leaf 6514 is only 3.879 nominal µm below the cube's upper-Y block boundary. Fluorescence occurs nearby across Z 22578–22593 µm, but the projected trace-to-signal depth correspondence is not precise enough to validate the end. The immediately adjoining upper-Y cube is absent from the local cache. The displayed edge is a cache/crop limit, not established tissue damage or an anatomical boundary.",
        "truncation_assessment": "possible_unresolved_continuation_or_tracing_truncation; missing_adjacent_image_prevents_decision",
    },
}


def build():
    directory = pilot.HERE
    output = directory / "codex_terminal_field_assessment.json"
    if output.exists():
        raise FileExistsError(output)
    generated = directory / "native_cube_pilot_current_rint"
    records = json.loads((generated / "pilot_inputs.json").read_text())
    atlas = np.asarray(pilot.nib.load(pilot.ATLAS).dataobj)[..., 0, 5]
    sensitivity = []
    for record in records:
        native_path = pilot.ROOT / record["native_swc"]["path"]
        atlas_path = pilot.ROOT / record["atlas_swc"]["path"]
        if pilot.sha(native_path) != record["native_swc"]["sha256"] or pilot.sha(atlas_path) != record["atlas_swc"]["sha256"]:
            raise ValueError("Source changed before review finalization")
        nodes = np.loadtxt(atlas_path)
        ri = np.rint(nodes[:, 2:5]/250).astype(int)
        fl = np.floor(nodes[:, 2:5]/250+.5).astype(int)
        valid = ((ri >= 0) & (ri < np.array(atlas.shape)) & (fl >= 0) & (fl < np.array(atlas.shape))).all(1)
        if not valid.all():
            raise ValueError("Pilot comparison needs explicit outside handling")
        rlabels = atlas[tuple(ri.T)]
        flabels = atlas[tuple(fl.T)]
        changed = np.flatnonzero(rlabels != flabels)
        sensitivity.append({"NeuronUID": record["NeuronUID"], "node_count": len(nodes),
                            "voxel_index_differences": int(np.any(ri != fl, axis=1).sum()),
                            "target_label_differences": len(changed),
                            "changed_nodes": [{"node_id": int(nodes[i, 0]), "current_rint_label": int(rlabels[i]),
                                               "halfopen_label": int(flabels[i])} for i in changed]})
        record.update(OBSERVATIONS[record["NeuronUID"]])
        record["reviewer"] = "Codex"
        record["review_date"] = "2026-10-09"
        record["review_state"] = "completed_bounded_morphology_and_source_plane_assessment"
        record["biological_terminal_state"] = "unassessed"
        record["bouton_synapse_state"] = "not_assessed"
        stem = record["NeuronUID"].replace("::", "_").removesuffix(".swc")
        inspected = [generated / stem / "target_component_morphology.png",
                     directory / "review_panels" / f"{stem}_ending_optical_planes.png"]
        record["primary_inspected_figures"] = [{"path": pilot.relative(path), "sha256": pilot.sha(path)} for path in inspected]
        record["image_review_denominator"] = {"selected_original_leaf_reviewed": 1,
                                               "component_original_leaves": len(record["component_full_graph_axon_leaf_ids"]),
                                               "complete_component_image_review": False}
    inventory = pd.read_csv(generated / "selected462_native_image_coverage.csv", dtype=str, keep_default_na=False)
    test = subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "unittest", "discover", "-s", str(directory),
                           "-p", "test_native_terminal_assessment.py"], capture_output=True, text=True)
    if test.returncode:
        raise RuntimeError(test.stdout + test.stderr)
    sources = [generated / "generation_provenance.json", directory / "optical_plane_checks_current_rint/optical_plane_provenance.json",
               directory / "review_panels/display_provenance.json", directory / "test_native_terminal_assessment.py",
               directory / "independent_terminal_review.json", Path(__file__)]
    passage = json.loads(sources[1].read_text())["passage"]
    passage.update(reviewer="Codex", review_date="2026-10-09",
                   review_state="completed_graph_and_projected_image_context_assessment",
                   observation="Original entry and exit links cross the target boundary, with no full-graph leaves or local branch points. This is reconstructed passage rather than a target-induced ending. The dense fluorescent projected context does not establish absence of en passant boutons or a uniquely resolved branch at every point.",
                   biological_terminal_state="unassessed", bouton_synapse_state="not_assessed")
    passage_path = directory / "review_panels/252790_032_passage_image_and_swc.png"
    passage["primary_inspected_figure"] = {"path": pilot.relative(passage_path), "sha256": pilot.sha(passage_path)}
    result = {"status": "completed_bounded_Codex_terminal_field_candidate_pilot", "reviewer": "Codex", "date": "2026-10-09",
              "scope": "purposive five neuron/target fields and one same-neuron reconstructed passage example; no complete cohort review",
              "cohort_unchanged": True, "selected_neurons": 462, "native_swcs_in_designated_cache": 201,
              "not_in_designated_native_cache": 261, "node_identity_type_parent_matches": 201,
              "native_candidate_leaves": 48574, "native_leaves_with_cached_cube": 9209,
              "neurons_with_at_least_one_cube_covered_native_leaf": int(sum(int(value or 0)>0 for value in inventory.leaves_with_cached_native_cube)),
              "availability_is_not_image_review": True, "reviewed_neurons": 5, "remaining_neurons_without_this_pilot_review": 457,
              "individually_image_reviewed_axon_leaves": 5, "branched_field_candidates": 2, "simple_ending_examples": 3,
              "reconstructed_passing_examples": 1, "biologically_accepted_terminals_or_synapses": 0,
              "registration_policy": "authoritative fMOST-to-NMT transform unavailable; recovery not required by current task; nominal native review plus declared ARM metadata",
              "source_image_scope": "local TIFF cache only; nominal sampling, not calibrated optical resolution; no network source-image acquisition",
              "current_target_lookup": "np.rint(atlas_xyz/250), existing current policy, ties-to-even, not anatomical acceptance",
              "halfopen_sensitivity": sensitivity, "reviews": records, "passage_review": passage,
              "tests": {"returncode": test.returncode, "output": test.stdout+test.stderr},
              "bindings": [{"path": pilot.relative(path), "sha256": pilot.sha(path)} for path in sources]}
    pilot.write_json(output, result)
    print(json.dumps({"status": result["status"], "report": pilot.relative(output), "sha256": pilot.sha(output)}, indent=2))


if __name__ == "__main__":
    build()
