"""Correct saved profile display only; preserve the numerical producer run."""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance_path = args.run / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    protected = {
        str(args.run / name): expected
        for name, expected in provenance["output_hashes"].items()
    }
    protected[str(provenance_path)] = sha(provenance_path)
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError("Changed primary output: " + path)
    profiles = pd.read_csv(args.run / "cluster_target_profiles.csv")
    args.output.mkdir(parents=True, exist_ok=False)
    for family in ["axon", "endpoint"]:
        analysis = f"all_{family}_hellinger"
        k = provenance["analyses"][analysis]["display_cut_k"]
        selected = profiles[(profiles.analysis == analysis) &
                            (profiles.requested_k == k)]
        table = selected.pivot(index="cluster", columns="TargetID",
                               values="equal_contributing_animal_relative_profile")
        top = table.mean().nlargest(min(18, len(table.columns))).index
        names = selected.drop_duplicates("TargetID").set_index("TargetID").ARMFullName
        counts = selected.groupby("cluster").n_neurons.first()
        labels = [textwrap.fill(names[target].replace("_", " "), 21) for target in top]
        fig, ax = plt.subplots(figsize=(16, 5.8 if family == "endpoint" else 5.3))
        image = ax.imshow(table[top], aspect="auto", cmap="magma", vmin=0)
        ax.set_xticks(range(len(top)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(table)),
                      [f"Diagnostic cluster {cluster} (n={counts[cluster]})"
                       for cluster in table.index], fontsize=9)
        if family == "axon":
            title = "Axon-labelled length: k=2 is strongly hemisphere-driven"
            caption = "Descriptive relative target allocation; diagnostic clusters are not cell types."
        else:
            title = "Candidate axon ends: fallback k=3, including a singleton (n=1)"
            caption = ("No tested cut meets the display size safeguards. Eligible neurons have at least one non-root "
                       "axon-labelled leaf anywhere, including outside the field of view; these are not reviewed terminals.")
        ax.set_title(title, fontsize=12, pad=14)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.025)
        colorbar.set_label("Mean relative allocation\n(equal contributing animals)", fontsize=9)
        fig.text(0.5, 0.015, textwrap.fill(caption, 145), ha="center", va="bottom", fontsize=9)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(args.output / f"{analysis}_target_profiles.png", dpi=160,
                    bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError("Primary output changed during display correction")
    receipt = {
        "status": "software_verified_display_only_correction",
        "primary_run": str(args.run.resolve()),
        "primary_provenance_SHA256": sha(provenance_path),
        "source_table": str((args.run / "cluster_target_profiles.csv").resolve()),
        "source_table_SHA256": sha(args.run / "cluster_target_profiles.csv"),
        "code_SHA256": sha(__file__),
        "numerical_results_changed": False,
        "primary_outputs_unchanged": True,
        "display_changes": ["wrapped full official names; underscores displayed as spaces",
                            "wrapped colorbar; tight output bounds",
                            "laterality and endpoint fallback/singleton limitations visible"],
        "output_hashes": {path.name: sha(path) for path in args.output.iterdir() if path.is_file()},
    }
    (args.output / "display_correction_provenance.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
