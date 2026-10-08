"""Reporting-only replay; historical source and frozen identities stay immutable."""
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "notes/region_analysis_review_20261004/runs/refined_subtables_20261004_072110/master_analysis/251637_20261004_072115_region_analysis/tables/251637_results_20261004_072515.xlsx"
CODE = [ROOT / "main_scripts/region_analysis" / name for name in ("utils.py", "plotting.py", "population.py")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def independent_status(value):
    # Independent oracle: no imports from region label/report helpers.
    if value is None or isinstance(value, float) and value != value:
        return "absent"
    if not isinstance(value, str):
        return "invalid"
    token = value.strip().upper()
    if not token or token in {"NONE", "NULL", "NAN"}:
        return "absent"
    token = re.sub(r"^(?:CL_|CR_|SL_|SR_|L-|R-|L_|R_)", "", token)
    if token == "OUT_OF_BOUNDS":
        return "outside"
    if token in {"UNMAPPED", "_UNMAPPED"}:
        return "unmapped"
    if token in {"UNKNOWN", "INSULAUNKNOWN"} or re.fullmatch(r"UNKNOWN_\d+", token):
        return "unknown"
    return "known"


def main(mode):
    import pandas as pd
    sys.path.insert(0, str(ROOT / "main_scripts"))
    from region_analysis.population import PopulationRegionAnalysis
    from region_analysis.plotting import plot_terminal_distribution_df, plot_projection_sites_count_df
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    frozen_path = OUT / "frozen_input.json"
    if mode == "before":
        original_hash = sha(SOURCE)
        assert original_hash == "508cea4b5c6ba8e1401e55302c2adcb9e319d6842cffe6d0d5f57e0c5080f282"
        summary = pd.read_excel(SOURCE, sheet_name="Summary")
        sites = pd.read_excel(SOURCE, sheet_name="Terminal_Sites")
        grouped = sites.groupby("NeuronID", sort=False)["Terminal_Region"].agg(list)
        records = []
        for row in summary.to_dict("records"):
            targets = grouped.get(row["NeuronID"], [])
            assert len(targets) == len(dict.fromkeys(targets)) == row["Terminal_Count"]
            records.append({key: row[key] for key in ("SampleID", "NeuronID", "Neuron_Type", "Terminal_Count", "Outlier_Count")} | {"Terminal_Regions": targets})
        assert len(records) == 562 and sum(r["Terminal_Count"] for r in records) == len(sites) == 2408
        dump_new(frozen_path, {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": original_hash, "status": "historical report replay; no dataset promotion", "rows": records})
        for path in CODE:
            with (OUT / ("before_" + path.name + ".txt")).open("xb") as stream:
                stream.write(path.read_bytes())
    frozen_hash = sha(frozen_path)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    source_hash = sha(SOURCE)
    assert source_hash == frozen["source_sha256"]
    df = pd.DataFrame(frozen["rows"])
    statuses = Counter(independent_status(v) for values in df.Terminal_Regions for v in values)
    legacy_known = sum("Unknown" not in str(v) for values in df.Terminal_Regions for v in values)
    corrected_known = statuses["known"]
    changed = []
    for row in frozen["rows"]:
        old = sum("Unknown" not in str(v) for v in row["Terminal_Regions"])
        new = sum(independent_status(v) == "known" for v in row["Terminal_Regions"])
        if old != new:
            changed.append({"NeuronID": row["NeuronID"], "old_known": old, "new_known": new, "reclassified_targets": [v for v in row["Terminal_Regions"] if ("Unknown" not in str(v)) != (independent_status(v) == "known")]})
    reports = {}
    pop = PopulationRegionAnalysis.__new__(PopulationRegionAnalysis)
    pop.plot_dataframe = df
    pop.sample_id = "251637 (historical frozen reporting replay)"
    pop.save_report = lambda content, name: reports.__setitem__(name, content)
    with contextlib.redirect_stdout(io.StringIO()):
        pop._save_terminal_report()
        pop._save_projection_sites_report()
        reports["plot_terminal_report"] = plot_terminal_distribution_df(df, top_n=1000, show=False)
        reports["plot_projection_sites_report"] = plot_projection_sites_count_df(df, show=False)
    expected_known = legacy_known if mode == "before" else corrected_known
    expected_unknown = 2408 - expected_known
    for name, report in reports.items():
        if "terminal_report" in name:
            assert f"Known sites: {expected_known} (" in report
            assert f"Unknown sites: {expected_unknown} (" in report
        else:
            assert f"Total known sites: {expected_known}\n" in report
            assert f"Total unknown sites excluded: {expected_unknown}\n" in report
        with (OUT / f"{mode}_{name}.txt").open("x", encoding="utf-8") as stream:
            stream.write(report + "\n")
    for index, number in enumerate(plt.get_fignums(), 1):
        plt.figure(number).savefig(OUT / f"{mode}_report_plot_{index}.png", dpi=100)
    plt.close("all")
    assert sha(SOURCE) == source_hash and sha(frozen_path) == frozen_hash
    result = {"mode": mode, "source_sha256": source_hash, "frozen_input_sha256": frozen_hash, "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in CODE}, "neurons": len(df), "unique_target_entries": 2408, "independent_status_counts": dict(statuses), "actual_known": expected_known, "actual_unresolved": expected_unknown, "affected_neurons": len(changed), "changed_rows": changed, "semantics": "Terminal_Count remains unique target regions across all reconstruction leaf types; no detector, biological terminal verification, classification, laterality or membership change", "source_and_frozen_input_unchanged": True}
    dump_new(OUT / f"{mode}_summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"changed_rows", "code_sha256"}}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
