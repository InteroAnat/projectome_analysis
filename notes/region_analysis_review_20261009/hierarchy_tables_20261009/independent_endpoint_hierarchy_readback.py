"""Read-only regional endpoint oracle; no production routines imported."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints"
OUT = Path(__file__).with_name("independent_endpoint_baseline_readback_20261009.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(name):
    with (RUN / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index(value):
    if not value:
        return None
    numeric = float(value)
    assert numeric.is_integer()
    return int(numeric)


def main():
    qc, source, saved = (rows(name) for name in ("per_neuron_qc.csv", "per_neuron_target_counts.csv", "animal_region_summaries.csv"))
    selected, eligible = Counter(), Counter()
    identity_group, eligibility = {}, {}
    for row in qc:
        uid = row["SampleID"], row["NeuronID"]
        assert uid not in identity_group
        group = row["AnimalID"], row["Subregion"]
        identity_group[uid] = group
        eligibility[uid] = int(row["n_computable"])
        selected[group] += 1
        eligible[group] += eligibility[uid]
        assert eligibility[uid] == int(int(row["candidate_axon_endpoint_count"]) > 0)
    counts, evidence, seen = Counter(), defaultdict(set), set()
    for row in source:
        uid = row["SampleID"], row["NeuronID"]
        group = row["AnimalID"], row["Subregion"]
        assert identity_group[uid] == group and eligibility[uid] == 1
        key = (*group, int(row["level"]), row["target_status"], index(row["target_index"]))
        assert (uid, key) not in seen
        seen.add((uid, key))
        assert int(row["neurons_with_target_evidence"]) == 1 and int(row["endpoint_count"]) > 0
        counts[key] += int(row["endpoint_count"])
        evidence[key].add(uid)
    actual_keys = set()
    for row in saved:
        group = row["AnimalID"], row["Subregion"]
        assert int(row["n_selected"]) == selected[group]
        assert int(row["n_computable"]) == eligible[group] and row["map_available"] == "True"
        key = (*group, int(row["level"]), row["target_status"], index(row["target_index"]))
        assert key not in actual_keys
        actual_keys.add(key)
        assert int(row["endpoint_count"]) == counts[key]
        assert int(row["neurons_with_target_evidence"]) == len(evidence[key])
        assert abs(float(row["frequency_per_computable_neuron"]) - len(evidence[key]) / eligible[group]) < 1e-12
        assert abs(float(row["mean_endpoints_per_computable_neuron"]) - counts[key] / eligible[group]) < 1e-12
    assert actual_keys == set(counts)
    assert len(qc) == 436 and sum(eligible.values()) == 403 and len(selected) == 47
    assert len(saved) == 4428 and len(source) == 11714
    total = sum(int(row["candidate_axon_endpoint_count"]) for row in qc)
    per_level = {level: sum(count for key, count in counts.items() if key[2] == level) for level in range(1, 7)}
    assert all(value == total for value in per_level.values())
    example = ("936", "ARM6_728_R", 1, "mapped", 638)
    report = {"status": "passed", "scope": "Existing main endpoint baseline CSVs only; new hierarchy export not yet checked",
              "script_sha256": sha(__file__), "n_selected": len(qc), "n_endpoint_eligible": sum(eligible.values()),
              "animal_source_groups": len(selected), "per_neuron_target_rows": len(source),
              "animal_target_rows_checked": len(saved), "total_candidate_endpoints": total,
              "per_level_endpoint_totals": per_level,
              "observed_target_status_rows": dict(Counter(row["target_status"] for row in source)),
              "dedup_example": {"AnimalID": example[0], "source": example[1], "level": 1, "target_index": 638,
                                "candidate_endpoint_count": counts[example], "neurons_with_target_evidence": len(evidence[example]),
                                "n_selected": selected[example[:2]], "n_eligible": eligible[example[:2]],
                                "regional_frequency": len(evidence[example]) / eligible[example[:2]]},
              "input_hashes": {name: sha(RUN / name) for name in ("run_provenance.json", "per_neuron_qc.csv", "per_neuron_target_counts.csv", "animal_region_summaries.csv")},
              "source_data_modified": False,
              "limitation": "CSV reconciliation does not establish source-image or terminal acceptance or validate an unavailable new exporter."}
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
