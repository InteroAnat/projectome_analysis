"""Combine reviewed, non-overlapping repairs without touching production files."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "notes/region_analysis_review_20261009"
CONSUMER = AUDIT / "consumer_audit/staged/main_scripts"
SPECIALIST = AUDIT / "code_audit/staged/main_scripts"
DEST = AUDIT / "validated_repairs/staged/main_scripts"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def methods(source):
    tree = ast.parse(source)
    population = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "PopulationRegionAnalysis")
    return {node.name: node for node in population.body if isinstance(node, ast.FunctionDef)}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CONSUMER / "region_analysis", DEST / "region_analysis", dirs_exist_ok=True)
    receipt = json.loads((AUDIT / "code_audit/expanded_staged_repair_receipt.json").read_text())
    path = DEST / "region_analysis/population.py"
    consumer = path.read_text(encoding="utf-8")
    specialist = (SPECIALIST / "region_analysis/population.py").read_text(encoding="utf-8")
    before, after = methods(consumer), methods(specialist)
    lines, additions = consumer.splitlines(keepends=True), specialist.splitlines(keepends=True)
    for name in sorted(receipt["population_method_scope"], key=lambda name: before[name].lineno, reverse=True):
        old, new = before[name], after[name]
        lines[old.lineno - 1:old.end_lineno] = additions[new.lineno - 1:new.end_lineno]
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    for name in ("region_analysis/hierarchy_table.py", "region_analysis/plotting.py", "neuro_tracer.py"):
        shutil.copyfile(SPECIALIST / name, DEST / name)

    paths = ["region_analysis/utils.py", "region_analysis/population.py",
             "region_analysis/laterality_projection_analysis.py", "region_analysis/hierarchy_table.py",
             "region_analysis/plotting.py", "neuro_tracer.py"]
    diffs, patch = [], ["*** Begin Patch\n"]
    bindings = {}
    for name in paths:
        source, proposed = ROOT / "main_scripts" / name, DEST / name
        original, modified = source.read_text(encoding="utf-8"), proposed.read_text(encoding="utf-8")
        ast.parse(modified)
        bindings["main_scripts/" + name] = {"before_sha256": sha(source), "staged_sha256": sha(proposed)}
        diff = list(difflib.unified_diff(original.splitlines(keepends=True), modified.splitlines(keepends=True),
                                       fromfile="main_scripts/" + name, tofile="main_scripts/" + name))
        diffs.extend(diff)
        if diff:
            patch.append("*** Update File: D:/projectome_analysis/main_scripts/" + name + "\n")
            patch.extend("@@\n" if line.startswith("@@") else line for line in diff[2:])
    patch.append("*** End Patch\n")
    output = AUDIT / "validated_repairs"
    (output / "proposed_software_repairs.patch").write_text("".join(diffs), encoding="utf-8")
    (output / "apply_software_repairs.txt").write_text("".join(patch), encoding="utf-8")
    (output / "before_and_staged_hashes.json").write_text(json.dumps(bindings, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(bindings), "patch_lines": len("".join(diffs).splitlines()),
                      "production_modified": False}))


if __name__ == "__main__":
    main()
