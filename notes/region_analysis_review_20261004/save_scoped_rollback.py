"""Preserve pre-region-review code and an exact scoped reversible patch."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "rollback"
OUT.mkdir(exist_ok=True)
names = ["laterality.py", "classifier.py", "neuron_analysis.py", "hierarchy.py", "population.py"]
patches, manifest = [], []
for name in names:
    relative = "main_scripts/region_analysis/" + name
    current = (ROOT / relative).read_text(encoding="utf-8")
    if name == "population.py" and (OUT / name).exists():
        before = (OUT / name).read_text(encoding="utf-8")
    elif name != "population.py":
        before = subprocess.check_output(["git", "-c", "safe.directory=" + str(ROOT), "show", "HEAD:" + relative], cwd=ROOT).decode("utf-8").replace("\r\n", "\n")
    else:
        # Population had earlier uncommitted edits. Reverse only the changes
        # introduced in this region review, keeping those existing edits.
        before = current
        replacements = [
            ("    add_soma_hierarchy_column,\n", ""),
            ("        neuron_list: Optional[List[dict]] = None,\n        swc_source_dir: str = None,\n", ""),
            ("        self.neuron_list = (list(neuron_list) if neuron_list is not None\n                            else _iondata.getNeuronListBySampleID(sample_id))\n        self.swc_source_dir = Path(swc_source_dir) if swc_source_dir is not None else None\n        self.load_failures = []\n", "        self.neuron_list = _iondata.getNeuronListBySampleID(sample_id)\n"),
            ("        data = []\n        self.load_failures = []\n", "        data = []\n"),
            ("                source = None\n                if self.swc_source_dir is not None:\n                    source_path = self.swc_source_dir / entry[\"name\"]\n                    if not source_path.is_file():\n                        raise FileNotFoundError(f\"Declared local SWC source missing: {source_path}\")\n                    source = str(source_path)\n", ""),
            ("                    entry[\"sampleid\"], entry[\"name\"], nii_space=self.nii_space,\n                    swc=source,\n                    output_dir=(str(self.output.output_dir / \"processed_neurons\")\n                                if self.swc_source_dir is not None and self.output else None),\n", "                    entry[\"sampleid\"], entry[\"name\"], nii_space=self.nii_space\n"),
            ("                self.load_failures.append({\"SampleID\": entry.get(\"sampleid\", self.sample_id),\n                                           \"NeuronID\": entry.get(\"name\"), \"error\": str(exc)})\n", ""),
            ("if \"Unknown\" in analysis.soma_region or analysis.soma_region == \"Out_of_Bounds\":", "if \"Unknown\" in analysis.soma_region:"),
            ("if \"Unknown\" in item[\"region\"] or item[\"region\"] == \"Out_of_Bounds\":", "if \"Unknown\" in item[\"region\"]:"),
            ("                    \"Length_Unit\": \"voxel\",\n", ""),
            ("        self.plot_dataframe = add_soma_hierarchy_column(\n            self.plot_dataframe, self.hierarchy, max_level=max_level,\n            hierarchy_table=hierarchy_source,\n        )\n        self.plot_dataframe = add_projection_hierarchy(\n            self.plot_dataframe, self.hierarchy, max_level=max_level,\n            hierarchy_table=hierarchy_source,\n        )\n", ""),
            ("            \"Soma_Side_Method\", \"Soma_Side_Conflict\", \"Soma_Side_Reference_Source\",\n", ""),
            ("            \"Total_Length\", \"Length_Unit\", \"Terminal_Count\",", "            \"Total_Length\", \"Terminal_Count\","),
            ("        for lv in range(1, 7):\n            col = f\"Soma_Level_{lv}\"\n            if col in self.plot_dataframe:\n                df[f\"L{lv}\"] = self.plot_dataframe[col]\n", ""),
            ("            if self.hierarchy is not None or self.hierarchy_table is not None or self.dual_hierarchy is not None:", "            if self.hierarchy is not None or self.hierarchy_table is not None:"),
        ]
        for new, old in replacements:
            if before.count(new) != 1:
                raise RuntimeError(f"Cannot reconstruct exact pre-review population state: {new[:70]}")
            before = before.replace(new, old, 1)
    backup = OUT / name
    if backup.exists() and backup.read_text(encoding="utf-8") != before:
        raise RuntimeError(f"Refusing to overwrite a different rollback snapshot: {name}")
    backup.write_text(before, encoding="utf-8", newline="\n")
    for line in difflib.unified_diff(before.splitlines(True), current.splitlines(True), fromfile="a/" + relative, tofile="b/" + relative):
        patches.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    manifest.append({"path": relative, "before_sha256_lf": hashlib.sha256(before.encode()).hexdigest(),
                     "after_sha256_lf": hashlib.sha256(current.encode()).hexdigest(), "backup": str(backup.relative_to(ROOT))})
for entry in json.loads((OUT / "entrypoint_snapshots.json").read_text(encoding="utf-8")):
    relative = entry["path"]
    before = (ROOT / entry["backup"]).read_text(encoding="utf-8")
    current = (ROOT / relative).read_text(encoding="utf-8")
    for line in difflib.unified_diff(before.splitlines(True), current.splitlines(True), fromfile="a/" + relative, tofile="b/" + relative):
        patches.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    manifest.append({**entry, "before_sha256_lf": hashlib.sha256(before.encode()).hexdigest(),
                     "after_sha256_lf": hashlib.sha256(current.encode()).hexdigest()})
(OUT / "region_review.patch").write_text("".join(patches), encoding="utf-8", newline="\n")
(OUT / "manifest.json").write_text(json.dumps({"scope": "Seven scoped production edits; preserves earlier population edits and the inherited untracked regenerator", "files": manifest}, indent=2), encoding="utf-8")
print(json.dumps({"status": "rollback_saved", "files": len(manifest), "patch": str((OUT / "region_review.patch").relative_to(ROOT))}))
