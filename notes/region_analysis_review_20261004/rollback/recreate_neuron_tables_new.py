import os
from pathlib import Path

import pandas as pd


RESULTS_XLSX = Path(
    r"d:\projectome_analysis\main_scripts\output\251637_20260423_165410_region_analysis\tables\251637_results.xlsx"
)
SOURCE_TABLES_DIR = Path(r"d:\projectome_analysis\neuron_tables")
OUTPUT_TABLES_DIR = Path(r"d:\projectome_analysis\neuron_tables_new")
REFINED_SOMA_XLSX = Path(r"d:\projectome_analysis\neuron_tables\251637_INS_HE_inferred.xlsx")


def load_refined_soma_map(ref_path: Path) -> dict:
    ref_summary = pd.read_excel(ref_path, sheet_name="Summary")
    ref_summary = ref_summary[["NeuronID", "Soma_Region"]].dropna(subset=["NeuronID"])
    ref_summary["NeuronID"] = ref_summary["NeuronID"].astype(str)
    return dict(zip(ref_summary["NeuronID"], ref_summary["Soma_Region"]))


def apply_refined_soma(df: pd.DataFrame, soma_map: dict) -> pd.DataFrame:
    if "NeuronID" not in df.columns or "Soma_Region" not in df.columns:
        return df
    out = df.copy()
    neuron_ids = out["NeuronID"].astype(str)
    mapped = neuron_ids.map(soma_map)
    out.loc[mapped.notna(), "Soma_Region"] = mapped[mapped.notna()]
    return out


def read_base_workbook(base_path: Path, soma_map: dict) -> dict:
    xls = pd.ExcelFile(base_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(base_path, sheet_name=sheet_name)
        sheets[sheet_name] = apply_refined_soma(df, soma_map)
    return sheets


def source_files_for_subtables(source_dir: Path) -> list[Path]:
    files = []
    for path in sorted(source_dir.glob("251637_*.xlsx")):
        if "results_clustered" in path.name.lower():
            continue
        try:
            xls = pd.ExcelFile(path)
            if "Summary" not in xls.sheet_names:
                continue
            summary = pd.read_excel(path, sheet_name="Summary", usecols=["NeuronID"])
            if summary.empty:
                continue
            files.append(path)
        except Exception:
            continue
    return files


def extract_neuron_ids(source_path: Path) -> list[str]:
    summary = pd.read_excel(source_path, sheet_name="Summary", usecols=["NeuronID"])
    neuron_ids = (
        summary["NeuronID"].dropna().astype(str).drop_duplicates().tolist()
    )
    return neuron_ids


def build_subtable(base_sheets: dict, neuron_ids: list[str]) -> dict:
    selected = set(neuron_ids)
    out = {}
    for sheet_name, df in base_sheets.items():
        if "NeuronID" in df.columns:
            key = df["NeuronID"].astype(str)
            out[sheet_name] = df[key.isin(selected)].copy()
        else:
            out[sheet_name] = df.copy()
    return out


def write_workbook(path: Path, sheets: dict) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    soma_map = load_refined_soma_map(REFINED_SOMA_XLSX)
    base_sheets = read_base_workbook(RESULTS_XLSX, soma_map)
    src_files = source_files_for_subtables(SOURCE_TABLES_DIR)

    print(f"Using base workbook: {RESULTS_XLSX}")
    print(f"Refined soma map size: {len(soma_map)}")
    print(f"Source subtable templates: {len(src_files)}")

    for src_file in src_files:
        neuron_ids = extract_neuron_ids(src_file)
        rebuilt = build_subtable(base_sheets, neuron_ids)
        out_file = OUTPUT_TABLES_DIR / src_file.name
        write_workbook(out_file, rebuilt)
        print(f"[OK] {src_file.name}: {len(neuron_ids)} neurons")

    print(f"\nDone. New subtables written to: {OUTPUT_TABLES_DIR}")


if __name__ == "__main__":
    main()
