from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_composition(
    df: pd.DataFrame, region_col: str, label: str
) -> pd.DataFrame:
    counts = (
        df.groupby(["SampleID", region_col], dropna=False)
        .size()
        .reset_index(name="n_neurons")
        .rename(columns={region_col: "Region"})
    )
    totals = counts.groupby("SampleID")["n_neurons"].transform("sum")
    counts["proportion"] = counts["n_neurons"] / totals
    counts["region_version"] = label
    counts["Region"] = counts["Region"].fillna("NA")
    return counts


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "multi_monkey_INS_combined.xlsx"
    output_png = base_dir / "monkey_neuron_composition_orig_vs_refined.png"
    output_csv = base_dir / "monkey_neuron_composition_orig_vs_refined.csv"

    df = pd.read_excel(input_path, sheet_name="Summary")

    auto_comp = build_composition(df, "Soma_Region_Auto", "Original (Auto)")
    refined_comp = build_composition(df, "Soma_Region_Refined", "Refined")
    composition = pd.concat([auto_comp, refined_comp], ignore_index=True)

    composition.to_csv(output_csv, index=False)

    sample_order = sorted(composition["SampleID"].unique())
    region_order = (
        composition.groupby("Region")["n_neurons"].sum().sort_values(ascending=False).index
    )

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=composition,
        kind="bar",
        x="SampleID",
        y="proportion",
        hue="Region",
        col="region_version",
        order=sample_order,
        hue_order=region_order,
        height=5.2,
        aspect=1.25,
        palette="tab20",
        errorbar=None,
    )
    g.set_axis_labels("Monkey (SampleID)", "Neuron proportion")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1))
    g.fig.subplots_adjust(top=0.84, bottom=0.15, wspace=0.08)
    g.fig.suptitle(
        "Neuron composition by monkey: original vs refined soma region", fontsize=13
    )

    if g._legend is not None:
        g._legend.set_title("Soma region")
        for txt in g._legend.texts:
            txt.set_fontsize(8)

    g.savefig(output_png, dpi=300)
    plt.close("all")

    print(f"Saved figure: {output_png}")
    print(f"Saved table:  {output_csv}")


if __name__ == "__main__":
    main()
