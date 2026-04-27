"""Combined per-region L/R balance check (251637 + new monkeys)."""
import pandas as pd
import os

df = pd.read_csv(
    r"d:\projectome_analysis\group_analysis\recovery\all_refined_neurons.csv"
)

print("Per-monkey x sub-region x side:")
print(pd.crosstab([df["SampleID"], df["Soma_Region_Refined"]],
                  df["Soma_Side_Inferred"]))
print()

new_L = int((df["Soma_Side_Inferred"] == "L").sum())
new_R = int((df["Soma_Side_Inferred"] == "R").sum())
print("Combined dataset (251637 untouched + new monkeys):")
print(f"  251637 only:  103 L : 157 R = 260 neurons")
print(f"  New monkeys:  {new_L} L : {new_R} R = {len(df)} neurons")
print(f"  COMBINED:     {103 + new_L} L : {157 + new_R} R = {260 + len(df)} neurons")
print()

print("Sub-region representation:")
ref251637 = {"IAL": (7, 95), "IAPM": (16, 0), "IDD5": (33, 30),
             "IDM": (44, 32), "IDV": (3, 0)}
new_pr = (pd.crosstab(df["Soma_Region_Refined"], df["Soma_Side_Inferred"])
          .fillna(0).astype(int))
all_regions = sorted(set(list(ref251637.keys()) + list(new_pr.index)))
for reg in all_regions:
    L0, R0 = ref251637.get(reg, (0, 0))
    L1 = int(new_pr.loc[reg, "L"]) if reg in new_pr.index and "L" in new_pr.columns else 0
    R1 = int(new_pr.loc[reg, "R"]) if reg in new_pr.index and "R" in new_pr.columns else 0
    flag = "  [NEW]" if reg not in ref251637 else ""
    print(f"  {reg:8s}  251637 = {L0:3d}L : {R0:3d}R   +   new = {L1:3d}L : {R1:3d}R   =>   total = {L0+L1:3d}L : {R0+R1:3d}R{flag}")
