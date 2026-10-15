# SQ3d Method (Mixed Explanation)

## What SQ3d is trying to answer

SQ3d estimates **group-level hemispheric asymmetry** of projection targeting:
- Compare **L-source** vs **R-source** neuron groups
- Region by region
- In a composition-focused way (not total projection magnitude)

---

## 1) Per-neuron normalization first

Start from ipsi projection matrix \(x_{rj}\):
- \(r\): target region
- \(j\): neuron

For each neuron \(j\), normalize by that neuron's total projection:

\[
p_{rj}=\frac{x_{rj}}{\sum_r x_{rj}}
\]

Interpretation:
- each neuron contributes a **targeting profile**
- analysis focuses on **where projections go**, not how large total output is

---

## 2) Strata used in current SQ3d

- `all_neurons`
- `region_<Soma_Region>` only for source soma regions that contain both L and R neurons

So SQ3d is currently: global stratum + bilateral-source regional strata.

---

## 3) Compute L/R means per target region

Within stratum \(s\), for each target region \(r\):

\[
\mu_{L,r,s}=\text{mean}(p_{rj},\ j\in L,s),\qquad
\mu_{R,r,s}=\text{mean}(p_{rj},\ j\in R,s)
\]

---

## 4) Compute laterality index

\[
LI_{r,s}=\frac{\mu_{L,r,s}-\mu_{R,r,s}}{\mu_{L,r,s}+\mu_{R,r,s}+\epsilon}
\]

Reading:
- \(LI>0\): relatively more incoming projection from L-source group
- \(LI<0\): relatively more from R-source group
- \(|LI|\): asymmetry magnitude

---

## 5) Statistical test per region

For each \((r,s)\), compare L vs R neuron values with Wilcoxon rank-sum:

\[
p_{r,s}
\]

Then BH correction within family:

\[
q_{r,s}=\mathrm{BH}(p_{r,s})
\]

Significance rule:
- \(q_{r,s}<0.05\)

---

## 6) What this method is and is not

This method **is**:
- group-level asymmetry in projection **composition**

This method is **not**:
- Gou-style per-neuron ipsi-vs-contra LI

---

## Quick practical interpretation

If a target region has:
- `mean_L > mean_R`, it tends to receive relatively more from L-source neurons
- `mean_R > mean_L`, vice versa

The significance test tells whether that difference is stable enough beyond sample noise after multiple-testing correction.
