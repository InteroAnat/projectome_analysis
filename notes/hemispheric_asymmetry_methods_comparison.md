# Hemispheric Asymmetry: Gou et al. vs. Current Analysis

**Date:** 2026-04-16
**Status:** Historical schematic comparison; factual method labels corrected 2026-10-09

The deposited Gou implementation distinguishes axon-length laterality from arbor-count laterality. The raw-length formula below describes one measurement, not every published arbor endpoint. Contra/ipsi describes a projection relative to its source hemisphere; it does not mean absolute right/left. In SQ3d, targets index hypotheses and neuron values enter the tests; animal independence is not established by target-wise BH correction.

---

## Two Approaches to Measuring Hemispheric Asymmetry

### 1. Gou et al. (reference implementation)

**Step 1 — per-neuron LI:**
For each neuron j, using raw axon lengths:

```
LI_j = (contra_len_j − ipsi_len_j) / (contra_len_j + ipsi_len_j)
```

- Each neuron gets one LI value (range −1 to +1)
- Purely ipsilateral → LI = −1
- Purely contralateral → LI = +1
- Equal projection → LI = 0

**Step 2 — test distributions:**
Compare the distribution of per-neuron LI values between groups (e.g., PT vs CT) using the Mann-Whitney U test.

```
Group_A_LIs = [LI_neuron1, LI_neuron2, ...]  ← many individual values
Group_B_LIs = [LI_neuron3, LI_neuron4, ...]  ← many individual values
→ Mann-Whitney U: are these two distributions different?
```

**Key:** every statistical unit is a neuron.

---

### 2. Current v3 Rmd — SQ3d

**Step 1 — per-neuron proportion normalization:**
Each neuron j's projection vector is normalized to sum to 1:

```
p_ij = projection_to_region_i / sum(all_regions_for_neuron_j)
```

**Step 2 — per-region group means:**
For each target region i, compute mean normalized projection from L-source vs R-source groups:

```
mean_L_i = average(p_ij for all LEFT-source neurons)
mean_R_i = average(p_ij for all RIGHT-source neurons)
```

**Step 3 — per-region LI:**

```
LI_region_i = (mean_L_i − mean_R_i) / (mean_L_i + mean_R_i + eps)
```

**Step 4:** Wilcoxon test per region comparing L vs R neuron values.

**Key:** each brain region defines a separate hypothesis; the tested observations are neuron values in L-source and R-source groups. Neurons nested within animals are not thereby independent animal replicates.

---

## Side-by-Side Comparison

| | Gou et al. | Current SQ3d |
|---|---|---|
| **What gets a LI value?** | Each neuron | Each target brain region |
| **Formula** | `(contra − ipsi) / (contra + ipsi)` on raw lengths | `(mean_L − mean_R) / (mean_L + mean_R + eps)` on normalized group-mean profiles |
| **Preprocessing** | None (raw lengths) | Per-neuron proportion normalization |
| **What is tested** | Distribution of per-neuron LIs across types | Per-region group-mean LI, tested neuron-by-neuron |
| **Test observations / hypothesis** | Neuron LI values / group contrast | Neuron normalized values / one hypothesis per target region |
| **Test used** | Mann-Whitney U | Wilcoxon rank-sum |
| **Interpretation** | "Higher LI denotes relatively more contralateral than ipsilateral projection" | "Which target regions receive systematically asymmetric inputs from left vs right source neurons?" |

---

## Why the Difference Matters

### Gou's approach
- Per-neuron asymmetry score — plottable, summarizable
- Easier biological interpretation
- Aligned with single-neuron projectome concept
- Natural unit for downstream type/region comparisons

### Current SQ3d
- Asymmetric **targeting** — which brain regions receive differential input
- Answers: "is the projection *field* left-right biased?"
- Does not give per-neuron LI scores
- More about network-level input balance

---

## Proposed Addition: Per-Neuron LI Block

To align more closely with Gou et al. while preserving current analysis, a new per-neuron LI block could be added:

1. For each neuron, compute: `LI_j = (contra_total_j − ipsi_total_j) / (contra_total_j + ipsi_total_j)`
2. Compare L-source vs R-source neuron LI distributions via Mann-Whitney U
3. Stratify by neuron type (IT, PT, CT, etc.)

This gives a complementary answer: "are left-sourced and right-sourced neurons globally different in projection asymmetry?"

---

## Statistical Tests Used in This Workflow

### Mann-Whitney U (Wilcoxon rank-sum)

**What it does:** Tests whether two independent groups have different distributions.

**How it works:** Pool all values, rank them, sum ranks per group → U statistic.

**Key property:** Non-parametric — does NOT assume normal distribution.

**Example:** Comparing LI values between IT neurons vs PT neurons (different neurons, independent groups).

```
Group A: [LI_neuron1, LI_neuron2, ...]  ← IT neurons
Group B: [LI_neuron3, LI_neuron4, ...]  ← PT neurons
→ Mann-Whitney U: are these two rank distributions different?
```

---

### Wilcoxon signed-rank test

**What it does:** Tests whether paired/matched observations differ.

**Key property:** Non-parametric, requires paired data.

**Example:** Same neurons measured before and after treatment.

```
Neuron 1: LI_before = 0.3, LI_after = 0.5  → diff = 0.2
Neuron 2: LI_before = 0.8, LI_after = 0.6  → diff = -0.2
→ Wilcoxon signed-rank: are paired differences centered at 0?
```

---

### t-test (independent samples)

**What it does:** Tests whether two independent group means differ.

**Key property:** Parametric — assumes normal distribution (or large enough n).

**Mann-Whitney is the non-parametric alternative to the t-test.**

---

### PERMANOVA

**What it does:** Multivariate extension of ANOVA for distance matrices.

**How it works:**
1. Compute pairwise distances (e.g., Bray-Curtis) between all observations
2. Test whether groups have different centroids in that distance space
3. Uses permutation (not analytic distribution) for p-values

**When to use:** When your "outcome" is a high-dimensional vector (e.g., projection to 100 brain regions), not a single number.

---

### BH (Benjamini-Hochberg) correction

Not a test — a **multiple testing correction** applied after tests.

**Problem:** Running 50 tests at α=0.05 → expect ~2.5 false positives by chance.

**BH method:** Controls the **False Discovery Rate** (FDR), not family-wise error rate. Less conservative than Bonferroni.

---

## Quick Reference Table

| Test | Groups | Paired? | Parametric? | What it tests |
|---|---|---|---|---|
| t-test (independent) | 2 | No | Yes | Difference in **means** |
| Mann-Whitney U | 2 | No | No | Difference in **rank distributions** |
| Wilcoxon signed-rank | 2 | Yes | No | Difference in **paired observations** |
| PERMANOVA | ≥2 | No | No | Difference in multivariate **centroids** (distance-based) |
| BH correction | Any | — | — | Controls FDR across multiple tests |

---

## Which Test to Use When

- **Single continuous outcome, two independent groups** → t-test (normal) or Mann-Whitney U (non-normal)
- **Same subjects, two conditions** → Wilcoxon signed-rank (non-normal paired)
- **High-dimensional "profile" (e.g., projection to 100 regions)** → PERMANOVA on distance matrix
- **Many tests at once (e.g., 50 regions)** → apply BH correction after

---

## Summary

- Gou et al. and the current v3 analysis answer **complementary but different questions**.
- Gou: "how lateralized are individual neurons?"
- Current v3: "which target regions receive biased inputs from left vs right source neurons?"
- Both are valid; the choice depends on scientific question.
- Adding a per-neuron LI block would allow direct comparison with Gou's framework.
