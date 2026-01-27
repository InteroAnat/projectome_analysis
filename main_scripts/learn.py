import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# ==========================================
# CONFIGURATION
# ==========================================
USE_REAL_DATA = True # Set to True to load your 'dist.txt'
FILE_PATH = 'dist.txt'

# ==========================================
# 1. LOAD OR SIMULATE DATA
# ==========================================
if USE_REAL_DATA:
    print(f"Loading {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH, sep='\t', header=0, names=['i', 'j', 'score', 'match', 'nomatch'])
    # Force numeric
    df['i'] = pd.to_numeric(df['i'], errors='coerce')
    df['j'] = pd.to_numeric(df['j'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
else:
    print("Generating Mock Data for demonstration...")
    # Simulate 10 neurons (IDs 0-9)
    # Scores range from 0.1 (Self) to 1,000,000,000 (Different)
    data = []
    ids = range(10)
    import random
    for i in ids:
        for j in ids:
            if i <= j: # Simulate FNT only calculating one direction
                if i == j: score = 0.23 # Self noise
                elif (i < 5 and j < 5): score = random.uniform(10, 100) # Cluster A (Similar)
                elif (i >= 5 and j >= 5): score = random.uniform(10, 100) # Cluster B (Similar)
                else: score = random.uniform(1e8, 1e9) # A vs B (Very Different)
                
                data.append({'i': i, 'j': j, 'score': score})
    df = pd.DataFrame(data)

# ==========================================
# STEP 1: THE RAW PIVOT (Visualizing Gaps)
# ==========================================
# Pivot to Matrix
matrix_raw = df.pivot(index='i', columns='j', values='score')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(matrix_raw, cmap='viridis', cbar=False)
plt.title("Step 1: Raw Output (Note the White Gaps!)")
plt.xlabel("Target Neuron (J)")
plt.ylabel("Source Neuron (I)")

# ==========================================
# STEP 2: SYMMETRY (Filling Gaps)
# ==========================================
matrix_sym = matrix_raw.combine_first(matrix_raw.T)
np.fill_diagonal(matrix_sym.values, 0)

plt.subplot(1, 2, 2)
sns.heatmap(matrix_sym, cmap='viridis', cbar=False)
plt.title("Step 2: Symmetrized (Gaps Filled)")
plt.show()

# ==========================================
# STEP 3: LOG TRANSFORMATION (The Scale Problem)
# ==========================================
# Why we do this?
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Plot Linear Histogram
ax[0].hist(matrix_sym.values.flatten(), bins=20, color='gray')
ax[0].set_title("Raw Distances (Linear Scale)")
ax[0].set_xlabel("Billions")
ax[0].text(0.5, 0.5, "Everything is bunched here ->", transform=ax[0].transAxes, ha='center')

# Plot Log Histogram
matrix_log = np.log1p(matrix_sym)
ax[1].hist(matrix_log.values.flatten(), bins=20, color='teal')
ax[1].set_title("Log Transformed (Log Scale)")
ax[1].set_xlabel("Log(Distance)")

plt.suptitle("Step 3: Why we use Log (To see structure)")
plt.show()

# ==========================================
# STEP 4: LINKAGE (The Tree Logic)
# ==========================================
# Convert to condensed distance vector
condensed_dist = squareform(matrix_log)

# Calculate Linkage
Z = linkage(condensed_dist, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z, labels=matrix_sym.index, leaf_rotation=0)
plt.title("Step 4: The Dendrogram (How Python groups them)")
plt.xlabel("Neuron ID")
plt.ylabel("Cluster Distance")
plt.show()

# ==========================================
# STEP 5: FINAL CLUSTERMAP
# ==========================================
print("Step 5: Final Result")
g = sns.clustermap(matrix_log, 
               row_linkage=Z, 
               col_linkage=Z, 
               cmap='viridis_r', 
               figsize=(8, 8))
g.fig.suptitle("Final: Reordered Matrix")
plt.show()