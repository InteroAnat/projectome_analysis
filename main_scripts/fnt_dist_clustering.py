#%%
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
df = pd.read_csv('dist.txt', sep='\t', header=0,
                 names=['i', 'j', 'score', 'match', 'nomatch'])
#%%
# 2. Pivot
print("Pivoting data...")

# Ensure numeric types
df['i'] = pd.to_numeric(df['i'], errors='coerce')
df['j'] = pd.to_numeric(df['j'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')

# Pivot using lowercase names
matrix = df.pivot(index='i', columns='j', values='score')

print(f"Matrix Shape: {matrix.shape}")
print(matrix.iloc[:10, :10])
#%%
# 1. Load Raw Data
# df was loaded earlier from 'dist.txt'
# df has columns: ['i', 'j', 'score']

# 2. Get Unique IDs
# We check both 'i' (Source) and 'j' (Target) to be safe
all_ids = pd.concat([df['i'], df['j']]).unique()

# 3. Determine Count
expected_count = len(all_ids)
min_id = all_ids.min()
max_id = all_ids.max()

print(f"--- DATA INTEGRITY CHECK ---")
print(f"Total Unique Neurons in Input File: {expected_count}")
print(f"ID Range: {min_id} to {max_id}")

# 4. Check for 'Ghost 0' immediately
if min_id == 0:
    print("Warning: ID 0 detected. This is likely a Ghost.")
    expected_count_real = expected_count - 1 # We plan to drop one
else:
    expected_count_real = expected_count

print(f"Real Expected Count (excluding ghosts): {expected_count_real}")
#%%
# 3. Fill Gaps
print("Symmetrizing...")

# Combine matrix with its Transpose to fill missing pairs
matrix = matrix.combine_first(matrix.T)

# Fill self-distance (Diagonal) with 0
np.fill_diagonal(matrix.values, 0)

# Check for remaining empty spots (NaN)
if matrix.isnull().values.any():
    print("Warning: Some pairs are missing. Filling with Max Value.")
    matrix = matrix.fillna(matrix.max().max())

print("Matrix ready.")
# %%
# 4. Plot
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage

print("Generating Heatmap...")

# A. Log Transform (Essential for FNT scale)
plot_data = np.log1p(matrix)
# plot_data = matrix

# B. Calculate Linkage Manually
# squareform converts the Matrix into the Vector format expected by 'linkage'
# This ensures it treats your data as Distances, not Coordinates.
condensed_dist = squareform(plot_data)
linkage_matrix = linkage(condensed_dist, method='ward')

# C. Plot
plt.figure(figsize=(10, 10))

sns.clustermap(plot_data, 
               row_linkage=linkage_matrix, # <--- PASS MANUAL LINKAGE
               col_linkage=linkage_matrix, # <--- PASS MANUAL LINKAGE
               cmap='viridis_r',           # Reversed: Dark=Similar
               xticklabels=True, 
               yticklabels=True)

plt.title("Neuron Clustering (FNT Distance)")
plt.show()
# %%

from scipy.cluster.hierarchy import linkage, fcluster

def extract_clusters(matrix, k=3):
    """
    Cuts the tree to get exactly 'k' clusters.
    Returns a Dictionary {NeuronID: ClusterID}
    """
    # 1. Prepare Data (Log transform)
    data = np.log1p(matrix)
    
    # 2. Calculate Linkage (Same math as clustermap)
    # 'ward' minimizes variance within clusters
    Z = linkage(data, method='ward', metric='euclidean')
    
    # 3. Cut the Tree
    # criterion='maxclust' means "Give me exactly k groups"
    labels = fcluster(Z, t=k, criterion='maxclust')
    
    # 4. Create Result Table
    results = pd.DataFrame({
        'NeuronID': matrix.index,
        'Cluster': labels
    })
    
    # Sort by Cluster for readability
    return results.sort_values('Cluster')

# --- RUN IT ---
# Try k=2 (PT vs IT) or k=3 (PT vs IT_A vs IT_B)
clusters = extract_clusters(matrix, k=3)

print(clusters)

# Save to CSV
clusters.to_csv("fnt_cluster_labels.csv", index=False)

# %%
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

def suggest_clusters(matrix, max_k=10):
    """
    Analyzes the matrix to suggest the best number of clusters (k).
    """
    print(f"Analyzing Cluster Quality (k=2 to {max_k})...")
    
    # 1. Prepare Data (Log Scale)
    data = np.log1p(matrix)
    
    # 2. Calculate Linkage (Ward)
    condensed = squareform(data)
    Z = linkage(condensed, method='ward')
    
    # Storage
    scores = []
    ks = range(2, max_k + 1)
    
    # 3. Test every k
    for k in ks:
        # Cut tree to get k clusters
        labels = fcluster(Z, t=k, criterion='maxclust')
        
        # Calculate Silhouette Score (-1 to +1)
        # +1 = Perfect clusters, 0 = Overlapping, -1 = Wrong
        # We pass the DISTANCE matrix (data) as the metric precomputed
        score = silhouette_score(data, labels, metric='precomputed')
        scores.append(score)
        print(f"  k={k}: Silhouette = {score:.4f}")

    # 4. Plot The Result (The Elbow)
    plt.figure(figsize=(10, 4))
    plt.plot(ks, scores, 'bo-', linewidth=2)
    plt.axvline(x=ks[np.argmax(scores)], color='r', linestyle='--', label='Best Math Score')
    
    plt.title("Silhouette Analysis (Higher is Better)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    best_k = ks[np.argmax(scores)]
    print(f"\n>> Mathematical Recommendation: k = {best_k}")
    return best_k

# --- RUN IT ---
best_k = suggest_clusters(matrix)
# %%
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np

def plot_tree(matrix, cut_threshold=None):
    # 1. Prepare Data
    data_log = np.log1p(matrix)
    Z = linkage(squareform(data_log), method='ward')

    # 2. Plot
    plt.figure(figsize=(12, 6))
    
    # Dendrogram
    ddata = dendrogram(Z, 
                       labels=matrix.index, 
                       color_threshold=cut_threshold if cut_threshold else 0) # Colors the clusters
    
    plt.title("Neuron Family Tree (Dendrogram)")
    plt.xlabel("Neuron ID")
    plt.ylabel("Distance (Dissimilarity)")
    
    # 3. Add a Cut Line (Optional visual aid)
    if cut_threshold:
        plt.axhline(y=cut_threshold, c='r', ls='--', label=f'Cut at {cut_threshold}')
        plt.legend()
        
    plt.tight_layout()
    plt.show()

# --- HOW TO USE ---
# 1. Run without threshold to see the heights
plot_tree(matrix)

# 2. Look at the Y-axis (Height). 
#    Find a gap where the vertical lines are long.
#    Pick a number in that gap (e.g., 50).

# 3. Run again with that threshold to see the color groups
plot_tree(matrix, cut_threshold=30)
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def manual_cluster_extract(matrix, k):
    """
    You choose K. I give you the list.
    """
    print(f"--- MANUAL EXTRACTION (K={k}) ---")
    
    # 1. Standard FNT Prep
    data_log = np.log1p(matrix)
    Z = linkage(squareform(data_log), method='ward')
    
    # 2. Cut Tree
    labels = fcluster(Z, t=k, criterion='maxclust')
    
    # 3. Create DataFrame
    df_clusters = pd.DataFrame({
        'NeuronID': matrix.index,
        'Cluster_ID': labels
    })
    
    # 4. Print Summary
    counts = df_clusters['Cluster_ID'].value_counts().sort_index()
    print("Cluster Sizes:")
    print(counts)
    
    return df_clusters


def merge_strict_intersection(fnt_clusters_df, region_summary_path):
    """
    Merges FNT Clusters with Biological Types.
    STRICTLY restricts data to neurons present in BOTH lists.
    """
    print("--- MERGING DATASETS (STRICT INTERSECTION) ---")
    
    # 1. Load the Biological Data (The Big List)
    bio_df = pd.read_csv(region_summary_path)
    
    # 2. Normalize IDs to ensure they match
    # We strip whitespace to be safe
    fnt_clusters_df['NeuronID'] = fnt_clusters_df['NeuronID'].astype(str).str.strip()
    bio_df['NeuronID'] = bio_df['NeuronID'].astype(str).str.strip()
    
    # 3. Validation Prints (Before Merge)
    count_fnt = len(fnt_clusters_df)
    count_bio = len(bio_df)
    print(f"1. FNT Cluster List (Insula Only):  {count_fnt} neurons")
    print(f"2. Region Analysis List (All):      {count_bio} neurons")
    
    # 4. THE MERGE (Inner Join)
    # how='inner' drops rows that don't match in both tables
    master_df = pd.merge(fnt_clusters_df, bio_df, on='NeuronID', how='inner')
    
    # 5. Validation Prints (After Merge)
    count_merged = len(master_df)
    dropped = count_bio - count_merged
    
    print("-" * 40)
    print(f"3. FINAL MERGED DATASET:            {count_merged} neurons")
    print(f"   -> {dropped} neurons were dropped (Non-Insula / No FNT data)")
    print("-" * 40)
    
    # Save
    master_df.to_csv("Insula_FNT_vs_Type_Master.csv", index=False)
    return master_df

def plot_correspondence(master_df):
    """
    Visualizes: Do Structural Clusters map to PT/IT/CT types?
    """
    if master_df.empty:
        print("Error: Merged dataframe is empty.")
        return

    # Cross-Tabulation
    ct = pd.crosstab(master_df['Cluster_ID'], master_df['Neuron_Type'])
    
    print("\nCluster Composition:")
    print(ct)
    
    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(ct, annot=True, fmt='d', cmap='Blues', linewidths=1)
    plt.title("Insula Structural Clusters vs. Projection Types")
    plt.ylabel("FNT Structural Cluster")
    plt.xlabel("Projection Type")
    plt.show()

# ==========================================
# RUN IT
# ==========================================
# Assuming 'df_clusters' is your output from the auto_cluster function
# And 'Summary_Results.csv' is your file from ProjectomeAnalysis
MY_CHOSEN_K = 7

# 3. Run Extraction
df_clusters = manual_cluster_extract(matrix, k=MY_CHOSEN_K)

# 4. Now run the Merger with this manual dataframe
master_data = merge_strict_intersection(df_clusters, "Summary_Results.csv")
if master_data is not None:
    plot_correspondence(master_data)
# %%
import os
import pandas as pd

def map_clusters_zero_based(matrix, k, fnt_folder_path):
    print(f"--- MAPPING (0-BASED INDEX) ---")
    
    # 1. Get File List (Sorted)
    # The tool processed these: [File_0, File_1, ... File_211]
    file_list = sorted([f for f in os.listdir(fnt_folder_path) if f.endswith('.decimate.fnt')])
    
    print(f"Files found: {len(file_list)} (Should be 212)")
    print(f"Matrix rows: {len(matrix)} (Should be 212)")
    
    # 2. Check Alignment
    if len(file_list) != len(matrix):
        print("CRITICAL WARNING: File count does not match Matrix row count!")
    
    # 3. Create Mapping Dictionary
    # Since matrix IDs are 0..211, and List indices are 0..211, they match perfectly.
    id_map = {}
    for numeric_id in matrix.index:
        # Direct mapping: ID 0 -> List[0]
        if 0 <= numeric_id < len(file_list):
            clean_name = file_list[numeric_id].replace('.decimate.fnt', '')
            id_map[numeric_id] = clean_name
        else:
            id_map[numeric_id] = f"Unknown_ID_{numeric_id}"

    # 4. Standard Clustering
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    data_log = np.log1p(matrix)
    Z = linkage(squareform(data_log), method='ward')
    labels = fcluster(Z, t=k, criterion='maxclust')
    
    # 5. Build Result
    # Map the Index (0..211) to the Name ("001.swc"...)
    mapped_names = [id_map[i] for i in matrix.index]
    
    df_clusters = pd.DataFrame({
        'NeuronID': mapped_names,
        'Cluster_ID': labels
    })
    
    print("\nMapping Verification:")
    print(f"ID 0   mapped to -> {id_map.get(0)}")
    print(f"ID 211 mapped to -> {id_map.get(211)}")
    
    return df_clusters

# ==========================================
# RUN IT
# ==========================================
FNT_FOLDER = r'./processed_neurons/251637/fnt_processed'

# Do NOT drop anything. Pass the full matrix.
df_clusters = map_clusters_zero_based(matrix, k=3, fnt_folder_path=FNT_FOLDER)

# Check the first few rows to ensure names look right
print(df_clusters.head())
# %%
