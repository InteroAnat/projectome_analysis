# ==============================================================================
# R Script: Hybrid Morphological Clustering with Penalty (Refactored)
# Features:
# 1. Uses tidyr::pivot_wider (equivalent to Python's pandas.pivot)
# 2. Spearman vs Log1p Toggle
# 3. Supervised Penalty (Force biological separation)
# 4. C-index Optimization
# ==============================================================================

library(tidyverse)
library(readxl)
library(writexl)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Update paths (Use forward slashes /)
DIST_FILE <- "D:/projectome_analysis/main_scripts/dist.txt"
TYPE_FILE <- "D:/projectome_analysis/main_scripts/neuron_tables/INS_df_v3.xlsx"

# --- SETTINGS ---
USE_SPEARMAN <- TRUE        # TRUE = Rank-based, FALSE = Magnitude-based
USE_PENALTY  <- TRUE        # TRUE = Apply Supervised Penalty
PENALTY_STRENGTH <- 1.5     # Multiplier (1.5x Max Distance)
MAX_K <- 65                 # Maximum clusters for C-index optimization

# ==============================================================================
# 2. LOAD DATA (Equivalent to Python pivot)
# ==============================================================================
message("--- Loading Data ---")

# Read raw distance data (skip header row, assign column names)
raw_df <- read_tsv(DIST_FILE, 
                   col_names = c("i", "j", "score", "match", "nomatch"), 
                   skip = 1,
                   show_col_types = FALSE)

# Convert to numeric
raw_df$i <- as.numeric(raw_df$i)
raw_df$j <- as.numeric(raw_df$j)
raw_df$score <- as.numeric(raw_df$score)

# Pivot to square matrix (equivalent to Python's df.pivot(index='i', columns='j', values='score'))
message(">> Creating distance matrix (pivot)...")

raw_matrix <- raw_df %>% 
  select(i, j, score) %>%
  pivot_wider(id_cols = i, 
              names_from = j, 
              values_from = score, 
              values_fill = 0) %>%
  column_to_rownames("i") %>%
  as.matrix()

# Ensure matrix is square (handle any missing rows/columns)
all_ids <- sort(union(as.numeric(rownames(raw_matrix)), 
                      as.numeric(colnames(raw_matrix))))

# Reindex to ensure square matrix
raw_matrix <- raw_matrix[as.character(all_ids), as.character(all_ids), drop = FALSE]
raw_matrix[is.na(raw_matrix)] <- 0

# Fill diagonal with 0
diag(raw_matrix) <- 0

message(paste("    Matrix size:", nrow(raw_matrix), "x", ncol(raw_matrix)))

# ==============================================================================
# 3. LOAD BIOLOGICAL TYPES & FILTER
# ==============================================================================
message(">> Loading neuron types...")

if (grepl(".xlsx$", TYPE_FILE)) {
  bio_df <- read_excel(TYPE_FILE)
} else {
  bio_df <- read_csv(TYPE_FILE, show_col_types = FALSE)
}

# Create type mapping
type_map <- setNames(bio_df$Neuron_Type, bio_df$NeuronID)

# Filter to common neurons
common_neurons <- intersect(rownames(raw_matrix), names(type_map))

if (length(common_neurons) == 0) {
  stop("No common neurons found between distance matrix and type file!")
}

raw_matrix <- raw_matrix[common_neurons, common_neurons, drop = FALSE]

message(paste("    Final Neurons:", length(common_neurons)))

# ==============================================================================
# 4. TRANSFORMATION (Spearman or Log1p)
# ==============================================================================
if (USE_SPEARMAN) {
  message("--- Mode: Spearman Correlation ---")
  
  # Calculate Spearman correlation between neurons (rows)
  spearman_corr <- cor(t(raw_matrix), method = "spearman", use = "pairwise.complete.obs")
  
  # Handle NA values (zero variance neurons)
  spearman_corr[is.na(spearman_corr)] <- 0
  
  # Convert correlation to distance (0 = identical, 1 = uncorrelated, 2 = opposite)
  dist_matrix <- 1 - spearman_corr
  
  # Clip negative values from floating point errors
  dist_matrix[dist_matrix < 0] <- 0
  
} else {
  message("--- Mode: Log1p Magnitude ---")
  dist_matrix <- log1p(raw_matrix)
}

# Safety checks
dist_matrix[!is.finite(dist_matrix)] <- max(dist_matrix[is.finite(dist_matrix)], na.rm = TRUE)
diag(dist_matrix) <- 0

# Ensure row/column names are preserved
rownames(dist_matrix) <- rownames(raw_matrix)
colnames(dist_matrix) <- colnames(raw_matrix)

# ==============================================================================
# 5. APPLY SUPERVISED PENALTY
# ==============================================================================
if (USE_PENALTY) {
  message(paste("--- Applying Penalty (Strength:", PENALTY_STRENGTH, ") ---"))
  
  max_val <- max(dist_matrix, na.rm = TRUE)
  if (max_val == 0) max_val <- 1  # Avoid zero penalty
  
  penalty_val <- max_val * PENALTY_STRENGTH
  message(paste("    Natural Max:", round(max_val, 4), "| Penalty:", round(penalty_val, 4)))
  
  # Get types for current neurons
  current_types <- type_map[rownames(dist_matrix)]
  current_types[is.na(current_types)] <- "Unknown"
  
  # Create boolean mask: TRUE where types are DIFFERENT
  diff_mask <- outer(current_types, current_types, "!=")
  
  # Apply penalty to between-type distances
  dist_matrix[diff_mask] <- dist_matrix[diff_mask] + penalty_val
  
  message("    Penalty applied successfully.")
}

# Convert to dist object for clustering
dist_obj <- as.dist(dist_matrix)

# ==============================================================================
# 6. C-INDEX CALCULATION
# ==============================================================================
calculate_c_index <- function(dist_matrix, dist_obj, max_k = 65) {
  message("--- Calculating C-index ---")
  
  # Get sorted pairwise distances
  all_dists <- sort(as.vector(dist_obj))
  
  c_indices <- numeric(max_k - 1)
  names(c_indices) <- 2:max_k
  
  for (k in 2:max_k) {
    # Hierarchical clustering
    hc <- hclust(dist_obj, method = "ward.D2")
    labels <- cutree(hc, k = k)
    
    # Calculate within-cluster sum
    S <- 0
    N_intra <- 0
    
    for (cid in unique(labels)) {
      idx <- which(labels == cid)
      if (length(idx) > 1) {
        sub <- dist_matrix[idx, idx, drop = FALSE]
        S <- S + sum(sub[upper.tri(sub)])
        N_intra <- N_intra + choose(length(idx), 2)
      }
    }
    
    if (N_intra == 0) {
      c_indices[k-1] <- 1.0
      next
    }
    
    S_min <- sum(all_dists[1:N_intra])
    S_max <- sum(tail(all_dists, N_intra))
    
    c <- if (S_max == S_min) 0 else (S - S_min) / (S_max - S_min)
    c_indices[k-1] <- c
    
    cat(sprintf("\r    k=%d | C=%.4f | N_intra=%d", k, c, N_intra))
  }
  cat("\n")
  
  # Find best k (minimum C-index)
  best_k <- as.integer(names(which.min(c_indices)))
  
  # Elbow detection (optional)
  deltas <- diff(c_indices)
  elbow_k <- NULL
  for (i in 2:length(deltas)) {
    if (deltas[i] > -0.01 && deltas[i-1] < -0.05) {
      elbow_k <- i + 1
      break
    }
  }
  
  # Plot C-index curve
  plot(2:max_k, c_indices, type = "o", col = "purple",
       main = paste("C-index Optimization (Penalty =", ifelse(USE_PENALTY, "ON", "OFF"), ")"),
       xlab = "Number of Clusters (k)", ylab = "C-index")
  abline(v = best_k, col = "red", lty = 2, lwd = 2)
  if (!is.null(elbow_k)) {
    abline(v = elbow_k, col = "orange", lty = 2)
  }
  grid()
  legend("topright", legend = c(paste("Best k =", best_k)), 
         col = "red", lty = 2, lwd = 2)
  
  message(paste(">> Optimal K (minimum C-index):", best_k))
  if (!is.null(elbow_k)) {
    message(paste(">> Elbow detected at k =", elbow_k))
  }
  
  return(best_k)
}

# Calculate optimal k
best_k <- calculate_c_index(dist_matrix, dist_obj, max_k = MAX_K)

# ==============================================================================
# 7. FINAL CLUSTERING & SAVE
# ==============================================================================
# Allow user to override
k_input <- readline(prompt = paste("Enter K (Press Enter for", best_k, "): "))
k_final <- suppressWarnings(as.integer(k_input))
if (is.na(k_final)) k_final <- best_k

message(paste("\n--- Final Clustering with k =", k_final, "---"))

# Perform hierarchical clustering
hc <- hclust(dist_obj, method = "ward.D2")
clusters <- cutree(hc, k = k_final)

# Create results dataframe
results_df <- data.frame(
  NeuronID = names(clusters),
  Bio_Type = type_map[names(clusters)],
  Subtype_Cluster = clusters,
  row.names = NULL
)

# Save results
filename <- paste0("R_Results_", ifelse(USE_PENALTY, "Penalty", "NoPenalty"), "_k", k_final, ".xlsx")
write_xlsx(results_df, filename)
message(paste(">> Saved:", filename))

# ==============================================================================
# 8. HEATMAP VISUALIZATION
# ==============================================================================
message("--- Generating Heatmap ---")

# Create annotation data
anno_row <- data.frame(BioType = factor(type_map[rownames(dist_matrix)]))
rownames(anno_row) <- rownames(dist_matrix)

# Choose colors based on mode
if (USE_SPEARMAN) {
  cols <- viridis::mako(100)
  title_suffix <- "Spearman"
} else {
  cols <- viridis::viridis(100, direction = -1)
  title_suffix <- "Log1p"
}

penalty_suffix <- ifelse(USE_PENALTY, paste("+ Penalty (", PENALTY_STRENGTH, "x)", sep = ""), "")

# Generate heatmap
pheatmap::pheatmap(dist_matrix,
                   clustering_distance_rows = dist_obj,
                   clustering_distance_cols = dist_obj,
                   clustering_method = "ward.D2",
                   annotation_row = anno_row,
                   show_rownames = FALSE, 
                   show_colnames = FALSE,
                   col = cols,
                   main = paste(title_suffix, penalty_suffix, "| k =", k_final))

message("--- Done ---")
