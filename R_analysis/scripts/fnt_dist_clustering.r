# ==============================================================================
# R Script: Hybrid Morphological Clustering with Penalty
# Features:
# 1. Natural Sorting (Fixes alignment issues)
# 2. Spearman vs Log1p Toggle
# 3. Supervised Penalty (Force biological separation)
# 4. NbClust C-index Optimization
# ==============================================================================

library(tidyverse)
library(readxl)
library(writexl)
library(NbClust)
library(pheatmap)
library(viridis)
library(stringr) # Required for natural sorting

# ==========================================
# 0. LOAD & PROCESS FNT-DIST → CLUSTERING MATRIX
# ==========================================

#' Build a processed FNT-distance matrix for clustering
#'
#' Reads a tab-separated pairwise distance file, symmetrizes it, aligns row/column
#' order with `*.decimate.fnt` basenames in `fnt_folder`, intersects neurons with
#' `neuron_table`, then optionally converts to a Spearman or log1p distance
#' matrix and applies the supervised type penalty.
#'
#' @param dist_file Path to TSV: columns i, j, score (with or without header row).
#' @param neuron_table Path to `.xlsx` or `.csv` with columns `NeuronID`, `Neuron_Type`.
#' @param fnt_folder Directory containing `*.decimate.fnt` files (ordering / ID labels).
#' @param use_spearman If TRUE, use `1 - cor(..., method = "spearman")`; else `log1p(raw)`.
#' @param use_penalty If TRUE, add `penalty_strength * max(dist)` between distinct types.
#' @param penalty_strength Multiplier applied to the max finite distance for the penalty.
#'
#' @return A list: `raw_matrix`, `dist_matrix`, `dist_obj` (`as.dist(dist_matrix)`),
#'   `type_map` (named vector NeuronID → Neuron_Type), `neuron_ids` (rownames order).
load_processed_fnt_dist_matrix <- function(dist_file,
                                           neuron_table,
                                           fnt_folder,
                                           use_spearman = TRUE,
                                           use_penalty = FALSE,
                                           penalty_strength = 1.5) {
  message("--- Loading Data ---")

  raw_df <- read_tsv(dist_file, col_names = FALSE, show_col_types = FALSE)

  first_val <- raw_df[[1, 3]]
  if (is.character(first_val) && !grepl("^[0-9.]+$", first_val)) {
    colnames(raw_df) <- as.character(raw_df[1, ])
    raw_df <- raw_df[-1, ]
    colnames(raw_df) <- c("i", "j", "score")
  } else {
    raw_df <- raw_df[, 1:3]
    colnames(raw_df) <- c("i", "j", "score")
  }

  raw_df$i <- as.numeric(as.character(raw_df$i))
  raw_df$j <- as.numeric(as.character(raw_df$j))
  raw_df$score <- as.numeric(as.character(raw_df$score))
  raw_df <- na.omit(raw_df)

  message(">> Creating & Symmetrizing Matrix...")

  all_ids <- sort(unique(c(raw_df$i, raw_df$j)))
  n <- length(all_ids)

  full_mat <- matrix(0, nrow = n, ncol = n, dimnames = list(all_ids, all_ids))
  full_mat[cbind(as.character(raw_df$i), as.character(raw_df$j))] <- raw_df$score
  full_mat <- pmax(full_mat, t(full_mat))
  diag(full_mat) <- 0

  fnt_files <- list.files(fnt_folder, pattern = "\\.decimate\\.fnt$")
  sorted_files <- str_sort(fnt_files, numeric = TRUE)

  if (nrow(full_mat) != length(sorted_files)) {
    warning(paste(
      "Size Mismatch: Matrix has", nrow(full_mat),
      "rows, but folder has", length(sorted_files), "files."
    ))
    min_len <- min(nrow(full_mat), length(sorted_files))
    full_mat <- full_mat[1:min_len, 1:min_len]
    sorted_files <- sorted_files[1:min_len]
  }

  neuron_names <- gsub("\\.decimate\\.fnt", "", sorted_files)
  rownames(full_mat) <- neuron_names
  colnames(full_mat) <- neuron_names

  if (grepl("\\.xlsx$", neuron_table, ignore.case = TRUE)) {
    bio_df <- read_excel(neuron_table)
  } else {
    bio_df <- read_csv(neuron_table, show_col_types = FALSE)
  }

  type_map <- setNames(bio_df$Neuron_Type, bio_df$NeuronID)
  soma_col_candidates <- c("Soma_Region", "SomaRegion", "soma_region", "Soma", "soma")
  soma_col <- soma_col_candidates[soma_col_candidates %in% colnames(bio_df)][1]
  if (!is.na(soma_col)) {
    soma_map <- setNames(as.character(bio_df[[soma_col]]), bio_df$NeuronID)
  } else {
    soma_map <- setNames(rep("Unknown", nrow(bio_df)), bio_df$NeuronID)
  }
  common_neurons <- intersect(rownames(full_mat), names(type_map))
  raw_matrix <- full_mat[common_neurons, common_neurons]

  message(paste("    Final Neurons:", length(common_neurons)))

  # --- transformation ---
  if (use_spearman) {
    message("--- Mode: Spearman Correlation ---")
    spearman_corr <- cor(t(raw_matrix), method = "spearman", use = "pairwise.complete.obs")
    spearman_corr[is.na(spearman_corr)] <- 0
    dist_matrix <- 1 - spearman_corr
    dist_matrix[dist_matrix < 0] <- 0
  } else {
    message("--- Mode: Log1p Magnitude ---")
    dist_matrix <- log1p(raw_matrix)
  }

  dist_matrix[!is.finite(dist_matrix)] <- max(dist_matrix[is.finite(dist_matrix)], na.rm = TRUE)
  diag(dist_matrix) <- 0

  if (use_penalty) {
    message(paste("--- Applying Penalty (Strength:", penalty_strength, ") ---"))
    max_val <- max(dist_matrix, na.rm = TRUE)
    if (max_val == 0) max_val <- 1
    penalty_val <- max_val * penalty_strength
    current_types <- type_map[rownames(dist_matrix)]
    current_types[is.na(current_types)] <- "Unknown"
    diff_mask <- outer(current_types, current_types, "!=")
    dist_matrix[diff_mask] <- dist_matrix[diff_mask] + penalty_val
    message("    Penalty applied.")
  }

  dist_obj <- as.dist(dist_matrix)

  list(
    raw_matrix = raw_matrix,
    dist_matrix = dist_matrix,
    dist_obj = dist_obj,
    type_map = type_map,
    soma_map = soma_map,
    neuron_ids = common_neurons
  )
}

# ==========================================

# 1. CONFIGURATION
# ==========================================
# Update paths (Use forward slashes /)
DIST_FILE <- "D:\\projectome_analysis\\processed_neurons\\251637\\fnt_processed\\251637_INS_HE_inferred\\251637_INS_HE_inferred_dist.txt"
TYPE_FILE <- "../tables/251637_INS_HE_inferred.xlsx"
FNT_FOLDER <- "D:\\projectome_analysis\\processed_neurons\\251637\\fnt_processed\\251637_INS_HE_inferred\\"

# --- SETTINGS ---
USE_SPEARMAN <- TRUE        # TRUE = Rank-based, FALSE = Magnitude-based
USE_PENALTY  <- TRUE        # TRUE = Apply Supervised Penalty
PENALTY_STRENGTH <- 1.5     # Multiplier (1.5x Max Distance)
FIGURE_DIR <- "./figures/clustering"
DATA_OUTPUT_DIR <- "./data_output/clustering"
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(DATA_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
# ==========================================
# 2–3. LOAD, FILTER, TRANSFORM (via helper)
# ==========================================
proc <- load_processed_fnt_dist_matrix(
  dist_file = DIST_FILE,
  neuron_table = TYPE_FILE,
  fnt_folder = FNT_FOLDER,
  use_spearman = USE_SPEARMAN,
  use_penalty = USE_PENALTY,
  penalty_strength = PENALTY_STRENGTH
)

raw_matrix <- proc$raw_matrix
dist_matrix <- proc$dist_matrix
dist_obj <- proc$dist_obj
type_map <- proc$type_map
soma_map <- proc$soma_map

write_xlsx(as.data.frame(t(raw_matrix)), file.path(DATA_OUTPUT_DIR, "raw_matrix_神经元矩阵.xlsx"))
# ==========================================
# 4. OPTIMIZATION (NbClust C-index)
# ==========================================
message("--- Calculating C-index (This takes time...) ---")

# NbClust handles the search range
nb <- NbClust(data = NULL, 
              diss = dist_obj, 
              distance = NULL,
              min.nc = 2, 
              max.nc = 65, 
              method = "ward.D2", 
              index = "cindex")

best_k <- nb$Best.nc["Number_clusters"]
message(paste(">> Optimal K:", best_k))

# Plot Curve
png(file.path(FIGURE_DIR, "cindex_optimization.png"), width = 1200, height = 800, res = 150)
plot(names(nb$All.index), nb$All.index, type="o", col="purple",
     main="C-index Optimization", xlab="K", ylab="Score")
abline(v=best_k, col="red", lty=2)
dev.off()

# ==========================================
# 5. CLUSTERING & SAVING
# ==========================================
# Allow user override
k_final = 9
# Ward's Method
hc <- hclust(dist_obj, method = "ward.D2")
clusters <- cutree(hc, k = k_final)

# Quick check: print head swcids in each cluster
message("--- Cluster preview (head swcids) ---")
for (cl in sort(unique(clusters))) {
  swcids_in_cluster <- names(clusters)[clusters == cl]
  swcids_head <- head(swcids_in_cluster, 6)
  message(
    paste0(
      "Cluster ", cl,
      " (n=", length(swcids_in_cluster), "): ",
      paste(swcids_head, collapse = ", ")
    )
  )
}

# Quick overview: cluster vs soma region
message("--- Cluster vs Soma Region overview ---")
cluster_soma_df <- data.frame(
  Cluster = clusters,
  SomaRegion = soma_map[names(clusters)],
  stringsAsFactors = FALSE
)
cluster_soma_df$SomaRegion[is.na(cluster_soma_df$SomaRegion) | cluster_soma_df$SomaRegion == ""] <- "Unknown"
cluster_soma_overview <- as.data.frame.matrix(table(cluster_soma_df$Cluster, cluster_soma_df$SomaRegion))
print(cluster_soma_overview)

# Plot: Cluster x Soma Region count heatmap
cluster_soma_plot_df <- as.data.frame(table(cluster_soma_df$Cluster, cluster_soma_df$SomaRegion))
colnames(cluster_soma_plot_df) <- c("Cluster", "SomaRegion", "Count")
cluster_soma_plot_df$Cluster <- factor(cluster_soma_plot_df$Cluster, levels = sort(unique(cluster_soma_plot_df$Cluster)))

cluster_soma_plot <- ggplot(cluster_soma_plot_df, aes(x = SomaRegion, y = Cluster, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 3) +
  scale_fill_viridis_c(option = "C", direction = -1) +
  labs(
    title = "Cluster vs Soma Region",
    x = "Soma Region",
    y = "Cluster",
    fill = "Count"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )
print(cluster_soma_plot)
ggsave(
  filename = file.path(FIGURE_DIR, "cluster_vs_soma_region.png"),
  plot = cluster_soma_plot,
  width = 10,
  height = 6,
  dpi = 300
)

# Plot: Region x Cluster x Neuron Type (faceted by neuron type)
cluster_region_type_df <- data.frame(
  Cluster = as.character(clusters),
  SomaRegion = soma_map[names(clusters)],
  NeuronType = type_map[names(clusters)],
  stringsAsFactors = FALSE
)
cluster_region_type_df$SomaRegion[is.na(cluster_region_type_df$SomaRegion) | cluster_region_type_df$SomaRegion == ""] <- "Unknown"
cluster_region_type_df$NeuronType[is.na(cluster_region_type_df$NeuronType) | cluster_region_type_df$NeuronType == ""] <- "Unknown"

cluster_region_type_count <- cluster_region_type_df %>%
  count(NeuronType, SomaRegion, Cluster, name = "Count")

region_cluster_type_plot <- ggplot(cluster_region_type_count, aes(x = SomaRegion, y = Cluster, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), size = 2.4) +
  scale_fill_viridis_c(option = "D", direction = -1) +
  facet_wrap(~ NeuronType, scales = "free_y") +
  labs(
    title = "Region x Cluster x Neuron Type",
    x = "Soma Region",
    y = "Cluster",
    fill = "Count"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank(),
    strip.text = element_text(face = "bold")
  )
print(region_cluster_type_plot)
ggsave(
  filename = file.path(FIGURE_DIR, "region_x_cluster_x_neuron_type.png"),
  plot = region_cluster_type_plot,
  width = 14,
  height = 8,
  dpi = 300
)

# Save
results_df <- data.frame(
  NeuronID = names(clusters),
  Original_Type = type_map[names(clusters)],
  Subtype_Cluster = clusters,
  row.names = NULL
)

filename <- paste0("fnt_dist_R_Results_", ifelse(USE_PENALTY, "Penalty", "NoPenalty"), "_k", k_final, ".xlsx")
results_path <- file.path(DATA_OUTPUT_DIR, filename)
write_xlsx(results_df, results_path)
message(paste("Saved:", results_path))

# ==========================================
# 6. HEATMAP
# ==========================================
message("--- Generating Heatmap ---")

# Annotation Bar
anno_row <- data.frame(BioType = factor(type_map[rownames(dist_matrix)]))
rownames(anno_row) <- rownames(dist_matrix)

# Colors
if (USE_SPEARMAN) {
  cols <- mako(100)
} else {
  cols <- viridis(100, direction = -1)
}

heatmap_path <- file.path(
  FIGURE_DIR,
  paste0("distance_heatmap_", ifelse(USE_PENALTY, "Penalty", "NoPenalty"), "_k", k_final, ".png")
)
pheatmap(
  dist_matrix,
  clustering_distance_rows = dist_obj,
  clustering_distance_cols = dist_obj,
  clustering_method = "ward.D2",
  annotation_row = anno_row,
  show_rownames = FALSE, show_colnames = FALSE,
  col = cols,
  name = "Distance",
  main = paste("Clustering (k =", k_final, ")"),
  filename = heatmap_path,
  width = 10,
  height = 10
)
message(paste("Saved:", heatmap_path))



# ==========================================
# 7. EXPORT CLUSTER INFO FOR EXTERNAL USE
# ==========================================

# Save complete clustering object
saveRDS(list(
  # Core clustering info
  hclust_obj = hc,                    # Full hclust object for dendrograms
  dist_matrix = dist_matrix,          # Distance matrix used
  dist_obj = dist_obj,                # dist object
  
  # Order and clusters
  dendro_order = hc$order,            # Integer indices of leaf order
  ordered_neurons = rownames(dist_matrix)[hc$order],  # Names in dendrogram order
  clusters = clusters,                # Named vector of cluster assignments
  
  # For ggplot2/annotation
  cluster_df = data.frame(
    NeuronID = names(clusters),
    Cluster = clusters,
    DendroOrder = match(names(clusters), rownames(dist_matrix)[hc$order]),
    BioType = type_map[names(clusters)],
    stringsAsFactors = FALSE
  ) %>% arrange(DendroOrder)
  
), file.path(DATA_OUTPUT_DIR, "clustering_results.rds"))

message(paste("Saved:", file.path(DATA_OUTPUT_DIR, "clustering_results.rds")))