# ============================================================
# Intra-insula interconnectivity analysis
#
# Question: do insula neurons project to OTHER insula sub-regions?
# If yes, is the connectivity asymmetric L vs R?
#
# Approach:
#   1. Identify which target columns in the projection matrix are
#      themselves insula sub-regions (intersect with atlas-derived
#      insula vocabulary).
#   2. Build a "from sub-region" x "to insula sub-region" projection
#      matrix (proportion of ipsi projection going to each insula target).
#   3. Test L vs R within each (source sub-region, insula target) cell.
#   4. Visualize as a heatmap.
# ============================================================
suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(readr)
  library(ggplot2); library(patchwork); library(writexl)
})

PROJECT_ROOT  <- "d:/projectome_analysis"
GROUP_DIR     <- file.path(PROJECT_ROOT, "group_analysis")
COMBINED_XLSX <- file.path(GROUP_DIR, "combined", "multi_monkey_INS_combined.xlsx")
SCRIPTS       <- file.path(GROUP_DIR, "scripts")

OUT_DIR  <- file.path(GROUP_DIR, "R_analysis", "outputs")
STATS_DIR <- file.path(OUT_DIR, "stats", "intra_insula")
FIG_DIR   <- file.path(OUT_DIR, "figures", "intra_insula")
TBL_DIR   <- file.path(OUT_DIR, "tables", "intra_insula")
for (d in c(STATS_DIR, FIG_DIR, TBL_DIR))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Insula vocabulary at FINEST atlas level (L6).
# These are the column names in Projection_Strength_ipsi/contra
# that correspond to insular sub-regions per ARM v2.1 + CHARM v2.
# ============================================================
INSULA_FINEST_TARGETS <- c(
  # Granular / parainsula / retroinsula (in floor_of_ls L3 family)
  "Ig", "Pi", "Ri",
  # Dysgranular sub-regions (in floor_of_ls L3 family; CHARM L4-L6)
  "Ia/Id",       # combined dysgranular
  # Agranular sub-regions (in caudal_OFC L3 family at finer levels)
  "Ial", "Iai", "Iapl",
  "Iam/Iapm",    # combined medial+post-medial agranular
  "lat_Ia"
)

normalize_label <- function(s) {
  v <- trimws(as.character(s))
  v <- sub("^(CL|CR|L|R)[_\\-]", "", v, perl = TRUE)
  v
}

# ============================================================
# Load combined data
# ============================================================
summ <- read_excel(COMBINED_XLSX, sheet = "Summary")
# Use FINEST level (L6) for intra-insula since it has Ial/Ig/Iam/Iapm/Iai/Pi/Iapl
# as separate target columns. L3 only has 'floor_of_ls' / 'caudal_OFC' aggregates.
ipsi <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_ipsi")
contra <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_contra")
common <- intersect(summ$NeuronUID, ipsi$NeuronUID)
summ <- summ[match(common, summ$NeuronUID), ]
ipsi <- ipsi[match(common, ipsi$NeuronUID), ]
contra <- contra[match(common, contra$NeuronUID), ]

summ <- summ %>%
  mutate(
    Soma_Side_Final = ifelse(!is.na(Soma_Side_Inferred) &
                                Soma_Side_Inferred %in% c("L", "R"),
                              Soma_Side_Inferred, Soma_Side),
    Soma_Region_Final = ifelse(!is.na(Soma_Region_Refined) &
                                  nzchar(Soma_Region_Refined),
                                Soma_Region_Refined,
                                gsub("^(CL|CR)_", "", Soma_Region_Auto))
  )
summ$Soma_Side_Final <- factor(summ$Soma_Side_Final, levels = c("L", "R"))

build_mat <- function(sheet, uids) {
  meta_cols <- c("NeuronUID", "SampleID", "NeuronID", "Neuron_Type")
  num_cols <- setdiff(colnames(sheet), meta_cols)
  m <- as.matrix(sheet[, num_cols, drop = FALSE])
  storage.mode(m) <- "numeric"
  m[is.na(m)] <- 0
  rownames(m) <- sheet$NeuronUID
  m[uids, , drop = FALSE]
}
ipsi_mat   <- build_mat(ipsi, summ$NeuronUID)
contra_mat <- build_mat(contra, summ$NeuronUID)

# ============================================================
# Identify finest-level insula targets in the projection matrix
# ============================================================
ipsi_targets <- colnames(ipsi_mat)
insula_target_idx <- which(ipsi_targets %in% INSULA_FINEST_TARGETS)
insula_targets <- ipsi_targets[insula_target_idx]

contra_targets <- colnames(contra_mat)
contra_insula_idx <- which(contra_targets %in% INSULA_FINEST_TARGETS)
contra_insula_targets <- contra_targets[contra_insula_idx]

cat("Insula finest-level targets in IPSI projection matrix:\n")
cat(sprintf("  %s\n", paste(insula_targets, collapse = ", ")))
cat(sprintf("  (n = %d / %d total ipsi targets)\n",
            length(insula_targets), length(ipsi_targets)))
cat("\nInsula finest-level targets in CONTRA projection matrix:\n")
cat(sprintf("  %s\n", paste(contra_insula_targets, collapse = ", ")))
cat(sprintf("  (n = %d / %d total contra targets)\n",
            length(contra_insula_targets), length(contra_targets)))

# ============================================================
# Per-neuron intra-insula projection signature
# ============================================================
# Total projection (all targets, ipsi + contra)
total_all_proj <- rowSums(ipsi_mat) + rowSums(contra_mat)
intra_ipsi <- if (length(insula_target_idx))
  rowSums(ipsi_mat[, insula_target_idx, drop = FALSE]) else rep(0, nrow(ipsi_mat))
intra_contra <- if (length(contra_insula_idx))
  rowSums(contra_mat[, contra_insula_idx, drop = FALSE]) else rep(0, nrow(contra_mat))

# Proportion of total projection that goes to OTHER insula sub-regions
intra_total <- intra_ipsi + intra_contra
intra_frac <- ifelse(total_all_proj > 0, intra_total / total_all_proj, 0)

scores <- summ %>%
  mutate(intra_ipsi = intra_ipsi,
         intra_contra = intra_contra,
         intra_total = intra_total,
         total_all = total_all_proj,
         intra_frac = intra_frac)

cat(sprintf("\nIntra-insula projection summary across all neurons (n=%d):\n",
            nrow(scores)))
cat(sprintf("  median intra_frac = %.4f\n", median(intra_frac)))
cat(sprintf("  mean intra_frac   = %.4f\n", mean(intra_frac)))
cat(sprintf("  neurons with intra_frac > 0:    %d (%.1f%%)\n",
            sum(intra_frac > 0), 100 * mean(intra_frac > 0)))
cat(sprintf("  neurons with intra_frac > 0.05: %d (%.1f%%)\n",
            sum(intra_frac > 0.05), 100 * mean(intra_frac > 0.05)))

# ============================================================
# (A) Source sub-region x ipsi insula-target heatmap (mean
#     proportion of ipsi-projection going to each insula target),
#     plotted per Soma_Side_Final
# ============================================================
ipsi_total <- rowSums(ipsi_mat, na.rm = TRUE)
ipsi_prop <- ipsi_mat
ipsi_prop[ipsi_total > 0, ] <- sweep(ipsi_mat[ipsi_total > 0, , drop = FALSE], 1,
                                      ipsi_total[ipsi_total > 0], "/")

build_heatmap_long <- function(side_label, region_filter = NULL) {
  m <- summ$Soma_Side_Final == side_label
  if (!is.null(region_filter)) m <- m & summ$Soma_Region_Final %in% region_filter
  ids <- which(m)
  if (length(ids) < 3) return(data.frame())
  sub <- ipsi_prop[ids, insula_target_idx, drop = FALSE]
  sr  <- summ$Soma_Region_Final[ids]
  out_rows <- list()
  for (src in unique(sr)) {
    src_ids <- which(sr == src)
    if (length(src_ids) < 3) next
    means <- colMeans(sub[src_ids, , drop = FALSE], na.rm = TRUE)
    fracs <- colMeans(sub[src_ids, , drop = FALSE] > 0, na.rm = TRUE)
    df <- data.frame(
      side = side_label,
      source_region = src,
      target_region = colnames(sub),
      mean_prop = means,
      frac_proj = fracs,
      n_neurons = length(src_ids),
      stringsAsFactors = FALSE
    )
    out_rows[[length(out_rows) + 1L]] <- df
  }
  bind_rows(out_rows)
}

heat_L <- build_heatmap_long("L")
heat_R <- build_heatmap_long("R")
heat_all <- bind_rows(heat_L, heat_R)

# Self-projection flag: when the soma sub-region overlaps the target
# sub-region label. At L6 this is direct (e.g. soma="IAL" projecting
# to target="Ial" = same region; or "IDD5" -> "Ia/Id" since IDD5 is
# under Ia/Id in CHARM hierarchy).
self_pairs <- list(
  c("IAL", "Ial"), c("IAI", "Iai"), c("IAM", "Iam/Iapm"),
  c("IAPM", "Iam/Iapm"), c("IAPL", "Iapl"),
  c("IDD5", "Ia/Id"), c("IDM", "Ia/Id"), c("IDV", "Ia/Id"),
  c("IDD", "Ia/Id"), c("IDI", "Ia/Id"), c("IA/ID", "Ia/Id"),
  c("IG", "Ig")
)
heat_all <- heat_all %>%
  rowwise() %>%
  mutate(self_proj = any(sapply(self_pairs, function(p)
    toupper(source_region) == p[1] && target_region == p[2]))) %>%
  ungroup()

write.csv(heat_all,
          file.path(STATS_DIR, "intra_insula_source_x_target_means.csv"),
          row.names = FALSE)

# Heatmap viz: source vs target, color = mean_prop, faceted by side
fig_heat <- ggplot(heat_all,
                   aes(x = target_region, y = source_region, fill = mean_prop)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.2f", mean_prop)),
            size = 2.7, color = "black") +
  geom_text(aes(label = ifelse(self_proj, "*", "")),
            size = 3.8, color = "red", nudge_y = 0.30) +
  scale_fill_gradient(low = "white", high = "#1976d2",
                      name = "mean prop ipsi") +
  facet_wrap(~ side, ncol = 2, labeller = labeller(side = function(x) paste("Soma side:", x))) +
  labs(title = "Intra-insula ipsilateral projection: source soma sub-region -> target insula sub-region",
       subtitle = "Cells = mean proportion of ipsi projection budget; * = self/within-sub-region (auto-projection back to soma sub-region)",
       x = "Target insula sub-region", y = "Source soma sub-region") +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(FIG_DIR, "intra_insula_heatmap_LR.png"),
       fig_heat, width = 13, height = 6.5, dpi = 200)

# ============================================================
# (B) Per-neuron intra-insula fraction L vs R, by sub-region
# ============================================================
df_for_test <- scores %>%
  filter(Soma_Side_Final %in% c("L", "R")) %>%
  filter(!is.na(Soma_Region_Final), nzchar(Soma_Region_Final))

# Test per sub-region
tests <- df_for_test %>%
  group_by(Soma_Region_Final) %>%
  summarise(
    n_total = n(),
    n_L = sum(Soma_Side_Final == "L"),
    n_R = sum(Soma_Side_Final == "R"),
    mean_intra_L = mean(intra_frac[Soma_Side_Final == "L"], na.rm = TRUE),
    mean_intra_R = mean(intra_frac[Soma_Side_Final == "R"], na.rm = TRUE),
    median_intra_L = median(intra_frac[Soma_Side_Final == "L"], na.rm = TRUE),
    median_intra_R = median(intra_frac[Soma_Side_Final == "R"], na.rm = TRUE),
    p_wilcox = if (sum(Soma_Side_Final == "L") >= 3 &&
                    sum(Soma_Side_Final == "R") >= 3)
      tryCatch(suppressWarnings(
        wilcox.test(intra_frac ~ Soma_Side_Final, exact = FALSE)$p.value),
        error = function(e) NA_real_)
      else NA_real_,
    .groups = "drop"
  ) %>%
  mutate(p_BH = p.adjust(p_wilcox, method = "BH"),
         direction = ifelse(mean_intra_L > mean_intra_R, "L>R", "R>L"))

write.csv(tests,
          file.path(STATS_DIR, "intra_frac_by_subregion_LR.csv"),
          row.names = FALSE)

cat("\nIntra-insula fraction L vs R by sub-region:\n")
print(tests, digits = 3, row.names = FALSE)

# ============================================================
# (C) Per-target test on insula-target columns (presence + magnitude)
# ============================================================
per_target_intra <- function(prop_mat, side_vec, sample_vec, label) {
  out <- bind_rows(lapply(insula_targets, function(rg) {
    if (!rg %in% colnames(prop_mat)) return(NULL)
    xL <- prop_mat[side_vec == "L", rg]
    xR <- prop_mat[side_vec == "R", rg]
    nL <- length(xL); nR <- length(xR)
    if (nL < 3 || nR < 3) return(NULL)
    tab <- matrix(c(sum(xL > 0), sum(xL == 0),
                    sum(xR > 0), sum(xR == 0)), nrow = 2)
    p_pres <- tryCatch(fisher.test(tab)$p.value, error = function(e) NA_real_)
    p_mag <- tryCatch(suppressWarnings(
      wilcox.test(xL, xR, exact = FALSE)$p.value), error = function(e) NA_real_)
    data.frame(
      stratum = label, target = rg,
      n_L = nL, n_R = nR, n_samples = length(unique(sample_vec)),
      frac_L = mean(xL > 0), frac_R = mean(xR > 0),
      mean_L = mean(xL), mean_R = mean(xR),
      p_presence = p_pres, p_magnitude = p_mag
    )
  }))
  if (!nrow(out)) return(out)
  out$direction <- ifelse(out$mean_L > out$mean_R, "L>R", "R>L")
  out$p_pres_BH <- p.adjust(out$p_presence, method = "BH")
  out$p_mag_BH  <- p.adjust(out$p_magnitude, method = "BH")
  out
}

strata_def <- list(
  all_combined = rep(TRUE, nrow(summ)),
  IDD5_plus_IDM = summ$SampleID == "251637" &
                   summ$Soma_Region_Final %in% c("IDD5", "IDM"),
  IAL_combined = summ$Soma_Region_Final == "IAL",
  IAL_251637 = summ$SampleID == "251637" & summ$Soma_Region_Final == "IAL",
  IAL_252385 = summ$SampleID == "252385" & summ$Soma_Region_Final == "IAL"
)

per_tgt_intra <- bind_rows(lapply(names(strata_def), function(lbl) {
  m <- strata_def[[lbl]]
  if (sum(m) < 6) return(NULL)
  per_target_intra(ipsi_prop[m, , drop = FALSE],
                    summ$Soma_Side_Final[m],
                    summ$SampleID[m], lbl)
}))
write.csv(per_tgt_intra,
          file.path(STATS_DIR, "intra_insula_per_target_LR.csv"),
          row.names = FALSE)

cat("\nIntra-insula per-target tests (BH q<0.10 in either presence or magnitude):\n")
sig <- per_tgt_intra %>%
  filter(p_pres_BH < 0.10 | p_mag_BH < 0.10) %>%
  arrange(stratum, p_magnitude)
if (nrow(sig)) {
  print(sig %>% dplyr::select(stratum, target, n_L, n_R, frac_L, frac_R,
                                mean_L, mean_R, p_pres_BH, p_mag_BH, direction),
        digits = 3, row.names = FALSE)
} else cat("  (none)\n")

# ============================================================
# (D) Visualization: per-neuron intra-insula fraction violins
# ============================================================
side_cols <- c(L = "#E74C3C", R = "#3498DB")
fig_intra <- ggplot(df_for_test,
                    aes(x = Soma_Region_Final, y = intra_frac,
                        fill = Soma_Side_Final)) +
  geom_violin(alpha = 0.5, trim = FALSE,
              position = position_dodge(width = 0.9)) +
  geom_boxplot(width = 0.18, alpha = 0.85, outlier.size = 0.6,
               position = position_dodge(width = 0.9)) +
  scale_fill_manual(values = side_cols, name = "Soma side") +
  labs(title = "Intra-insula projection fraction per neuron, by source soma sub-region",
       subtitle = sprintf("intra_frac = (insula_ipsi + insula_contra) / total_projection;  n_total = %d", nrow(df_for_test)),
       x = "Source sub-region", y = "Intra-insula fraction") +
  theme_minimal(base_size = 11)
ggsave(file.path(FIG_DIR, "intra_frac_by_subregion_LR.png"),
       fig_intra, width = 11, height = 6, dpi = 200)

# ============================================================
# (E) Save consolidated workbook
# ============================================================
write_xlsx(list(
  summary_per_subregion = tests,
  per_target_LR = per_tgt_intra,
  source_x_target = heat_all
), file.path(STATS_DIR, "intra_insula_consolidated.xlsx"))

cat(sprintf("\n[saved] %s\n",
            file.path(STATS_DIR, "intra_insula_consolidated.xlsx")))
cat("[Intra-insula analysis done]\n")
