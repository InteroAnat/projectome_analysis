# ============================================================
# Hypothesis-driven L/R insula projectome analysis
# - Tier-1: IAL-excluded sensitivity for SQ3d
# - Tier-1: hurdle decomposition (presence + magnitude) for top targets
# - Tier-2: autonomic target panel + sympathetic/parasympathetic scoring
# - Tier-2: AP-axis stratification (anterior / mid / posterior insula)
# - Tier-2: per-soma-region L vs R within balanced strata (IDD5 + IDM)
#
# Frame: Oppenheimer 1992; Hassanpour 2017; Craig 2009; Cechetto/Saper.
# Hypothesis (anatomical): R-insula neurons preferentially target
#   sympathetic-network nodes (caudal OFC, hypothalamus, PAG/midbrain,
#   central amygdala); L-insula neurons preferentially target
#   parasympathetic-network nodes (medulla, NTS region) and somatomotor.
# ============================================================

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(writexl)
  library(vegan)
  library(patchwork)
})

set.seed(42)

PATH_INS_XLSX <- "../../neuron_tables_new/251637_INS_HE_inferred.xlsx"
PATH_CLUSTER_RDS <- "data_output/clustering/clustering_results.rds"
OUT_DIR <- "LR_analysis_hypothesis"

setwd("d:/projectome_analysis/R_analysis/scripts")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "stats"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "tables"), recursive = TRUE, showWarnings = FALSE)

# ---- Load data ----
cluster_info <- readRDS(PATH_CLUSTER_RDS)
ipsi_raw <- read_excel(PATH_INS_XLSX, sheet = "Projection_Strength_L3_ipsi")
contra_raw <- read_excel(PATH_INS_XLSX, sheet = "Projection_Strength_L3_contra")
summ <- read_excel(PATH_INS_XLSX, sheet = "Summary")

# Strip CL_/CR_/L-/R- prefixes from soma region for hemisphere-neutral labels
strip_prefix <- function(x) {
  v <- trimws(as.character(x))
  v <- sub("^(CL|CR)[_\\-]", "", v, perl = TRUE)
  v <- sub("^[LR][_\\-]", "", v, perl = TRUE)
  ifelse(nzchar(v), v, NA_character_)
}

# Build merged metadata
meta_full <- summ %>%
  transmute(
    NeuronID,
    Soma_Region_raw = Soma_Region,
    Soma_Region = strip_prefix(Soma_Region),
    Soma_Side = factor(Soma_Side, levels = c("L", "R")),
    AP_coord = as.numeric(Soma_NII_Y),
    ML_coord = as.numeric(Soma_NII_X),
    DV_coord = as.numeric(Soma_NII_Z)
  ) %>%
  left_join(
    cluster_info$cluster_df %>%
      dplyr::select(NeuronID, Cluster, Neuron_Type = BioType),
    by = "NeuronID"
  )

# ---- Build target matrices (neuron x target) ----
# ipsi sheet has many target columns; same for contra
ipsi_tgt_cols <- setdiff(names(ipsi_raw), c("NeuronID", "Neuron_Type"))
contra_tgt_cols <- setdiff(names(contra_raw), c("NeuronID", "Neuron_Type"))

ipsi_mat <- ipsi_raw %>%
  dplyr::select(NeuronID, dplyr::all_of(ipsi_tgt_cols)) %>%
  as.data.frame()
contra_mat <- contra_raw %>%
  dplyr::select(NeuronID, dplyr::all_of(contra_tgt_cols)) %>%
  as.data.frame()

stopifnot(all(meta_full$NeuronID %in% ipsi_mat$NeuronID))

# Reorder to common neuron order
meta <- meta_full %>%
  arrange(match(NeuronID, ipsi_mat$NeuronID))
ipsi_mat <- ipsi_mat[match(meta$NeuronID, ipsi_mat$NeuronID), ]
contra_mat <- contra_mat[match(meta$NeuronID, contra_mat$NeuronID), ]

# Numeric matrices
ipsi_num <- as.matrix(ipsi_mat[, -1])
storage.mode(ipsi_num) <- "numeric"
ipsi_num[is.na(ipsi_num)] <- 0
rownames(ipsi_num) <- meta$NeuronID

contra_num <- as.matrix(contra_mat[, -1])
storage.mode(contra_num) <- "numeric"
contra_num[is.na(contra_num)] <- 0
rownames(contra_num) <- meta$NeuronID

cat(sprintf("Loaded %d neurons; %d ipsi targets; %d contra targets\n",
            nrow(meta), ncol(ipsi_num), ncol(contra_num)))
cat("L-side n =", sum(meta$Soma_Side == "L"),
    " R-side n =", sum(meta$Soma_Side == "R"), "\n\n")

# ============================================================
# AUTONOMIC TARGET PANEL — pre-registered before testing
# Anchors: Oppenheimer & Cechetto 2016 Compr Physiol;
#          Saper 2002 Annu Rev Neurosci; Craig 2002 NRN;
#          Carmichael & Price 1995 J Comp Neurol;
#          Cechetto & Saper 1990 J Comp Neurol (rat insula);
#          Oppenheimer 1992 Neurology (intraop stim).
# ============================================================

# Sympathetic-related (descending): central autonomic network nodes
panel_sym <- c(
  "caudal_OFC", "lat_OFC", "med_OFC",   # CAN cortical (areas 12/13)
  "ACC", "MCC", "PCgG",                 # cingulate autonomic
  "spAmy", "pAmy",                       # amygdala (sympathetic descending)
  "THy", "PHy", "ZI-H",                  # hypothalamus (sympathetic)
  "VMid", "DMid", "MMid", "LMid", "IMed",# midbrain incl. PAG
  "VPons", "DPons", "LPons"              # pontine autonomic
)

# Parasympathetic-related (vagal / NTS / DMV equivalents)
panel_para <- c(
  "VMed", "DMed"                         # medulla — NTS / DMV proxies
)

# Interoceptive sensory (thalamic relays — VPM/VMpo lineage)
panel_intero <- c(
  "VThal", "MThal", "MLThal", "PThal", "GThal", "Rh", "Rt"
)

# Cognitive/limbic non-autonomic (negative-control panel)
panel_cog <- c(
  "vlPFC", "dlPFC", "area_8A", "SMA/preSMA", "M1/PM"
)

# Sensory parietal/temporal (interoceptive-adjacent)
panel_sens <- c(
  "SI", "SII", "area_5", "area_7m", "area_7_in_IPL",
  "lat_IPS", "vm_IPS", "TE", "STGr/STSd", "STGc", "STSf",
  "belt", "core", "POC", "TG", "HF", "PrT", "MST"
)

panels <- list(
  sympathetic = panel_sym,
  parasympathetic = panel_para,
  interoceptive = panel_intero,
  cognitive = panel_cog,
  sensory = panel_sens
)

# Diagnostic: which panel targets exist in data?
ipsi_targets <- colnames(ipsi_num)
panel_diag <- bind_rows(lapply(names(panels), function(p) {
  in_data <- intersect(panels[[p]], ipsi_targets)
  missing <- setdiff(panels[[p]], ipsi_targets)
  data.frame(
    panel = p,
    n_targets_defined = length(panels[[p]]),
    n_targets_in_data = length(in_data),
    targets_in_data = paste(in_data, collapse = ";"),
    targets_missing = paste(missing, collapse = ";"),
    stringsAsFactors = FALSE
  )
}))
write.csv(panel_diag,
          file.path(OUT_DIR, "tables", "autonomic_panel_diagnostic.csv"),
          row.names = FALSE)
print(panel_diag[, c("panel", "n_targets_defined", "n_targets_in_data")])

# ============================================================
# Compute per-neuron panel scores (proportion of total ipsi
# projection going to each panel)
# ============================================================
ipsi_total <- rowSums(ipsi_num, na.rm = TRUE)
ok <- ipsi_total > 0
ipsi_prop <- ipsi_num
ipsi_prop[ok, ] <- sweep(ipsi_num[ok, , drop = FALSE], 1,
                         ipsi_total[ok], "/")

panel_score <- function(prop_mat, target_set) {
  use <- intersect(target_set, colnames(prop_mat))
  if (!length(use)) return(rep(NA_real_, nrow(prop_mat)))
  rowSums(prop_mat[, use, drop = FALSE], na.rm = TRUE)
}

scores <- as.data.frame(meta)
for (p in names(panels)) {
  scores[[paste0("score_", p)]] <- panel_score(ipsi_prop, panels[[p]])
}

# Sympathetic vs parasympathetic asymmetry per neuron
scores$sym_minus_para <- scores$score_sympathetic - scores$score_parasympathetic
scores$sym_para_LI <- with(scores, (score_sympathetic - score_parasympathetic) /
                                    (score_sympathetic + score_parasympathetic + 1e-8))

# ============================================================
# Tier-1 step 1: IAL-excluded sensitivity for SQ3d
# ============================================================
keep_balanced <- meta$Soma_Region %in% c("IDD5", "IDM")  # most balanced strata
cat("\nIAL-excluded subset: n =", sum(keep_balanced),
    " (", sum(keep_balanced & meta$Soma_Side == "L"), "L, ",
    sum(keep_balanced & meta$Soma_Side == "R"), "R)\n")

# Recompute LI per target on IAL-excluded subset
ipsi_prop_bal <- ipsi_prop[keep_balanced, , drop = FALSE]
meta_bal <- meta[keep_balanced, , drop = FALSE]

calc_li_targets <- function(prop_mat, side_vec, label) {
  ids_L <- which(side_vec == "L")
  ids_R <- which(side_vec == "R")
  if (length(ids_L) < 3 || length(ids_R) < 3)
    return(NULL)
  out <- data.frame(
    stratum = label,
    target_region = colnames(prop_mat),
    mean_L = colMeans(prop_mat[ids_L, , drop = FALSE], na.rm = TRUE),
    mean_R = colMeans(prop_mat[ids_R, , drop = FALSE], na.rm = TRUE),
    n_L = length(ids_L),
    n_R = length(ids_R),
    stringsAsFactors = FALSE
  )
  out$LI <- (out$mean_L - out$mean_R) /
            (out$mean_L + out$mean_R + 1e-8)
  out$abs_LI <- abs(out$LI)
  out$p_wilcox <- vapply(seq_len(ncol(prop_mat)), function(j) {
    x <- prop_mat[ids_L, j]
    y <- prop_mat[ids_R, j]
    tryCatch(suppressWarnings(wilcox.test(x, y, exact = FALSE)$p.value),
             error = function(e) NA_real_)
  }, numeric(1))
  # Hurdle: presence test (Fisher) and conditional magnitude test
  out$frac_L_proj <- vapply(seq_len(ncol(prop_mat)), function(j)
    mean(prop_mat[ids_L, j] > 0, na.rm = TRUE), numeric(1))
  out$frac_R_proj <- vapply(seq_len(ncol(prop_mat)), function(j)
    mean(prop_mat[ids_R, j] > 0, na.rm = TRUE), numeric(1))
  out$p_presence <- vapply(seq_len(ncol(prop_mat)), function(j) {
    tab <- matrix(c(sum(prop_mat[ids_L, j] > 0),
                    sum(prop_mat[ids_L, j] == 0),
                    sum(prop_mat[ids_R, j] > 0),
                    sum(prop_mat[ids_R, j] == 0)), nrow = 2)
    tryCatch(fisher.test(tab)$p.value, error = function(e) NA_real_)
  }, numeric(1))
  out$p_magnitude_cond <- vapply(seq_len(ncol(prop_mat)), function(j) {
    xL <- prop_mat[ids_L, j]; xR <- prop_mat[ids_R, j]
    xL <- xL[xL > 0]; xR <- xR[xR > 0]
    if (length(xL) < 2 || length(xR) < 2) return(NA_real_)
    tryCatch(suppressWarnings(wilcox.test(xL, xR, exact = FALSE)$p.value),
             error = function(e) NA_real_)
  }, numeric(1))
  out$p_BH <- p.adjust(out$p_wilcox, method = "BH")
  out$p_presence_BH <- p.adjust(out$p_presence, method = "BH")
  out$p_magnitude_BH <- p.adjust(out$p_magnitude_cond, method = "BH")
  out$direction <- ifelse(out$LI >= 0, "L>R", "R>L")
  out$receipt_class <- ifelse(out$mean_L > 0 & out$mean_R > 0,
                              "bilateral_receiving",
                              "one_side_only_or_extreme")
  out
}

li_full <- calc_li_targets(ipsi_prop, meta$Soma_Side, "all_neurons")
li_bal <- calc_li_targets(ipsi_prop_bal, meta_bal$Soma_Side,
                          "IAL_excluded_IDD5_IDM_only")
li_idd5 <- calc_li_targets(ipsi_prop[meta$Soma_Region == "IDD5", , drop = FALSE],
                           meta$Soma_Side[meta$Soma_Region == "IDD5"],
                           "IDD5_only")
li_idm <- calc_li_targets(ipsi_prop[meta$Soma_Region == "IDM", , drop = FALSE],
                          meta$Soma_Side[meta$Soma_Region == "IDM"],
                          "IDM_only")

li_all <- bind_rows(li_full, li_bal, li_idd5, li_idm)
write.csv(li_all,
          file.path(OUT_DIR, "stats", "sq3d_with_hurdle_decomposition.csv"),
          row.names = FALSE)

# Compare top BH-significant targets across strata
cat("\n=== SQ3d sensitivity: top targets per stratum (BH-significant only) ===\n")
top <- li_all %>%
  filter(p_BH < 0.05, receipt_class == "bilateral_receiving") %>%
  arrange(stratum, desc(abs_LI)) %>%
  dplyr::select(stratum, target_region, n_L, n_R, LI, direction,
                p_wilcox, p_BH, p_presence_BH, p_magnitude_BH)
print(top)

# ============================================================
# Tier-2: Sympathetic / parasympathetic score testing
# ============================================================
test_score_lr <- function(df, score_col, label) {
  if (!score_col %in% names(df)) return(NULL)
  d <- df[!is.na(df[[score_col]]), ]
  if (length(unique(d$Soma_Side)) < 2) return(NULL)
  nL <- sum(d$Soma_Side == "L")
  nR <- sum(d$Soma_Side == "R")
  if (nL < 3 || nR < 3) return(NULL)
  p <- tryCatch(suppressWarnings(
    wilcox.test(d[[score_col]] ~ d$Soma_Side, exact = FALSE)$p.value),
    error = function(e) NA_real_)
  data.frame(
    stratum = label,
    score = score_col,
    n_L = nL, n_R = nR,
    median_L = median(d[[score_col]][d$Soma_Side == "L"], na.rm = TRUE),
    median_R = median(d[[score_col]][d$Soma_Side == "R"], na.rm = TRUE),
    mean_L = mean(d[[score_col]][d$Soma_Side == "L"], na.rm = TRUE),
    mean_R = mean(d[[score_col]][d$Soma_Side == "R"], na.rm = TRUE),
    p_raw = p,
    stringsAsFactors = FALSE
  )
}

run_score_panel <- function(scores_df, label) {
  res <- bind_rows(lapply(
    c("score_sympathetic", "score_parasympathetic", "score_interoceptive",
      "score_cognitive", "score_sensory", "sym_minus_para", "sym_para_LI"),
    function(s) test_score_lr(scores_df, s, label)))
  res$p_BH <- p.adjust(res$p_raw, method = "BH")
  res$direction <- ifelse(res$mean_L > res$mean_R, "L>R", "R>L")
  res
}

score_all <- run_score_panel(scores, "all_neurons")
score_bal <- run_score_panel(scores[keep_balanced, ],
                             "IAL_excluded_IDD5_IDM_only")
score_idd5 <- run_score_panel(scores[meta$Soma_Region == "IDD5", ], "IDD5_only")
score_idm <- run_score_panel(scores[meta$Soma_Region == "IDM", ], "IDM_only")

score_combined <- bind_rows(score_all, score_bal, score_idd5, score_idm)
write.csv(score_combined,
          file.path(OUT_DIR, "stats", "autonomic_panel_lr_tests.csv"),
          row.names = FALSE)
cat("\n=== Autonomic panel L vs R tests ===\n")
print(score_combined[, c("stratum", "score", "n_L", "n_R",
                         "median_L", "median_R",
                         "p_raw", "p_BH", "direction")],
      digits = 3, row.names = FALSE)

# ============================================================
# Tier-2: AP-axis stratification (anterior / mid / posterior)
# split AP_coord by tertile
# ============================================================
ap_breaks <- quantile(scores$AP_coord, c(1/3, 2/3), na.rm = TRUE)
scores$AP_bin <- cut(scores$AP_coord,
                     breaks = c(-Inf, ap_breaks, Inf),
                     labels = c("posterior", "mid", "anterior"))

cat("\nAP tertile counts:\n")
print(table(scores$AP_bin, scores$Soma_Side, useNA = "ifany"))

ap_tests <- bind_rows(lapply(levels(scores$AP_bin), function(b) {
  sub <- scores[!is.na(scores$AP_bin) & scores$AP_bin == b, ]
  res <- run_score_panel(sub, paste0("AP_", b))
  res
}))
write.csv(ap_tests,
          file.path(OUT_DIR, "stats", "autonomic_panel_AP_stratified.csv"),
          row.names = FALSE)
cat("\n=== AP-axis stratified panel tests ===\n")
print(ap_tests[, c("stratum", "score", "n_L", "n_R",
                   "median_L", "median_R", "p_raw", "p_BH", "direction")],
      digits = 3, row.names = FALSE)

# ============================================================
# Save scores per neuron
# ============================================================
write.csv(scores,
          file.path(OUT_DIR, "tables", "per_neuron_panel_scores.csv"),
          row.names = FALSE)

# ============================================================
# Plots
# ============================================================
side_cols <- c(L = "#E74C3C", R = "#3498DB")

# Panel score boxplots (full + IAL-excluded)
plot_score_panel <- function(df, score_col, title_lbl, sub_lbl) {
  d <- df[!is.na(df[[score_col]]), ]
  if (!nrow(d)) return(NULL)
  p <- tryCatch(suppressWarnings(
    wilcox.test(d[[score_col]] ~ d$Soma_Side)$p.value), error = function(e) NA)
  ggplot(d, aes(x = Soma_Side, y = .data[[score_col]], fill = Soma_Side)) +
    geom_violin(alpha = 0.45, trim = FALSE) +
    geom_boxplot(width = 0.18, outlier.size = 0.7, alpha = 0.9) +
    scale_fill_manual(values = side_cols) +
    labs(title = title_lbl,
         subtitle = sprintf("%s | Wilcoxon p=%.3g", sub_lbl, p),
         x = "Soma side", y = score_col) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none")
}

plots_full <- lapply(c("score_sympathetic", "score_parasympathetic",
                       "score_interoceptive", "score_cognitive",
                       "score_sensory", "sym_para_LI"), function(s)
  plot_score_panel(scores, s, paste0(s, " — all neurons"),
                   "All soma regions"))
plots_full <- plots_full[!sapply(plots_full, is.null)]
fig_full <- patchwork::wrap_plots(plots_full, ncol = 3)
ggsave(file.path(OUT_DIR, "figures", "panel_scores_all_neurons.png"),
       fig_full, width = 14, height = 8, dpi = 200)

plots_bal <- lapply(c("score_sympathetic", "score_parasympathetic",
                      "score_interoceptive", "score_cognitive",
                      "score_sensory", "sym_para_LI"), function(s)
  plot_score_panel(scores[keep_balanced, ], s,
                   paste0(s, " — IAL-excluded"),
                   "IDD5 + IDM only"))
plots_bal <- plots_bal[!sapply(plots_bal, is.null)]
fig_bal <- patchwork::wrap_plots(plots_bal, ncol = 3)
ggsave(file.path(OUT_DIR, "figures", "panel_scores_IAL_excluded.png"),
       fig_bal, width = 14, height = 8, dpi = 200)

# IAL-excluded SQ3d barplot of significant targets
li_bal_sig <- li_bal %>%
  filter(p_BH < 0.05, receipt_class == "bilateral_receiving") %>%
  arrange(LI) %>%
  mutate(target_region = factor(target_region, levels = unique(target_region)))

if (nrow(li_bal_sig) > 0) {
  p_bar <- ggplot(li_bal_sig, aes(x = target_region, y = LI,
                                  fill = direction)) +
    geom_col() +
    geom_hline(yintercept = 0, linetype = 2) +
    coord_flip() +
    scale_fill_manual(values = c("L>R" = "#E74C3C", "R>L" = "#3498DB")) +
    labs(title = "SQ3d (IAL-excluded): BH-significant LI targets",
         subtitle = sprintf("IDD5 + IDM only; n_L=%d, n_R=%d",
                            sum(meta_bal$Soma_Side == "L"),
                            sum(meta_bal$Soma_Side == "R")),
         x = "Target region", y = "Laterality Index (mean_L - mean_R)/(sum)") +
    theme_minimal(base_size = 11)
  ggsave(file.path(OUT_DIR, "figures", "sq3d_IAL_excluded_significant.png"),
         p_bar, width = 9, height = max(4, nrow(li_bal_sig) * 0.35), dpi = 200)
} else {
  cat("\nNo BH-significant LI targets in IAL-excluded subset\n")
}

# ============================================================
# Caudal_OFC focused diagnostic
# ============================================================
cat("\n=== caudal_OFC diagnostic across strata ===\n")
caudal_ofc <- li_all %>%
  filter(target_region == "caudal_OFC") %>%
  dplyr::select(stratum, n_L, n_R, mean_L, mean_R, LI, direction,
                p_wilcox, p_BH, p_presence, p_magnitude_cond,
                frac_L_proj, frac_R_proj)
print(caudal_ofc, digits = 3, row.names = FALSE)
write.csv(caudal_ofc,
          file.path(OUT_DIR, "stats", "caudal_OFC_sensitivity.csv"),
          row.names = FALSE)

# ============================================================
# Summary report
# ============================================================
cat("\n\n========== SUMMARY ==========\n")
cat("Total neurons:", nrow(meta), " (L:", sum(meta$Soma_Side == "L"),
    ", R:", sum(meta$Soma_Side == "R"), ")\n")
cat("\nSoma region x side:\n")
print(table(meta$Soma_Region, meta$Soma_Side))

cat("\nIAL-excluded balanced subset n =", sum(keep_balanced),
    " (L:", sum(keep_balanced & meta$Soma_Side == "L"),
    ", R:", sum(keep_balanced & meta$Soma_Side == "R"), ")\n")

cat("\nFiles written to:", OUT_DIR, "\n")
cat("  stats/sq3d_with_hurdle_decomposition.csv\n")
cat("  stats/autonomic_panel_lr_tests.csv\n")
cat("  stats/autonomic_panel_AP_stratified.csv\n")
cat("  stats/caudal_OFC_sensitivity.csv\n")
cat("  tables/autonomic_panel_diagnostic.csv\n")
cat("  tables/per_neuron_panel_scores.csv\n")
cat("  figures/panel_scores_all_neurons.png\n")
cat("  figures/panel_scores_IAL_excluded.png\n")
cat("  figures/sq3d_IAL_excluded_significant.png (if any)\n")

cat("\nDone.\n")
