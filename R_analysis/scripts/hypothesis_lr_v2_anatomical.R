# ============================================================
# Hypothesis-driven L/R insula projectome analysis -- v2
# Anatomical functional panels derived from established macaque
# insula efferent literature, in the spirit of Gou et al. (2025)
# 3-tier function table (function_class_big > _small > _name).
#
# Anatomical references:
#   Mesulam & Mufson 1982 J Comp Neurol (insula architectonics)
#   Mufson, Mesulam & Pandya 1981 Brain Res (insula→amygdala)
#   Carmichael & Price 1995 J Comp Neurol (OFC connectivity)
#   Saleem, Kondo & Price 2008 J Comp Neurol (OFC subdivisions)
#   Evrard 2019 Curr Opin Neurobiol (modular insula)
#   Saper 2002 Annu Rev Neurosci (central autonomic network)
#   Oppenheimer & Cechetto 2016 Compr Physiol (insula & cardiac)
#   Saper 1982 J Comp Neurol (insula→hypothalamus, rat→primate)
#   Cechetto & Saper 1990 J Comp Neurol (rat insula efferents)
#   Augustine 1996 Brain Res Rev (primate insula review)
#   Craig 2002 Nat Rev Neurosci (interoception)
# ============================================================

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr)
  library(ggplot2); library(patchwork); library(writexl)
})
set.seed(42)

setwd("d:/projectome_analysis/R_analysis/scripts")
PATH_INS_XLSX <- "../../neuron_tables_new/251637_INS_HE_inferred.xlsx"
PATH_CLUSTER_RDS <- "data_output/clustering/clustering_results.rds"
OUT_DIR <- "LR_analysis_hypothesis_v2"

dir.create(file.path(OUT_DIR, "stats"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "tables"), recursive = TRUE, showWarnings = FALSE)

# ---- Load ----
cluster_info <- readRDS(PATH_CLUSTER_RDS)
ipsi_raw <- read_excel(PATH_INS_XLSX, sheet = "Projection_Strength_L3_ipsi")
contra_raw <- read_excel(PATH_INS_XLSX, sheet = "Projection_Strength_L3_contra")
summ <- read_excel(PATH_INS_XLSX, sheet = "Summary")

strip_prefix <- function(x) {
  v <- trimws(as.character(x))
  v <- sub("^(CL|CR)[_\\-]", "", v, perl = TRUE)
  v <- sub("^[LR][_\\-]", "", v, perl = TRUE)
  ifelse(nzchar(v), v, NA_character_)
}

meta <- summ %>%
  transmute(NeuronID,
            Soma_Region = strip_prefix(Soma_Region),
            Soma_Side = factor(Soma_Side, levels = c("L", "R")),
            AP_coord = as.numeric(Soma_NII_Y),
            ML_coord = as.numeric(Soma_NII_X),
            DV_coord = as.numeric(Soma_NII_Z)) %>%
  left_join(cluster_info$cluster_df %>%
              dplyr::select(NeuronID, Cluster, Neuron_Type = BioType),
            by = "NeuronID")

ipsi_num <- as.matrix(ipsi_raw[, setdiff(names(ipsi_raw),
                                          c("NeuronID", "Neuron_Type"))])
storage.mode(ipsi_num) <- "numeric"
ipsi_num[is.na(ipsi_num)] <- 0
rownames(ipsi_num) <- ipsi_raw$NeuronID
ipsi_num <- ipsi_num[match(meta$NeuronID, rownames(ipsi_num)), ]

ipsi_total <- rowSums(ipsi_num, na.rm = TRUE)
ipsi_prop <- ipsi_num
ok <- ipsi_total > 0
ipsi_prop[ok, ] <- sweep(ipsi_num[ok, , drop = FALSE], 1,
                         ipsi_total[ok], "/")

ipsi_targets <- colnames(ipsi_num)

# ============================================================
# REVISED PANELS -- anatomically grounded, three-tier hierarchy
# in the spirit of Gou et al 2025's functional annotation
# (function_class_big > function_class_small > function_name)
#
# Tier 1 (function_class_big): broad functional domain
# Tier 2 (function_class_small): anatomical/functional subgroup
# Tier 3 (region label as in your data)
# ============================================================

panel_hierarchy <- tribble(
  ~big,                ~small,                       ~region,
  # --- AUTONOMIC / LIMBIC (Oppenheimer 1992, Saper 2002, Cechetto&Saper 1990,
  #     Carmichael&Price 1995, Mesulam&Mufson 1982) ---
  "autonomic_limbic",  "OFC_central_autonomic",      "caudal_OFC",
  "autonomic_limbic",  "OFC_central_autonomic",      "lat_OFC",
  "autonomic_limbic",  "OFC_central_autonomic",      "med_OFC",
  "autonomic_limbic",  "cingulate_autonomic",        "ACC",
  "autonomic_limbic",  "cingulate_autonomic",        "MCC",
  "autonomic_limbic",  "cingulate_autonomic",        "PCgG",
  "autonomic_limbic",  "amygdala",                   "spAmy",
  "autonomic_limbic",  "amygdala",                   "pAmy",
  "autonomic_limbic",  "hypothalamus",               "THy",
  "autonomic_limbic",  "hypothalamus",               "PHy",
  "autonomic_limbic",  "hypothalamus",               "ZI-H",
  "autonomic_limbic",  "hippocampal",                "HF",
  # --- BRAINSTEM (mixed nuclei; keep tier-2 anatomical only) ---
  "brainstem",         "midbrain",                   "VMid",
  "brainstem",         "midbrain",                   "DMid",
  "brainstem",         "midbrain",                   "MMid",
  "brainstem",         "midbrain",                   "LMid",
  "brainstem",         "pons",                       "VPons",
  "brainstem",         "pons",                       "DPons",
  "brainstem",         "pons",                       "LPons",
  "brainstem",         "medulla",                    "VMed",
  "brainstem",         "medulla",                    "DMed",
  "brainstem",         "medulla",                    "IMed",
  # --- BASAL GANGLIA ---
  "basal_ganglia",     "striatum",                   "Str",
  "basal_ganglia",     "pallidum",                   "Pd",
  "basal_ganglia",     "pallidum",                   "LPal",
  "basal_ganglia",     "pallidum",                   "VPal",
  # --- THALAMUS (interoceptive + others) ---
  "thalamus",          "thalamic_relay",             "VThal",
  "thalamus",          "thalamic_relay",             "MThal",
  "thalamus",          "thalamic_relay",             "MLThal",
  "thalamus",          "thalamic_relay",             "PThal",
  "thalamus",          "thalamic_relay",             "GThal",
  "thalamus",          "thalamic_relay",             "Rh",
  "thalamus",          "thalamic_relay",             "Rt",
  # --- SOMATOSENSORY ---
  "somatosensory",     "primary_secondary_SS",       "SI",
  "somatosensory",     "primary_secondary_SS",       "SII",
  "somatosensory",     "trigeminal_orofacial",       "TG",
  # --- AUDITORY (negative control: insula→aud rare) ---
  "auditory",          "auditory_belt_core",         "belt",
  "auditory",          "auditory_belt_core",         "core",
  "auditory",          "STG_aud_assoc",              "STGr/STSd",
  "auditory",          "STG_aud_assoc",              "STGc",
  # --- HIGHER-ORDER PARIETAL ---
  "parietal_higher",   "IPL_IPS",                    "area_7_in_IPL",
  "parietal_higher",   "IPL_IPS",                    "area_7m",
  "parietal_higher",   "IPL_IPS",                    "lat_IPS",
  "parietal_higher",   "IPL_IPS",                    "vm_IPS",
  "parietal_higher",   "SPL",                        "area_5",
  "parietal_higher",   "MST",                        "MST",
  # --- TEMPORAL ASSOCIATION ---
  "temporal_assoc",    "TE",                         "TE",
  "temporal_assoc",    "STSf",                       "STSf",
  "temporal_assoc",    "POC",                        "POC",
  "temporal_assoc",    "PrT",                        "PrT",
  # --- MOTOR / PREMOTOR (negative control for autonomic) ---
  "motor",             "primary_motor",              "M1/PM",
  "motor",             "premotor_SMA",               "SMA/preSMA",
  # --- COGNITIVE PFC (cognitive control control) ---
  "cognitive_PFC",     "lateral_PFC",                "vlPFC",
  "cognitive_PFC",     "lateral_PFC",                "dlPFC",
  "cognitive_PFC",     "lateral_PFC",                "area_8A",
  # --- VISUOSPATIAL / OTHER ---
  "visual_aux",        "visual_aux",                 "DSP",
  "visual_aux",        "visual_aux",                 "VMid"  # placeholder duplicate ignored
)
panel_hierarchy <- panel_hierarchy %>% distinct(big, small, region)

# Validate against data
in_data <- panel_hierarchy %>%
  mutate(present = region %in% ipsi_targets)

cat("\n=== Panel coverage in data ===\n")
print(in_data %>% group_by(big) %>%
        summarise(n_defined = n(), n_in_data = sum(present),
                  missing = paste(region[!present], collapse = ";")))

# Compute per-neuron score = proportion of total ipsi projection going to
# regions in each tier-1 panel
panel_score <- function(prop_mat, target_set) {
  use <- intersect(target_set, colnames(prop_mat))
  if (!length(use)) return(rep(NA_real_, nrow(prop_mat)))
  rowSums(prop_mat[, use, drop = FALSE], na.rm = TRUE)
}

scores <- as.data.frame(meta)
big_levels <- unique(panel_hierarchy$big)
for (b in big_levels) {
  regs <- panel_hierarchy$region[panel_hierarchy$big == b]
  scores[[paste0("big_", b)]] <- panel_score(ipsi_prop, regs)
}
small_levels <- unique(panel_hierarchy$small)
for (s in small_levels) {
  regs <- panel_hierarchy$region[panel_hierarchy$small == s]
  scores[[paste0("sml_", s)]] <- panel_score(ipsi_prop, regs)
}

# ============================================================
# Test L vs R for each panel x stratum
# Strata:
#   1. all_neurons (n=260) -- known confounded
#   2. IAL_excluded (IDD5+IDM, balanced)
#   3. IDD5_only
#   4. IDM_only
# ============================================================

test_lr <- function(df, score_col, label) {
  d <- df[!is.na(df[[score_col]]), ]
  if (length(unique(d$Soma_Side)) < 2) return(NULL)
  nL <- sum(d$Soma_Side == "L"); nR <- sum(d$Soma_Side == "R")
  if (nL < 3 || nR < 3) return(NULL)
  p <- tryCatch(suppressWarnings(
    wilcox.test(d[[score_col]] ~ d$Soma_Side, exact = FALSE)$p.value),
    error = function(e) NA_real_)
  vL <- d[[score_col]][d$Soma_Side == "L"]
  vR <- d[[score_col]][d$Soma_Side == "R"]
  data.frame(stratum = label, panel = score_col,
             n_L = nL, n_R = nR,
             median_L = median(vL, na.rm = TRUE),
             median_R = median(vR, na.rm = TRUE),
             mean_L = mean(vL, na.rm = TRUE),
             mean_R = mean(vR, na.rm = TRUE),
             p_raw = p)
}

run_panel <- function(df, label, score_cols) {
  res <- bind_rows(lapply(score_cols, function(s) test_lr(df, s, label)))
  if (!nrow(res)) return(res)
  res$p_BH <- p.adjust(res$p_raw, method = "BH")
  res$direction <- ifelse(res$mean_L > res$mean_R, "L>R", "R>L")
  res
}

big_cols <- paste0("big_", big_levels)
small_cols <- paste0("sml_", small_levels)

keep_balanced <- meta$Soma_Region %in% c("IDD5", "IDM")
keep_idd5 <- meta$Soma_Region == "IDD5"
keep_idm <- meta$Soma_Region == "IDM"

big_results <- bind_rows(
  run_panel(scores, "all_neurons", big_cols),
  run_panel(scores[keep_balanced, ], "IAL_excluded_IDD5_IDM", big_cols),
  run_panel(scores[keep_idd5, ], "IDD5_only", big_cols),
  run_panel(scores[keep_idm, ], "IDM_only", big_cols)
)

small_results <- bind_rows(
  run_panel(scores, "all_neurons", small_cols),
  run_panel(scores[keep_balanced, ], "IAL_excluded_IDD5_IDM", small_cols),
  run_panel(scores[keep_idd5, ], "IDD5_only", small_cols),
  run_panel(scores[keep_idm, ], "IDM_only", small_cols)
)

write.csv(big_results,
          file.path(OUT_DIR, "stats", "panels_tier1_big_lr_tests.csv"),
          row.names = FALSE)
write.csv(small_results,
          file.path(OUT_DIR, "stats", "panels_tier2_small_lr_tests.csv"),
          row.names = FALSE)

cat("\n=== Tier-1 (function_class_big) panel L vs R ===\n")
print(big_results, digits = 3, row.names = FALSE)

cat("\n=== Tier-2 (function_class_small) panel L vs R ===\n")
sig_small <- small_results[!is.na(small_results$p_BH) &
                            small_results$p_BH < 0.10, ]
if (nrow(sig_small)) {
  print(sig_small, digits = 3, row.names = FALSE)
} else {
  cat("(none with p_BH < 0.10)\n")
}

# ============================================================
# Per-target detection-probability table for autonomic_limbic
# panel within balanced strata (the only place inference is OK)
# ============================================================
auto_regs <- panel_hierarchy$region[panel_hierarchy$big == "autonomic_limbic"]
auto_regs <- intersect(auto_regs, ipsi_targets)

per_target_test <- function(prop_mat, side_vec, label, target_regs) {
  out <- bind_rows(lapply(target_regs, function(rg) {
    xL <- prop_mat[side_vec == "L", rg]
    xR <- prop_mat[side_vec == "R", rg]
    nL <- length(xL); nR <- length(xR)
    if (nL < 3 || nR < 3) return(NULL)
    tab <- matrix(c(sum(xL > 0), sum(xL == 0),
                    sum(xR > 0), sum(xR == 0)), nrow = 2)
    p_pres <- tryCatch(fisher.test(tab)$p.value,
                       error = function(e) NA_real_)
    p_mag <- tryCatch(suppressWarnings(
      wilcox.test(xL, xR, exact = FALSE)$p.value),
      error = function(e) NA_real_)
    data.frame(stratum = label, target = rg,
               n_L = nL, n_R = nR,
               frac_L = mean(xL > 0), frac_R = mean(xR > 0),
               mean_L = mean(xL), mean_R = mean(xR),
               p_presence = p_pres, p_magnitude = p_mag)
  }))
  if (!nrow(out)) return(out)
  out$p_pres_BH <- p.adjust(out$p_presence, method = "BH")
  out$p_mag_BH <- p.adjust(out$p_magnitude, method = "BH")
  out$direction <- ifelse(out$mean_L > out$mean_R, "L>R", "R>L")
  out
}

per_target_auto <- bind_rows(
  per_target_test(ipsi_prop, meta$Soma_Side, "all_neurons", auto_regs),
  per_target_test(ipsi_prop[keep_balanced, ],
                  meta$Soma_Side[keep_balanced],
                  "IAL_excluded_IDD5_IDM", auto_regs),
  per_target_test(ipsi_prop[keep_idd5, ],
                  meta$Soma_Side[keep_idd5], "IDD5_only", auto_regs),
  per_target_test(ipsi_prop[keep_idm, ],
                  meta$Soma_Side[keep_idm], "IDM_only", auto_regs)
)
write.csv(per_target_auto,
          file.path(OUT_DIR, "stats", "autonomic_limbic_per_target.csv"),
          row.names = FALSE)

cat("\n=== Autonomic_limbic per-target test (BH q<0.10) ===\n")
sig_auto <- per_target_auto[!is.na(per_target_auto$p_pres_BH) &
                              (per_target_auto$p_pres_BH < 0.10 |
                                 per_target_auto$p_mag_BH < 0.10), ]
if (nrow(sig_auto)) {
  print(sig_auto, digits = 3, row.names = FALSE)
} else {
  cat("(none)\n")
}

# ============================================================
# Plots: tier-1 score boxplots, full vs balanced subset
# ============================================================
side_cols <- c(L = "#E74C3C", R = "#3498DB")

plot_grid <- function(df, score_cols, lbl_prefix) {
  plots <- lapply(score_cols, function(s) {
    if (!s %in% names(df)) return(NULL)
    d <- df[!is.na(df[[s]]), ]
    if (!nrow(d) || length(unique(d$Soma_Side)) < 2) return(NULL)
    p <- tryCatch(suppressWarnings(
      wilcox.test(d[[s]] ~ d$Soma_Side, exact = FALSE)$p.value),
      error = function(e) NA_real_)
    ggplot(d, aes(x = Soma_Side, y = .data[[s]], fill = Soma_Side)) +
      geom_violin(alpha = 0.45, trim = FALSE) +
      geom_boxplot(width = 0.18, outlier.size = 0.7, alpha = 0.9) +
      scale_fill_manual(values = side_cols) +
      labs(title = sub("^big_", "", s),
           subtitle = sprintf("%s | p=%.3g", lbl_prefix, p),
           x = "Soma side", y = "Score") +
      theme_minimal(base_size = 10) +
      theme(legend.position = "none")
  })
  plots <- plots[!sapply(plots, is.null)]
  patchwork::wrap_plots(plots, ncol = 3)
}

fig_full <- plot_grid(scores, big_cols, "all_neurons")
ggsave(file.path(OUT_DIR, "figures", "tier1_panels_all_neurons.png"),
       fig_full, width = 14, height = 8, dpi = 200)
fig_bal <- plot_grid(scores[keep_balanced, ], big_cols,
                     "IAL_excluded (IDD5+IDM only)")
ggsave(file.path(OUT_DIR, "figures", "tier1_panels_IAL_excluded.png"),
       fig_bal, width = 14, height = 8, dpi = 200)

# Save scores
write.csv(scores, file.path(OUT_DIR, "tables", "per_neuron_panel_scores_v2.csv"),
          row.names = FALSE)
write.csv(panel_hierarchy,
          file.path(OUT_DIR, "tables", "panel_hierarchy.csv"),
          row.names = FALSE)

cat("\n\n========== DONE ==========\nFiles written to", OUT_DIR, "\n")
