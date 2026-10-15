# ============================================================
# v3: L/R insula projectome analysis using Gou et al. 2025
# 6-category functional hierarchy (Table_S2(B) + (C) joined).
#
# This replaces my ad-hoc "sympathetic"/"sensory" panels with
# the published macaque PFC functional schema.
# ============================================================

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr)
  library(ggplot2); library(patchwork); library(writexl)
})
set.seed(42)
setwd("d:/projectome_analysis/R_analysis/scripts")

PATH_INS_XLSX <- "../../neuron_tables_new/251637_INS_HE_inferred.xlsx"
PATH_CLUSTER_RDS <- "data_output/clustering/clustering_results.rds"
PATH_GOU_FUNC <- "data_output/gou_function_table/area_function_category_full.csv"
OUT_DIR <- "LR_analysis_hypothesis_v3"
dir.create(file.path(OUT_DIR, "stats"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "figures"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "tables"), recursive = TRUE, showWarnings = FALSE)

# ---- Load data ----
cluster_info <- readRDS(PATH_CLUSTER_RDS)
ipsi_raw <- read_excel(PATH_INS_XLSX, sheet = "Projection_Strength_L3_ipsi")
summ <- read_excel(PATH_INS_XLSX, sheet = "Summary")
gou <- read.csv(PATH_GOU_FUNC, stringsAsFactors = FALSE)

cat(sprintf("Gou table rows: %d  unique areas: %d  unique categories: %d\n",
            nrow(gou), length(unique(gou$BrainArea)),
            length(unique(gou$Category))))

# ============================================================
# Map user's 57 ipsi target regions to Gou's 6 categories
# Strategy: where direct match exists, use it. Where user's
# label is coarser, take the union of Gou subareas it contains.
# ============================================================

# User's region labels in this data
ipsi_targets <- setdiff(names(ipsi_raw), c("NeuronID", "Neuron_Type"))

# Build mapping table. Where the region has multiple Gou categories,
# we keep ALL of them (multi-label). For regions not in Gou table,
# we annotate from the macaque anatomy literature.
region_map <- tribble(
  ~user_region,    ~gou_areas,                 ~category_assigned,
  # --- direct matches in Gou's table ---
  "ACC",           "ACC",                      "auto",                    
  "MCC",           "MCC",                      "auto",
  "PCgG",          "PCC",                      "emo",
  "STSf",          "STSf",                     "sens",
  "STGr/STSd",     "STGr;STSd",                "sens",
  "STGc",          "STGr",                     "sens",
  "TE",            "TE",                       "sens",
  "SII",           "SII",                      "sens",
  "Pd",            "Pd",                       "motor",
  "Rt",            "Rt",                       "sens",
  "M1/PM",         "PMd;PMdr",                 "motor",
  "SMA/preSMA",    "preSMA",                   "motor",
  "VMid",          "PAG;VTA;SN",               "auto",
  "DMid",          "PAG;VTA;SN",               "auto",
  "MMid",          "PAG;VTA;SN",               "auto",
  "LMid",          "SC",                       "sens",
  "VPons",         "Pons",                     "auto",
  "DPons",         "Pons",                     "auto",
  "LPons",         "Pons",                     "sens",
  "VMed",          "Med",                      "auto",
  "DMed",          "Med",                      "auto",
  "IMed",          "Med",                      "auto",
  "Rh",            "Rt",                       "sens",
  "MThal",         "MD",                       "memory",
  "VThal",         "VA",                       "motor",
  "MLThal",        "MD;VA",                    "memory",
  "PThal",         "Pul",                      "sens",
  "GThal",         "MD",                       "memory",
  "vlPFC",         "area_45a;area_45b;area_46v;area_12l;area_12r;area_12o", "cog",
  "dlPFC",         "area_46d;area_9d;area_8Ad;area_8Bd",  "cog",
  "area_8A",       "area_8Ad",                 "cog",
  "caudal_OFC",    "area_13m;area_11;area_11m","emo",
  "lat_OFC",       "area_12l;area_12o;area_12r","emo",
  "med_OFC",       "area_10mr;area_11m;area_11","emo",
  "area_7m",       "area_7a",                  "sens",
  "area_7_in_IPL", "area_7a;LIP",              "sens",
  "lat_IPS",       "LIP",                      "sens",
  "vm_IPS",        "LIP",                      "sens",
  "area_5",        NA,                         "sens",
  "MST",           NA,                         "sens",
  "POC",           NA,                         "sens",
  "PrT",           NA,                         "sens",
  "TG",            NA,                         "sens",
  "DSP",           NA,                         "motor",
  # --- not in Gou table; literature-based ---
  "spAmy",         NA,                         "emo",   # central + extended amygdala
  "pAmy",          NA,                         "emo",
  "THy",           "LHy",                      "auto",
  "PHy",           "LHy",                      "auto",
  "ZI-H",          NA,                         "auto",  # zona incerta-hypothalamus
  "Str",           "CdH;Pu",                   "motor", # primarily motor in cortico-striatal
  "LPal",          "Pd",                       "motor",
  "VPal",          "Pd",                       "motor",
  "HF",            NA,                         "memory",
  "SI",            NA,                         "sens",
  "belt",          NA,                         "sens",
  "core",          NA,                         "sens"
)

# Coverage check
missing_user <- setdiff(ipsi_targets, region_map$user_region)
cat("\nUser regions NOT mapped:\n"); print(missing_user)

# Translate short codes to Gou 6-category names
cat_lookup <- c(
  auto   = "Autonomic and Physiological Regulation",
  emo    = "Emotional, Social and Reward Functions",
  cog    = "Cognitive and Executive Function",
  motor  = "Motor Function",
  sens   = "Sensory Processing and Cross-Functional Integration",
  memory = "Learning and Memory"
)
region_map$Category_full <- cat_lookup[region_map$category_assigned]

write.csv(region_map,
          file.path(OUT_DIR, "tables", "region_to_gou_category_map.csv"),
          row.names = FALSE)

# Build per-category region sets
panels_v3 <- split(region_map$user_region, region_map$category_assigned)
panel_full_names <- cat_lookup[names(panels_v3)]
cat("\nGou-aligned panels (n regions per category):\n")
print(sapply(panels_v3, length))

# ============================================================
# Build matrices and metadata (same as v1/v2)
# ============================================================
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
            AP_coord = as.numeric(Soma_NII_Y)) %>%
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
ipsi_prop[ok, ] <- sweep(ipsi_num[ok, , drop = FALSE], 1, ipsi_total[ok], "/")

# Per-neuron Gou-category scores
panel_score <- function(prop_mat, target_set) {
  use <- intersect(target_set, colnames(prop_mat))
  if (!length(use)) return(rep(NA_real_, nrow(prop_mat)))
  rowSums(prop_mat[, use, drop = FALSE], na.rm = TRUE)
}
scores <- as.data.frame(meta)
for (k in names(panels_v3)) {
  scores[[paste0("gou_", k)]] <- panel_score(ipsi_prop, panels_v3[[k]])
}

# ============================================================
# Test L vs R per category x stratum
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
             mean_L = mean(vL), mean_R = mean(vR),
             median_L = median(vL), median_R = median(vR),
             p_raw = p)
}

run_panel <- function(df, lbl, score_cols) {
  res <- bind_rows(lapply(score_cols, function(s) test_lr(df, s, lbl)))
  if (!nrow(res)) return(res)
  res$p_BH <- p.adjust(res$p_raw, method = "BH")
  res$direction <- ifelse(res$mean_L > res$mean_R, "L>R", "R>L")
  res
}

gou_cols <- paste0("gou_", names(panels_v3))
keep_balanced <- meta$Soma_Region %in% c("IDD5", "IDM")
keep_idd5 <- meta$Soma_Region == "IDD5"
keep_idm <- meta$Soma_Region == "IDM"

results <- bind_rows(
  run_panel(scores, "all_neurons", gou_cols),
  run_panel(scores[keep_balanced, ], "IAL_excluded_IDD5_IDM", gou_cols),
  run_panel(scores[keep_idd5, ], "IDD5_only", gou_cols),
  run_panel(scores[keep_idm, ], "IDM_only", gou_cols)
)

# Add full category name
results$Category <- cat_lookup[sub("^gou_", "", results$panel)]
write.csv(results,
          file.path(OUT_DIR, "stats", "gou_categories_lr_tests.csv"),
          row.names = FALSE)

cat("\n============================================================\n")
cat("Gou 6-category L vs R tests by stratum (sorted by stratum)\n")
cat("============================================================\n")
res_pretty <- results %>%
  dplyr::select(stratum, Category, n_L, n_R,
                mean_L, mean_R, p_raw, p_BH, direction) %>%
  arrange(stratum, desc(mean_R - mean_L))
print(res_pretty, digits = 3, row.names = FALSE)

# ============================================================
# Visualization
# ============================================================
side_cols <- c(L = "#E74C3C", R = "#3498DB")

scores_long <- scores %>%
  dplyr::select(NeuronID, Soma_Region, Soma_Side, all_of(gou_cols)) %>%
  pivot_longer(cols = all_of(gou_cols), names_to = "panel", values_to = "score") %>%
  mutate(Category = cat_lookup[sub("^gou_", "", panel)],
         stratum = case_when(
           Soma_Region %in% c("IDD5", "IDM") ~ "IAL-excluded\n(IDD5+IDM, balanced)",
           TRUE ~ "Other regions\n(unbalanced)"))

# Boxplots per category, faceted by stratum
fig_box <- ggplot(scores_long %>%
                    filter(stratum == "IAL-excluded\n(IDD5+IDM, balanced)"),
                  aes(x = Soma_Side, y = score, fill = Soma_Side)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.18, outlier.size = 0.6, alpha = 0.85) +
  facet_wrap(~ Category, ncol = 3, scales = "free_y") +
  scale_fill_manual(values = side_cols) +
  labs(title = "Gou 6-category projection scores: L vs R in balanced subset",
       subtitle = "IDD5 + IDM only (n_L = 77, n_R = 62) — IAL sampling artefact removed",
       x = "Soma side", y = "Proportion of ipsi projection") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none",
        strip.text = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "figures", "gou_categories_balanced.png"),
       fig_box, width = 13, height = 8, dpi = 200)

fig_box_full <- ggplot(scores_long, aes(x = Soma_Side, y = score, fill = Soma_Side)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.18, outlier.size = 0.6, alpha = 0.85) +
  facet_grid(stratum ~ Category, scales = "free_y") +
  scale_fill_manual(values = side_cols) +
  labs(title = "Gou 6-category projection scores: full data vs IAL-excluded subset",
       x = "Soma side", y = "Proportion") +
  theme_minimal(base_size = 9) +
  theme(legend.position = "none",
        strip.text.x = element_text(face = "bold", size = 8),
        strip.text.y = element_text(face = "bold", size = 8))
ggsave(file.path(OUT_DIR, "figures", "gou_categories_full_vs_balanced.png"),
       fig_box_full, width = 16, height = 7, dpi = 200)

# Per-region detection within Autonomic + Emotional categories,
# IAL-excluded only
auto_emo <- region_map$user_region[region_map$category_assigned %in% c("auto", "emo")]
auto_emo <- intersect(auto_emo, ipsi_targets)

per_target_test <- function(prop_mat, side_vec, label, target_regs) {
  out <- bind_rows(lapply(target_regs, function(rg) {
    xL <- prop_mat[side_vec == "L", rg]; xR <- prop_mat[side_vec == "R", rg]
    nL <- length(xL); nR <- length(xR)
    if (nL < 3 || nR < 3) return(NULL)
    tab <- matrix(c(sum(xL > 0), sum(xL == 0),
                    sum(xR > 0), sum(xR == 0)), nrow = 2)
    p_pres <- tryCatch(fisher.test(tab)$p.value, error = function(e) NA_real_)
    p_mag <- tryCatch(suppressWarnings(
      wilcox.test(xL, xR, exact = FALSE)$p.value), error = function(e) NA_real_)
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

per_tgt <- bind_rows(
  per_target_test(ipsi_prop, meta$Soma_Side, "all_neurons", auto_emo),
  per_target_test(ipsi_prop[keep_balanced, ],
                  meta$Soma_Side[keep_balanced], "IAL_excluded", auto_emo)
)
write.csv(per_tgt,
          file.path(OUT_DIR, "stats", "auto_emo_per_target_LR.csv"),
          row.names = FALSE)

cat("\n============================================================\n")
cat("Per-target tests for Autonomic + Emotional regions (BH q < 0.10)\n")
cat("============================================================\n")
sig <- per_tgt[!is.na(per_tgt$p_pres_BH) &
                 (per_tgt$p_pres_BH < 0.10 | per_tgt$p_mag_BH < 0.10), ]
if (nrow(sig)) {
  print(sig[, c("stratum", "target", "n_L", "n_R", "frac_L", "frac_R",
                "mean_L", "mean_R", "p_pres_BH", "p_mag_BH", "direction")],
        digits = 3, row.names = FALSE)
} else cat("(none)\n")

# Save scores
write.csv(scores, file.path(OUT_DIR, "tables", "per_neuron_gou_scores.csv"),
          row.names = FALSE)

cat("\n========== DONE ==========\nOutputs in:", OUT_DIR, "\n")
