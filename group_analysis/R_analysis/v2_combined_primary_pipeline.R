inp <- knitr::current_input()
pipeline_dir <- dirname(normalizePath(
  if (is.null(inp) || !nzchar(inp)) getwd() else inp,
  winslash = "/", mustWork = FALSE
))
knitr::opts_knit$set(root.dir = pipeline_dir)
knitr::opts_chunk$set(
  echo = TRUE,
  message = FALSE,
  warning = FALSE,
  fig.show = "hide",
  results = "asis",
  cache = FALSE
)

suppressPackageStartupMessages({
  library(readxl)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(vegan)
  library(scales)
  library(ComplexHeatmap)
  library(circlize)
  library(writexl)
})

set.seed(42)

PROJECT_ROOT <- "d:/projectome_analysis"
GROUP_DIR <- file.path(PROJECT_ROOT, "group_analysis")
COMBINED_XLSX <- file.path(
  GROUP_DIR, "combined", "multi_monkey_INS_combined_harmonized.xlsx"
)
GOU_MAP <- file.path(
  PROJECT_ROOT, "R_analysis", "scripts",
  "LR_analysis_hypothesis_v3", "tables", "region_to_gou_category_map.csv"
)
DIST_FILE <- file.path(GROUP_DIR, "fnt", "multi_monkey_INS_dist.txt")
JOINED_FNT <- file.path(GROUP_DIR, "fnt", "multi_monkey_INS_joined.fnt")

OUT_ROOT <- file.path(GROUP_DIR, "R_analysis", "outputs", "combined_primary_v2")
OUT_SPEC <- file.path(OUT_ROOT, "spec")
OUT_STATS <- file.path(OUT_ROOT, "stats")
OUT_FIGS <- file.path(OUT_ROOT, "figures")
OUT_OVERLAY <- file.path(OUT_ROOT, "flatmap_overlays")
OUT_TABLES <- file.path(OUT_ROOT, "tables")
for (d in c(OUT_ROOT, OUT_SPEC, OUT_STATS, OUT_FIGS, OUT_OVERLAY, OUT_TABLES)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
build_mat <- function(sheet, uids) {
  meta_cols <- c("NeuronUID", "SampleID", "NeuronID", "Neuron_Type")
  num_cols <- setdiff(colnames(sheet), meta_cols)
  m <- as.matrix(sheet[, num_cols, drop = FALSE])
  storage.mode(m) <- "numeric"
  m[is.na(m)] <- 0
  rownames(m) <- sheet$NeuronUID
  m[uids, , drop = FALSE]
}

normalize_rows <- function(m) {
  rs <- rowSums(m, na.rm = TRUE)
  out <- m
  ok <- rs > 0
  out[ok, ] <- sweep(m[ok, , drop = FALSE], 1, rs[ok], "/")
  out
}

cliffs_delta <- function(a, b) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (!length(a) || !length(b)) return(NA_real_)
  na <- length(a); nb <- length(b)
  more <- 0; less <- 0
  for (xa in a) {
    more <- more + sum(xa > b)
    less <- less + sum(xa < b)
  }
  (more - less) / (na * nb)
}

safe_wilcox <- function(x, y) {
  tryCatch(
    suppressWarnings(wilcox.test(x, y, exact = FALSE))$p.value,
    error = function(e) NA_real_
  )
}

safe_fisher <- function(p_l, n_l, p_r, n_r) {
  m <- matrix(c(p_l, n_l - p_l, p_r, n_r - p_r), nrow = 2)
  if (any(rowSums(m) == 0)) return(NA_real_)
  tryCatch(fisher.test(m)$p.value, error = function(e) NA_real_)
}

# ------------------------------------------------------------
# Load + finalize columns from harmonized file
# ------------------------------------------------------------
cat("[v2] loading harmonized combined workbook\n")
summ <- read_excel(COMBINED_XLSX, sheet = "Summary")
ipsi_l3 <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_L3_ipsi")
ipsi_l6 <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_ipsi")
contra_l6 <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_contra")
length_l6 <- read_excel(COMBINED_XLSX, sheet = "Projection_Length_ipsi")
length_l6_contra <- read_excel(COMBINED_XLSX, sheet = "Projection_Length_contra")

uids <- Reduce(intersect, list(summ$NeuronUID, ipsi_l3$NeuronUID, ipsi_l6$NeuronUID))
summ <- summ[match(uids, summ$NeuronUID), , drop = FALSE]

summ <- summ %>%
  mutate(
    Side = ifelse(
      !is.na(Soma_Side_Final) & Soma_Side_Final %in% c("L", "R"),
      Soma_Side_Final,
      Soma_Side
    ),
    Region = Soma_Region_Refined
  ) %>%
  filter(Side %in% c("L", "R"))
uids <- summ$NeuronUID

m_l3 <- build_mat(ipsi_l3, uids)
m_l6 <- build_mat(ipsi_l6, uids)
m_l6_contra <- build_mat(contra_l6, uids)
m_len_l6 <- build_mat(length_l6, uids)
m_len_l6_contra <- build_mat(length_l6_contra, uids)
p_l3 <- normalize_rows(m_l3)
p_l6 <- normalize_rows(m_l6)
# Intra-insula targets: L6 atlas columns; all other targets: L3. One row-normalization per neuron.
INSULA_TARGETS <- c("Ial", "Iai", "Iapl", "Iam/Iapm", "Ia/Id", "Ig", "Pi", "Ri")
ins_l6_cols <- intersect(INSULA_TARGETS, colnames(m_l6))
l3_extrinsic <- setdiff(colnames(m_l3), INSULA_TARGETS)
ins_l3_only <- setdiff(intersect(INSULA_TARGETS, colnames(m_l3)), ins_l6_cols)
l3_hybrid_names <- c(l3_extrinsic, ins_l3_only)
m_hybrid <- cbind(
  m_l3[, l3_hybrid_names, drop = FALSE],
  m_l6[, ins_l6_cols, drop = FALSE]
)
colnames(m_hybrid) <- c(
  paste0(l3_hybrid_names, "@L3"),
  paste0(ins_l6_cols, "@L6")
)
p_combo <- normalize_rows(m_hybrid)
ap_axis_lbl <- "Soma NII Y (NMT RAS: greater = anterior, lesser = posterior)"

# ------------------------------------------------------------
# P2 - strata + metrics spec
# ------------------------------------------------------------
cat("[v2] writing strata + metrics spec\n")
strata_spec <- tibble::tribble(
  ~stratum_id,                ~filter_R,                                               ~role,                  ~min_n_inferential,
  "IDD5_plus_IDM_balanced",   "Region %in% c('IDD5','IDM')",                            "primary inferential",  10,
  "IDD5_balanced",            "Region == 'IDD5'",                                       "stricter check",       10,
  "IDM_balanced",             "Region == 'IDM'",                                        "replication",          10,
  "all_combined",             "TRUE",                                                   "overall sensitivity",  10,
  "IAL_combined",             "Region == 'IAL'",                                        "v3 caudal-OFC sanity", 10,
  "IAPM_combined",            "Region == 'IAPM'",                                       "descriptive only",     999,
  "IDV_combined",             "Region == 'IDV'",                                        "descriptive only",     999
)
metric_spec <- tibble::tribble(
  ~metric,                    ~description,
  "frac_projecting",          "fraction of source-side neurons with prop > 0 (Fisher's exact)",
  "mean_prop",                "mean ipsi projection proportion (Wilcoxon)",
  "cliffs_delta",             "rank-based effect size",
  "log_odds_presence",        "log-odds ratio of presence",
  "regression_slope",         "OLS slope of target ~ soma_pos with permutation p"
)
write.csv(strata_spec, file.path(OUT_SPEC, "strata_spec.csv"), row.names = FALSE)
write.csv(metric_spec, file.path(OUT_SPEC, "metric_spec.csv"), row.names = FALSE)

# ------------------------------------------------------------
# P3 - context tables (per-subregion sample sizes + label provenance)
# ------------------------------------------------------------
cat("[v2] writing context tables\n")
context_n <- summ %>%
  group_by(Region, Side) %>% summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = Side, values_from = n, values_fill = 0) %>%
  mutate(total = L + R)
write.csv(context_n, file.path(OUT_STATS, "00_context_per_subregion_n.csv"), row.names = FALSE)

context_prov <- summ %>%
  count(Region, Soma_Region_Source, name = "n") %>%
  arrange(Region, Soma_Region_Source)
write.csv(context_prov, file.path(OUT_STATS, "00_context_label_provenance.csv"), row.names = FALSE)
cat("[v2] per-subregion L:R after harmonize:\n")
print(context_n)

# ------------------------------------------------------------
# Helper: stratum filtering
# ------------------------------------------------------------
stratum_idx <- function(stratum_id) {
  switch(
    stratum_id,
    "IDD5_plus_IDM_balanced" = which(summ$Region %in% c("IDD5", "IDM")),
    "IDD5_balanced"          = which(summ$Region == "IDD5"),
    "IDM_balanced"           = which(summ$Region == "IDM"),
    "all_combined"           = seq_len(nrow(summ)),
    "IAL_combined"           = which(summ$Region == "IAL"),
    "IAPM_combined"          = which(summ$Region == "IAPM"),
    "IDV_combined"           = which(summ$Region == "IDV")
  )
}

# Order rows by anatomy, then label-source for ambiguity awareness
SUBREGION_ROW_ORDER <- c("IAL", "IAPM", "IDD5", "IDM", "IDV")
# F1 adds pooled insula row ahead of subregions
F1_REGION_ORDER <- c("All_insula", SUBREGION_ROW_ORDER)
# Gou 6 short codes (heatmap column order)
PANEL_CODE_ORDER <- c("auto", "emo", "sens", "motor", "cog", "memory")

# ------------------------------------------------------------
# P4 - F1 per-subregion x Gou functional-domain panels (L+R insula)
# Domain mass = row sum of p_combo columns assigned to each Gou category.
# p_combo = L3 columns for non-insula targets + L6 columns for intra-insula targets (L3-only insula fallback).
# Assignment: (1) region_to_gou_category_map user_region (L3 table);
# (2) else modal Category from Gou et al. area_function_category_full.csv;
# (3) else insula L6 interoceptive leaves -> sensory panel.
# L6-only extras (e.g. PAG) are excluded from p_combo under this policy.
# ------------------------------------------------------------
cat("[v2] F1 per-subregion x Gou panels (hybrid L3/L6 domain profile)\n")
gou_map <- read_csv(GOU_MAP, show_col_types = FALSE)
GOU_ONTOLOGY_CSV <- file.path(
  PROJECT_ROOT, "R_analysis", "scripts", "data_output", "gou_function_table",
  "area_function_category_full.csv"
)
short_cat <- c(
  "Autonomic and Physiological Regulation" = "auto",
  "Emotional, Social and Reward Functions" = "emo",
  "Cognitive and Executive Function" = "cog",
  "Motor Function" = "motor",
  "Sensory Processing and Cross-Functional Integration" = "sens",
  "Learning and Memory" = "memory"
)
gou_full <- read_csv(GOU_ONTOLOGY_CSV, show_col_types = FALSE)
ontol_panel <- gou_full %>%
  count(.data$BrainArea, .data$Category, name = "nn") %>%
  group_by(.data$BrainArea) %>%
  slice_max(order_by = .data$nn, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(panel_code = unname(short_cat[.data$Category]))
onto_map <- stats::setNames(ontol_panel$panel_code, ontol_panel$BrainArea)
insula_l6_interoceptive <- c(
  "Ig", "Pi", "Iai", "Iapl", "Ial", "Ia/Id", "Iam/Iapm", "Ri", "PrCO"
)

cn_pc <- colnames(p_combo)
m_col <- str_match(cn_pc, "^(.+)@(L3|L6)$")
if (any(is.na(m_col[, 1]))) {
  warning("Some p_combo columns do not match name@(L3|L6); domain map may be incomplete.")
}
base_cn <- m_col[, 2]
hier_cn <- m_col[, 3]

map_panel_for_base <- function(b) {
  if (length(b) != 1L || is.na(b) || !nzchar(b)) return(NA_character_)
  j <- match(b, gou_map$user_region, nomatch = NA_integer_)
  if (!is.na(j)) return(as.character(gou_map$category_assigned[j]))
  if (b %in% names(onto_map)) {
    oc <- onto_map[[b]]
    if (!is.na(oc) && nzchar(as.character(oc))) return(as.character(oc))
  }
  if (b %in% insula_l6_interoceptive) return("sens")
  NA_character_
}
map_src_for_base <- function(b) {
  if (length(b) != 1L || is.na(b) || !nzchar(b)) return("invalid_column")
  j <- match(b, gou_map$user_region, nomatch = NA_integer_)
  if (!is.na(j)) return("gou_map_table")
  if (b %in% names(onto_map)) {
    oc <- onto_map[[b]]
    if (!is.na(oc) && nzchar(as.character(oc))) return("gou_ontology_modal")
  }
  if (b %in% insula_l6_interoceptive) return("insula_l6_interoceptive")
  "unmapped"
}
panel_per_col <- vapply(base_cn, map_panel_for_base, character(1))
names(panel_per_col) <- cn_pc
src_per_col <- vapply(base_cn, map_src_for_base, character(1))
write.csv(
  tibble::tibble(
    p_combo_column = cn_pc,
    atlas_base = base_cn,
    hierarchy = hier_cn,
    panel_code = panel_per_col,
    mapping_source = src_per_col
  ),
  file.path(OUT_STATS, "01_gou_domain_column_map.csv"),
  row.names = FALSE
)

f1_rows <- list()
for (reg in F1_REGION_ORDER) {
  idx <- if (reg == "All_insula") {
    which(summ$Region %in% SUBREGION_ROW_ORDER)
  } else {
    which(summ$Region == reg)
  }
  n_total <- length(idx)
  n_l <- sum(summ$Side[idx] == "L")
  n_r <- sum(summ$Side[idx] == "R")
  uids <- summ$NeuronUID[idx]
  for (pn in PANEL_CODE_ORDER) {
    cols <- cn_pc[!is.na(panel_per_col) & panel_per_col == pn]
    if (!length(cols)) {
      sc <- rep(0, n_total)
    } else {
      sc <- rowSums(p_combo[uids, cols, drop = FALSE], na.rm = TRUE)
    }
    sc_l <- sc[summ$Side[idx] == "L"]
    sc_r <- sc[summ$Side[idx] == "R"]
    p_lr <- if (length(sc_l) >= 3 && length(sc_r) >= 3) safe_wilcox(sc_l, sc_r) else NA_real_
    delta <- cliffs_delta(sc_l, sc_r)
    f1_rows[[length(f1_rows) + 1]] <- tibble::tibble(
      Region = reg,
      panel = pn,
      n_domain_targets = length(cols),
      n_total = n_total, n_L = n_l, n_R = n_r,
      mean_combined = mean(sc, na.rm = TRUE),
      mean_L = mean(sc_l, na.rm = TRUE),
      mean_R = mean(sc_r, na.rm = TRUE),
      cliffs_delta = delta,
      p_LR = p_lr
    )
  }
}
f1_df <- bind_rows(f1_rows) %>%
  group_by(Region) %>%
  mutate(p_LR_BH = p.adjust(p_LR, method = "BH")) %>%
  ungroup() %>%
  mutate(
    inferential = n_total >= 10 & n_L >= 3 & n_R >= 3,
    sig_label = case_when(
      is.na(p_LR_BH)        ~ "",
      p_LR_BH < 0.001        ~ "***",
      p_LR_BH < 0.01         ~ "**",
      p_LR_BH < 0.05         ~ "*",
      TRUE                   ~ ""
    ),
    Region = factor(Region, levels = F1_REGION_ORDER),
    panel_label = recode(
      panel, auto = "Autonomic", emo = "Emotional/Reward",
      cog = "Cognitive", motor = "Motor", sens = "Sensory", memory = "Memory"
    ),
    panel_label = factor(panel_label, levels = c(
      "Autonomic", "Emotional/Reward", "Sensory", "Motor", "Cognitive", "Memory"
    ))
  )
write.csv(f1_df, file.path(OUT_STATS, "01_subregion_x_gou_panels.csv"), row.names = FALSE)

# Two-panel figure: combined heatmap + L-R contrast heatmap
fig1_combined <- ggplot(f1_df, aes(panel_label, Region, fill = mean_combined)) +
  geom_tile(color = "white") +
  geom_text(aes(
    label = ifelse(mean_combined > 0.005,
                   sprintf("%.2f\nn=%d", mean_combined, n_total), sprintf("n=%d", n_total))
  ), size = 2.6, color = "black") +
  scale_fill_gradient(low = "white", high = "#2a6fbb", name = "Mean prop\n(L+R)") +
  labs(
    title = "F1A. Functional-domain projection (Gou 6 panels, L+R)",
    subtitle = "Row-normalized ipsi profile: L3 extrinsic + L6 intra-insula; cell = summed target prop per domain.",
    caption = "Domain→column map: stats/01_gou_domain_column_map.csv (v3 table, Gou ontology, L6 interoceptive→Sensory).",
    x = NULL, y = "Source (All insula pooled, then sub-region)"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1),
    plot.caption = element_text(size = 7.5, hjust = 0)
  )

fig1_lr <- ggplot(
  f1_df %>% filter(inferential),
  aes(panel_label, Region, fill = cliffs_delta)
) +
  geom_tile(color = "white") +
  geom_text(aes(
    label = ifelse(!is.na(cliffs_delta),
                   paste0(sprintf("%.2f", cliffs_delta), sig_label), "")
  ), size = 2.6, color = "black") +
  scale_fill_gradient2(
    low = "#c1272d", mid = "white", high = "#0072b2",
    midpoint = 0, name = "Cliff's d\n(L vs R)",
    limits = c(-1, 1)
  ) +
  labs(
    title = "F1B. L vs R domain contrast (Cliff's d, BH within row)",
    subtitle = "Inferential rows only (n≥10; nL,nR≥3). * p<.05  ** p<.01  *** p<.001",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave(file.path(OUT_FIGS, "F1A_subregion_x_gou_panels_combined.png"),
       fig1_combined, width = 9.0, height = 5.6, dpi = 220)
ggsave(file.path(OUT_FIGS, "F1B_subregion_x_gou_panels_LRcontrast.png"),
       fig1_lr, width = 9.0, height = 5.4, dpi = 220)

# ------------------------------------------------------------
# P5 - F2 per-subregion x key targets
# ------------------------------------------------------------
cat("[v2] F2 per-subregion x key targets\n")
# L3 for non-insula keys; L6 only for intra-insula keys (same policy as p_combo).
KEY_TARGETS_L3 <- c("caudal_OFC", "lat_OFC", "med_OFC", "spAmy", "pAmy",
                    "LPal", "VPal", "Str", "MThal", "MLThal",
                    "VMid", "VMed", "VPons", "ZI-H")
KEY_TARGETS_L6_ONLY <- c("Cl", "Ig", "Pi", "Iam/Iapm", "MD", "CM", "VTA", "PAG", "PH", "Acb")
key_from_l3 <- c(
  intersect(KEY_TARGETS_L3, colnames(p_l3)),
  intersect(setdiff(KEY_TARGETS_L6_ONLY, INSULA_TARGETS), colnames(p_l3))
)
key_from_l3 <- key_from_l3[!duplicated(key_from_l3)]
key_intra_l6 <- intersect(
  intersect(KEY_TARGETS_L6_ONLY, INSULA_TARGETS),
  colnames(p_l6)
)

f2_rows <- list()
for (reg in SUBREGION_ROW_ORDER) {
  idx <- which(summ$Region == reg)
  n_total <- length(idx)
  n_l <- sum(summ$Side[idx] == "L"); n_r <- sum(summ$Side[idx] == "R")
  for (tgt in key_from_l3) {
    mat <- p_l3
    lev <- "L3"
    vec <- mat[idx, tgt]
    vec_l <- vec[summ$Side[idx] == "L"]
    vec_r <- vec[summ$Side[idx] == "R"]
    p_lr <- if (length(vec_l) >= 3 && length(vec_r) >= 3) safe_wilcox(vec_l, vec_r) else NA_real_
    p_pres <- safe_fisher(sum(vec_l > 0), length(vec_l), sum(vec_r > 0), length(vec_r))
    f2_rows[[length(f2_rows) + 1]] <- tibble::tibble(
      Region = reg, target = tgt, hier = lev, target_id = paste0(tgt, "@", lev),
      n_total = n_total, n_L = n_l, n_R = n_r,
      frac_combined = mean(vec > 0, na.rm = TRUE),
      frac_L = if (length(vec_l)) mean(vec_l > 0) else NA_real_,
      frac_R = if (length(vec_r)) mean(vec_r > 0) else NA_real_,
      mean_combined = mean(vec, na.rm = TRUE),
      mean_L = mean(vec_l, na.rm = TRUE),
      mean_R = mean(vec_r, na.rm = TRUE),
      cliffs_delta = cliffs_delta(vec_l, vec_r),
      p_pres = p_pres,
      p_mag = p_lr
    )
  }
  for (tgt in key_intra_l6) {
    mat <- p_l6
    lev <- "L6"
    vec <- mat[idx, tgt]
    vec_l <- vec[summ$Side[idx] == "L"]
    vec_r <- vec[summ$Side[idx] == "R"]
    p_lr <- if (length(vec_l) >= 3 && length(vec_r) >= 3) safe_wilcox(vec_l, vec_r) else NA_real_
    p_pres <- safe_fisher(sum(vec_l > 0), length(vec_l), sum(vec_r > 0), length(vec_r))
    f2_rows[[length(f2_rows) + 1]] <- tibble::tibble(
      Region = reg, target = tgt, hier = lev, target_id = paste0(tgt, "@", lev),
      n_total = n_total, n_L = n_l, n_R = n_r,
      frac_combined = mean(vec > 0, na.rm = TRUE),
      frac_L = if (length(vec_l)) mean(vec_l > 0) else NA_real_,
      frac_R = if (length(vec_r)) mean(vec_r > 0) else NA_real_,
      mean_combined = mean(vec, na.rm = TRUE),
      mean_L = mean(vec_l, na.rm = TRUE),
      mean_R = mean(vec_r, na.rm = TRUE),
      cliffs_delta = cliffs_delta(vec_l, vec_r),
      p_pres = p_pres,
      p_mag = p_lr
    )
  }
}
f2_df <- bind_rows(f2_rows) %>%
  group_by(Region) %>%
  mutate(
    p_pres_BH = p.adjust(p_pres, method = "BH"),
    p_mag_BH  = p.adjust(p_mag,  method = "BH")
  ) %>%
  ungroup() %>%
  mutate(
    inferential = n_total >= 10 & n_L >= 3 & n_R >= 3,
    Region = factor(Region, levels = SUBREGION_ROW_ORDER)
  )
write.csv(f2_df, file.path(OUT_STATS, "02_subregion_x_key_targets.csv"), row.names = FALSE)

target_order_f2 <- c(
  paste0(key_from_l3, "@L3"),
  paste0(key_intra_l6, "@L6")
)
target_order_f2 <- target_order_f2[target_order_f2 %in% f2_df$target_id]
f2_df$target_id <- factor(f2_df$target_id, levels = target_order_f2)

fig2 <- ggplot(f2_df, aes(target_id, Region, fill = mean_combined)) +
  geom_tile(color = "white") +
  geom_text(aes(
    label = ifelse(mean_combined > 0.005, sprintf("%.2f", mean_combined), "")
  ), size = 2.4, color = "black") +
  scale_fill_gradient(low = "white", high = "#3a7ab8", name = "Mean prop\n(L+R)") +
  labs(
    title = "F2. Sub-region x key target (L+R combined)",
    subtitle = "L3 extrinsic targets, then L6 intra-insula; rows IAL→IDV.",
    x = "Target (region@layer)", y = "Source sub-region"
  ) +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 8))
ggsave(file.path(OUT_FIGS, "F2_subregion_x_key_targets.png"),
       fig2, width = 11.0, height = 4.4, dpi = 220)

# Paired L vs R contrast (Cliff's d) for inferential rows only.
fig2c <- ggplot(f2_df %>% filter(inferential), aes(target_id, Region, fill = cliffs_delta)) +
  geom_tile(color = "white") +
  geom_text(aes(
    label = case_when(
      !is.finite(cliffs_delta)                           ~ "",
      !is.na(p_mag_BH) & p_mag_BH < 0.001                 ~ paste0(sprintf("%.2f", cliffs_delta), "***"),
      !is.na(p_mag_BH) & p_mag_BH < 0.01                  ~ paste0(sprintf("%.2f", cliffs_delta), "**"),
      !is.na(p_mag_BH) & p_mag_BH < 0.05                  ~ paste0(sprintf("%.2f", cliffs_delta), "*"),
      TRUE                                                ~ sprintf("%.2f", cliffs_delta)
    )
  ), size = 2.4, color = "black") +
  scale_fill_gradient2(low = "#c1272d", mid = "white", high = "#0072b2",
                       midpoint = 0, limits = c(-1, 1), name = "Cliff's d\n(L vs R)") +
  labs(
    title = "F2B. Sub-region x key target L vs R contrast (inferential rows)",
    subtitle = "BH within row. * p<.05  ** p<.01  *** p<.001",
    x = "Target (region@layer)", y = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 8))
ggsave(file.path(OUT_FIGS, "F2B_subregion_x_key_targets_LRcontrast.png"),
       fig2c, width = 11.0, height = 4.4, dpi = 220)

# ------------------------------------------------------------
# P6 - F3 intra-insula heatmap (Gou-style)
# ------------------------------------------------------------
cat("[v2] F3 intra-insula heatmap (Gou-style)\n")
ins_cols <- intersect(INSULA_TARGETS, colnames(p_l6))

# Long table: source sub-region x target (L+R combined)
f3_rows <- list()
for (reg in SUBREGION_ROW_ORDER) {
  idx <- which(summ$Region == reg)
  if (length(idx) < 3) next
  for (tgt in ins_cols) {
    vec <- p_l6[idx, tgt]
    vec_l <- vec[summ$Side[idx] == "L"]
    vec_r <- vec[summ$Side[idx] == "R"]
    p_lr <- if (length(vec_l) >= 3 && length(vec_r) >= 3) safe_wilcox(vec_l, vec_r) else NA_real_
    p_pres <- safe_fisher(sum(vec_l > 0), length(vec_l), sum(vec_r > 0), length(vec_r))
    f3_rows[[length(f3_rows) + 1]] <- tibble::tibble(
      Region = reg, target = tgt,
      n_total = length(idx),
      n_with_proj = sum(vec > 0),
      frac_projecting = mean(vec > 0, na.rm = TRUE),
      mean_prop_combined = mean(vec, na.rm = TRUE),
      mean_prop_L = mean(vec_l, na.rm = TRUE),
      mean_prop_R = mean(vec_r, na.rm = TRUE),
      cliffs_delta = cliffs_delta(vec_l, vec_r),
      p_presence = p_pres,
      p_magnitude = p_lr
    )
  }
}
f3_df <- bind_rows(f3_rows) %>%
  group_by(Region) %>%
  mutate(
    p_presence_BH = p.adjust(p_presence, method = "BH"),
    p_magnitude_BH = p.adjust(p_magnitude, method = "BH")
  ) %>%
  ungroup() %>%
  mutate(
    edge_above_thr = n_with_proj >= 3,
    inferential_row = n_total >= 10
  )
write.csv(f3_df %>% select(Region, target, mean_prop_combined),
          file.path(OUT_STATS, "03_intra_insula_meanprop.csv"), row.names = FALSE)
write.csv(f3_df %>% select(Region, target, frac_projecting),
          file.path(OUT_STATS, "03_intra_insula_prevalence.csv"), row.names = FALSE)
write.csv(f3_df, file.path(OUT_STATS, "03_intra_insula_full.csv"), row.names = FALSE)

# Build wide matrix for hclust ordering
mat_prev <- f3_df %>%
  select(Region, target, frac_projecting) %>%
  pivot_wider(names_from = target, values_from = frac_projecting, values_fill = 0) %>%
  as.data.frame()
rownames(mat_prev) <- mat_prev$Region; mat_prev$Region <- NULL
mat_prev <- as.matrix(mat_prev)

# Hierarchical reorder of insula targets (Gou recipe: ward.D2, no enforced barjoseph in base R hclust)
if (ncol(mat_prev) >= 2) {
  d_t <- dist(t(mat_prev))
  if (all(is.finite(d_t)) && length(d_t) > 0) {
    hc_t <- hclust(d_t, method = "ward.D2")
    target_order_f3 <- colnames(mat_prev)[hc_t$order]
  } else {
    target_order_f3 <- INSULA_TARGETS
  }
} else {
  target_order_f3 <- INSULA_TARGETS
}

f3_plot <- f3_df %>%
  mutate(
    target = factor(target, levels = target_order_f3),
    Region = factor(Region, levels = SUBREGION_ROW_ORDER),
    is_self_proj = paste0("(self?)") # marker for self-projection cells handled below
  )

self_proj_lookup <- list(
  IAL = c("Ial"), IAPM = c("Iam/Iapm"),
  IDD5 = c(), IDM = c("Ia/Id"), IDV = c()
)
f3_plot$is_self <- mapply(function(reg, tgt) {
  tgt %in% self_proj_lookup[[reg]]
}, as.character(f3_plot$Region), as.character(f3_plot$target))

# Visual: combined mean-prop heatmap with sample-size annotations and self-projection de-emphasized
fig3a <- ggplot(f3_plot, aes(target, Region)) +
  geom_tile(aes(fill = mean_prop_combined), color = "white") +
  geom_tile(
    data = filter(f3_plot, is_self),
    aes(target, Region), fill = NA, color = "grey30", linetype = "dotted",
    inherit.aes = FALSE, linewidth = 0.4
  ) +
  geom_text(aes(
    label = ifelse(edge_above_thr & mean_prop_combined > 0.01,
                   sprintf("%.2f\nn=%d", mean_prop_combined, n_with_proj),
                   ifelse(n_with_proj > 0, sprintf("n=%d", n_with_proj), ""))
  ), size = 2.4, color = "black") +
  scale_fill_gradient(low = "white", high = "#1f6f8b", name = "mean prop\n(L+R)") +
  labs(
    title = "F3A. Intra-insula source x target (L+R combined)",
    subtitle = "Columns: Ward.D2 on prevalence; dotted outline = same-subregion “self” edge.",
    x = "Target insula sub-region", y = "Source soma sub-region"
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

# Prevalence heatmap (paired)
fig3b <- ggplot(f3_plot, aes(target, Region, fill = frac_projecting)) +
  geom_tile(color = "white") +
  geom_text(aes(
    label = ifelse(edge_above_thr,
                   sprintf("%d%%\nn=%d/%d",
                           round(100 * frac_projecting),
                           n_with_proj, n_total),
                   "")
  ), size = 2.4, color = "black") +
  scale_fill_gradient(low = "white", high = "#b85a3a", name = "fraction\nprojecting") +
  labs(
    title = "F3B. Intra-insula prevalence (fraction-of-source projecting)",
    subtitle = "Edges with <3 projecting neurons blank; n shown when above threshold.",
    x = "Target insula sub-region", y = "Source soma sub-region"
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

# L-R contrast (Cliff's d)
fig3c <- ggplot(f3_plot %>% filter(inferential_row), aes(target, Region, fill = cliffs_delta)) +
  geom_tile(color = "white") +
  geom_text(aes(
    label = case_when(
      !is.finite(cliffs_delta)            ~ "",
      !is.na(p_magnitude_BH) & p_magnitude_BH < 0.001 ~ paste0(sprintf("%.2f", cliffs_delta), "***"),
      !is.na(p_magnitude_BH) & p_magnitude_BH < 0.01  ~ paste0(sprintf("%.2f", cliffs_delta), "**"),
      !is.na(p_magnitude_BH) & p_magnitude_BH < 0.05  ~ paste0(sprintf("%.2f", cliffs_delta), "*"),
      TRUE                                ~ sprintf("%.2f", cliffs_delta)
    )
  ), size = 2.4, color = "black") +
  scale_fill_gradient2(low = "#c1272d", mid = "white", high = "#0072b2",
                       midpoint = 0, limits = c(-1, 1), name = "Cliff's d\n(L vs R)") +
  labs(
    title = "F3C. Intra-insula L vs R contrast (inferential rows)",
    subtitle = "BH within row. * p<.05  ** p<.01  *** p<.001",
    x = "Target insula sub-region", y = "Source soma sub-region"
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))

ggsave(file.path(OUT_FIGS, "F3A_intra_insula_combined.png"),  fig3a, width = 9.6, height = 4.6, dpi = 220)
ggsave(file.path(OUT_FIGS, "F3B_intra_insula_prevalence.png"), fig3b, width = 9.6, height = 4.6, dpi = 220)
ggsave(file.path(OUT_FIGS, "F3C_intra_insula_LRcontrast.png"), fig3c, width = 9.6, height = 4.6, dpi = 220)

# ------------------------------------------------------------
# P7 - F4 interoceptive gradient as fitted regression
# We model: target_proportion ~ soma_NII_Y (AP axis), per side,
# for granular-Ig target and a granular_to_agranular contrast vector.
# NMT RAS: Soma_NII_Y increases toward anterior (same convention as v3 Rmd).
# ------------------------------------------------------------
cat("[v2] F4 interoceptive gradient regression\n")
ap <- summ$Soma_NII_Y
gradient_targets <- c("Ig", "Pi", "Ial", "Iam/Iapm", "Ia/Id")
gradient_targets <- intersect(gradient_targets, colnames(p_l6))

f4_rows <- list()
f4_scatter <- list()
for (tgt in gradient_targets) {
  for (sd in c("L", "R", "all")) {
    sel <- if (sd == "all") seq_len(nrow(summ)) else which(summ$Side == sd)
    sel <- sel[summ$Region[sel] %in% SUBREGION_ROW_ORDER]
    if (length(sel) < 5) next
    y <- p_l6[sel, tgt]
    x <- ap[sel]
    valid <- is.finite(x) & is.finite(y)
    if (sum(valid) < 5) next
    fit <- lm(y[valid] ~ x[valid])
    co <- coef(summary(fit))
    slope <- co[2, "Estimate"]
    se    <- co[2, "Std. Error"]
    p     <- co[2, "Pr(>|t|)"]
    f4_rows[[length(f4_rows) + 1]] <- tibble::tibble(
      target = tgt, side = sd, n = sum(valid),
      slope = slope, slope_se = se, slope_p = p,
      r2 = summary(fit)$r.squared
    )
    f4_scatter[[length(f4_scatter) + 1]] <- tibble::tibble(
      target = tgt, side = sd,
      ap = x[valid], prop = y[valid],
      Region = summ$Region[sel][valid]
    )
  }
}
f4_df <- bind_rows(f4_rows) %>%
  group_by(target) %>%
  mutate(slope_p_BH = p.adjust(slope_p, method = "BH")) %>%
  ungroup()
f4_scatter_df <- bind_rows(f4_scatter)
write.csv(f4_df, file.path(OUT_STATS, "04_interoceptive_gradient_regressions.csv"), row.names = FALSE)
write.csv(f4_scatter_df, file.path(OUT_STATS, "04_interoceptive_gradient_scatter_points.csv"), row.names = FALSE)

if (nrow(f4_scatter_df)) {
  f4_plot <- f4_scatter_df %>% filter(side %in% c("L", "R"))
  fig4 <- ggplot(f4_plot, aes(ap, prop, color = side)) +
    geom_point(alpha = 0.45, size = 1.2) +
    geom_smooth(method = "lm", se = TRUE, alpha = 0.18, formula = y ~ x) +
    facet_wrap(~ target, ncol = 5, scales = "free_y") +
    scale_color_manual(values = c(L = "#0072b2", R = "#c1272d"), name = "Side") +
    labs(
      title = "F4. Interoceptive-gradient regressions (per-side fits)",
      subtitle = "OLS lines per facet; slopes and p in stats/04_interoceptive_gradient_regressions.csv",
      x = ap_axis_lbl,
      y = "Projection proportion to target (row-normalized L6)"
    ) +
    theme_minimal(base_size = 10)
  ggsave(file.path(OUT_FIGS, "F4_interoceptive_gradient.png"), fig4,
         width = 12.0, height = 4.6, dpi = 220)
}

# ------------------------------------------------------------
# P8 - F5 continuous per-neuron Ibias on projection STRENGTH
# Use Projection_Length_ipsi vs contra; if both zero we treat as bias = -1 (all ipsi by default)
# ------------------------------------------------------------
cat("[v2] F5 per-neuron Ibias (continuous)\n")
len_ipsi  <- rowSums(m_len_l6, na.rm = TRUE)
len_contra <- rowSums(m_len_l6_contra, na.rm = TRUE)
ibias_df <- tibble::tibble(
  NeuronUID = summ$NeuronUID,
  SampleID = summ$SampleID,
  Region = summ$Region,
  Side = summ$Side,
  ap_y = summ$Soma_NII_Y,
  ml_x = summ$Soma_NII_X,
  total_ipsi  = len_ipsi,
  total_contra = len_contra,
  has_any_contra = len_contra > 0
) %>%
  mutate(
    Ibias = ifelse(total_ipsi + total_contra > 0,
                   (total_contra - total_ipsi) / (total_contra + total_ipsi),
                   NA_real_)
  )
write.csv(ibias_df, file.path(OUT_STATS, "05_per_neuron_ibias.csv"), row.names = FALSE)

ibias_summary <- ibias_df %>%
  group_by(Region, Side) %>%
  summarise(
    n = n(),
    n_with_contra = sum(has_any_contra),
    pct_bilateral = mean(has_any_contra),
    median_Ibias = median(Ibias, na.rm = TRUE),
    mean_Ibias = mean(Ibias, na.rm = TRUE),
    .groups = "drop"
  )
write.csv(ibias_summary, file.path(OUT_STATS, "05_per_neuron_ibias_summary.csv"), row.names = FALSE)

fig5 <- ggplot(ibias_df %>% filter(Region %in% SUBREGION_ROW_ORDER),
               aes(ap_y, Ibias, color = Side)) +
  geom_jitter(alpha = 0.5, size = 1.2, height = 0.02) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.18, formula = y ~ x) +
  facet_wrap(~ Region, ncol = 5) +
  scale_color_manual(values = c(L = "#0072b2", R = "#c1272d"), name = "Side") +
  ylim(-1.05, 1.05) +
  labs(
    title = "F5. Per-neuron continuous hemispheric bias (Gou-style)",
    subtitle = "Ibias = (contra−ipsi)/(contra+ipsi) on total axon length; −1 = purely ipsilateral.",
    x = ap_axis_lbl, y = "Hemispheric bias (Ibias)"
  ) +
  theme_minimal(base_size = 10)
ggsave(file.path(OUT_FIGS, "F5_per_neuron_ibias.png"), fig5,
       width = 11.5, height = 4.4, dpi = 220)

# ------------------------------------------------------------
# P8b - F6 primary: SQ3d projection laterality index (hybrid-aligned)
# Per-neuron hybrid composition profile (p_combo); LI = (mean_L-mean_R)/(mean_L+mean_R) on
# group-mean proportions. Bilateral soma strata repeat LI within subregions that
# have both L and R sampling (reduces misleading "asymmetry" from missing sides).
# receipt_class flags targets that truly receive from both source groups.
# ------------------------------------------------------------
cat("[v2] F6 SQ3d projection laterality index (composition LI)\n")
eps_li <- 1e-8
mat_ipsi_prop_li <- t(p_combo)
colnames(mat_ipsi_prop_li) <- rownames(p_combo)
region_names <- rownames(mat_ipsi_prop_li)
meta_li <- summ %>%
  transmute(
    NeuronUID,
    Soma_Side = Side,
    Soma_Region_chr = trimws(as.character(Region))
  )

calc_li_table <- function(ids, stratum_name) {
  ids <- intersect(ids, colnames(mat_ipsi_prop_li))
  if (length(ids) < 6L) return(NULL)
  sub_meta <- meta_li[match(ids, meta_li$NeuronUID), , drop = FALSE]
  sub_meta <- sub_meta[!is.na(sub_meta$NeuronUID), , drop = FALSE]
  if (n_distinct(sub_meta$Soma_Side) < 2L) return(NULL)
  ids_L <- sub_meta$NeuronUID[sub_meta$Soma_Side == "L"]
  ids_R <- sub_meta$NeuronUID[sub_meta$Soma_Side == "R"]
  if (length(ids_L) < 3L || length(ids_R) < 3L) return(NULL)
  mean_L <- rowMeans(mat_ipsi_prop_li[, ids_L, drop = FALSE], na.rm = TRUE)
  mean_R <- rowMeans(mat_ipsi_prop_li[, ids_R, drop = FALSE], na.rm = TRUE)
  data.frame(
    stratum = stratum_name,
    target_region = region_names,
    mean_L = mean_L,
    mean_R = mean_R,
    n_L = length(ids_L),
    n_R = length(ids_R),
    LI = (mean_L - mean_R) / (mean_L + mean_R + eps_li),
    p_wilcox = vapply(region_names, function(rg) {
      x <- as.numeric(mat_ipsi_prop_li[rg, ids_L, drop = TRUE])
      y <- as.numeric(mat_ipsi_prop_li[rg, ids_R, drop = TRUE])
      safe_wilcox(x, y)
    }, numeric(1)),
    stringsAsFactors = FALSE
  )
}

li_all <- calc_li_table(meta_li$NeuronUID, "all_neurons")
bilateral_soma_regions <- meta_li %>%
  filter(!is.na(Soma_Region_chr), nzchar(Soma_Region_chr)) %>%
  group_by(Soma_Region_chr) %>%
  filter(n_distinct(Soma_Side) == 2L) %>%
  group_keys() %>%
  pull(Soma_Region_chr)

li_region_list <- lapply(bilateral_soma_regions, function(reg) {
  ids <- meta_li$NeuronUID[meta_li$Soma_Region_chr == reg]
  calc_li_table(ids, paste0("region_", reg))
})
li_region <- bind_rows(li_region_list)

li_all_tab <- bind_rows(li_all, li_region) %>%
  filter(!is.na(LI), is.finite(LI)) %>%
  group_by(stratum) %>%
  mutate(
    p_BH = p.adjust(p_wilcox, method = "BH"),
    significant = !is.na(p_BH) & p_BH < 0.05
  ) %>%
  ungroup() %>%
  mutate(
    abs_LI = abs(LI),
    direction = ifelse(LI >= 0, "L > R", "R > L"),
    receipt_class = case_when(
      mean_L > 0 & mean_R > 0 ~ "bilateral_receiving",
      TRUE ~ "one_side_only_or_extreme"
    )
  )

write.csv(
  li_all_tab,
  file.path(OUT_STATS, "06b_projection_laterality_index_all_targets.csv"),
  row.names = FALSE
)

li_sig <- li_all_tab %>%
  filter(significant) %>%
  mutate(
    LI_dir = ifelse(LI >= 0, "L>R", "R>L")
  )

make_li_bars <- function(df, title_txt) {
  if (!nrow(df)) {
    return(
      ggplot() +
        annotate("text", x = 0.5, y = 0.5, label = "No BH-significant regions in this class") +
        theme_void() +
        labs(title = title_txt)
    )
  }
  df2 <- df %>%
    group_by(stratum) %>%
    arrange(LI, .by_group = TRUE) %>%
    mutate(
      target_stratum = paste0(stratum, "__", target_region),
      target_stratum = factor(target_stratum, levels = unique(target_stratum)),
      LI_dir = ifelse(LI >= 0, "L>R", "R>L")
    ) %>%
    ungroup()
  ggplot(df2, aes(target_stratum, LI, fill = LI_dir)) +
    geom_col(width = 0.82) +
    geom_hline(yintercept = 0, linetype = 2, linewidth = 0.35) +
    coord_flip() +
    facet_grid(stratum ~ ., scales = "free_y", space = "free_y", switch = "y") +
    scale_x_discrete(labels = function(x) sub("^.*__", "", x)) +
    scale_fill_manual(values = c("L>R" = "#E74C3C", "R>L" = "#3498DB"), name = "Dir") +
    labs(
      title = title_txt,
      subtitle = "BH across targets within each stratum; nL/nR in stats CSV.",
      x = "Target (region@layer)", y = "Laterality index (L−R)/(L+R)"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      strip.text.y.left = element_text(face = "bold", angle = 0),
      axis.text.y = element_text(size = 7.5),
      plot.subtitle = element_text(size = 8.5)
    )
}

g6b <- make_li_bars(
  dplyr::filter(li_sig, receipt_class == "bilateral_receiving"),
  "F6. Bilateral-receiving targets (preferred LI interpretation)"
)
g6c <- make_li_bars(
  dplyr::filter(li_sig, receipt_class == "one_side_only_or_extreme"),
  "F6 supplement. One-sided / extreme receipt (QC; not symmetric sampling)"
)
ggsave(
  file.path(OUT_FIGS, "F6_projection_LI_bilateral_receiving.png"),
  g6b, width = 14, height = 12, dpi = 220
)
ggsave(
  file.path(OUT_FIGS, "F6_projection_LI_one_side_or_extreme.png"),
  g6c, width = 14, height = 10, dpi = 220
)

li_plot_df <- li_all_tab %>%
  mutate(
    stratum_family = case_when(
      stratum == "all_neurons" ~ "all_neurons",
      grepl("^region_", stratum) ~ "bilateral_soma_subregions",
      TRUE ~ "other"
    )
  )
g6v <- ggplot(li_plot_df, aes(stratum_family, LI, fill = stratum_family)) +
  geom_violin(alpha = 0.45, trim = FALSE) +
  geom_boxplot(width = 0.18, outlier.size = 0.5, alpha = 0.9) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.35) +
  labs(
    title = "F6. Target-level LI distributions by stratum family",
    subtitle = "Includes all-neurons and per-subregion strata with both L and R somas.",
    x = NULL, y = "Laterality index"
  ) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none")
ggsave(
  file.path(OUT_FIGS, "F6_projection_LI_violin_by_stratum_family.png"),
  g6v, width = 8, height = 4.5, dpi = 220
)

if (nrow(li_sig)) {
  write.csv(
    li_sig,
    file.path(OUT_STATS, "06b_projection_laterality_index_BH_significant.csv"),
    row.names = FALSE
  )
}

# F10 — Gou-inspired AP-octile heatmap (after F5 so AP axis convention is shared)
cat("[v2] F10 AP-octile interoceptive target profile (Gou-style heatmap)\n")
summ_ap <- summ %>%
  filter(Region %in% SUBREGION_ROW_ORDER, is.finite(Soma_NII_Y))
if (nrow(summ_ap) >= 40) {
  ap_brks <- unique(as.numeric(quantile(
    summ_ap$Soma_NII_Y, probs = seq(0, 1, by = 0.125), na.rm = TRUE
  )))
  if (length(ap_brks) >= 3) {
    summ_ap$ap_oct <- cut(
      summ_ap$Soma_NII_Y, breaks = ap_brks, include.lowest = TRUE,
      labels = FALSE
    )
    f10_tgt <- intersect(
      c("Ig", "Pi", "Ial", "Iam/Iapm", "Ia/Id"), colnames(p_l6)
    )
    f10_rows <- list()
    if (length(f10_tgt)) {
      for (k in sort(unique(summ_ap$ap_oct[!is.na(summ_ap$ap_oct)]))) {
        u <- summ_ap$NeuronUID[summ_ap$ap_oct == k]
        if (length(u) < 3) next
        med_y <- median(summ_ap$Soma_NII_Y[summ_ap$ap_oct == k], na.rm = TRUE)
        cm <- colMeans(p_l6[u, f10_tgt, drop = FALSE], na.rm = TRUE)
        f10_rows[[length(f10_rows) + 1]] <- bind_cols(
          tibble::tibble(ap_oct = k, median_nii_y = med_y),
          tibble::as_tibble_row(cm)
        )
      }
    }
    if (length(f10_rows)) {
      f10_df <- bind_rows(f10_rows) %>%
        arrange(median_nii_y) %>%
        mutate(
          ap_label = sprintf("Oct %d\n(Y~%.0f)", ap_oct, median_nii_y),
          ap_label = factor(ap_label, levels = unique(ap_label))
        )
      write.csv(
        f10_df, file.path(OUT_STATS, "10_ap_octile_interoceptive_profile.csv"),
        row.names = FALSE
      )
      f10_long <- f10_df %>%
        tidyr::pivot_longer(
          all_of(f10_tgt), names_to = "target", values_to = "mean_prop"
        )
      fig10 <- ggplot(f10_long, aes(target, ap_label, fill = mean_prop)) +
        geom_tile(color = "grey85") +
        geom_text(aes(label = sprintf("%.2f", mean_prop)), size = 2.8, color = "grey15") +
        scale_fill_gradient(low = "white", high = "#1a3353", name = "Mean\nprop") +
        labs(
          title = "F10. Interoceptive targets vs soma AP (octiles)",
          subtitle = "Rows: soma NII-Y octiles (posterior low Y bottom); L6 row-normalized mean prop.",
          x = "Target", y = "Soma AP (posterior ↑ anterior)"
        ) +
        theme_minimal(base_size = 9) +
        theme(
          panel.grid = element_blank(),
          axis.text.x = element_text(angle = 30, hjust = 1)
        )
      ggsave(
        file.path(OUT_FIGS, "F10_interoceptive_AP_octile_heatmap.png"), fig10,
        width = 6.2, height = 5.0, dpi = 220
      )
    }
  }
}

# ------------------------------------------------------------
# P9 - F6 supplement: rank-based receiver tests (Cliff / Fisher; hybrid L3/L6)
# Test each target once at its policy hierarchy; BH within (stratum x family)
# Family = "intra_insula" if target in INSULA_TARGETS else "extra_insula"
# ------------------------------------------------------------
cat("[v2] F6 supplement: asymmetric receivers ranked (proper BH; hybrid L3/L6 policy)\n")
strata_to_test <- c("IDD5_plus_IDM_balanced", "IDD5_balanced", "IDM_balanced", "all_combined")
f6_tgt_l3 <- setdiff(colnames(p_l3), intersect(INSULA_TARGETS, colnames(p_l6)))
f6_tgt_l6 <- intersect(INSULA_TARGETS, colnames(p_l6))
f6_targets <- unique(c(f6_tgt_l3, f6_tgt_l6))
f6_rows <- list()
for (st in strata_to_test) {
  idx <- stratum_idx(st)
  if (length(idx) < 10) next
  side <- summ$Side[idx]
  for (tgt in f6_targets) {
    if (tgt %in% INSULA_TARGETS && tgt %in% colnames(p_l6)) {
      lev <- "L6"
      mat <- p_l6
    } else if (tgt %in% colnames(p_l3)) {
      lev <- "L3"
      mat <- p_l3
    } else {
      next
    }
    vec <- mat[idx, tgt]
    vec_l <- vec[side == "L"]
    vec_r <- vec[side == "R"]
    if (length(vec_l) < 3 || length(vec_r) < 3) next
    n_proj <- sum(vec > 0)
    if (n_proj < 3) next
    fam <- ifelse(tgt %in% INSULA_TARGETS, "intra_insula", "extra_insula")
    p_pres <- safe_fisher(sum(vec_l > 0), length(vec_l),
                          sum(vec_r > 0), length(vec_r))
    p_mag  <- safe_wilcox(vec_l, vec_r)
    f6_rows[[length(f6_rows) + 1]] <- tibble::tibble(
      stratum = st, target = tgt, hier = lev,
      target_id = paste0(tgt, "@", lev),
      family = fam,
      n_L = length(vec_l), n_R = length(vec_r),
      frac_L = mean(vec_l > 0), frac_R = mean(vec_r > 0),
      mean_L = mean(vec_l), mean_R = mean(vec_r),
      cliffs_delta = cliffs_delta(vec_l, vec_r),
      p_pres = p_pres, p_mag = p_mag
    )
  }
}
f6_df <- bind_rows(f6_rows) %>%
  group_by(stratum, hier, family) %>%
  mutate(
    p_pres_BH = p.adjust(p_pres, method = "BH"),
    p_mag_BH  = p.adjust(p_mag,  method = "BH"),
    best_q = pmin(p_pres_BH, p_mag_BH, na.rm = TRUE),
    direction = ifelse(mean_L > mean_R, "L>R", "R>L")
  ) %>%
  ungroup() %>%
  arrange(best_q, desc(abs(cliffs_delta)))
write.csv(f6_df, file.path(OUT_STATS, "06_asymmetric_receivers_BH.csv"), row.names = FALSE)

# Top targets that survive BH<0.05 in any balanced stratum.
# Group by target_id (region@hierarchy) so LPal@L3 is distinct from L6 pallidal sub-leaves.
balanced_strata <- c("IDD5_plus_IDM_balanced", "IDD5_balanced", "IDM_balanced")
top_targets <- f6_df %>%
  filter(stratum %in% balanced_strata, best_q < 0.05) %>%
  group_by(target_id) %>%
  summarise(
    best_q = min(best_q, na.rm = TRUE),
    best_stratum = stratum[which.min(best_q)],
    direction_at_best = direction[which.min(best_q)],
    abs_delta = max(abs(cliffs_delta), na.rm = TRUE),
    family = first(family),
    .groups = "drop"
  ) %>%
  arrange(best_q, desc(abs_delta)) %>%
  slice_head(n = 18)
if (nrow(top_targets)) {
  top_targets <- top_targets %>%
    mutate(target_id = factor(target_id, levels = rev(target_id)))
  fig6 <- ggplot(top_targets, aes(target_id, abs_delta, fill = direction_at_best)) +
    geom_col() +
    geom_text(aes(label = sprintf("q=%.2g\n%s\n%s", best_q, best_stratum, family)),
              hjust = -0.05, size = 2.2, lineheight = 0.95) +
    coord_flip() +
    expand_limits(y = max(top_targets$abs_delta, na.rm = TRUE) * 1.55) +
    scale_fill_manual(values = c(`L>R` = "#0072b2", `R>L` = "#c1272d"), name = "Direction") +
    labs(
      title = "F6 supplement. Top asymmetric receiver targets (rank-based)",
      subtitle = "BH within stratum × layer × family (intra vs extra insula).",
      caption = "Composition LI: F6_projection_LI_*.png",
      x = "Target (region@layer)", y = "max |Cliff's δ|"
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.caption = element_text(size = 7.5, hjust = 0))
} else {
  fig6 <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No rank-based BH hits in balanced strata") +
    theme_void() +
    labs(
      title = "F6 supplement. Top asymmetric receiver targets (rank-based)",
      subtitle = "No BH hits in balanced strata at q<.05.",
      caption = "Composition LI: F6_projection_LI_*.png"
    ) +
    theme(plot.caption = element_text(size = 7.5, hjust = 0))
}
ggsave(file.path(OUT_FIGS, "F6_supplement_rank_based_receivers.png"), fig6,
       width = 9.5, height = 6.0, dpi = 220)

# ------------------------------------------------------------
# P10 - F7 hierarchy sensitivity at TARGET level for headline targets
# ------------------------------------------------------------
cat("[v2] F7 headline targets under hybrid L3/L6 policy\n")
HIER_TARGETS <- list(
  Ig           = list(tgt = "Ig",         lev = "L6"),
  Cl           = list(tgt = "Cl",         lev = "L3"),
  LPal         = list(tgt = "LPal",       lev = "L3"),
  VPal         = list(tgt = "VPal",       lev = "L3"),
  PAG          = list(tgt = "VMid",       lev = "L3"),
  caudal_OFC   = list(tgt = "caudal_OFC", lev = "L3")
)
f7_rows <- list()
for (st in c("IDD5_plus_IDM_balanced", "IDD5_balanced", "IDM_balanced", "all_combined")) {
  idx <- stratum_idx(st)
  if (length(idx) < 10) next
  side <- summ$Side[idx]
  for (nm in names(HIER_TARGETS)) {
    cfg <- HIER_TARGETS[[nm]]
    lev <- cfg$lev
    tgt <- cfg$tgt
    mat <- if (lev == "L3") p_l3 else p_l6
    if (!(tgt %in% colnames(mat))) next
    vec <- mat[idx, tgt]
    vec_l <- vec[side == "L"]; vec_r <- vec[side == "R"]
    f7_rows[[length(f7_rows) + 1]] <- tibble::tibble(
      target = nm, hierarchy = lev, stratum = st,
      mean_L = mean(vec_l), mean_R = mean(vec_r),
      cliffs_delta = cliffs_delta(vec_l, vec_r),
      p_mag = safe_wilcox(vec_l, vec_r),
      p_pres = safe_fisher(sum(vec_l > 0), length(vec_l),
                           sum(vec_r > 0), length(vec_r))
    )
  }
}
f7_df <- bind_rows(f7_rows) %>%
  group_by(stratum) %>%
  mutate(
    p_mag_BH = p.adjust(p_mag, method = "BH"),
    p_pres_BH = p.adjust(p_pres, method = "BH")
  ) %>%
  ungroup() %>%
  mutate(
    direction = ifelse(mean_L > mean_R, "L>R", "R>L"),
    sig_label = case_when(
      is.na(p_mag_BH)        ~ "",
      p_mag_BH < 0.001        ~ "***",
      p_mag_BH < 0.01         ~ "**",
      p_mag_BH < 0.05         ~ "*",
      TRUE                    ~ ""
    )
  )
write.csv(f7_df, file.path(OUT_STATS, "07_hierarchy_sensitivity_per_target.csv"), row.names = FALSE)

fig7 <- ggplot(f7_df, aes(target, cliffs_delta, fill = direction)) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_text(aes(label = paste0(hierarchy, sig_label)),
            position = position_dodge(width = 0.9),
            vjust = -0.3, size = 2.6) +
  facet_wrap(~ stratum, ncol = 2) +
  scale_fill_manual(values = c(`L>R` = "#0072b2", `R>L` = "#c1272d")) +
  labs(
    title = "F7. Headline targets (resolved hierarchy)",
    subtitle = "Cliff's d (L vs R); PAG mapped to VMid@L3. * p_mag BH<.05",
    caption = "Layer (L3/L6) on bars; stats/07_hierarchy_sensitivity_per_target.csv",
    x = "Target", y = "Cliff's d"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1),
    plot.caption = element_text(size = 7.5, hjust = 0)
  )
ggsave(file.path(OUT_FIGS, "F7_hierarchy_sensitivity.png"), fig7,
       width = 10.5, height = 6.0, dpi = 220)

# ------------------------------------------------------------
# P11 - F8 LOSO sensitivity (drop-one-monkey)
# ------------------------------------------------------------
cat("[v2] F8 LOSO sensitivity\n")
LOSO_TARGETS <- c("Ig", "Cl", "LPal", "VPal", "Pi", "Ia/Id", "caudal_OFC", "VPons", "VMid")

samples_loso <- unique(summ$SampleID)
f8_rows <- list()
for (drop in c("__none__", samples_loso)) {
  keep <- if (drop == "__none__") seq_len(nrow(summ)) else which(summ$SampleID != drop)
  for (st in c("IDD5_plus_IDM_balanced", "IDD5_balanced", "all_combined")) {
    idx <- intersect(keep, stratum_idx(st))
    if (length(idx) < 10) next
    side <- summ$Side[idx]
    if (sum(side == "L") < 3 || sum(side == "R") < 3) next
    for (tgt in LOSO_TARGETS) {
      mat <- if (tgt %in% INSULA_TARGETS && tgt %in% colnames(p_l6)) p_l6 else p_l3
      if (!(tgt %in% colnames(mat))) next
      vec <- mat[idx, tgt]
      vec_l <- vec[side == "L"]; vec_r <- vec[side == "R"]
      f8_rows[[length(f8_rows) + 1]] <- tibble::tibble(
        dropped = drop, stratum = st, target = tgt,
        n_L = length(vec_l), n_R = length(vec_r),
        cliffs_delta = cliffs_delta(vec_l, vec_r),
        p_mag = safe_wilcox(vec_l, vec_r)
      )
    }
  }
}
f8_df <- bind_rows(f8_rows) %>%
  group_by(stratum, dropped) %>%
  mutate(p_mag_BH = p.adjust(p_mag, method = "BH")) %>%
  ungroup() %>%
  mutate(direction = ifelse(cliffs_delta > 0, "L>R", "R>L"))
write.csv(f8_df, file.path(OUT_STATS, "09_loso_sensitivity.csv"), row.names = FALSE)

fig8 <- ggplot(f8_df %>% filter(stratum == "IDD5_plus_IDM_balanced"),
               aes(dropped, cliffs_delta, fill = direction)) +
  geom_col() +
  geom_text(aes(label = ifelse(!is.na(p_mag_BH) & p_mag_BH < 0.05,
                               sprintf("q=%.2g", p_mag_BH), "")),
            vjust = -0.3, size = 2.4) +
  facet_wrap(~ target, ncol = 4) +
  scale_fill_manual(values = c(`L>R` = "#0072b2", `R>L` = "#c1272d")) +
  labs(
    title = "F8. Leave-one-monkey-out (IDD5+IDM stratum)",
    subtitle = "Cliff's d L vs R; __none__ = full data. nL/nR in stats/09_loso_sensitivity.csv",
    x = "Dropped sample", y = "Cliff's d (L vs R)"
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 8))
ggsave(file.path(OUT_FIGS, "F8_loso_sensitivity.png"), fig8,
       width = 12.0, height = 5.0, dpi = 220)

# ------------------------------------------------------------
# P12 - Mantel + SQ4-style FNT vs projection pairwise plot (hybrid Rmd)
# FNT `Score` in multi_monkey_INS_dist.txt is a distance (do not invert).
# Projection: Bray-Curtis on per-neuron p_combo proportion profile (vegdist rows).
# ------------------------------------------------------------
cat("[v2] Mantel replication aligned with SQ4\n")
# SQ4 convention: multi_monkey_INS_dist.txt 'Score' is already a distance
# (self-pairs near zero, cross-pairs ~1e9). Use directly, do NOT invert.
raw_dist <- read_tsv(DIST_FILE, show_col_types = FALSE)
fnt_lines <- readLines(JOINED_FNT)
nname_pat <- str_match(fnt_lines, "^\\d+\\s+Neuron\\s+(\\S+)\\s*$")
fnt_names <- nname_pat[, 2][!is.na(nname_pat[, 2])]
all_idx <- sort(unique(c(raw_dist$I, raw_dist$J)))
fnt_dist <- matrix(NA_real_, nrow = length(all_idx), ncol = length(all_idx),
                    dimnames = list(as.character(all_idx), as.character(all_idx)))
fnt_dist[cbind(as.character(raw_dist$I), as.character(raw_dist$J))] <- raw_dist$Score
fnt_dist[cbind(as.character(raw_dist$J), as.character(raw_dist$I))] <- raw_dist$Score
diag(fnt_dist) <- 0

if (length(fnt_names) >= nrow(fnt_dist)) {
  rownames(fnt_dist) <- colnames(fnt_dist) <- fnt_names[seq_len(nrow(fnt_dist))]
} else {
  m <- min(length(fnt_names), nrow(fnt_dist))
  fnt_dist <- fnt_dist[1:m, 1:m]
  rownames(fnt_dist) <- colnames(fnt_dist) <- fnt_names[1:m]
}

summ$NeuronFnt <- paste0(summ$SampleID, "_", gsub("\\.swc$", "", summ$NeuronID))
common_ids <- intersect(rownames(fnt_dist), summ$NeuronFnt[rowSums(m_hybrid, na.rm = TRUE) > 0])
common_ids <- common_ids[!is.na(rowSums(fnt_dist[common_ids, common_ids, drop = FALSE]))]
n_f9_neurons <- length(common_ids)
n_f9_pairs <- NA_integer_

mantel_rep <- tibble::tibble(
  endpoint   = "Mantel rho (FNT distance vs Bray-prop projection hybrid L3/L6)",
  rho = NA_real_, p = NA_real_, n = length(common_ids),
  spearman_pairs = NA_real_
)
if (length(common_ids) > 30) {
  fnt_to_uid <- setNames(summ$NeuronUID, summ$NeuronFnt)
  uid_sub <- fnt_to_uid[common_ids]
  p_sub <- p_combo[uid_sub, , drop = FALSE]
  rownames(p_sub) <- common_ids
  fnt_sub <- fnt_dist[common_ids, common_ids, drop = FALSE]
  fnt_sub[is.na(fnt_sub)] <- max(fnt_sub, na.rm = TRUE)
  n_f9_pairs <- sum(upper.tri(fnt_sub))
  proj_dist <- vegdist(p_sub, method = "bray")
  pd_mat_full <- as.matrix(proj_dist)
  mt <- mantel(as.dist(fnt_sub), proj_dist, method = "spearman", permutations = 999)
  mantel_rep$rho <- unname(mt$statistic)
  mantel_rep$p   <- unname(mt$signif)

  # FNT01 display matrix + rank mismatch (hybrid SQ5)
  ut_h <- upper.tri(fnt_sub)
  fnt01_h <- fnt_sub
  fnt_h_lo <- min(fnt_sub[ut_h], na.rm = TRUE)
  fnt_h_hi <- max(fnt_sub[ut_h], na.rm = TRUE)
  if (is.finite(fnt_h_lo) && is.finite(fnt_h_hi) && fnt_h_hi > fnt_h_lo) {
    fnt01_h[ut_h] <- (fnt_sub[ut_h] - fnt_h_lo) / (fnt_h_hi - fnt_h_lo)
    fnt01_h[lower.tri(fnt01_h)] <- t(fnt01_h)[lower.tri(fnt01_h)]
  } else {
    fnt01_h[ut_h] <- 0
    fnt01_h[lower.tri(fnt01_h)] <- t(fnt01_h)[lower.tri(fnt01_h)]
  }
  diag(fnt01_h) <- 0

  f_vals <- fnt01_h[ut_h]
  p_vals <- pd_mat_full[ut_h]
  f_rank01 <- (rank(f_vals, ties.method = "average") - 1) / (length(f_vals) - 1)
  p_rank01 <- (rank(p_vals, ties.method = "average") - 1) / (length(p_vals) - 1)
  diff_rank <- abs(f_rank01 - p_rank01)
  rank_mismatch <- matrix(
    NA_real_, nrow = nrow(fnt01_h), ncol = ncol(fnt01_h),
    dimnames = dimnames(fnt01_h)
  )
  rank_mismatch[ut_h] <- diff_rank
  rank_mismatch[lower.tri(rank_mismatch)] <- t(rank_mismatch)[lower.tri(rank_mismatch)]
  diag(rank_mismatch) <- NA

  combo <- (fnt01_h + pd_mat_full / max(pd_mat_full, na.rm = TRUE)) / 2
  ord <- stats::hclust(stats::as.dist(combo))$order

  fv <- f_vals
  pv <- p_vals
  sp_rho <- suppressWarnings(cor(fv, pv, method = "spearman", use = "complete.obs"))
  mantel_rep$spearman_pairs <- sp_rho
  p_txt <- if (is.na(mantel_rep$p)) "NA" else format.pval(mantel_rep$p, digits = 3)
  fig_fnt_proj <- ggplot(data.frame(FNT01 = fv, Proj = pv), aes(FNT01, Proj)) +
    geom_hex(bins = 50) +
    scale_fill_distiller(palette = "Blues", direction = 1) +
    labs(
      title = "F9. Pairwise FNT vs projection distance",
      subtitle = sprintf(
        "n=%d neurons; %d unique pairs (upper triangle). Spearman rho=%.3f; Mantel p=%s (999 perm).",
        n_f9_neurons, n_f9_pairs, sp_rho, p_txt
      ),
      x = "FNT distance (min–max scaled, display only)",
      y = "Bray–Curtis dissimilarity (hybrid profile)",
      caption = "Each point is one neuron pair; hex = 2D bin counts."
    ) +
    theme_minimal(base_size = 10) +
    theme(plot.caption = element_text(size = 7.5, hjust = 0))
  ggsave(
    file.path(OUT_FIGS, "F9_FNT_vs_projection_pairwise.png"),
    fig_fnt_proj, width = 6.5, height = 5.0, dpi = 220
  )

  # SQ5: same row/column order; FNT01 | Proj | |Δrank|
  cat("[v2] F9b SQ5 parallel FNT vs projection heatmaps\n")
  side_lab <- summ$Side[match(rownames(fnt_sub), summ$NeuronFnt)]
  ha <- rowAnnotation(
    Side = side_lab,
    col = list(Side = c(L = "#E74C3C", R = "#3498DB"))
  )
  rcols_sq5 <- function(x, utm) {
    colorRamp2(
      quantile(x[utm], c(0.05, 0.5, 0.95), na.rm = TRUE),
      c("white", "orange", "red4")
    )
  }
  rq <- quantile(rank_mismatch, c(0.05, 0.5, 0.95), na.rm = TRUE)
  f9b_col_title <- sprintf(
    "F9b | n=%d neurons | pairs=%d (upper tri) | order=hclust(combined dist)",
    n_f9_neurons, n_f9_pairs
  )
  ht <- Heatmap(
    fnt01_h, name = "FNT01", col = rcols_sq5(fnt01_h, ut_h),
    column_title = f9b_col_title,
    column_title_gp = grid::gpar(fontsize = 9),
    left_annotation = ha, cluster_rows = FALSE, cluster_columns = FALSE,
    row_order = ord, column_order = ord,
    show_row_names = FALSE, show_column_names = FALSE
  ) +
    Heatmap(
      pd_mat_full, name = "Proj", col = rcols_sq5(pd_mat_full, ut_h),
      left_annotation = ha, cluster_rows = FALSE, cluster_columns = FALSE,
      row_order = ord, column_order = ord,
      show_row_names = FALSE, show_column_names = FALSE
    ) +
    Heatmap(
      rank_mismatch, name = "|d_rank|",
      col = colorRamp2(rq, c("white", "tan", "brown4")),
      left_annotation = ha, cluster_rows = FALSE, cluster_columns = FALSE,
      row_order = ord, column_order = ord,
      show_row_names = FALSE, show_column_names = FALSE,
      column_title = "|rank(FNT01) - rank(Proj)|"
    )
  png(
    file.path(OUT_FIGS, "F9b_FNT_vs_projection_heatmaps_SQ5.png"),
    width = 14, height = 5, units = "in", res = 220
  )
  draw(ht)
  dev.off()
}
write.csv(mantel_rep, file.path(OUT_STATS, "08_mantel_replication.csv"), row.names = FALSE)
cat("[v2] mantel replication:\n"); print(mantel_rep)

# ------------------------------------------------------------
# P13–P16 Plan v2: combined spec, stat registry, README (flatmap script separate)
# ------------------------------------------------------------
cat("[v2] writing strata_and_metrics_spec + stat_registry + README\n")
ss <- read.csv(file.path(OUT_SPEC, "strata_spec.csv"), stringsAsFactors = FALSE)
ms <- read.csv(file.path(OUT_SPEC, "metric_spec.csv"), stringsAsFactors = FALSE)
strata_metrics_combined <- bind_rows(
  tibble::tibble(
    section = "stratum",
    name = ss$stratum_id,
    detail = paste(ss$filter_R, ss$role, paste0("min_n=", ss$min_n_inferential), sep = " | ")
  ),
  tibble::tibble(
    section = "metric",
    name = ms$metric,
    detail = ms$description
  )
)
write.csv(
  strata_metrics_combined,
  file.path(OUT_SPEC, "strata_and_metrics_spec.csv"),
  row.names = FALSE
)

stat_registry <- tibble::tribble(
  ~figure_file, ~primary_stats_csv, ~analysis_summary,
  "F1A_subregion_x_gou_panels_combined.png", "01_subregion_x_gou_panels.csv", "Hybrid domain prop; All_insula row",
  "F1B_subregion_x_gou_panels_LRcontrast.png", "01_subregion_x_gou_panels.csv", "Cliff d L vs R domains",
  "F2_subregion_x_key_targets.png", "02_subregion_x_key_targets.csv", "L+R key targets",
  "F2B_subregion_x_key_targets_LRcontrast.png", "02_subregion_x_key_targets.csv", "L vs R key targets",
  "F3A_intra_insula_combined.png", "03_intra_insula_meanprop.csv", "Intra-insula mean prop",
  "F3B_intra_insula_prevalence.png", "03_intra_insula_prevalence.csv", "Fraction projecting",
  "F3C_intra_insula_LRcontrast.png", "03_intra_insula_meanprop.csv", "Intra L vs R",
  "F4_interoceptive_gradient.png", "04_interoceptive_gradient_regressions.csv", "OLS slope soma AP x prop",
  "F5_per_neuron_ibias.png", "05_per_neuron_ibias.csv", "Length-based Ibias vs AP",
  "F6_projection_LI_bilateral_receiving.png", "06b_projection_laterality_index_all_targets.csv", "SQ3d LI bilateral targets",
  "F6_projection_LI_one_side_or_extreme.png", "06b_projection_laterality_index_all_targets.csv", "SQ3d LI QC class",
  "F6_projection_LI_violin_by_stratum_family.png", "06b_projection_laterality_index_all_targets.csv", "LI distribution",
  "F6_supplement_rank_based_receivers.png", "06_asymmetric_receivers_BH.csv", "Rank tests Cliff/Fisher",
  "F7_hierarchy_sensitivity.png", "07_hierarchy_sensitivity_per_target.csv", "Headline targets resolved L3/L6",
  "F8_loso_sensitivity.png", "09_loso_sensitivity.csv", "Leave-one-monkey-out",
  "F9_FNT_vs_projection_pairwise.png", "08_mantel_replication.csv", "Hex + Mantel rho/p",
  "F9b_FNT_vs_projection_heatmaps_SQ5.png", "08_mantel_replication.csv", "Parallel matrices + d_rank",
  "F10_interoceptive_AP_octile_heatmap.png", "10_ap_octile_interoceptive_profile.csv", "AP octile x targets",
  "flatmap_overlays/P13_flatmap_with_LR_context_strip.png", "00_context_per_subregion_n.csv", "Flatmap + L/R n strip (Python)"
)
write.csv(stat_registry, file.path(OUT_SPEC, "stat_registry.csv"), row.names = FALSE)

readme_txt <- c(
  "# combined_primary_v2 outputs",
  "",
  "Generated by: `group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd` (knit or `Rscript v2_combined_primary_pipeline.R`).",
  paste("Input workbook:", COMBINED_XLSX),
  "",
  "## Reproduce",
  "",
  "```bash",
  "Rscript group_analysis/R_analysis/v2_combined_primary_pipeline.R",
  "# or knit v2_combined_primary_pipeline.Rmd in RStudio",
  "```",
  "",
  "## Layout",
  "",
  "- `figures/FIGURE_CAPTIONS.md` — publication-style captions + stats pointers",
  "- `spec/strata_spec.csv` — inferential strata",
  "- `spec/metric_spec.csv` — metric definitions",
  "- `spec/strata_and_metrics_spec.csv` — merged long table",
  "- `spec/stat_registry.csv` — figure PNG → primary stats CSV",
  "- `stats/` — numeric outputs",
  "- `figures/` — PNG panels (F1–F10; F9b = SQ5 heatmaps)",
  "- `flatmap_overlays/` — `python group_analysis/scripts/13_flatmap_context_strip.py`",
  "- `stats/99_figure_QC_report.csv` — P15 registry alignment (runs with pipeline)",
  "- `tables/manuscript_quoted_numbers.xlsx` — figure registry + headline numbers (fill MS refs)",
  "",
  "## Figure index (short)",
  "",
  "| Fig | Stats |",
  "|-----|-------|",
  "| F1A/B | 01_subregion_x_gou_panels.csv (+ 01_gou_domain_column_map.csv) |",
  "| F2 / F2B | 02_subregion_x_key_targets.csv |",
  "| F3* | 03_intra_insula_*.csv |",
  "| F4 | 04_interoceptive_gradient_*.csv |",
  "| F5 | 05_per_neuron_ibias*.csv |",
  "| F6 LI | 06b_projection_laterality_index_*.csv |",
  "| F6 supplement | 06_asymmetric_receivers_BH.csv |",
  "| F7 | 07_hierarchy_sensitivity_per_target.csv |",
  "| F8 | 09_loso_sensitivity.csv |",
  "| F9 / F9b | 08_mantel_replication.csv |",
  "| F10 | 10_ap_octile_interoceptive_profile.csv |",
  "",
  "## QC notes (P15)",
  "",
  "- `99_figure_QC_report.csv` checks each registry figure and that its primary stats CSV is readable; `bh_summary` is filled only when expected BH columns exist.",
  "- F3C stars come from `03_intra_insula_full.csv` at build time; registry still points at `03_intra_insula_meanprop.csv` for the mean-prop layer.",
  "- `99_orphan_stats_files.csv` lists auxiliary CSVs not linked in `stat_registry.csv`.",
  ""
)
writeLines(readme_txt, file.path(OUT_ROOT, "README.md"))

# ------------------------------------------------------------
# P15 - Figure / stat alignment QC (registry-driven; no OCR)
# ------------------------------------------------------------
cat("[v2] P15 figure QC report\n")
resolve_fig_path <- function(rel) {
  if (startsWith(rel, "flatmap_overlays/")) file.path(OUT_ROOT, rel) else file.path(OUT_FIGS, rel)
}
reg_qc <- read.csv(file.path(OUT_SPEC, "stat_registry.csv"), stringsAsFactors = FALSE)
qc_list <- list()
for (i in seq_len(nrow(reg_qc))) {
  fig_rel <- reg_qc$figure_file[i]
  stat_fn <- reg_qc$primary_stats_csv[i]
  fp <- resolve_fig_path(fig_rel)
  sp <- file.path(OUT_STATS, stat_fn)
  fig_ok <- file.exists(fp) && file.info(fp)$size > 0
  stat_ok <- file.exists(sp)
  st <- if (stat_ok) utils::read.csv(sp, stringsAsFactors = FALSE, check.names = FALSE) else NULL
  nr <- if (!is.null(st)) nrow(st) else NA_integer_
  nc <- if (!is.null(st)) ncol(st) else NA_integer_
  bh_snippet <- ""
  if (!is.null(st)) {
    if (all(c("inferential_row", "p_magnitude_BH") %in% names(st))) {
      n_sig <- sum(st$inferential_row & !is.na(st$p_magnitude_BH) & st$p_magnitude_BH < 0.05, na.rm = TRUE)
      bh_snippet <- sprintf("F3-style inferential p_BH<.05: %d", n_sig)
    } else if ("p_BH" %in% names(st)) {
      bh_snippet <- sprintf("rows p_BH<.05: %d", sum(!is.na(st$p_BH) & st$p_BH < 0.05, na.rm = TRUE))
    } else if ("p_mag_BH" %in% names(st)) {
      bh_snippet <- sprintf("rows p_mag_BH<.05: %d", sum(!is.na(st$p_mag_BH) & st$p_mag_BH < 0.05, na.rm = TRUE))
    }
  }
  iss <- character(0)
  if (!fig_ok) iss <- c(iss, "missing_or_empty_figure")
  if (!stat_ok) iss <- c(iss, "missing_stats_csv")
  qc_list[[i]] <- tibble::tibble(
    figure_file = fig_rel,
    figure_ok = fig_ok,
    figure_bytes = if (fig_ok) file.info(fp)$size else NA_real_,
    stats_csv = stat_fn,
    stats_ok = stat_ok,
    stats_nrow = nr,
    stats_ncol = nc,
    bh_summary = bh_snippet,
    issues = if (!length(iss)) NA_character_ else paste(iss, collapse = ";")
  )
}
qc_df <- dplyr::bind_rows(qc_list)
write.csv(qc_df, file.path(OUT_STATS, "99_figure_QC_report.csv"), row.names = FALSE)

all_stats <- list.files(OUT_STATS, pattern = "\\.csv$", full.names = FALSE)
ref_stats <- unique(reg_qc$primary_stats_csv)
orphan_stats <- setdiff(all_stats, c(ref_stats, "99_figure_QC_report.csv"))
if (length(orphan_stats)) {
  write.csv(
    tibble::tibble(csv_file = sort(orphan_stats), note = "not referenced in stat_registry"),
    file.path(OUT_STATS, "99_orphan_stats_files.csv"),
    row.names = FALSE
  )
}

# ------------------------------------------------------------
# Plan v2 tables/manuscript_quoted_numbers.xlsx (template + headline pulls)
# ------------------------------------------------------------
cat("[v2] writing tables/manuscript_quoted_numbers.xlsx\n")
fig_reg_book <- reg_qc %>%
  mutate(manuscript_quote_ref = NA_character_, manuscript_section = NA_character_)
mantel_df <- utils::read.csv(file.path(OUT_STATS, "08_mantel_replication.csv"), stringsAsFactors = FALSE)
ctx_df <- utils::read.csv(file.path(OUT_STATS, "00_context_per_subregion_n.csv"), stringsAsFactors = FALSE)
ibias_n <- nrow(utils::read.csv(file.path(OUT_STATS, "05_per_neuron_ibias.csv"), stringsAsFactors = FALSE))
li_path <- file.path(OUT_STATS, "06b_projection_laterality_index_all_targets.csv")
li_bh <- if (file.exists(li_path)) utils::read.csv(li_path, stringsAsFactors = FALSE) else data.frame()
n_li_sig <- if (nrow(li_bh) && "significant" %in% names(li_bh)) {
  sum(li_bh$significant, na.rm = TRUE)
} else NA_integer_
headline <- tibble::tibble(
  quantity_label = c(
    "Mantel Spearman rho (FNT vs Bray projection)",
    "Mantel permutation p",
    "Neurons in Mantel (n)",
    "Spearman pairs (upper triangle, descriptive)",
    "Neurons in Ibias table (n)",
    "Projection LI table rows BH-significant (all strata)",
    "Subregions in context L/R table (n rows)",
    "Total L soma (sum Region L)",
    "Total R soma (sum Region R)"
  ),
  value = c(
    mantel_df$rho[1],
    mantel_df$p[1],
    mantel_df$n[1],
    mantel_df$spearman_pairs[1],
    ibias_n,
    n_li_sig,
    nrow(ctx_df),
    sum(ctx_df$L, na.rm = TRUE),
    sum(ctx_df$R, na.rm = TRUE)
  ),
  source_csv = c(
    rep("08_mantel_replication.csv", 4),
    "05_per_neuron_ibias.csv",
    "06b_projection_laterality_index_all_targets.csv",
    rep("00_context_per_subregion_n.csv", 3)
  ),
  manuscript_quote_ref = NA_character_
)
provenance <- tibble::tibble(
  item = c("pipeline_rmarkdown", "pipeline_launcher_r", "input_workbook", "qc_report", "plan_doc", "figure_captions"),
  path = c(
    "group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd",
    "group_analysis/R_analysis/v2_combined_primary_pipeline.R",
    as.character(COMBINED_XLSX),
    "stats/99_figure_QC_report.csv",
    "notes/whole_insula_lr_continuation_plan_v2.md",
    "outputs/combined_primary_v2/figures/FIGURE_CAPTIONS.md"
  )
)
writexl::write_xlsx(
  list(
    Figure_registry = fig_reg_book,
    Headline_numbers = headline,
    Provenance = provenance
  ),
  path = file.path(OUT_TABLES, "manuscript_quoted_numbers.xlsx")
)

# ------------------------------------------------------------
# Publication-oriented captions (single markdown next to PNGs)
# ------------------------------------------------------------
mantel_one <- if (exists("mantel_rep") && nrow(mantel_rep)) mantel_rep[1, ] else NULL
mantel_txt <- if (!is.null(mantel_one) && is.finite(as.numeric(mantel_one$rho))) {
  sprintf(
    paste0(
      "> **Mantel / F9 (this run).** Neurons in test *n* = %s; Mantel correlation ≈ %.3f; ",
      "permutation *p* = %s. Upper-triangle pair count (F9 hex / F9b heatmaps) = %s. ",
      "Full row: `stats/08_mantel_replication.csv`."
    ),
    mantel_one$n, as.numeric(mantel_one$rho),
    format.pval(as.numeric(mantel_one$p), digits = 3),
    if (exists("n_f9_pairs") && is.finite(n_f9_pairs)) as.character(n_f9_pairs) else "NA"
  )
} else {
  "> **Mantel / F9.** See `stats/08_mantel_replication.csv`."
}

caption_md <- c(
  "# Figure captions — whole-insula combined primary (v2)",
  "",
  "Pipeline: `group_analysis/R_analysis/v2_combined_primary_pipeline.Rmd` (source); `.R` launches render.",
  "Stats directory: `../stats/`; spec: `../spec/`.",
  "",
  sprintf(
    "**Cohort.** After harmonization, **N = %d** neurons in Summary with L/R soma and ipsilateral L3/L6 projection matrices.",
    nrow(summ)
  ),
  "",
  "**`p_combo` policy.** Extra-insula targets from L3; intra-insula from L6 (L3 insula fallback if no L6 column). Row-normalized proportions per neuron.",
  "",
  mantel_txt,
  "",
  "---",
  "",
  "## F1A — `F1A_subregion_x_gou_panels_combined.png`",
  "",
  "**What.** Mean summed target proportion (L+R pooled) assigned to each Gou *functional domain* (six columns), by insula source sub-region and **All_insula** row.",
  "**Stats.** `01_subregion_x_gou_panels.csv`, `01_gou_domain_column_map.csv` (column→domain mapping). Cliff's *d* (L vs R) with BH within row.",
  "",
  "## F1B — `F1B_subregion_x_gou_panels_LRcontrast.png`",
  "",
  "**What.** Same layout as F1A; inferential rows only; color = Cliff's *d* for domain mass L vs R.",
  "",
  "## F2 — `F2_subregion_x_key_targets.png`",
  "",
  "**What.** Mean proportion to curated targets; labels `region@L3` or `@L6` per hybrid policy.",
  "**Stats.** `02_subregion_x_key_targets.csv`.",
  "",
  "## F2B — `F2B_subregion_x_key_targets_LRcontrast.png`",
  "",
  "**What.** Cliff's *d* L vs R for inferential rows; BH within row.",
  "",
  "## F3A–C — intra-insula heatmaps",
  "",
  "**What.** L6 row-normalized proportion (A), prevalence (B), Cliff's *d* L vs R (C, inferential rows). Column order: Ward.D2 on prevalence.",
  "**Stats.** `03_intra_insula_meanprop.csv`, `03_intra_insula_prevalence.csv`, `03_intra_insula_full.csv`.",
  "",
  "## F4 — `F4_interoceptive_gradient.png`",
  "",
  "**What.** Per-hemisphere scatter: interoceptive target proportion vs soma NII-Y (AP); linear fit.",
  "**Stats.** `04_interoceptive_gradient_regressions.csv`, scatter points in `04_interoceptive_gradient_scatter_points.csv`.",
  "",
  "## F5 — `F5_per_neuron_ibias.png`",
  "",
  "**What.** Ibias = (contra − ipsi)/(contra + ipsi) **total axon length** (L6 length sheets) vs soma AP.",
  "**Stats.** `05_per_neuron_ibias.csv`.",
  "",
  "## F6 — projection laterality index",
  "",
  "**What.** Target-level **composition LI** = (mean_L − mean_R)/(mean_L + mean_R) on group-mean `p_combo` proportions; Wilcoxon L vs R per target, BH within stratum.",
  "**Panels.** `F6_projection_LI_bilateral_receiving.png` (both sides receive); `F6_projection_LI_one_side_or_extreme.png` (QC); `F6_projection_LI_violin_by_stratum_family.png` (distribution).",
  "**Stats.** `06b_projection_laterality_index_all_targets.csv` (`n_L`, `n_R` per stratum); BH-significant subset in `06b_projection_laterality_index_BH_significant.csv` when non-empty.",
  "",
  "## F6 supplement — `F6_supplement_rank_based_receivers.png`",
  "",
  "**What.** Top targets by Cliff's *d* / Fisher asymmetry (hybrid L3/L6); BH within stratum × layer × family.",
  "**Stats.** `06_asymmetric_receivers_BH.csv`.",
  "",
  "## F7 — `F7_hierarchy_sensitivity.png`",
  "",
  "**What.** Headline endpoints under resolved layer (e.g. Ig@L6, VMid as PAG proxy@L3); Cliff's *d*, BH within stratum for magnitude.",
  "**Stats.** `07_hierarchy_sensitivity_per_target.csv`.",
  "",
  "## F8 — `F8_loso_sensitivity.png`",
  "",
  "**What.** Leave-one-**SampleID**-out Cliff's *d* (IDD5+IDM stratum); `__none__` = full data.",
  "**Stats.** `09_loso_sensitivity.csv` includes *n*<sub>L</sub>, *n*<sub>R</sub> per drop.",
  "",
  "## F9 — `F9_FNT_vs_projection_pairwise.png`",
  "",
  "**What.** Hex bin of upper-triangle neuron pairs: display-scaled FNT distance vs Bray–Curtis on hybrid profile. Subtitle reports *n* neurons and pair count; Spearman on displayed pairs; Mantel on matrix correlation (999 perm).",
  "**Stats.** `08_mantel_replication.csv`.",
  "",
  "## F9b — `F9b_FNT_vs_projection_heatmaps_SQ5.png`",
  "",
  "**What.** Three aligned heatmaps (identical neuron order from hclust on combined distance): FNT01 (display scale), projection Bray, |Δrank| between the two. Left track = soma side. Column title repeats *n* and pair count.",
  "",
  "## F10 — `F10_interoceptive_AP_octile_heatmap.png`",
  "",
  "**What.** Mean L6 proportion to interoceptive targets by soma AP octile (NII-Y).",
  "**Stats.** `10_ap_octile_interoceptive_profile.csv`.",
  "",
  "## P13 flatmap — `../flatmap_overlays/P13_flatmap_with_LR_context_strip.png`",
  "",
  "**What.** Flatmap overlay with L/R *n* strip; Python script in repo. **Stats.** `00_context_per_subregion_n.csv`.",
  ""
)
writeLines(caption_md, file.path(OUT_FIGS, "FIGURE_CAPTIONS.md"))

cat("[v2] R pipeline done; outputs at", OUT_ROOT, "\n")
