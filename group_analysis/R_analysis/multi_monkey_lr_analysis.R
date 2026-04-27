# ============================================================
# Phase 6 - Multi-monkey L/R analysis with SampleID stratification
#
# Inputs:
#   - group_analysis/combined/multi_monkey_INS_combined.xlsx
#   - group_analysis/fnt/multi_monkey_INS_dist.txt
#   - group_analysis/fnt/fnt_work/*.decimate.fnt  (for index ordering)
#   - data_output/gou_function_table/area_function_category_full.csv
#   - R_analysis/scripts/LR_analysis_hypothesis_v3/tables/
#                       region_to_gou_category_map.csv
#
# Outputs (under group_analysis/R_analysis/outputs/):
#   stats/  - L/R test tables per stratum (Gou 6 categories)
#   stats/permanova_combined.csv  - PERMANOVA with strata=SampleID
#   stats/mantel_combined.csv     - FNT vs projection Mantel
#   figures/panels_combined.png   - boxplots
#
# Built on top of the existing fnt_dist_clustering.r loader semantics
# (Spearman rank-transform; no penalty for now since types are mixed).
# ============================================================

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(readr)
  library(ggplot2); library(patchwork); library(vegan); library(stringr)
  library(writexl)
})
set.seed(42)

PROJECT_ROOT  <- "d:/projectome_analysis"
GROUP_DIR     <- file.path(PROJECT_ROOT, "group_analysis")
COMBINED_XLSX <- file.path(GROUP_DIR, "combined", "multi_monkey_INS_combined.xlsx")
DIST_FILE     <- file.path(GROUP_DIR, "fnt", "multi_monkey_INS_dist.txt")
FNT_FOLDER    <- file.path(GROUP_DIR, "fnt", "fnt_work")
GOU_MAP       <- file.path(PROJECT_ROOT, "R_analysis", "scripts",
                            "LR_analysis_hypothesis_v3", "tables",
                            "region_to_gou_category_map.csv")
GOU_FUNC_FULL <- file.path(PROJECT_ROOT, "R_analysis", "scripts",
                            "data_output", "gou_function_table",
                            "area_function_category_full.csv")

OUT_DIR  <- file.path(GROUP_DIR, "R_analysis", "outputs")
STATS_DIR <- file.path(OUT_DIR, "stats")
FIG_DIR   <- file.path(OUT_DIR, "figures")
TBL_DIR   <- file.path(OUT_DIR, "tables")
for (d in c(STATS_DIR, FIG_DIR, TBL_DIR))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 1. Load combined Summary + projection sheets
# ============================================================
cat("[6] Loading combined table\n")
summ <- read_excel(COMBINED_XLSX, sheet = "Summary")
ipsi_sheet <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_L3_ipsi")
contra_sheet <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_L3_contra")
cat(sprintf("  summary: %d rows  ipsi: %d x %d  contra: %d x %d\n",
            nrow(summ), nrow(ipsi_sheet), ncol(ipsi_sheet),
            nrow(contra_sheet), ncol(contra_sheet)))

# Keep only neurons that appear in both Summary and ipsi sheet
common <- intersect(summ$NeuronUID, ipsi_sheet$NeuronUID)
summ <- summ[match(common, summ$NeuronUID), , drop = FALSE]
ipsi_sheet <- ipsi_sheet[match(common, ipsi_sheet$NeuronUID), , drop = FALSE]
contra_sheet <- contra_sheet[match(common, contra_sheet$NeuronUID), , drop = FALSE]

# Build clean Soma_Side / Soma_Region cols
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

cat("  combined Soma_Side breakdown:\n")
print(table(summ$Soma_Side_Final, summ$SampleID, useNA = "ifany"))
cat("\n  combined Soma_Region breakdown:\n")
print(table(summ$Soma_Region_Final, summ$Soma_Side_Final, useNA = "ifany"))

# ============================================================
# 2. Build numeric projection matrices (neuron x target region)
# ============================================================
build_mat <- function(sheet, summ_uids) {
  meta_cols <- c("NeuronUID", "SampleID", "NeuronID", "Neuron_Type")
  num_cols <- setdiff(colnames(sheet), meta_cols)
  m <- as.matrix(sheet[, num_cols, drop = FALSE])
  storage.mode(m) <- "numeric"
  m[is.na(m)] <- 0
  rownames(m) <- sheet$NeuronUID
  m <- m[summ_uids, , drop = FALSE]
  m
}
ipsi_mat <- build_mat(ipsi_sheet, summ$NeuronUID)
contra_mat <- build_mat(contra_sheet, summ$NeuronUID)
cat(sprintf("\n  ipsi_mat: %d x %d   contra_mat: %d x %d\n",
            nrow(ipsi_mat), ncol(ipsi_mat),
            nrow(contra_mat), ncol(contra_mat)))

# Per-neuron proportion-normalized ipsi (for Gou-category scores)
ipsi_total <- rowSums(ipsi_mat, na.rm = TRUE)
ipsi_prop <- ipsi_mat
ok <- ipsi_total > 0
ipsi_prop[ok, ] <- sweep(ipsi_mat[ok, , drop = FALSE], 1, ipsi_total[ok], "/")

# ============================================================
# 3. Load Gou 6-category mapping
# ============================================================
gou_map <- read_csv(GOU_MAP, show_col_types = FALSE)
cat_lookup <- c(
  auto   = "Autonomic and Physiological Regulation",
  emo    = "Emotional, Social and Reward Functions",
  cog    = "Cognitive and Executive Function",
  motor  = "Motor Function",
  sens   = "Sensory Processing and Cross-Functional Integration",
  memory = "Learning and Memory"
)
panels <- split(gou_map$user_region, gou_map$category_assigned)

target_regions_in_data <- colnames(ipsi_prop)
panels_in_data <- lapply(panels, function(rs)
  intersect(rs, target_regions_in_data))

cat("\n  Panel coverage in combined data:\n")
for (k in names(panels_in_data))
  cat(sprintf("    %-7s: %d/%d targets\n",
              k, length(panels_in_data[[k]]), length(panels[[k]])))

panel_score <- function(prop_mat, target_set) {
  use <- intersect(target_set, colnames(prop_mat))
  if (!length(use)) return(rep(NA_real_, nrow(prop_mat)))
  rowSums(prop_mat[, use, drop = FALSE], na.rm = TRUE)
}
scores <- as.data.frame(summ)
for (k in names(panels_in_data)) {
  scores[[paste0("gou_", k)]] <- panel_score(ipsi_prop, panels_in_data[[k]])
}

# ============================================================
# 4. L/R tests by stratum (Gou 6 categories)
# ============================================================
test_lr <- function(df, score_col, label) {
  d <- df[!is.na(df[[score_col]]) & df$Soma_Side_Final %in% c("L", "R"), ]
  if (length(unique(d$Soma_Side_Final)) < 2) return(NULL)
  nL <- sum(d$Soma_Side_Final == "L"); nR <- sum(d$Soma_Side_Final == "R")
  if (nL < 3 || nR < 3) return(NULL)
  p <- tryCatch(
    suppressWarnings(wilcox.test(d[[score_col]] ~ d$Soma_Side_Final,
                                  exact = FALSE)$p.value),
    error = function(e) NA_real_)
  vL <- d[[score_col]][d$Soma_Side_Final == "L"]
  vR <- d[[score_col]][d$Soma_Side_Final == "R"]
  data.frame(
    stratum = label, panel = score_col,
    n_L = nL, n_R = nR,
    mean_L = mean(vL), mean_R = mean(vR),
    median_L = median(vL), median_R = median(vR),
    p_raw = p
  )
}

run_panels <- function(df, lbl, score_cols) {
  res <- bind_rows(lapply(score_cols, function(s) test_lr(df, s, lbl)))
  if (!nrow(res)) return(res)
  res$p_BH <- p.adjust(res$p_raw, method = "BH")
  res$direction <- ifelse(res$mean_L > res$mean_R, "L>R", "R>L")
  res$Category <- cat_lookup[sub("^gou_", "", res$panel)]
  res
}

gou_cols <- paste0("gou_", names(panels_in_data))

# Stratum filters
strata_def <- list(
  all_combined          = rep(TRUE, nrow(scores)),
  per_251637            = scores$SampleID == "251637",
  IDD5_251637           = scores$SampleID == "251637" & scores$Soma_Region_Final == "IDD5",
  IDM_251637            = scores$SampleID == "251637" & scores$Soma_Region_Final == "IDM",
  IDD5_plus_IDM_251637  = scores$SampleID == "251637" & scores$Soma_Region_Final %in% c("IDD5", "IDM"),
  IAL_combined          = scores$Soma_Region_Final == "IAL",
  IAL_251637_only       = scores$SampleID == "251637" & scores$Soma_Region_Final == "IAL",
  IAL_252384            = scores$SampleID == "252384" & scores$Soma_Region_Final == "IAL",
  IAL_252385            = scores$SampleID == "252385" & scores$Soma_Region_Final == "IAL",
  IG_combined           = scores$Soma_Region_Final == "IG"
)
results_list <- list()
for (k in names(strata_def)) {
  m <- strata_def[[k]]
  if (sum(m) < 6) {
    cat(sprintf("[skip stratum] %s: n=%d (<6)\n", k, sum(m)))
    next
  }
  r <- run_panels(scores[m, , drop = FALSE], k, gou_cols)
  if (nrow(r)) {
    cat(sprintf("[%s] n=%d  testing %d panels\n", k, sum(m), nrow(r)))
    results_list[[k]] <- r
  }
}
results <- bind_rows(results_list)
write.csv(results, file.path(STATS_DIR, "gou_categories_lr_tests_combined.csv"),
          row.names = FALSE)

cat("\n[combined Gou-category L/R results, BH-significant only]\n")
sig <- results %>% filter(p_BH < 0.05) %>%
  arrange(stratum, panel)
print(sig %>% dplyr::select(stratum, Category, n_L, n_R,
                              mean_L, mean_R, p_BH, direction),
      digits = 3, row.names = FALSE)

# ============================================================
# 5. PERMANOVA on combined ipsi profile (with strata=SampleID)
# ============================================================
keep_pmv <- summ$Soma_Side_Final %in% c("L", "R") & ipsi_total > 0
ipsi_prop_pmv <- ipsi_prop[keep_pmv, , drop = FALSE]
side_pmv <- droplevels(summ$Soma_Side_Final[keep_pmv])
sample_pmv <- factor(summ$SampleID[keep_pmv])
type_pmv <- factor(summ$Neuron_Type[keep_pmv])
region_pmv <- factor(summ$Soma_Region_Final[keep_pmv])

cat(sprintf("\n[PERMANOVA] n=%d (L=%d, R=%d) across %d samples\n",
            length(side_pmv), sum(side_pmv == "L"), sum(side_pmv == "R"),
            nlevels(sample_pmv)))

PERM_N <- 999L
permanova_rows <- list()
for (dist_method in c("bray", "jaccard")) {
  d <- vegdist(ipsi_prop_pmv,
               method = dist_method,
               binary = (dist_method == "jaccard"))
  # Side-only with SampleID strata
  a1 <- adonis2(d ~ side_pmv, permutations = PERM_N,
                strata = sample_pmv, by = "margin")
  permanova_rows[[length(permanova_rows) + 1L]] <- data.frame(
    distance = dist_method, model = "side | strata=SampleID",
    R2_side = a1$R2[1], pval = a1$`Pr(>F)`[1])
  # Adjusted: type + region + side
  a2 <- adonis2(d ~ type_pmv + region_pmv + side_pmv,
                permutations = PERM_N, strata = sample_pmv, by = "margin")
  side_row <- a2[rownames(a2) == "side_pmv", , drop = FALSE]
  permanova_rows[[length(permanova_rows) + 1L]] <- data.frame(
    distance = dist_method,
    model = "type + region + side | strata=SampleID",
    R2_side = side_row$R2[1], pval = side_row$`Pr(>F)`[1])
}
permanova_df <- bind_rows(permanova_rows)
write.csv(permanova_df, file.path(STATS_DIR, "permanova_combined.csv"),
          row.names = FALSE)
cat("[PERMANOVA results]\n")
print(permanova_df, digits = 4)

# ============================================================
# 6. Mantel test: FNT distance vs projection Bray-Curtis distance
# ============================================================
cat("\n[Mantel] Loading multi-monkey FNT distance\n")
raw_dist <- read_tsv(DIST_FILE, show_col_types = FALSE)
# Align indices to neuron names via the joined FNT
joined_fnt <- file.path(GROUP_DIR, "fnt", "multi_monkey_INS_joined.fnt")
fnt_lines <- readLines(joined_fnt)
nname_pat <- str_match(fnt_lines, "^\\d+\\s+Neuron\\s+(\\S+)\\s*$")
fnt_names <- nname_pat[, 2][!is.na(nname_pat[, 2])]
cat(sprintf("  joined FNT contains %d neurons (expect 306)\n", length(fnt_names)))

# Build symmetric square matrix of fnt scores indexed by neuron name
all_idx <- sort(unique(c(raw_dist$I, raw_dist$J)))
fnt_mat <- matrix(0, nrow = length(all_idx), ncol = length(all_idx),
                   dimnames = list(as.character(all_idx),
                                    as.character(all_idx)))
fnt_mat[cbind(as.character(raw_dist$I),
              as.character(raw_dist$J))] <- raw_dist$Score
fnt_mat <- pmax(fnt_mat, t(fnt_mat))
diag(fnt_mat) <- 0

# Map indices to names
if (length(fnt_names) == nrow(fnt_mat)) {
  rownames(fnt_mat) <- colnames(fnt_mat) <- fnt_names
} else {
  cat("  size mismatch; trimming to min\n")
  m <- min(length(fnt_names), nrow(fnt_mat))
  fnt_mat <- fnt_mat[1:m, 1:m]
  rownames(fnt_mat) <- colnames(fnt_mat) <- fnt_names[1:m]
}

# Spearman-rank-transform FNT scores -> distance (replicates user pipeline)
sp_corr <- cor(t(fnt_mat), method = "spearman", use = "pairwise.complete.obs")
sp_corr[is.na(sp_corr)] <- 0
fnt_dist <- 1 - sp_corr
fnt_dist[fnt_dist < 0] <- 0
diag(fnt_dist) <- 0

# Align FNT names ("251637_001") with Summary NeuronUID ("251637::001.swc")
# by normalizing both to "{SampleID}_{NeuronBase}" format.
summ$NeuronFnt <- paste0(summ$SampleID, "_",
                          gsub("\\.swc$", "", summ$NeuronID))
common_ids <- intersect(rownames(fnt_dist), summ$NeuronFnt[ipsi_total > 0])
cat(sprintf("  common ids for Mantel: %d\n", length(common_ids)))

if (length(common_ids) > 30) {
  # Map ipsi_prop (indexed by NeuronUID) -> indexed by NeuronFnt
  fnt_to_uid <- setNames(summ$NeuronUID, summ$NeuronFnt)
  ipsi_prop_for_mantel <- ipsi_prop[fnt_to_uid[common_ids], , drop = FALSE]
  rownames(ipsi_prop_for_mantel) <- common_ids
  fnt_sub <- fnt_dist[common_ids, common_ids]
  proj_dist <- vegdist(ipsi_prop_for_mantel, method = "bray")
  fnt_dobj <- as.dist(fnt_sub)
  m <- mantel(fnt_dobj, proj_dist, method = "spearman", permutations = 999)
  print(m)
  mantel_df <- data.frame(
    statistic = unname(m$statistic),
    pval = unname(m$signif),
    n = length(common_ids),
    stringsAsFactors = FALSE
  )
  write.csv(mantel_df, file.path(STATS_DIR, "mantel_combined.csv"),
            row.names = FALSE)

  # Also break down within-monkey vs cross-monkey if useful
  sample_lookup <- setNames(summ$SampleID, summ$NeuronFnt)
  s_i <- sample_lookup[common_ids]
  same_mat <- outer(s_i, s_i, "==")
  fnt_long <- as.numeric(fnt_sub[upper.tri(fnt_sub)])
  proj_long <- as.numeric(as.matrix(proj_dist)[upper.tri(fnt_sub)])
  same_long <- as.logical(same_mat[upper.tri(same_mat)])
  cor_within <- suppressWarnings(cor(fnt_long[same_long],
                                      proj_long[same_long],
                                      method = "spearman",
                                      use = "complete.obs"))
  cor_cross <- suppressWarnings(cor(fnt_long[!same_long],
                                     proj_long[!same_long],
                                     method = "spearman",
                                     use = "complete.obs"))
  cat(sprintf("  Spearman corr within-monkey pairs:  rho=%.3f (n=%d)\n",
              cor_within, sum(same_long)))
  cat(sprintf("  Spearman corr cross-monkey pairs:   rho=%.3f (n=%d)\n",
              cor_cross, sum(!same_long)))
}

# ============================================================
# 7. Plots
# ============================================================
side_cols <- c(L = "#E74C3C", R = "#3498DB")

plot_panel <- function(df, score_col, lbl) {
  d <- df[!is.na(df[[score_col]]) & df$Soma_Side_Final %in% c("L", "R"), ]
  if (!nrow(d) || length(unique(d$Soma_Side_Final)) < 2) return(NULL)
  p <- tryCatch(
    suppressWarnings(wilcox.test(d[[score_col]] ~ d$Soma_Side_Final,
                                  exact = FALSE)$p.value),
    error = function(e) NA_real_)
  ggplot(d, aes(x = Soma_Side_Final, y = .data[[score_col]],
                fill = Soma_Side_Final)) +
    geom_violin(alpha = 0.4, trim = FALSE) +
    geom_boxplot(width = 0.18, alpha = 0.85, outlier.size = 0.6) +
    scale_fill_manual(values = side_cols) +
    labs(title = sub("^gou_", "", score_col),
         subtitle = sprintf("%s | p=%.3g | n_L=%d n_R=%d",
                            lbl, p, sum(d$Soma_Side_Final == "L"),
                            sum(d$Soma_Side_Final == "R")),
         x = "Soma side", y = "Proportion") +
    theme_minimal(base_size = 10) +
    theme(legend.position = "none")
}

plot_grid <- function(df, lbl) {
  ps <- lapply(gou_cols, function(s) plot_panel(df, s, lbl))
  ps <- ps[!sapply(ps, is.null)]
  if (!length(ps)) return(NULL)
  patchwork::wrap_plots(ps, ncol = 3)
}

fig_all <- plot_grid(scores, "all combined")
if (!is.null(fig_all))
  ggsave(file.path(FIG_DIR, "panels_all_combined.png"),
         fig_all, width = 14, height = 8, dpi = 200)

fig_idd5_idm <- plot_grid(scores[scores$SampleID == "251637" &
                                  scores$Soma_Region_Final %in% c("IDD5", "IDM"), ],
                          "251637 IDD5+IDM only")
if (!is.null(fig_idd5_idm))
  ggsave(file.path(FIG_DIR, "panels_IDD5_IDM_251637.png"),
         fig_idd5_idm, width = 14, height = 8, dpi = 200)

fig_ial_combined <- plot_grid(scores[scores$Soma_Region_Final == "IAL", ],
                              "IAL combined (251637 + 252384 + 252385)")
if (!is.null(fig_ial_combined))
  ggsave(file.path(FIG_DIR, "panels_IAL_combined.png"),
         fig_ial_combined, width = 14, height = 8, dpi = 200)

# Save scores for downstream use
write.csv(scores, file.path(TBL_DIR, "per_neuron_scores_combined.csv"),
          row.names = FALSE)

cat("\n========== Phase 6 DONE ==========\n")
cat("Outputs in:", OUT_DIR, "\n")
