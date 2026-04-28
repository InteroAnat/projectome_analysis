# ============================================================
# Combined-first (L+R pooled) insula analysis
# - Primary outputs are pooled across hemisphere
# - L/R asymmetry is retained as supplemental diagnostics
# - Includes:
#   (1) L3 vs L6 hierarchy sensitivity check
#   (2) Combined interoceptive-gradient / intra-insula heatmap
#   (3) Concise functional-domain overview
#   (4) Ranked asymmetric receiver-site summary
#   (5) Group Mantel replication (FNT vs projection)
# ============================================================

suppressPackageStartupMessages({
  library(readxl)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(vegan)
})

set.seed(42)

PROJECT_ROOT <- "d:/projectome_analysis"
GROUP_DIR <- file.path(PROJECT_ROOT, "group_analysis")
COMBINED_XLSX <- file.path(GROUP_DIR, "combined", "multi_monkey_INS_combined.xlsx")
GOU_MAP <- file.path(
  PROJECT_ROOT, "R_analysis", "scripts",
  "LR_analysis_hypothesis_v3", "tables", "region_to_gou_category_map.csv"
)
DIST_FILE <- file.path(GROUP_DIR, "fnt", "multi_monkey_INS_dist.txt")
JOINED_FNT <- file.path(GROUP_DIR, "fnt", "multi_monkey_INS_joined.fnt")
HUBS_DIR <- file.path(GROUP_DIR, "R_analysis", "outputs", "stats", "hubs_L6")
INTRA_PER_TARGET <- file.path(
  GROUP_DIR, "R_analysis", "outputs", "stats", "intra_insula", "intra_insula_per_target_LR.csv"
)

OUT_ROOT <- file.path(GROUP_DIR, "R_analysis", "outputs", "combined_primary")
OUT_STATS <- file.path(OUT_ROOT, "stats")
OUT_FIGS <- file.path(OUT_ROOT, "figures")
for (d in c(OUT_ROOT, OUT_STATS, OUT_FIGS)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

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

cat("[combined-primary] loading combined workbook\n")
summ <- read_excel(COMBINED_XLSX, sheet = "Summary")
ipsi_l3 <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_L3_ipsi")
ipsi_l6 <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_ipsi")

uids <- Reduce(intersect, list(summ$NeuronUID, ipsi_l3$NeuronUID, ipsi_l6$NeuronUID))
summ <- summ[match(uids, summ$NeuronUID), , drop = FALSE]
ipsi_l3 <- ipsi_l3[match(uids, ipsi_l3$NeuronUID), , drop = FALSE]
ipsi_l6 <- ipsi_l6[match(uids, ipsi_l6$NeuronUID), , drop = FALSE]

summ <- summ %>%
  mutate(
    Soma_Side_Final = ifelse(
      !is.na(Soma_Side_Inferred) & Soma_Side_Inferred %in% c("L", "R"),
      Soma_Side_Inferred,
      Soma_Side
    ),
    Soma_Region_Final = ifelse(
      !is.na(Soma_Region_Refined) & nzchar(Soma_Region_Refined),
      Soma_Region_Refined,
      gsub("^(CL|CR)_", "", Soma_Region_Auto)
    )
  )

summ <- summ %>% filter(Soma_Side_Final %in% c("L", "R"))
uids <- summ$NeuronUID

m_l3 <- build_mat(ipsi_l3, uids)
m_l6 <- build_mat(ipsi_l6, uids)
p_l3 <- normalize_rows(m_l3)
p_l6 <- normalize_rows(m_l6)

# ------------------------------------------------------------
# (1) L3 vs L6 hierarchy sensitivity check
# ------------------------------------------------------------
cat("[combined-primary] hierarchy sensitivity L3 vs L6\n")
gou_map <- read_csv(GOU_MAP, show_col_types = FALSE)
panels <- split(gou_map$user_region, gou_map$category_assigned)

panel_score <- function(prop_mat, target_set) {
  use <- intersect(target_set, colnames(prop_mat))
  if (!length(use)) return(rep(NA_real_, nrow(prop_mat)))
  rowSums(prop_mat[, use, drop = FALSE], na.rm = TRUE)
}

hier_rows <- lapply(names(panels), function(k) {
  s_l3 <- panel_score(p_l3, panels[[k]])
  s_l6 <- panel_score(p_l6, panels[[k]])
  n_t3 <- length(intersect(panels[[k]], colnames(p_l3)))
  n_t6 <- length(intersect(panels[[k]], colnames(p_l6)))
  cor_s <- suppressWarnings(cor(s_l3, s_l6, method = "spearman", use = "pairwise.complete.obs"))

  d <- data.frame(
    side = summ$Soma_Side_Final,
    l3 = s_l3,
    l6 = s_l6
  )
  p_l3_lr <- tryCatch(wilcox.test(l3 ~ side, data = d, exact = FALSE)$p.value, error = function(e) NA_real_)
  p_l6_lr <- tryCatch(wilcox.test(l6 ~ side, data = d, exact = FALSE)$p.value, error = function(e) NA_real_)

  data.frame(
    panel = k,
    n_targets_l3 = n_t3,
    n_targets_l6 = n_t6,
    mean_l3_combined = mean(s_l3, na.rm = TRUE),
    mean_l6_combined = mean(s_l6, na.rm = TRUE),
    spearman_l3_l6 = cor_s,
    p_lr_l3 = p_l3_lr,
    p_lr_l6 = p_l6_lr,
    stringsAsFactors = FALSE
  )
})

hier_df <- bind_rows(hier_rows) %>%
  mutate(
    p_lr_l3_BH = p.adjust(p_lr_l3, method = "BH"),
    p_lr_l6_BH = p.adjust(p_lr_l6, method = "BH")
  )
write.csv(hier_df, file.path(OUT_STATS, "hierarchy_sensitivity_L3_vs_L6.csv"), row.names = FALSE)

# ------------------------------------------------------------
# (2) Combined interoceptive-gradient heatmap (L+R pooled)
# ------------------------------------------------------------
cat("[combined-primary] pooled intra-insula heatmap\n")
insula_targets <- c("Ig", "Pi", "Ri", "Ia/Id", "Ial", "Iai", "Iapl", "Iam/Iapm", "lat_Ia")
ins_cols <- intersect(insula_targets, colnames(p_l6))

src_levels <- c("IDV", "IDM", "IDD5", "IA/ID", "IAPM", "IAL", "IG")
grad_df <- bind_rows(lapply(src_levels, function(src) {
  idx <- which(summ$Soma_Region_Final == src)
  if (length(idx) < 3 || !length(ins_cols)) return(NULL)
  sub <- p_l6[idx, ins_cols, drop = FALSE]
  data.frame(
    source_region = src,
    target_region = ins_cols,
    mean_prop = colMeans(sub, na.rm = TRUE),
    frac_projecting = colMeans(sub > 0, na.rm = TRUE),
    n_source = length(idx),
    stringsAsFactors = FALSE
  )
}))

if (nrow(grad_df)) {
  grad_df$source_region <- factor(grad_df$source_region, levels = src_levels)
  grad_df$target_region <- factor(grad_df$target_region, levels = ins_cols)
}

write.csv(grad_df, file.path(OUT_STATS, "combined_interoceptive_gradient_table.csv"), row.names = FALSE)

fig_grad <- ggplot(grad_df, aes(x = target_region, y = source_region, fill = mean_prop)) +
  geom_tile(color = "white") +
  geom_text(
    aes(label = ifelse(mean_prop >= 0.01, sprintf("%.2f", mean_prop), "")),
    size = 2.8
  ) +
  scale_fill_gradient(low = "white", high = "#2a6fbb", name = "mean\nprop") +
  labs(
    title = "Whole-insula interoceptive gradient (L+R pooled)",
    subtitle = "Pooled mean ipsi projection fraction from each source sub-region to insula targets",
    x = "Target insula sub-region",
    y = "Source soma sub-region"
  ) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))
ggsave(file.path(OUT_FIGS, "combined_intra_insula_gradient_heatmap.png"), fig_grad, width = 10.5, height = 5.8, dpi = 220)

# ------------------------------------------------------------
# (3) Combined functional-domain overview (L+R pooled)
# ------------------------------------------------------------
cat("[combined-primary] pooled functional-domain overview\n")
dom_df <- bind_rows(lapply(names(panels), function(k) {
  sc <- panel_score(p_l3, panels[[k]])
  data.frame(
    panel = k,
    mean_prop = mean(sc, na.rm = TRUE),
    sem_prop = sd(sc, na.rm = TRUE) / sqrt(sum(is.finite(sc))),
    n_targets = length(intersect(panels[[k]], colnames(p_l3))),
    stringsAsFactors = FALSE
  )
})) %>%
  mutate(
    panel_label = recode(
      panel,
      auto = "Autonomic",
      emo = "Emotional/Reward",
      cog = "Cognitive",
      motor = "Motor",
      sens = "Sensory",
      memory = "Memory"
    )
  )

write.csv(dom_df, file.path(OUT_STATS, "combined_functional_domain_overview.csv"), row.names = FALSE)

fig_dom <- ggplot(dom_df, aes(x = reorder(panel_label, -mean_prop), y = mean_prop)) +
  geom_col(fill = "#5f9ea0", width = 0.72) +
  geom_errorbar(aes(ymin = pmax(0, mean_prop - sem_prop), ymax = mean_prop + sem_prop), width = 0.15) +
  geom_text(aes(label = sprintf("nT=%d", n_targets)), vjust = -0.35, size = 3) +
  labs(
    title = "Whole-insula functional domain overview (L+R pooled)",
    subtitle = "Mean pooled projection proportion by Gou domain (L3 mapping)",
    x = "Functional domain",
    y = "Mean projection proportion"
  ) +
  theme_minimal(base_size = 10)
ggsave(file.path(OUT_FIGS, "combined_functional_domain_overview.png"), fig_dom, width = 8.2, height = 5.4, dpi = 220)

# ------------------------------------------------------------
# (4) Ranked asymmetric receiver-site summary
# ------------------------------------------------------------
cat("[combined-primary] ranked asymmetric receiver sites\n")
hub_files <- list.files(HUBS_DIR, pattern = "^hub_L6_.*_per_target\\.csv$", full.names = TRUE)
hub_all <- bind_rows(lapply(hub_files, read.csv, stringsAsFactors = FALSE))
intra_df <- if (file.exists(INTRA_PER_TARGET)) read.csv(INTRA_PER_TARGET, stringsAsFactors = FALSE) else data.frame()
recv_all <- bind_rows(
  mutate(hub_all, source_family = "hubs_L6"),
  mutate(intra_df, source_family = "intra_insula")
)

recv_rank <- recv_all %>%
  filter(stratum %in% c("all_combined", "IDD5_plus_IDM", "IDD5_251637")) %>%
  mutate(
    effect_abs = abs(mean_L - mean_R),
    best_q = pmin(
      ifelse(is.na(p_presence_BH), 1, p_presence_BH),
      ifelse(is.na(p_pres_BH), 1, p_pres_BH),
      ifelse(is.na(p_magnitude_BH), 1, p_magnitude_BH),
      ifelse(is.na(p_mag_BH), 1, p_mag_BH)
    )
  ) %>%
  group_by(target) %>%
  summarise(
    source_family = paste(unique(source_family), collapse = ";"),
    best_stratum = stratum[which.min(best_q)],
    best_q = min(best_q, na.rm = TRUE),
    max_effect_abs = max(effect_abs, na.rm = TRUE),
    direction_at_best = direction[which.min(best_q)],
    support_balanced = any(stratum %in% c("IDD5_plus_IDM", "IDD5_251637") & best_q < 0.05, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    confidence = ifelse(support_balanced, "high_confidence_balanced_support", "illustrative_only")
  ) %>%
  arrange(best_q, desc(max_effect_abs))

write.csv(recv_rank, file.path(OUT_STATS, "asymmetric_receiver_sites_ranked.csv"), row.names = FALSE)

top_recv <- recv_rank %>% slice_head(n = 15) %>%
  mutate(target = factor(target, levels = rev(target)))
fig_recv <- ggplot(top_recv, aes(x = target, y = max_effect_abs, fill = confidence)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(
    values = c(
      high_confidence_balanced_support = "#d95f02",
      illustrative_only = "#7570b3"
    )
  ) +
  labs(
    title = "Top asymmetric receiver sites (supplemental L/R scan)",
    subtitle = "Ranked by BH significance and effect size; balanced-stratum support highlighted",
    x = "Receiver target",
    y = "Max |mean_L - mean_R|"
  ) +
  theme_minimal(base_size = 10)
ggsave(file.path(OUT_FIGS, "asymmetric_receiver_sites_top.png"), fig_recv, width = 8.8, height = 6.2, dpi = 220)

# ------------------------------------------------------------
# (5) Group Mantel replication summary
# ------------------------------------------------------------
cat("[combined-primary] group Mantel replication\n")
raw_dist <- read_tsv(DIST_FILE, show_col_types = FALSE)
fnt_lines <- readLines(JOINED_FNT)
nname_pat <- str_match(fnt_lines, "^\\d+\\s+Neuron\\s+(\\S+)\\s*$")
fnt_names <- nname_pat[, 2][!is.na(nname_pat[, 2])]
all_idx <- sort(unique(c(raw_dist$I, raw_dist$J)))
fnt_mat <- matrix(0, nrow = length(all_idx), ncol = length(all_idx),
                  dimnames = list(as.character(all_idx), as.character(all_idx)))
fnt_mat[cbind(as.character(raw_dist$I), as.character(raw_dist$J))] <- raw_dist$Score
fnt_mat <- pmax(fnt_mat, t(fnt_mat))
diag(fnt_mat) <- 0
if (length(fnt_names) == nrow(fnt_mat)) {
  rownames(fnt_mat) <- colnames(fnt_mat) <- fnt_names
} else {
  m <- min(length(fnt_names), nrow(fnt_mat))
  fnt_mat <- fnt_mat[1:m, 1:m]
  rownames(fnt_mat) <- colnames(fnt_mat) <- fnt_names[1:m]
}

sp_corr <- cor(t(fnt_mat), method = "spearman", use = "pairwise.complete.obs")
sp_corr[is.na(sp_corr)] <- 0
fnt_dist <- 1 - sp_corr
fnt_dist[fnt_dist < 0] <- 0
diag(fnt_dist) <- 0

summ$NeuronFnt <- paste0(summ$SampleID, "_", gsub("\\.swc$", "", summ$NeuronID))
common_ids <- intersect(rownames(fnt_dist), summ$NeuronFnt[rowSums(m_l3, na.rm = TRUE) > 0])

mantel_rep <- data.frame(
  statistic = NA_real_,
  pval = NA_real_,
  n = length(common_ids),
  stringsAsFactors = FALSE
)
if (length(common_ids) > 30) {
  fnt_to_uid <- setNames(summ$NeuronUID, summ$NeuronFnt)
  p_l3_sub <- p_l3[fnt_to_uid[common_ids], , drop = FALSE]
  rownames(p_l3_sub) <- common_ids
  fnt_sub <- fnt_dist[common_ids, common_ids]
  proj_dist <- vegdist(p_l3_sub, method = "bray")
  mt <- mantel(as.dist(fnt_sub), proj_dist, method = "spearman", permutations = 999)
  mantel_rep$statistic <- unname(mt$statistic)
  mantel_rep$pval <- unname(mt$signif)
}
write.csv(mantel_rep, file.path(OUT_STATS, "mantel_replication_group.csv"), row.names = FALSE)

cat("[combined-primary] done\n")
cat(" outputs root:", OUT_ROOT, "\n")

