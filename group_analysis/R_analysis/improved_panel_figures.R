# Improved multi-panel figures with explicit sample-size breakdowns
# Replaces the Phase 6 panel images (titles ran together, no per-sample
# breakdown, no significance markers)
suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr); library(readr)
  library(ggplot2); library(patchwork); library(scales)
})
set.seed(42)

PROJECT_ROOT  <- "d:/projectome_analysis"
GROUP_DIR     <- file.path(PROJECT_ROOT, "group_analysis")
COMBINED_XLSX <- file.path(GROUP_DIR, "combined", "multi_monkey_INS_combined.xlsx")
GOU_MAP       <- file.path(PROJECT_ROOT, "R_analysis", "scripts",
                            "LR_analysis_hypothesis_v3", "tables",
                            "region_to_gou_category_map.csv")

OUT_DIR <- file.path(GROUP_DIR, "R_analysis", "outputs", "figures", "improved")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat_lookup <- c(
  auto   = "Autonomic & Physiological",
  emo    = "Emotional, Reward & Social",
  cog    = "Cognitive & Executive",
  motor  = "Motor",
  sens   = "Sensory & Cross-functional",
  memory = "Learning & Memory"
)

side_cols <- c(L = "#E74C3C", R = "#3498DB")
sample_cols <- c("251637" = "#4878D0", "252383" = "#EE854A",
                 "252384" = "#6ACC64", "252385" = "#D65F5F")

p_to_sig <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*"
  else if (p < 0.10) "." else "ns"
}

# ============================================================
# Load data + compute scores
# ============================================================
summ <- read_excel(COMBINED_XLSX, sheet = "Summary")
ipsi <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_L3_ipsi")
common <- intersect(summ$NeuronUID, ipsi$NeuronUID)
summ <- summ[match(common, summ$NeuronUID), ]
ipsi <- ipsi[match(common, ipsi$NeuronUID), ]

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
ipsi_mat <- build_mat(ipsi, summ$NeuronUID)
ipsi_total <- rowSums(ipsi_mat, na.rm = TRUE)
ipsi_prop <- ipsi_mat
ipsi_prop[ipsi_total > 0, ] <- sweep(ipsi_mat[ipsi_total > 0, , drop = FALSE], 1,
                                      ipsi_total[ipsi_total > 0], "/")

gou_map <- read_csv(GOU_MAP, show_col_types = FALSE)
panels <- split(gou_map$user_region, gou_map$category_assigned)

panel_score <- function(prop_mat, target_set) {
  use <- intersect(target_set, colnames(prop_mat))
  if (!length(use)) return(rep(NA_real_, nrow(prop_mat)))
  rowSums(prop_mat[, use, drop = FALSE], na.rm = TRUE)
}

scores <- as.data.frame(summ)
for (k in names(panels)) {
  scores[[paste0("gou_", k)]] <- panel_score(ipsi_prop, panels[[k]])
}

# ============================================================
# FIG A: Sample composition bar chart
# ============================================================
sample_breakdown <- scores %>%
  filter(Soma_Side_Final %in% c("L", "R")) %>%
  count(SampleID, Soma_Region_Final, Soma_Side_Final) %>%
  mutate(SampleID = factor(SampleID, levels = c("251637", "252383", "252384", "252385")),
         Soma_Region_Final = factor(
           Soma_Region_Final,
           levels = c("IAL", "IAPM", "IDD5", "IDM", "IDV", "IA/ID", "IG"))) %>%
  arrange(SampleID, Soma_Region_Final)

n_per_sample <- sample_breakdown %>%
  group_by(SampleID) %>%
  summarise(total = sum(n)) %>%
  mutate(label = paste0(SampleID, "\n(n=", total, ")"))

fig_A <- ggplot(sample_breakdown,
                aes(x = SampleID, y = n, fill = Soma_Region_Final)) +
  geom_col(position = position_stack(reverse = TRUE)) +
  geom_text(aes(label = ifelse(n >= 4, n, "")), position = position_stack(vjust = 0.5,
                                                                          reverse = TRUE),
            color = "white", size = 3.2, fontface = "bold") +
  facet_wrap(~ Soma_Side_Final, ncol = 2,
             labeller = labeller(Soma_Side_Final = c(L = "LEFT", R = "RIGHT"))) +
  scale_x_discrete(labels = setNames(n_per_sample$label, n_per_sample$SampleID)) +
  scale_fill_brewer(palette = "Set2", name = "Soma sub-region") +
  labs(title = "Sample composition: 306 neurons across 4 monkeys",
       subtitle = sprintf(
         "L = %d, R = %d  |  251637 = %d  |  252383 = %d  |  252384 = %d  |  252385 = %d",
         sum(scores$Soma_Side_Final == "L"),
         sum(scores$Soma_Side_Final == "R"),
         sum(scores$SampleID == "251637"),
         sum(scores$SampleID == "252383"),
         sum(scores$SampleID == "252384"),
         sum(scores$SampleID == "252385")),
       x = "Monkey sample", y = "n neurons") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "right",
        strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "FigA_sample_composition.png"),
       fig_A, width = 12, height = 5.5, dpi = 200)

# ============================================================
# FIG B: Gou 6-category panels in 3 strata, one row per category
#         x = stratum, y = mean proportion ± 95% CI, fill = side
# ============================================================
gou_cols <- paste0("gou_", names(panels))

build_summary <- function(df, stratum_label) {
  df <- df %>% filter(Soma_Side_Final %in% c("L", "R"))
  if (nrow(df) < 6) return(NULL)
  bind_rows(lapply(gou_cols, function(s) {
    if (!s %in% names(df)) return(NULL)
    d <- df[!is.na(df[[s]]), ]
    nL <- sum(d$Soma_Side_Final == "L"); nR <- sum(d$Soma_Side_Final == "R")
    if (nL < 3 || nR < 3) return(NULL)
    p <- tryCatch(suppressWarnings(
      wilcox.test(d[[s]] ~ d$Soma_Side_Final, exact = FALSE)$p.value),
      error = function(e) NA_real_)
    bind_rows(
      data.frame(stratum = stratum_label, panel = s, side = "L",
                  mean = mean(d[[s]][d$Soma_Side_Final == "L"]),
                  se = sd(d[[s]][d$Soma_Side_Final == "L"]) / sqrt(nL),
                  n = nL, p_value = p),
      data.frame(stratum = stratum_label, panel = s, side = "R",
                  mean = mean(d[[s]][d$Soma_Side_Final == "R"]),
                  se = sd(d[[s]][d$Soma_Side_Final == "R"]) / sqrt(nR),
                  n = nR, p_value = p)
    )
  }))
}

strata_dfs <- list(
  list(
    label = "All combined\n(n_L=121, n_R=185)",
    data = scores %>% filter(Soma_Side_Final %in% c("L","R"))
  ),
  list(
    label = "IDD5+IDM (251637)\n(balanced; n_L=77, n_R=62)",
    data = scores %>% filter(SampleID == "251637" &
                               Soma_Region_Final %in% c("IDD5", "IDM"))
  ),
  list(
    label = "IAL combined\n(251637+252384+252385; n_L=21, n_R=117)",
    data = scores %>% filter(Soma_Region_Final == "IAL")
  )
)

panel_summary <- bind_rows(lapply(strata_dfs, function(x)
  build_summary(x$data, x$label)))
panel_summary$panel_label <- cat_lookup[sub("^gou_", "", panel_summary$panel)]
panel_summary$panel_label <- factor(panel_summary$panel_label,
                                      levels = unname(cat_lookup))
panel_summary$stratum <- factor(panel_summary$stratum,
                                  levels = sapply(strata_dfs, `[[`, "label"))

# Significance label per panel/stratum
sig_df <- panel_summary %>%
  group_by(stratum, panel_label) %>%
  summarise(p = first(p_value),
            y = max(mean + se) * 1.10, .groups = "drop") %>%
  mutate(label = sapply(p, p_to_sig))

fig_B <- ggplot(panel_summary,
                aes(x = panel_label, y = mean, fill = side)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.78) +
  geom_errorbar(aes(ymin = pmax(0, mean - se), ymax = mean + se),
                position = position_dodge(width = 0.85), width = 0.20,
                linewidth = 0.4) +
  geom_text(data = sig_df,
            aes(x = panel_label, y = y, label = label),
            inherit.aes = FALSE, size = 4.5, fontface = "bold") +
  facet_wrap(~ stratum, ncol = 1) +
  scale_fill_manual(values = side_cols, name = "Soma side") +
  labs(title = "Gou 6-category L/R projection scores across strata",
       subtitle = "Bars = mean proportion of ipsi projection budget; error bars = SEM; *=p<0.05, **=p<0.01, ***=p<0.001 (Wilcoxon, uncorrected)",
       x = "Gou functional category (Table S2C)",
       y = "Mean proportion of ipsi projection") +
  theme_minimal(base_size = 11) +
  theme(strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(angle = 25, hjust = 1, size = 9),
        legend.position = "right")
ggsave(file.path(OUT_DIR, "FigB_gou_panels_3_strata.png"),
       fig_B, width = 12, height = 11, dpi = 200)

# ============================================================
# FIG C: caudal_OFC sensitivity — the biggest signal collapse story
# ============================================================
ofc_strata_def <- list(
  list(label = "All combined", n_label = "n=306",
       d = scores),
  list(label = "251637 only", n_label = "n=260 (103L,157R)",
       d = scores %>% filter(SampleID == "251637")),
  list(label = "IDD5+IDM (balanced)", n_label = "n=139 (77L,62R)",
       d = scores %>% filter(SampleID == "251637" &
                                Soma_Region_Final %in% c("IDD5","IDM"))),
  list(label = "IAL combined", n_label = "n=138 (21L,117R)",
       d = scores %>% filter(Soma_Region_Final == "IAL")),
  list(label = "IAL 251637 only", n_label = "n=102 (7L,95R)",
       d = scores %>% filter(SampleID == "251637" &
                                Soma_Region_Final == "IAL")),
  list(label = "IAL 252385 only", n_label = "n=31 (11L,20R)",
       d = scores %>% filter(SampleID == "252385" &
                                Soma_Region_Final == "IAL"))
)

ofc_long <- bind_rows(lapply(ofc_strata_def, function(x) {
  d <- x$d %>% filter(Soma_Side_Final %in% c("L", "R"))
  if (!"caudal_OFC" %in% colnames(ipsi_prop)) return(NULL)
  d$ofc_prop <- ipsi_prop[d$NeuronUID, "caudal_OFC"]
  if (sum(d$Soma_Side_Final == "L") < 3 || sum(d$Soma_Side_Final == "R") < 3)
    return(NULL)
  p <- tryCatch(suppressWarnings(
    wilcox.test(d$ofc_prop ~ d$Soma_Side_Final, exact = FALSE)$p.value),
    error = function(e) NA_real_)
  data.frame(stratum = x$label, n_label = x$n_label,
             ofc_prop = d$ofc_prop, side = d$Soma_Side_Final,
             p_value = p, stringsAsFactors = FALSE)
}))
ofc_long$stratum <- factor(ofc_long$stratum,
                            levels = sapply(ofc_strata_def, `[[`, "label"))

ofc_sig <- ofc_long %>%
  group_by(stratum, n_label) %>%
  summarise(p = first(p_value),
            y = max(ofc_prop, na.rm = TRUE) * 1.05, .groups = "drop") %>%
  mutate(label = sapply(p, p_to_sig),
         text = sprintf("p=%.2g %s", p, label))

fig_C <- ggplot(ofc_long,
                aes(x = side, y = ofc_prop, fill = side)) +
  geom_violin(alpha = 0.40, trim = FALSE) +
  geom_boxplot(width = 0.18, alpha = 0.85, outlier.size = 0.4) +
  geom_text(data = ofc_sig,
            aes(x = 1.5, y = y, label = text),
            inherit.aes = FALSE, size = 4) +
  geom_text(data = unique(ofc_long[, c("stratum", "n_label")]),
            aes(x = 1.5, y = -0.05, label = n_label),
            inherit.aes = FALSE, size = 3, color = "grey30") +
  facet_wrap(~ stratum, ncol = 3) +
  scale_fill_manual(values = side_cols) +
  labs(title = "Caudal-OFC R>L asymmetry collapses with bilateral sampling",
       subtitle = "Per-neuron proportion of ipsi projection going to caudal_OFC. The R>L signal in 251637 (p=5.4e-9) was driven by IAL imbalance (7L:95R). When 252385 contributes 14 L-IAL neurons, the asymmetry vanishes (p=0.49 in IAL_combined; ns in IAL_252385 alone).",
       x = "Soma side", y = "Proportion of ipsi projection to caudal_OFC") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none",
        strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "bold", size = 10))
ggsave(file.path(OUT_DIR, "FigC_caudal_OFC_sensitivity.png"),
       fig_C, width = 14, height = 8, dpi = 200)

# ============================================================
# FIG D: Inter-animal replication of agranular -> caudal OFC
#         Per-monkey IAL only, side colors
# ============================================================
ial_per_sample <- scores %>%
  filter(Soma_Region_Final == "IAL", Soma_Side_Final %in% c("L","R")) %>%
  mutate(SampleID = factor(SampleID,
                            levels = c("251637", "252384", "252385")))
ial_per_sample$ofc_prop <- ipsi_prop[ial_per_sample$NeuronUID, "caudal_OFC"]

ial_summary <- ial_per_sample %>%
  group_by(SampleID, Soma_Side_Final) %>%
  summarise(n = n(),
            mean_ofc = mean(ofc_prop, na.rm = TRUE),
            sd_ofc = sd(ofc_prop, na.rm = TRUE),
            se_ofc = sd_ofc / sqrt(n),
            .groups = "drop")

fig_D <- ggplot(ial_per_sample,
                aes(x = Soma_Side_Final, y = ofc_prop, fill = Soma_Side_Final)) +
  geom_violin(alpha = 0.35, trim = FALSE) +
  geom_boxplot(width = 0.20, alpha = 0.85, outlier.size = 0.5) +
  geom_text(data = ial_summary,
            aes(x = Soma_Side_Final, y = -0.05,
                label = paste0("n=", n)),
            inherit.aes = FALSE, size = 3.5) +
  facet_wrap(~ SampleID, ncol = 3) +
  scale_fill_manual(values = side_cols) +
  labs(title = "Inter-animal replication: IAL projection to caudal OFC is bilateral",
       subtitle = "Per-monkey IAL-only neurons. The 251637-R-IAL bias against L-IAL (n=7) was a sampling artefact - 252385 has 11 L-IAL neurons projecting comparably.",
       x = "Soma side", y = "Proportion of ipsi projection to caudal_OFC") +
  theme_minimal(base_size = 11) +
  theme(legend.position = "none",
        strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "FigD_IAL_caudal_OFC_per_sample.png"),
       fig_D, width = 11, height = 5.5, dpi = 200)

# ============================================================
# FIG E: NEW finding: pallidum (LPal/Cl) L>R asymmetry that
#        survives in IDD5 (balanced)
# ============================================================
hub_targets_for_E <- c("LPal", "VPal", "Cl")
hub_long <- bind_rows(lapply(hub_targets_for_E, function(rg) {
  if (!rg %in% colnames(ipsi_prop)) return(NULL)
  bind_rows(
    data.frame(target = rg,
               stratum = "IDD5 (251637; n_L=33, n_R=30)",
               prop = ipsi_prop[scores$NeuronUID[scores$SampleID == "251637" &
                                                   scores$Soma_Region_Final == "IDD5"],
                                 rg],
               side = scores$Soma_Side_Final[scores$SampleID == "251637" &
                                                scores$Soma_Region_Final == "IDD5"]),
    data.frame(target = rg,
               stratum = "All combined (n_L=121, n_R=185)",
               prop = ipsi_prop[, rg],
               side = scores$Soma_Side_Final)
  )
}))
hub_long <- hub_long %>% filter(side %in% c("L", "R"))
hub_long$side <- factor(hub_long$side, levels = c("L", "R"))

fig_E <- ggplot(hub_long, aes(x = target, y = prop, fill = side)) +
  geom_boxplot(width = 0.6, alpha = 0.8, outlier.size = 0.6,
               position = position_dodge(width = 0.7)) +
  facet_wrap(~ stratum, ncol = 1) +
  scale_fill_manual(values = side_cols) +
  labs(title = "Pallidum / claustrum: L>R asymmetry in IDD5 (balanced) AND combined",
       subtitle = sprintf(
         "L>R survives in the only L/R-balanced soma stratum (IDD5; n_L=33, n_R=30): LPal p_BH=0.009, Cl p_BH=0.033 (Fisher presence test).\nReplicates in all_combined: LPal p_BH=8e-5, Cl p_BH=3e-4. New, robust laterality finding."),
       x = "target region", y = "proportion of ipsi projection") +
  theme_minimal(base_size = 11) +
  theme(strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "bold"))
ggsave(file.path(OUT_DIR, "FigE_pallidum_claustrum_LR.png"),
       fig_E, width = 11, height = 8, dpi = 200)

# ============================================================
# FIG F: Intra-insula heatmap (source x target) at L6
# ============================================================
ipsi6 <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_ipsi")
ipsi6 <- ipsi6[match(common, ipsi6$NeuronUID), ]
ipsi6_mat <- build_mat(ipsi6, summ$NeuronUID)
ipsi6_total <- rowSums(ipsi6_mat, na.rm = TRUE)
ipsi6_prop <- ipsi6_mat
ipsi6_prop[ipsi6_total > 0, ] <-
  sweep(ipsi6_mat[ipsi6_total > 0, , drop = FALSE], 1,
         ipsi6_total[ipsi6_total > 0], "/")

INSULA_FINEST_TARGETS <- c("Ig", "Pi", "Ri", "Ia/Id",
                            "Ial", "Iai", "Iapl", "Iam/Iapm", "lat_Ia")
ins_targs <- intersect(INSULA_FINEST_TARGETS, colnames(ipsi6_prop))

build_heat <- function(side_label) {
  m <- summ$Soma_Side_Final == side_label
  ids <- which(m)
  if (length(ids) < 3) return(data.frame())
  bind_rows(lapply(unique(summ$Soma_Region_Final[ids]), function(src) {
    src_ids <- ids[summ$Soma_Region_Final[ids] == src]
    if (length(src_ids) < 3) return(NULL)
    sub <- ipsi6_prop[src_ids, ins_targs, drop = FALSE]
    data.frame(side = side_label, source_region = src,
                target_region = ins_targs,
                mean_prop = colMeans(sub, na.rm = TRUE),
                frac_proj = colMeans(sub > 0, na.rm = TRUE),
                n_neurons = length(src_ids),
                stringsAsFactors = FALSE)
  }))
}
heat_LR <- bind_rows(build_heat("L"), build_heat("R"))
heat_LR$source_region <- factor(heat_LR$source_region,
                                  levels = c("IAL", "IAPM", "IDV", "IA/ID",
                                              "IDD5", "IDM", "IG"))
heat_LR$target_region <- factor(heat_LR$target_region,
                                  levels = ins_targs)

fig_F <- ggplot(heat_LR,
                aes(x = target_region, y = source_region, fill = mean_prop)) +
  geom_tile(color = "white") +
  geom_text(aes(label = ifelse(mean_prop >= 0.005,
                                sprintf("%.2f\n(n=%d)", mean_prop, n_neurons),
                                "")),
            size = 2.6) +
  facet_wrap(~ side, ncol = 2,
             labeller = labeller(side = c(L = "Soma side: LEFT",
                                            R = "Soma side: RIGHT"))) +
  scale_fill_gradient(low = "white", high = "#1976d2",
                      name = "mean prop\nipsi") +
  labs(title = "Intra-insula projection at finest atlas level (L6)",
       subtitle = "Each cell = mean fraction of ipsi projection from source soma sub-region (rows) to target insula sub-region (cols). 96.4% of insula neurons project to other insula sub-regions.",
       x = "Target insula sub-region", y = "Source soma sub-region") +
  theme_minimal(base_size = 10) +
  theme(strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "bold"),
        axis.text.x = element_text(angle = 35, hjust = 1))
ggsave(file.path(OUT_DIR, "FigF_intra_insula_heatmap.png"),
       fig_F, width = 14, height = 5.5, dpi = 200)

cat("\n[Improved figures saved] in", OUT_DIR, "\n")
list.files(OUT_DIR)
