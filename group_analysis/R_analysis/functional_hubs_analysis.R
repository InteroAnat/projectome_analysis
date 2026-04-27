# ============================================================
# Functional hub-region L/R analysis: thalamus + brainstem
# Per-target hurdle decomposition (presence + magnitude)
# Stratified across sub-region + sample
# ============================================================
suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(tidyr)
  library(ggplot2); library(patchwork); library(writexl); library(readr)
})
set.seed(42)

PROJECT_ROOT  <- "d:/projectome_analysis"
GROUP_DIR     <- file.path(PROJECT_ROOT, "group_analysis")
COMBINED_XLSX <- file.path(GROUP_DIR, "combined", "multi_monkey_INS_combined.xlsx")

OUT_DIR  <- file.path(GROUP_DIR, "R_analysis", "outputs")
STATS_DIR <- file.path(OUT_DIR, "stats", "hubs")
FIG_DIR   <- file.path(OUT_DIR, "figures", "hubs")
TBL_DIR   <- file.path(OUT_DIR, "tables", "hubs")
for (d in c(STATS_DIR, FIG_DIR, TBL_DIR))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Hub definitions (anatomical literature)
# ============================================================
# Thalamus: relay nuclei known to integrate insular interoceptive afferents
hub_thalamus <- c("VThal", "MThal", "MLThal", "PThal", "GThal", "Rh", "Rt")
# Brainstem subdivisions; PAG/VTA/SN sit in midbrain, autonomic outputs in medulla
hub_brainstem_midbrain <- c("VMid", "DMid", "MMid", "LMid", "IMed")
hub_brainstem_pons <- c("VPons", "DPons", "LPons")
hub_brainstem_medulla <- c("VMed", "DMed")
# Limbic subcortical (autonomic-network)
hub_limbic <- c("spAmy", "pAmy", "THy", "PHy", "ZI-H")
# Pallidum / striatum
hub_basal_ganglia <- c("Str", "LPal", "VPal", "Pd")

hubs_all <- list(
  thalamus = hub_thalamus,
  brainstem_midbrain = hub_brainstem_midbrain,
  brainstem_pons = hub_brainstem_pons,
  brainstem_medulla = hub_brainstem_medulla,
  limbic_subcortical = hub_limbic,
  basal_ganglia = hub_basal_ganglia
)

# ============================================================
# Load combined table
# ============================================================
summ <- read_excel(COMBINED_XLSX, sheet = "Summary")
ipsi <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_L3_ipsi")
contra <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_L3_contra")

# Use NeuronUID order
common <- intersect(summ$NeuronUID, ipsi$NeuronUID)
summ <- summ[match(common, summ$NeuronUID), ]
ipsi <- ipsi[match(common, ipsi$NeuronUID), ]
contra <- contra[match(common, contra$NeuronUID), ]

# Final side / region
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

# Build numeric matrices
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

ipsi_total   <- rowSums(ipsi_mat, na.rm = TRUE)
contra_total <- rowSums(contra_mat, na.rm = TRUE)

ipsi_prop <- ipsi_mat
ipsi_prop[ipsi_total > 0, ] <- sweep(ipsi_mat[ipsi_total > 0, , drop = FALSE], 1,
                                      ipsi_total[ipsi_total > 0], "/")

# ============================================================
# Per-target test: hurdle (presence + magnitude conditional)
# ============================================================
per_target_test <- function(prop_mat, side_vec, sample_vec, label, target_regs) {
  out <- bind_rows(lapply(target_regs, function(rg) {
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
    p_mag_cond <- tryCatch({
      xLp <- xL[xL > 0]; xRp <- xR[xR > 0]
      if (length(xLp) >= 2 && length(xRp) >= 2)
        suppressWarnings(wilcox.test(xLp, xRp, exact = FALSE)$p.value)
      else NA_real_
    }, error = function(e) NA_real_)
    safe_med <- function(x) {
      v <- x[x > 0]
      if (length(v) == 0) return(NA_real_)
      as.numeric(median(v, na.rm = TRUE))
    }
    data.frame(
      stratum = label, target = rg,
      n_L = nL, n_R = nR,
      n_samples = length(unique(sample_vec)),
      frac_L = mean(xL > 0), frac_R = mean(xR > 0),
      mean_L = mean(xL), mean_R = mean(xR),
      median_pos_L = safe_med(xL),
      median_pos_R = safe_med(xR),
      p_presence = p_pres,
      p_magnitude_overall = p_mag,
      p_magnitude_conditional = p_mag_cond,
      stringsAsFactors = FALSE,
      row.names = NULL
    )
  }))
  if (!nrow(out)) return(out)
  out$direction <- ifelse(out$mean_L > out$mean_R, "L>R", "R>L")
  out
}

# Strata to test
strata_def <- list(
  all_combined = rep(TRUE, nrow(summ)),
  per_251637 = summ$SampleID == "251637",
  IDD5_251637 = summ$SampleID == "251637" & summ$Soma_Region_Final == "IDD5",
  IDM_251637 = summ$SampleID == "251637" & summ$Soma_Region_Final == "IDM",
  IDD5_plus_IDM = summ$SampleID == "251637" & summ$Soma_Region_Final %in% c("IDD5", "IDM"),
  IAL_combined = summ$Soma_Region_Final == "IAL",
  IAL_251637 = summ$SampleID == "251637" & summ$Soma_Region_Final == "IAL",
  IAL_252385 = summ$SampleID == "252385" & summ$Soma_Region_Final == "IAL"
)

# Run per hub family
all_hub_results <- list()
for (hub_name in names(hubs_all)) {
  targets <- hubs_all[[hub_name]]
  cat(sprintf("\n=== HUB: %s  (targets: %s) ===\n", hub_name,
              paste(targets, collapse = ", ")))
  hub_dfs <- list()
  for (slabel in names(strata_def)) {
    mask <- strata_def[[slabel]]
    if (sum(mask) < 6) next
    res <- per_target_test(ipsi_prop[mask, , drop = FALSE],
                            summ$Soma_Side_Final[mask],
                            summ$SampleID[mask],
                            slabel, targets)
    if (nrow(res)) hub_dfs[[slabel]] <- res
  }
  combined <- bind_rows(hub_dfs)
  if (nrow(combined)) {
    # BH per stratum (within-stratum family of targets)
    combined <- combined %>%
      group_by(stratum) %>%
      mutate(
        p_presence_BH = p.adjust(p_presence, method = "BH"),
        p_magnitude_overall_BH = p.adjust(p_magnitude_overall, method = "BH"),
        p_magnitude_conditional_BH = p.adjust(p_magnitude_conditional, method = "BH")
      ) %>% ungroup()
    out_csv <- file.path(STATS_DIR, paste0("hub_", hub_name, "_per_target.csv"))
    write.csv(combined, out_csv, row.names = FALSE)
    cat(sprintf("  saved %d rows -> %s\n", nrow(combined), out_csv))

    sig_rows <- combined %>%
      filter(p_presence_BH < 0.10 | p_magnitude_overall_BH < 0.10) %>%
      arrange(stratum, p_magnitude_overall)
    if (nrow(sig_rows)) {
      cat("  significant rows (BH q<0.10 in either presence or magnitude):\n")
      print(sig_rows %>% dplyr::select(stratum, target, n_L, n_R,
                                        frac_L, frac_R, mean_L, mean_R,
                                        p_presence_BH, p_magnitude_overall_BH,
                                        direction),
            digits = 3, row.names = FALSE)
    } else {
      cat("  no BH-significant per-target results\n")
    }
    all_hub_results[[hub_name]] <- combined
  }
}

# Save consolidated workbook
wb <- list()
for (h in names(all_hub_results)) wb[[h]] <- all_hub_results[[h]]
write_xlsx(wb, file.path(STATS_DIR, "hubs_per_target_all.xlsx"))
cat(sprintf("\n[saved consolidated] %s\n",
            file.path(STATS_DIR, "hubs_per_target_all.xlsx")))

# ============================================================
# Visualization: per-hub mean L vs mean R bar plots, faceted by stratum
# ============================================================
side_cols <- c(L = "#E74C3C", R = "#3498DB")

plot_hub <- function(df, hub_name, top_strata) {
  if (nrow(df) == 0) return(NULL)
  d <- df %>%
    filter(stratum %in% top_strata) %>%
    pivot_longer(cols = c("mean_L", "mean_R"),
                 names_to = "side",
                 values_to = "mean_proportion") %>%
    mutate(side = sub("mean_", "", side),
           side = factor(side, levels = c("L", "R")),
           label = ifelse(p_magnitude_overall_BH < 0.05, "*",
                          ifelse(p_magnitude_overall_BH < 0.10, ".", "")))
  if (!nrow(d)) return(NULL)
  ggplot(d, aes(x = target, y = mean_proportion, fill = side)) +
    geom_col(position = position_dodge(width = 0.85), width = 0.75) +
    facet_wrap(~ stratum, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = side_cols) +
    labs(title = sprintf("Hub: %s", hub_name),
         x = "target region", y = "mean proportion of ipsi projection") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "bottom")
}

top_strata <- c("all_combined", "IDD5_plus_IDM", "IAL_combined")
for (h in names(all_hub_results)) {
  p <- plot_hub(all_hub_results[[h]], h, top_strata)
  if (!is.null(p)) {
    ggsave(file.path(FIG_DIR, paste0("hub_", h, ".png")),
           p, width = 9, height = 8, dpi = 200)
  }
}
cat("\n[Hub analysis done]\n")
