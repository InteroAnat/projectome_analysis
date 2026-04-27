# ============================================================
# Functional hub-region L/R analysis at FINEST level (L6)
# Per-target hurdle (presence + magnitude), stratified by sample/sub-region
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
STATS_DIR <- file.path(OUT_DIR, "stats", "hubs_L6")
FIG_DIR   <- file.path(OUT_DIR, "figures", "hubs_L6")
for (d in c(STATS_DIR, FIG_DIR))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Hub definitions at L6 (atlas finest names from ARM v2.1 / CHARM /SARM)
# ============================================================
# Thalamus L6 sub-nuclei (from columns observed in 251637 finest sheet):
#   MD = mediodorsal     IMD = intermediodorsal
#   CM = centromedian    PF/CMn-PF = parafascicular
#   VPM-VPL = ventral posterior (somatosensory + visceral / VMpo region)
#   VPI = ventral posterior inferior   APul = anterior pulvinar
#   MPul = medial pulvinar     SPFPC = subparafascicular
#   VLA = ventral lateral anterior     VLPV/VLPD = ventral lateral pv/pd
#   VM = ventral medial    VA = ventral anterior   MG = medial geniculate
#   DLG = dorsal lateral geniculate   PR-RI = posterior thalamus
#   PaL = paralaminar  Pa = paraventricular  R = reticular  Re-Rh-Xi = midline
#   Rt = reticular thalamic
hub_thalamus <- c("MD", "IMD", "CM", "CMn-PF", "VLA", "VLPV", "VLPD",
                  "VM", "VA", "VPM-VPL", "VPI", "VMPo-VMB", "APul",
                  "MPul", "MG", "DLG", "SG", "Pa", "Re-Rh-Xi", "PaL",
                  "PR-RI", "Lim", "MM", "PH", "Rt", "SPFPC", "PCom_MCPC",
                  "p1Rt", "PO")

# Brainstem L6: midbrain, pons, medulla
hub_brainstem_midbrain <- c("PAG", "SCo", "ICo", "VTA", "SN", "SubC",
                             "RtTg", "MiTg", "RM", "PnO", "PnC",
                             "MPB", "KF", "RF", "InO", "PCom_MCPC", "ll")
hub_brainstem_medulla  <- c("Gi", "LRt", "ml", "RM", "py", "scp", "ic",
                             "cp", "mcp", "Pn", "PR-RI")
# Limbic + autonomic subcortical
hub_amygdala <- c("Ce", "BM", "BLD", "BLI", "BLV", "AA", "Pir", "Me",
                  "STIA", "PaL", "ASt", "AON/TTv", "EA", "APir",
                  "LaD", "LaV", "PR-RI", "EGP", "IGP", "B")
hub_hypothalamus <- c("ZI-H", "PH", "PLH", "Pa", "MM")
# Pallidum / striatum L6
hub_basal_ganglia <- c("CdH", "CdT", "Pu", "VP", "EGP", "IGP", "Cl",
                        "DGP", "B", "Acb", "STh", "ST", "Tu", "IPAC",
                        "PeB")

hubs_all <- list(
  thalamus = hub_thalamus,
  brainstem_midbrain = hub_brainstem_midbrain,
  brainstem_medulla = hub_brainstem_medulla,
  amygdala = hub_amygdala,
  hypothalamus = hub_hypothalamus,
  basal_ganglia = hub_basal_ganglia
)

# ============================================================
# Load combined table (L6 sheet)
# ============================================================
summ <- read_excel(COMBINED_XLSX, sheet = "Summary")
ipsi <- read_excel(COMBINED_XLSX, sheet = "Projection_Strength_ipsi")

common <- intersect(summ$NeuronUID, ipsi$NeuronUID)
summ <- summ[match(common, summ$NeuronUID), ]
ipsi <- ipsi[match(common, ipsi$NeuronUID), ]

summ <- summ %>%
  mutate(Soma_Side_Final = ifelse(!is.na(Soma_Side_Inferred) &
                                    Soma_Side_Inferred %in% c("L", "R"),
                                   Soma_Side_Inferred, Soma_Side),
         Soma_Region_Final = ifelse(!is.na(Soma_Region_Refined) &
                                      nzchar(Soma_Region_Refined),
                                     Soma_Region_Refined,
                                     gsub("^(CL|CR)_", "", Soma_Region_Auto)))
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

cat(sprintf("[L6] ipsi matrix: %d x %d targets\n",
            nrow(ipsi_prop), ncol(ipsi_prop)))

# Coverage check
cat("\n[hub coverage at L6]\n")
for (h in names(hubs_all)) {
  in_data <- intersect(hubs_all[[h]], colnames(ipsi_prop))
  cat(sprintf("  %-22s %d/%d targets in data\n",
              h, length(in_data), length(hubs_all[[h]])))
}

# ============================================================
# Per-target test
# ============================================================
per_target_test <- function(prop_mat, side_vec, sample_vec, label, target_regs) {
  out <- bind_rows(lapply(intersect(target_regs, colnames(prop_mat)),
                            function(rg) {
    xL <- prop_mat[side_vec == "L", rg]
    xR <- prop_mat[side_vec == "R", rg]
    nL <- length(xL); nR <- length(xR)
    if (nL < 3 || nR < 3) return(NULL)
    tab <- matrix(c(sum(xL > 0), sum(xL == 0),
                    sum(xR > 0), sum(xR == 0)), nrow = 2)
    p_pres <- tryCatch(fisher.test(tab)$p.value, error = function(e) NA_real_)
    p_mag <- tryCatch(suppressWarnings(
      wilcox.test(xL, xR, exact = FALSE)$p.value), error = function(e) NA_real_)
    safe_med <- function(x) {
      v <- x[x > 0]; if (length(v) == 0) NA_real_ else as.numeric(median(v, na.rm = TRUE))
    }
    data.frame(
      stratum = label, target = rg,
      n_L = nL, n_R = nR,
      n_samples = length(unique(sample_vec)),
      frac_L = mean(xL > 0), frac_R = mean(xR > 0),
      mean_L = mean(xL), mean_R = mean(xR),
      median_pos_L = safe_med(xL), median_pos_R = safe_med(xR),
      p_presence = p_pres, p_magnitude = p_mag,
      stringsAsFactors = FALSE, row.names = NULL
    )
  }))
  if (!nrow(out)) return(out)
  out$direction <- ifelse(out$mean_L > out$mean_R, "L>R", "R>L")
  out
}

strata_def <- list(
  all_combined = rep(TRUE, nrow(summ)),
  per_251637 = summ$SampleID == "251637",
  IDD5_251637 = summ$SampleID == "251637" & summ$Soma_Region_Final == "IDD5",
  IDM_251637 = summ$SampleID == "251637" & summ$Soma_Region_Final == "IDM",
  IDD5_plus_IDM = summ$SampleID == "251637" &
                   summ$Soma_Region_Final %in% c("IDD5", "IDM"),
  IAL_combined = summ$Soma_Region_Final == "IAL",
  IAL_251637 = summ$SampleID == "251637" & summ$Soma_Region_Final == "IAL",
  IAL_252385 = summ$SampleID == "252385" & summ$Soma_Region_Final == "IAL"
)

all_hub_results <- list()
for (hub_name in names(hubs_all)) {
  cat(sprintf("\n=== HUB %s (n=%d targets in data) ===\n",
              hub_name, length(intersect(hubs_all[[hub_name]],
                                          colnames(ipsi_prop)))))
  hub_dfs <- list()
  for (slabel in names(strata_def)) {
    mask <- strata_def[[slabel]]
    if (sum(mask) < 6) next
    res <- per_target_test(ipsi_prop[mask, , drop = FALSE],
                            summ$Soma_Side_Final[mask],
                            summ$SampleID[mask],
                            slabel, hubs_all[[hub_name]])
    if (nrow(res)) hub_dfs[[slabel]] <- res
  }
  combined <- bind_rows(hub_dfs)
  if (!nrow(combined)) next
  combined <- combined %>%
    group_by(stratum) %>%
    mutate(p_presence_BH = p.adjust(p_presence, method = "BH"),
           p_magnitude_BH = p.adjust(p_magnitude, method = "BH")) %>%
    ungroup()
  out_csv <- file.path(STATS_DIR, paste0("hub_L6_", hub_name, "_per_target.csv"))
  write.csv(combined, out_csv, row.names = FALSE)
  cat(sprintf("  saved %d rows -> %s\n", nrow(combined), out_csv))
  sig <- combined %>%
    filter(p_presence_BH < 0.05 | p_magnitude_BH < 0.05) %>%
    arrange(stratum, p_magnitude)
  if (nrow(sig)) {
    cat("  Significant rows (BH < 0.05):\n")
    print(sig %>% dplyr::select(stratum, target, n_L, n_R,
                                  frac_L, frac_R, mean_L, mean_R,
                                  p_presence_BH, p_magnitude_BH, direction),
          digits = 3, row.names = FALSE)
  } else {
    cat("  no BH<0.05 results\n")
  }
  all_hub_results[[hub_name]] <- combined
}

# Save consolidated
write_xlsx(all_hub_results,
           file.path(STATS_DIR, "hubs_L6_per_target_all.xlsx"))
cat(sprintf("\n[saved consolidated] %s\n",
            file.path(STATS_DIR, "hubs_L6_per_target_all.xlsx")))

# ============================================================
# Visualization: top BH-significant targets per hub
# ============================================================
side_cols <- c(L = "#E74C3C", R = "#3498DB")
for (h in names(all_hub_results)) {
  d <- all_hub_results[[h]] %>%
    filter(stratum %in% c("all_combined", "IDD5_plus_IDM", "IAL_combined",
                           "IAL_251637", "IAL_252385")) %>%
    pivot_longer(cols = c("mean_L", "mean_R"),
                 names_to = "side", values_to = "mean_prop") %>%
    mutate(side = sub("mean_", "", side),
           side = factor(side, levels = c("L", "R")))
  if (nrow(d) == 0) next
  p <- ggplot(d, aes(x = target, y = mean_prop, fill = side)) +
    geom_col(position = position_dodge(width = 0.85), width = 0.78) +
    facet_wrap(~ stratum, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = side_cols) +
    labs(title = sprintf("Hub L6: %s", h),
         x = "target region", y = "mean proportion of ipsi projection") +
    theme_minimal(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
          legend.position = "bottom")
  ggsave(file.path(FIG_DIR, paste0("hub_L6_", h, ".png")),
         p, width = 12, height = 11, dpi = 200)
}

cat("\n[Hub L6 analysis done]\n")
