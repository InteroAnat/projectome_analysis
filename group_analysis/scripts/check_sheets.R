library(readxl)
cat("251637 sheets:\n")
for (s in excel_sheets("D:/projectome_analysis/neuron_tables_new/251637_INS_HE_inferred.xlsx"))
  cat("  ", s, "\n")
cat("\n252385 sheets:\n")
for (s in excel_sheets("D:/projectome_analysis/group_analysis/step1_results/252385_20260427_131131_region_analysis/tables/252385_results_20260427_131418.xlsx"))
  cat("  ", s, "\n")

cat("\n251637 finest ipsi columns (sample):\n")
m <- read_excel("D:/projectome_analysis/neuron_tables_new/251637_INS_HE_inferred.xlsx",
                 sheet = "Projection_Strength_ipsi", n_max = 1)
cat("  ", paste(names(m), collapse = " | "), "\n")
cat("\n251637 L3 ipsi columns:\n")
m2 <- read_excel("D:/projectome_analysis/neuron_tables_new/251637_INS_HE_inferred.xlsx",
                 sheet = "Projection_Strength_L3_ipsi", n_max = 1)
cat("  ", paste(names(m2), collapse = " | "), "\n")
