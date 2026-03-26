# ============================================================
# Subset FULL table by excluding INS and M1 neurons
# ============================================================

library(dplyr)
library(readxl)

# Set paths
base_path <- "D:/projectome_analysis/main_scripts/neuron_tables"

# Load tables
full_table <- read_excel(file.path(base_path, "251637_FULL.xlsx"))
ins_table <- read_excel(file.path(base_path, "251637_INS.xlsx"))
m1_table <- read_excel(file.path(base_path, "251637_M1.xlsx"))

# Get NeuronIDs to exclude
exclude_ids <- c(ins_table$NeuronID, m1_table$NeuronID) %>% unique()

cat("Full table:", nrow(full_table), "neurons\n")
cat("INS table:", nrow(ins_table), "neurons\n")
cat("M1 table:", nrow(m1_table), "neurons\n")
cat("Unique IDs to exclude:", length(exclude_ids), "neurons\n")

# Method 1: Using anti_join (most readable)
subset_result <- full_table %>%
  anti_join(ins_table, by = "NeuronID") %>%
  anti_join(m1_table, by = "NeuronID")

cat("\nResult:", nrow(subset_result), "neurons (excluded", nrow(full_table) - nrow(subset_result), ")\n")

# Save result
output_path <- file.path(base_path, "251637_FULL_excluding_INS_M1.xlsx")
writexl::write_xlsx(subset_result, output_path)
cat("Saved to:", output_path, "\n")
