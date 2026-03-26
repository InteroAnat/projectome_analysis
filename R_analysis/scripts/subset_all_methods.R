# ============================================================
# Complete Guide: Subset FULL table excluding INS and M1
# ============================================================

library(dplyr)
library(readxl)
library(writexl)

# -----------------------------------------------------------
# 1. Load Data
# -----------------------------------------------------------
base_path <- "D:/projectome_analysis/main_scripts/neuron_tables"

full_table <- read_excel(file.path(base_path, "251637_FULL.xlsx"))
ins_table <- read_excel(file.path(base_path, "251637_INS.xlsx"))
m1_table <- read_excel(file.path(base_path, "251637_M1.xlsx"))

cat("=== Input Tables ===\n")
cat(sprintf("FULL: %d neurons\n", nrow(full_table)))
cat(sprintf("INS:  %d neurons\n", nrow(ins_table)))
cat(sprintf("M1:   %d neurons\n", nrow(m1_table)))

# -----------------------------------------------------------
# Method 1: anti_join (cleanest, preserves all columns)
# -----------------------------------------------------------
cat("\n=== Method 1: anti_join() ===\n")

result1 <- full_table %>%
  anti_join(ins_table, by = "NeuronID") %>%
  anti_join(m1_table, by = "NeuronID")

cat(sprintf("Result: %d neurons (removed %d)\n", 
            nrow(result1), nrow(full_table) - nrow(result1)))

# -----------------------------------------------------------
# Method 2: filter() with negation
# -----------------------------------------------------------
cat("\n=== Method 2: filter() with !%in% ===\n")

exclude_ids <- unique(c(ins_table$NeuronID, m1_table$NeuronID))

result2 <- full_table %>%
  filter(!NeuronID %in% exclude_ids)

cat(sprintf("Result: %d neurons (removed %d)\n",
            nrow(result2), nrow(full_table) - nrow(result2)))

# -----------------------------------------------------------
# Method 3: Base R subset()
# -----------------------------------------------------------
cat("\n=== Method 3: Base R subset() ===\n")

result3 <- subset(full_table, 
                  !(NeuronID %in% ins_table$NeuronID) & 
                    !(NeuronID %in% m1_table$NeuronID))

cat(sprintf("Result: %d neurons (removed %d)\n",
            nrow(result3), nrow(full_table) - nrow(result3)))

# -----------------------------------------------------------
# Verification: Check overlap
# -----------------------------------------------------------
cat("\n=== Verification ===\n")

# Check if any INS neurons are in result
ins_in_result <- sum(result1$NeuronID %in% ins_table$NeuronID)
m1_in_result <- sum(result1$NeuronID %in% m1_table$NeuronID)

cat(sprintf("INS neurons still in result: %d (should be 0)\n", ins_in_result))
cat(sprintf("M1 neurons still in result: %d (should be 0)\n", m1_in_result))

# -----------------------------------------------------------
# Summary of excluded neurons by type
# -----------------------------------------------------------
cat("\n=== Summary of Excluded Neurons ===\n")

excluded <- full_table %>%
  filter(NeuronID %in% exclude_ids) %>%
  count(Neuron_Type, name = "Count")

print(excluded)

# -----------------------------------------------------------
# Summary of remaining neurons by type
# -----------------------------------------------------------
cat("\n=== Summary of Remaining Neurons ===\n")

remaining_summary <- result1 %>%
  count(Neuron_Type, Soma_Region, name = "Count") %>%
  arrange(desc(Count))

print(head(remaining_summary, 10))

# -----------------------------------------------------------
# Save Result
# -----------------------------------------------------------
output_file <- file.path(base_path, "251637_FULL_excluding_INS_M1.xlsx")
write_xlsx(result1, output_file)

cat(sprintf("\n=== Saved to ===\n%s\n", output_file))
cat(sprintf("Final count: %d neurons\n", nrow(result1)))
