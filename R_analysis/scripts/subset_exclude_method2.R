# ============================================================
# Method 2: Using %in% with negation
# ============================================================

library(readxl)

base_path <- "D:/projectome_analysis/main_scripts/neuron_tables"

# Load tables
full <- read_excel(file.path(base_path, "251637_FULL.xlsx"))
ins <- read_excel(file.path(base_path, "251637_INS.xlsx"))
m1 <- read_excel(file.path(base_path, "251637_M1.xlsx"))

# Create subset excluding INS and M1
subset_table <- full[!(full$NeuronID %in% ins$NeuronID | full$NeuronID %in% m1$NeuronID), ]

# Alternative: two-step
# subset_table <- full[!(full$NeuronID %in% ins$NeuronID), ]
# subset_table <- subset_table[!(subset_table$NeuronID %in% m1$NeuronID), ]

cat("Original:", nrow(full), "→ After exclusion:", nrow(subset_table), "\n")

# Save
writexl::write_xlsx(subset_table, file.path(base_path, "251637_excluding_INS_M1.xlsx"))
