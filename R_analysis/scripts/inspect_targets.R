suppressPackageStartupMessages(library(readxl))
suppressPackageStartupMessages(library(dplyr))

f <- "d:/projectome_analysis/neuron_tables_new/251637_INS_HE_inferred.xlsx"
ipsi <- read_excel(f, sheet = "Projection_Strength_L3_ipsi")
contra <- read_excel(f, sheet = "Projection_Strength_L3_contra")
summ <- read_excel(f, sheet = "Summary")

cat("Ipsi nrow=", nrow(ipsi), " ncol=", ncol(ipsi), "\n", sep = "")
cat("Contra nrow=", nrow(contra), " ncol=", ncol(contra), "\n", sep = "")
cat("Summary nrow=", nrow(summ), " ncol=", ncol(summ), "\n", sep = "")

cat("\nSummary columns:\n")
print(names(summ))

cat("\nSoma_Region x Soma_Side table:\n")
print(table(summ$Soma_Region, summ$Soma_Side, useNA = "ifany"))

cat("\nIpsi target regions (all):\n")
ipsi_targets <- setdiff(names(ipsi), c("NeuronID"))
print(ipsi_targets)

cat("\nContra target regions (all):\n")
contra_targets <- setdiff(names(contra), c("NeuronID"))
print(contra_targets)

cat("\nUnique union of targets:\n")
print(union(ipsi_targets, contra_targets))
