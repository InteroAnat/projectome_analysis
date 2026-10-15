library(dplyr)
library(readxl)
library(writexl)
library(purrr)
library(stringr)
dist_file <- "D:/projectome_analysis/main_scripts/processed_neurons/251637/fnt_processed/ins/ins_dist.txt"
base <- "D:/projectome_analysis/neuron_tables/"

# Get sheet names from the first file
sheet_names <- excel_sheets(file.path(base, "251637_FULL.xlsx"))

# Read all sheets from each file into named lists
full_list <- setNames(map(sheet_names, ~read_excel(file.path(base, "251637_FULL.xlsx"), sheet = .x)), sheet_names)
ins_list  <- setNames(map(sheet_names, ~read_excel(file.path(base, "251637_INS.xlsx"),  sheet = .x)), sheet_names)
m1_list   <- setNames(map(sheet_names, ~read_excel(file.path(base, "251637_M1.xlsx"),   sheet = .x)), sheet_names)


full_summary <- read_excel(file.path(base, "251637_FULL.xlsx"))
           
           # Define region filters from the first sheet (neuron metadata)
acc <- full_list[[1]] %>% filter(str_detect(Soma_Region, 'area_23|24|ACC')) %>% dplyr::select(NeuronID) 
prco <- full_list[[1]] %>% filter(str_detect(Soma_Region, regex('PrCO', ignore_case = TRUE))) %>% dplyr::select(NeuronID)
unknown <- full_list[[1]] %>% filter(str_detect(Soma_Region, regex('unknown', ignore_case = TRUE))) %>% dplyr::select(NeuronID)

# Apply filtering to each corresponding sheet (remove INS and M1 neurons)
result_list <- map(seq_along(sheet_names), function(i) {
  full_list[[i]] %>%
    anti_join(ins_list[[i]], by = "NeuronID") %>%
    anti_join(m1_list[[i]],  by = "NeuronID")
})

# Filter by specific regions
acc_list <- map(seq_along(sheet_names), function(i) {
  result_list[[i]] %>% filter(NeuronID %in% acc$NeuronID)
})

prco_list <- map(seq_along(sheet_names), function(i) {
  result_list[[i]] %>% filter(NeuronID %in% prco$NeuronID)
})

unknown_list <- map(seq_along(sheet_names), function(i) {
  result_list[[i]] %>% filter(NeuronID %in% unknown$NeuronID)
})

other_list <- map(seq_along(sheet_names), function(i) {
  result_list[[i]] %>% 
    filter(!NeuronID %in% acc$NeuronID & 
             !NeuronID %in% prco$NeuronID & 
             !NeuronID %in% unknown$NeuronID)
})

# Name all lists with original sheet names
names(result_list) <- sheet_names
names(acc_list) <- sheet_names
names(prco_list) <- sheet_names
names(unknown_list) <- sheet_names
names(other_list) <- sheet_names

# Save all datasets to Excel
write_xlsx(result_list, file.path(base, "251637_subset.xlsx"))
write_xlsx(acc_list, file.path(base, "251637_ACC.xlsx"))
write_xlsx(prco_list, file.path(base, "251637_PrCO.xlsx"))
write_xlsx(unknown_list, file.path(base, "251637_Unknown.xlsx"))
write_xlsx(other_list, file.path(base, "251637_Other.xlsx"))

# Print summary
cat("\n=== RESULT (Full - INS - M1) ===\n")
iwalk(result_list, ~cat("Sheet", .y, ":", nrow(.x), "neurons\n"))

cat("\n=== ACC Subset ===\n")
iwalk(acc_list, ~cat("Sheet", .y, ":", nrow(.x), "neurons\n"))

cat("\n=== PrCO Subset ===\n")
iwalk(prco_list, ~cat("Sheet", .y, ":", nrow(.x), "neurons\n"))

cat("\n=== Unknown Region Subset ===\n")
iwalk(unknown_list, ~cat("Sheet", .y, ":", nrow(.x), "neurons\n"))

cat("\n=== Other Region Subset ===\n")
iwalk(other_list, ~cat("Sheet", .y, ":", nrow(.x), "neurons\n"))

# Summary table
summary_df <- tibble(
  Sheet = sheet_names,
  Full_Filtered = map_int(result_list, nrow),
  ACC = map_int(acc_list, nrow),
  PrCO = map_int(prco_list, nrow),
  Unknown = map_int(unknown_list, nrow),
  Other = map_int(other_list, nrow)
)

print(summary_df)
# write_xlsx(summary_df, file.path(base, "251637_summary.xlsx"))

# Verify totals match
cat("\n=== Verification ===\n")
print(summary_df %>% 
        mutate(Sum = ACC + PrCO + Unknown + Other,
               Match = Sum == Full_Filtered))



library(dplyr)
library(ggplot2)
soma_table <- table(full_list[[1]]$Soma_Region)
soma_distribution <- tibble(
  Soma_Region = names(soma_table),
  Count = as.numeric(soma_table)
) %>%
  arrange(desc(Count)) %>%
  mutate(Percentage = Count / sum(Count) * 100)

print(soma_distribution)
ggplot(soma_distribution, aes(x = "", y = Count, fill = reorder(Soma_Region, Count))) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y", start = 0) +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "right",
    legend.title = element_text(face = "bold")
  ) +
  labs(
    title = "Soma Region Distribution (251637 FULL)",
    subtitle = paste("Total:", sum(soma_distribution$Count), "neurons"),
    fill = "Soma Region"
  ) +
  scale_fill_viridis_d(option = "turbo") +
  geom_text(
    aes(label = ifelse(Percentage > 3, paste0(round(Percentage, 1), "%"), "")),
    position = position_stack(vjust = 0.5),
    color = "white", fontface = "bold", size = 3.5
  )

# Create summary data from first sheet (neuron metadata)
soma_stats <- full_summary%>%
  count(Soma_Region) %>%
  mutate(percentage = n / sum(n) * 100)

# Basic pie chart
ggplot(soma_stats, aes(x = "", y = n, fill = Soma_Region)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y") +
  theme_void() +
  labs(title = "Soma Region Distribution")