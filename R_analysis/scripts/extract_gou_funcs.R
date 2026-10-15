suppressPackageStartupMessages({library(readxl); library(dplyr); library(tidyr)})

fS2 <- "C:/Users/binbi/xwechat_files/wxid_ky7h36zl302f11_d2b8/msg/file/2025-11/macaque_PFC_paper_20250710/Supplemental Information/Table_S2.xlsx"

biofun <- read_excel(fS2, sheet = "(B) Biofunction")
funclass <- read_excel(fS2, sheet = "(C) Function class")

# Forward-fill empty Brain Area entries (merged cells in xlsx)
biofun <- biofun %>%
  rename(BrainArea = `Brain Area`, FunctionTerm = Function) %>%
  fill(BrainArea, .direction = "down")

funclass <- funclass %>%
  rename(FunctionTerm = Function,
         Subcategory = Subcatagory,
         Category = Catagory) %>%
  fill(Subcategory, .direction = "down") %>%
  fill(Category, .direction = "down")

cat("== Biofunction (region -> function term) ==\n")
cat(sprintf("n_unique_areas=%d, n_unique_functions=%d, n_rows=%d\n",
            length(unique(biofun$BrainArea)),
            length(unique(biofun$FunctionTerm)),
            nrow(biofun)))

cat("\n-- All BrainArea values (Gou's Macaque atlas labels) --\n")
print(sort(unique(biofun$BrainArea)))

cat("\n== Function class (function -> subcat -> cat) ==\n")
cat(sprintf("n_unique_functions=%d, n_unique_subcat=%d, n_unique_cat=%d\n",
            length(unique(funclass$FunctionTerm)),
            length(unique(funclass$Subcategory)),
            length(unique(funclass$Category))))

cat("\n-- All Categories (function_class_big) --\n")
print(sort(unique(funclass$Category)))

cat("\n-- All Subcategories (function_class_small) --\n")
print(sort(unique(funclass$Subcategory)))

# Save to CSV for downstream use
dir.create("d:/projectome_analysis/R_analysis/scripts/data_output/gou_function_table",
           recursive = TRUE, showWarnings = FALSE)
write.csv(biofun,
          "d:/projectome_analysis/R_analysis/scripts/data_output/gou_function_table/area_to_function.csv",
          row.names = FALSE)
write.csv(funclass,
          "d:/projectome_analysis/R_analysis/scripts/data_output/gou_function_table/function_to_category.csv",
          row.names = FALSE)

# Join to make region -> Subcategory / Category
joined <- biofun %>%
  left_join(funclass, by = "FunctionTerm")
write.csv(joined,
          "d:/projectome_analysis/R_analysis/scripts/data_output/gou_function_table/area_function_category_full.csv",
          row.names = FALSE)

cat("\n\n-- Sample of full join (region -> function -> subcat -> cat) --\n")
print(head(joined, 25))
