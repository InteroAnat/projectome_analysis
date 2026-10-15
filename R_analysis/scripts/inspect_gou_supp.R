suppressPackageStartupMessages(library(readxl))
fS1 <- "C:/Users/binbi/xwechat_files/wxid_ky7h36zl302f11_d2b8/msg/file/2025-11/macaque_PFC_paper_20250710/Supplemental Information/Table_S1.xlsx"
fS2 <- "C:/Users/binbi/xwechat_files/wxid_ky7h36zl302f11_d2b8/msg/file/2025-11/macaque_PFC_paper_20250710/Supplemental Information/Table_S2.xlsx"
cat("=== Table_S1 sheets ===\n"); print(excel_sheets(fS1))
cat("\n=== Table_S2 sheets ===\n"); print(excel_sheets(fS2))

for (s in excel_sheets(fS1)) {
  cat("\n--- Table_S1 / ", s, "---\n", sep="")
  d <- tryCatch(read_excel(fS1, sheet = s, n_max = 5), error = function(e) NULL)
  if (!is.null(d)) print(d)
}
for (s in excel_sheets(fS2)) {
  cat("\n--- Table_S2 / ", s, "---\n", sep="")
  d <- tryCatch(read_excel(fS2, sheet = s, n_max = 5), error = function(e) NULL)
  if (!is.null(d)) print(d)
}
