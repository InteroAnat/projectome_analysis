cat(.libPaths(), sep = "\n")
cat("\n--- pkgs ---\n")
for (p in c("readxl","readr","dplyr","tidyr","ggplot2","stringr","vegan","scales","tibble")) {
  cat(p, p %in% installed.packages()[,"Package"], "\n")
}
