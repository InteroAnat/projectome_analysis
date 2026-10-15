# Core CRAN packages for projectome_analysis R_analysis scripts.
user_lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(user_lib)) {
  user_lib <- file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library",
                        paste0(R.version$major, ".", strsplit(R.version$minor, "\\.")[[1]][1]))
}
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(user_lib, .libPaths()))
message("R library path: ", user_lib)

pkgs <- c(
  "readxl", "dplyr", "tidyr", "purrr", "ggplot2", "scales", "patchwork",
  "cowplot", "ggpubr", "knitr", "RColorBrewer", "viridis",
  "vegan", "cluster", "factoextra", "igraph", "randomForest",
  "circlize", "fossil", "GGally"
)

install_if_missing <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message("Installing ", p, " ...")
    install.packages(
      p,
      repos = "https://cloud.r-project.org",
      lib = user_lib,
      dependencies = TRUE,
      type = if (.Platform$OS.type == "windows") "both" else "source"
    )
  } else {
    message("OK ", p)
  }
}

# Bioconductor for ComplexHeatmap if needed
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = user_lib)
}
bioc <- c("ComplexHeatmap")
for (p in bioc) {
  if (!requireNamespace(p, quietly = TRUE)) {
    message("Installing Bioconductor ", p, " ...")
    BiocManager::install(p, ask = FALSE, update = FALSE)
  } else {
    message("OK ", p)
  }
}

invisible(lapply(setdiff(pkgs, "ComplexHeatmap"), install_if_missing))

cat("\nR package bootstrap complete.\n")
