#!/usr/bin/env Rscript
# Smoke tests for projectome_analysis R code.
# Run from repo root:
#   Rscript scripts/smoke_test_r.R

options(warn = 1)
root <- Sys.getenv("PROJECTOME_ROOT")
if (!nzchar(root)) {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  root <- if (length(file_arg)) {
    normalizePath(file.path(dirname(file_arg), ".."), winslash = "/")
  } else {
    normalizePath(getwd(), winslash = "/")
  }
}
setwd(root)
cat("Project root:", root, "\n\n")

user_lib <- file.path(Sys.getenv("USERPROFILE"), "AppData", "Local", "R", "win-library",
                      paste0(R.version$major, ".", strsplit(R.version$minor, "\\.")[[1]][1]))
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

fail <- 0L
ok <- function(msg) cat("  OK   ", msg, "\n", sep = "")
bad <- function(msg) { cat("  FAIL ", msg, "\n", sep = ""); fail <<- fail + 1L }

cat("[1] Core packages\n")
core_pkgs <- c(
  "readxl", "readr", "dplyr", "tidyr", "ggplot2", "scales", "patchwork",
  "cowplot", "ggpubr", "knitr", "RColorBrewer", "viridis", "vegan",
  "cluster", "factoextra", "igraph", "randomForest", "ComplexHeatmap",
  "circlize", "fossil", "GGally", "grid", "dendextend"
)
for (p in core_pkgs) {
  if (requireNamespace(p, quietly = TRUE)) ok(paste("package", p))
  else bad(paste("package", p))
}

cat("\n[2] R script syntax (parse only)\n")
r_files <- c(
  "R_analysis/plot_figure2a.R",
  "R_analysis/advanced_visualization.R",
  "R_analysis/summarise_plot.R",
  "R_analysis/merge_bilateral.R",
  "group_analysis/R_analysis/plot_dataset_status_overview.R",
  "group_analysis/R_analysis/combined_lr_primary_analysis.R",
  "group_analysis/R_analysis/multi_monkey_lr_analysis.R"
)
for (f in r_files) {
  path <- file.path(root, f)
  if (!file.exists(path)) {
    bad(paste("missing file", f))
    next
  }
  res <- tryCatch(parse(path), error = function(e) e)
  if (inherits(res, "error")) bad(paste(f, ":", conditionMessage(res)))
  else ok(paste("parse", f))
}

cat("\n[3] Data dependencies\n")
data_checks <- list(
  c("R_analysis/plot_figure2a.R", "R_analysis/neuron_projection_data.xlsx"),
  c("group_analysis/R_analysis/plot_dataset_status_overview.R",
    "group_analysis/docs/dataset_status_manifest.csv"),
  c("R_analysis/tables/251637_INS_HE_inferred.xlsx", "R_analysis/tables/251637_INS_HE_inferred.xlsx")
)
for (row in data_checks) {
  script <- row[[1]]
  data_path <- file.path(root, row[[2]])
  if (file.exists(data_path)) ok(paste("data for", basename(script), "->", row[[2]]))
  else bad(paste("missing data for", script, ":", row[[2]]))
}

cat("\n[4] Executable smoke runs\n")

# 4a: read real insula table
cat("  --- read insula xlsx ---\n")
insula_xlsx <- file.path(root, "R_analysis/tables/251637_INS_HE_inferred.xlsx")
if (file.exists(insula_xlsx)) {
  res <- tryCatch({
    df <- readxl::read_excel(insula_xlsx)
    cat("       rows:", nrow(df), "cols:", ncol(df), "\n")
    ok("read_excel 251637_INS_HE_inferred.xlsx")
  }, error = function(e) bad(paste("read insula xlsx:", conditionMessage(e))))
} else {
  bad("insula xlsx not found")
}

# 4b: group status plot script
cat("  --- plot_dataset_status_overview.R ---\n")
status_script <- file.path(root, "group_analysis/R_analysis/plot_dataset_status_overview.R")
if (file.exists(status_script)) {
  res <- tryCatch({
    sys.source(status_script, envir = new.env(), keep.source = TRUE)
    ok("sourced plot_dataset_status_overview.R")
  }, error = function(e) bad(paste("plot_dataset_status_overview:", conditionMessage(e))))
}

# 4c: plot_figure2a package block only (avoid auto-install to Program Files)
cat("  --- plot_figure2a.R (packages only) ---\n")
fig2a <- file.path(root, "R_analysis/plot_figure2a.R")
if (file.exists(fig2a)) {
  pkgs_fig2a <- c("ComplexHeatmap", "circlize", "dendextend", "RColorBrewer",
                  "readxl", "dplyr", "tidyr", "grid", "ggplot2")
  pkg_ok <- all(vapply(pkgs_fig2a, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1)))
  demo_xlsx <- file.path(root, "R_analysis/neuron_projection_data.xlsx")
  if (pkg_ok) ok("plot_figure2a package deps")
  else bad("plot_figure2a missing packages")
  if (file.exists(demo_xlsx)) {
    res <- tryCatch({
      env <- new.env()
      sys.source(fig2a, envir = env, keep.source = TRUE)
      ok("sourced full plot_figure2a.R")
    }, error = function(e) bad(paste("plot_figure2a run:", conditionMessage(e))))
  } else {
    cat("  SKIP full plot_figure2a.R (no R_analysis/neuron_projection_data.xlsx)\n")
  }
}

cat("\n")
if (fail == 0L) {
  cat("SMOKE TEST PASSED\n")
  quit(status = 0)
}
cat("SMOKE TEST: ", fail, " failure(s)\n", sep = "")
quit(status = 1)
