# Launcher only: renders `v2_combined_primary_pipeline.Rmd` (canonical pipeline source).
# Usage:
#   Rscript v2_combined_primary_pipeline.R
#   Rscript v2_combined_primary_pipeline.R --verbose

args <- commandArgs(trailingOnly = TRUE)
verbose <- "--verbose" %in% args

ca <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", ca, value = TRUE)
if (!length(file_arg)) {
  stop("Use: Rscript v2_combined_primary_pipeline.R (needs --file= from Rscript)")
}
this_file <- sub("^--file=", "", file_arg[1])
dir_r <- dirname(normalizePath(this_file))
rmd <- file.path(dir_r, "v2_combined_primary_pipeline.Rmd")
if (!file.exists(rmd)) {
  stop("Missing Rmd: ", rmd)
}
if (!requireNamespace("rmarkdown", quietly = TRUE)) {
  stop("Install rmarkdown: install.packages('rmarkdown')")
}
rmarkdown::render(
  input = rmd,
  output_dir = dir_r,
  quiet = !verbose
)
