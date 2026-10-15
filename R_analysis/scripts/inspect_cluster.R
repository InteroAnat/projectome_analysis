cluster_info <- readRDS("d:/projectome_analysis/R_analysis/scripts/data_output/clustering/clustering_results.rds")
cat("Names of cluster_info:\n"); print(names(cluster_info))
cat("\nNames of cluster_info$cluster_df:\n"); print(names(cluster_info$cluster_df))
cat("\nHead:\n"); print(head(cluster_info$cluster_df))
