# ============================================================================
# 高级神经投射数据可视化脚本
# 包含多种图表类型和分析方法
# ============================================================================

# ==================== 0. 初始化 ====================

# 清空环境
rm(list = ls())

# 设置工作目录 (请修改为你的路径)
# setwd("/path/to/your/data")

# ==================== 1. 安装和加载包 ====================

packages <- c("ComplexHeatmap", "circlize", "RColorBrewer", "readxl", 
              "dplyr", "tidyr", "ggplot2", "grid", "viridis", 
              "ggdendro", "factoextra", "cluster")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

# Bioconductor包
if (!require("ComplexHeatmap", quietly = TRUE)) {
  if (!require("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
  BiocManager::install("ComplexHeatmap")
  library(ComplexHeatmap)
}

cat("✅ 所有包加载完成\n")

# ==================== 2. 数据读取函数 ====================

read_projection_data <- function(file_path) {
  #' 读取投射数据
  #' @param file_path Excel文件路径
  #' @return 包含三个数据框的列表

  cat("📖 读取数据:", file_path, "\n")

  df_projection <- read_excel(file_path, sheet = "Projection_Data")
  df_metadata <- read_excel(file_path, sheet = "Metadata")
  df_regions <- read_excel(file_path, sheet = "Region_Info")

  # 设置行名
  df_projection <- df_projection %>% column_to_rownames(var = colnames(df_projection)[1])

  cat("  ✓ 投射矩阵:", nrow(df_projection), "x", ncol(df_projection), "\n")
  cat("  ✓ 元数据:", nrow(df_metadata), "行\n")
  cat("  ✓ 脑区信息:", nrow(df_regions), "行\n")

  return(list(
    projection = df_projection,
    metadata = df_metadata,
    regions = df_regions
  ))
}

# ==================== 3. 数据预处理函数 ====================

preprocess_data <- function(data_list, z_score = FALSE, log_transform = FALSE) {
  #' 数据预处理
  #' @param data_list 数据列表
  #' @param z_score 是否进行Z-score标准化
  #' @param log_transform 是否进行log转换
  #' @return 处理后的数据列表

  mat <- as.matrix(data_list$projection)

  # Log转换
  if (log_transform) {
    mat <- log1p(mat)
    cat("✓ 应用log1p转换\n")
  }

  # Z-score标准化 (按行)
  if (z_score) {
    mat <- t(scale(t(mat)))
    cat("✓ 应用Z-score标准化\n")
  }

  data_list$projection_matrix <- mat
  return(data_list)
}

# ==================== 4. 颜色方案 ====================

create_color_schemes <- function(data_list) {
  #' 创建颜色方案

  # Type颜色 (参考Cell文章)
  type_colors <- c(
    "ITc" = "#4DAF4A",  # 绿色
    "ITs" = "#984EA3",  # 紫色
    "CT"  = "#FF7F00",  # 橙色
    "PT"  = "#E41A1C",  # 红色
    "ITi" = "#377EB8"   # 蓝色
  )

  # Soma区域颜色
  soma_regions <- unique(data_list$metadata$Soma_Region)
  n_soma <- length(soma_regions)
  soma_colors <- setNames(
    colorRampPalette(RColorBrewer::brewer.pal(min(8, n_soma), "Set2"))(n_soma),
    soma_regions
  )

  # Hemisphere颜色
  hemisphere_colors <- c(
    "Ipsilateral" = "#FDB462",
    "Contralateral" = "#80B1D3"
  )

  # 投射强度颜色 (蓝色渐变)
  mat <- data_list$projection_matrix
  projection_colors <- colorRamp2(
    c(min(mat), quantile(mat, 0.25), median(mat), quantile(mat, 0.75), max(mat)),
    c("#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
  )

  return(list(
    type = type_colors,
    soma = soma_colors,
    hemisphere = hemisphere_colors,
    projection = projection_colors
  ))
}

# ==================== 5. 主热图函数 ====================

plot_main_heatmap <- function(data_list, colors, output_prefix = "Figure2a") {
  #' 绘制主热图 (Figure 2a风格)

  mat <- data_list$projection_matrix
  meta <- data_list$metadata

  # 确保顺序一致
  meta <- meta[match(rownames(mat), meta$Subtype), ]

  # 创建顶部注释
  ha_top <- HeatmapAnnotation(
    Type = meta$Type,
    Soma = meta$Soma_Region,
    N_Neurons = anno_barplot(
      meta$N_Neurons,
      gp = gpar(fill = "gray70", col = NA),
      height = unit(1.2, "cm"),
      axis_param = list(gp = gpar(fontsize = 6))
    ),
    Axon_Length = anno_barplot(
      log10(meta$Total_Axon_Length),
      gp = gpar(fill = "steelblue", col = NA),
      height = unit(1.2, "cm"),
      axis_param = list(gp = gpar(fontsize = 6))
    ),
    col = list(
      Type = colors$type,
      Soma = colors$soma
    ),
    annotation_name_side = "left",
    annotation_name_gp = gpar(fontsize = 9, fontface = "bold"),
    show_legend = c(TRUE, TRUE, FALSE, FALSE)
  )

  # 创建右侧注释 (脑区分组)
  regions <- data_list$regions
  region_hemi <- regions$Hemisphere[match(colnames(mat), regions$Region)]

  ha_right <- rowAnnotation(
    Hemisphere = region_hemi,
    col = list(Hemisphere = colors$hemisphere),
    show_legend = TRUE,
    annotation_name_side = "top",
    width = unit(0.4, "cm")
  )

  # 层次聚类
  dist_rows <- dist(mat, method = "euclidean")
  hc_rows <- hclust(dist_rows, method = "ward.D2")

  dist_cols <- dist(t(mat), method = "euclidean")
  hc_cols <- hclust(dist_cols, method = "ward.D2")

  # 创建热图
  ht <- Heatmap(
    mat,
    name = "Log Axon Length",
    col = colors$projection,

    # 聚类
    cluster_rows = hc_rows,
    cluster_columns = hc_cols,
    row_dend_gp = gpar(col = "gray40", lwd = 0.6),
    column_dend_gp = gpar(col = "gray40", lwd = 0.6),
    row_dend_height = unit(2.5, "cm"),
    column_dend_height = unit(2.5, "cm"),

    # 标签
    row_names_side = "left",
    row_names_gp = gpar(fontsize = 7),
    column_names_side = "bottom",
    column_names_gp = gpar(fontsize = 7),
    column_names_rot = 45,

    # 单元格
    rect_gp = gpar(col = NA),

    # 注释
    top_annotation = ha_top,
    right_annotation = ha_right,

    # 图例
    heatmap_legend_param = list(
      title = "Log Axon Length",
      title_gp = gpar(fontsize = 10, fontface = "bold"),
      labels_gp = gpar(fontsize = 8),
      legend_height = unit(4, "cm"),
      direction = "vertical"
    ),

    # 尺寸
    width = unit(18, "cm"),
    height = unit(14, "cm"),

    # 标题
    column_title = "Target Brain Regions",
    column_title_gp = gpar(fontsize = 11, fontface = "bold"),
    column_title_side = "bottom",
    row_title = "Neuron Subtypes",
    row_title_gp = gpar(fontsize = 11, fontface = "bold"),
    row_title_side = "left"
  )

  # 保存PDF
  pdf(paste0(output_prefix, "_Main_Heatmap.pdf"), width = 14, height = 11)
  draw(ht, merge_legend = TRUE, heatmap_legend_side = "right")
  dev.off()

  # 保存PNG
  png(paste0(output_prefix, "_Main_Heatmap.png"), width = 2800, height = 2200, res = 150)
  draw(ht, merge_legend = TRUE, heatmap_legend_side = "right")
  dev.off()

  cat("✅ 主热图已保存:", output_prefix, "_Main_Heatmap.pdf/png\n")

  return(ht)
}

# ==================== 6. 按Type分组的子热图 ====================

plot_subtype_heatmaps <- function(data_list, colors, output_prefix = "Figure2a") {
  #' 为每个Type创建单独的热图

  mat <- data_list$projection_matrix
  meta <- data_list$metadata

  types <- unique(meta$Type)

  for (t in types) {
    idx <- which(meta$Type == t)

    if (length(idx) < 2) next

    sub_mat <- mat[idx, ]
    sub_meta <- meta[idx, ]

    # 过滤掉全为0的列
    col_sums <- colSums(sub_mat)
    sub_mat <- sub_mat[, col_sums > 0]

    # 创建子热图
    sub_ht <- Heatmap(
      sub_mat,
      name = paste0("Type_", t),
      col = colors$projection,
      cluster_rows = TRUE,
      cluster_columns = TRUE,
      row_names_side = "left",
      row_names_gp = gpar(fontsize = 8),
      column_names_side = "bottom",
      column_names_rot = 45,
      column_names_gp = gpar(fontsize = 7),
      top_annotation = HeatmapAnnotation(
        Soma = sub_meta$Soma_Region,
        N_Neurons = anno_barplot(sub_meta$N_Neurons, 
                                  gp = gpar(fill = "gray70"),
                                  height = unit(1, "cm")),
        col = list(Soma = colors$soma),
        annotation_name_side = "left"
      ),
      column_title = paste0("Type ", t, " Projection Pattern (n=", length(idx), ")"),
      column_title_gp = gpar(fontsize = 12, fontface = "bold"),
      width = unit(14, "cm"),
      height = unit(length(idx) * 0.6, "cm")
    )

    # 保存
    pdf(paste0(output_prefix, "_Type_", t, "_Heatmap.pdf"), 
        width = 12, height = max(6, length(idx) * 0.4))
    draw(sub_ht)
    dev.off()

    cat("  ✓ Type", t, "热图已保存\n")
  }
}

# ==================== 7. UMAP/PCA可视化 ====================

plot_dimension_reduction <- function(data_list, output_prefix = "Figure2a") {
  #' 绘制降维图 (类似原文UMAP)

  mat <- data_list$projection_matrix
  meta <- data_list$metadata

  # PCA
  pca_result <- prcomp(mat, scale. = TRUE)
  pca_df <- as.data.frame(pca_result$x[, 1:2])
  pca_df$Subtype <- rownames(mat)
  pca_df$Type <- meta$Type[match(rownames(mat), meta$Subtype)]
  pca_df$Soma <- meta$Soma_Region[match(rownames(mat), meta$Subtype)]

  # 绘制PCA
  p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Type)) +
    geom_point(size = 4, alpha = 0.8) +
    scale_color_manual(values = c(
      "ITc" = "#4DAF4A", "ITs" = "#984EA3", 
      "CT" = "#FF7F00", "PT" = "#E41A1C", "ITi" = "#377EB8"
    )) +
    theme_minimal() +
    labs(title = "PCA of Projection Patterns",
         x = paste0("PC1 (", round(summary(pca_result)$importance[2,1]*100, 1), "%)"),
         y = paste0("PC2 (", round(summary(pca_result)$importance[2,2]*100, 1), "%)")) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "right"
    )

  ggsave(paste0(output_prefix, "_PCA.pdf"), p_pca, width = 8, height = 6)
  cat("✅ PCA图已保存\n")

  # 如果安装了umap包，绘制UMAP
  if (require("umap", quietly = TRUE)) {
    umap_result <- umap(mat)
    umap_df <- as.data.frame(umap_result$layout)
    colnames(umap_df) <- c("UMAP1", "UMAP2")
    umap_df$Subtype <- rownames(mat)
    umap_df$Type <- meta$Type[match(rownames(mat), meta$Subtype)]

    p_umap <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Type)) +
      geom_point(size = 4, alpha = 0.8) +
      scale_color_manual(values = c(
        "ITc" = "#4DAF4A", "ITs" = "#984EA3", 
        "CT" = "#FF7F00", "PT" = "#E41A1C", "ITi" = "#377EB8"
      )) +
      theme_minimal() +
      labs(title = "UMAP of Projection Patterns") +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

    ggsave(paste0(output_prefix, "_UMAP.pdf"), p_umap, width = 8, height = 6)
    cat("✅ UMAP图已保存\n")
  }
}

# ==================== 8. 投射强度分布图 ====================

plot_projection_distribution <- function(data_list, output_prefix = "Figure2a") {
  #' 绘制投射强度的分布图

  mat <- data_list$projection_matrix
  meta <- data_list$metadata

  # 转换为长格式
  df_long <- as.data.frame(mat) %>%
    mutate(Subtype = rownames(mat)) %>%
    pivot_longer(cols = -Subtype, names_to = "Region", values_to = "LogLength") %>%
    left_join(meta, by = "Subtype")

  # 按Type分组的箱线图
  p_box <- ggplot(df_long, aes(x = Type, y = LogLength, fill = Type)) +
    geom_boxplot(alpha = 0.7) +
    geom_jitter(width = 0.2, alpha = 0.3, size = 0.5) +
    scale_fill_manual(values = c(
      "ITc" = "#4DAF4A", "ITs" = "#984EA3", 
      "CT" = "#FF7F00", "PT" = "#E41A1C", "ITi" = "#377EB8"
    )) +
    theme_minimal() +
    labs(title = "Projection Strength Distribution by Type",
         x = "Neuron Type",
         y = "Log Axon Length") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = "none"
    )

  ggsave(paste0(output_prefix, "_Distribution.pdf"), p_box, width = 8, height = 6)
  cat("✅ 分布图已保存\n")
}

# ==================== 9. 主程序 ====================

main <- function() {
  #' 主程序

  cat("=====================================\n")
  cat("  神经投射数据可视化\n")
  cat("=====================================\n\n")

  # 读取数据 (请修改为你的文件路径)
  file_path <- "neuron_projection_data.xlsx"

  if (!file.exists(file_path)) {
    cat("❌ 错误: 找不到文件", file_path, "\n")
    cat("请修改脚本中的 file_path 变量\n")
    return(NULL)
  }

  # 读取数据
  data_list <- read_projection_data(file_path)

  # 预处理 (可选)
  data_list <- preprocess_data(data_list, z_score = FALSE, log_transform = FALSE)

  # 创建颜色方案
  colors <- create_color_schemes(data_list)

  # 绘制主热图
  cat("\n📊 绘制主热图...\n")
  plot_main_heatmap(data_list, colors)

  # 绘制子热图
  cat("\n📊 绘制Type子热图...\n")
  plot_subtype_heatmaps(data_list, colors)

  # 降维分析
  cat("\n📊 绘制降维图...\n")
  plot_dimension_reduction(data_list)

  # 分布图
  cat("\n📊 绘制分布图...\n")
  plot_projection_distribution(data_list)

  cat("\n=====================================\n")
  cat("  ✅ 所有图表生成完成!\n")
  cat("=====================================\n")
}

# 运行主程序
main()
