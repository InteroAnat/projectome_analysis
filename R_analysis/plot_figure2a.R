# ============================================================================
# 猕猴前额叶皮层单神经元投射组热图绘制
# 参考: Gou et al., Cell 2025 - Figure 2a
# ============================================================================

# ==================== 1. 安装和加载必要的包 ====================

# 定义需要安装的包
packages <- c("ComplexHeatmap", "circlize", "dendextend", "RColorBrewer", 
              "readxl", "dplyr", "tidyr", "grid", "ggplot2")

# 安装未安装的包
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  }
}

invisible(sapply(packages, install_if_missing))

# 特别安装Bioconductor包 (如果ComplexHeatmap安装失败)
if (!require("ComplexHeatmap", quietly = TRUE)) {
  if (!require("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
  }
  BiocManager::install("ComplexHeatmap")
  library(ComplexHeatmap)
}

# ==================== 2. 数据读取和预处理 ====================

# 读取Excel文件 (请修改为你的文件路径)
file_path <- "neuron_projection_data.xlsx"

# 读取各个sheet
df_projection <- read_excel(file_path, sheet = "Projection_Data")
df_metadata <- read_excel(file_path, sheet = "Metadata")
df_regions <- read_excel(file_path, sheet = "Region_Info")

# 设置行名
df_projection <- df_projection %>% column_to_rownames(var = "...1")

# 查看数据结构
cat("=== 数据概览 ===\n")
cat("投射矩阵维度:", nrow(df_projection), "x", ncol(df_projection), "\n")
cat("亚型数量:", nrow(df_metadata), "\n")
cat("脑区数量:", nrow(df_regions), "\n")

# ==================== 3. 数据转换和标准化 ====================

# 将数据转换为矩阵
projection_matrix <- as.matrix(df_projection)

# 可选：对数据进行标准化 (Z-score)
# projection_matrix <- t(scale(t(projection_matrix)))

# ==================== 4. 创建注释信息 ====================

# 创建Type的颜色映射
type_colors <- c(
  "ITc" = "#4DAF4A",  # 绿色 - 皮层内投射 (cortical)
  "ITs" = "#984EA3",  # 紫色 - 纹状体投射 (striatum)
  "CT"  = "#FF7F00",  # 橙色 - 皮层丘脑投射 (corticothalamic)
  "PT"  = "#E41A1C",  # 红色 - 锥体束投射 (pyramidal tract)
  "ITi" = "#377EB8"   # 蓝色 - 皮层内长程投射 (ipsilateral long-range)
)

# 创建Soma区域的颜色映射
soma_regions <- unique(df_metadata$Soma_Region)
soma_colors <- setNames(
  colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(length(soma_regions)),
  soma_regions
)

# 创建顶部注释条 (Top Annotation)
# 注意：确保metadata的顺序与projection_matrix的行顺序一致
df_metadata <- df_metadata[match(rownames(projection_matrix), df_metadata$Subtype), ]

ha_top <- HeatmapAnnotation(
  # 主要Type分类
  Type = df_metadata$Type,
  # Soma位置
  Soma = df_metadata$Soma_Region,
  # 神经元数量 (条形图)
  N_Neurons = anno_barplot(df_metadata$N_Neurons, 
                           gp = gpar(fill = "gray70"),
                           height = unit(1.5, "cm")),
  # 总轴突长度 (条形图)
  Axon_Length = anno_barplot(log10(df_metadata$Total_Axon_Length), 
                             gp = gpar(fill = "steelblue"),
                             height = unit(1.5, "cm")),

  col = list(
    Type = type_colors,
    Soma = soma_colors
  ),

  annotation_name_side = "left",
  annotation_name_gp = gpar(fontsize = 10, fontface = "bold"),

  # 添加图例
  show_legend = c(TRUE, TRUE, FALSE, FALSE)
)

# ==================== 5. 创建右侧脑区分组注释 ====================

# 根据Hemisphere分组
region_hemisphere <- df_regions$Hemisphere[match(colnames(projection_matrix), df_regions$Region)]

# 创建右侧注释
ha_right <- rowAnnotation(
  Hemisphere = region_hemisphere,
  col = list(
    Hemisphere = c("Ipsilateral" = "#FDB462", "Contralateral" = "#80B1D3")
  ),
  show_legend = TRUE,
  annotation_name_side = "top",
  width = unit(0.5, "cm")
)

# ==================== 6. 定义颜色方案 ====================

# 创建蓝色渐变颜色方案 (类似原文)
col_fun <- colorRamp2(
  c(0, quantile(projection_matrix, 0.5), max(projection_matrix)),
  c("white", "#9ECAE1", "#08306B")  # 从白到深蓝
)

# 或者使用更复杂的颜色方案
col_fun2 <- colorRamp2(
  c(0, quantile(projection_matrix, 0.25), 
    quantile(projection_matrix, 0.5), 
    quantile(projection_matrix, 0.75), 
    max(projection_matrix)),
  c("#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
)

# ==================== 7. 创建聚类树状图 ====================

# 对亚型进行层次聚类 (使用Ward方法)
dist_subtypes <- dist(projection_matrix, method = "euclidean")
hc_subtypes <- hclust(dist_subtypes, method = "ward.D2")

# 对脑区进行层次聚类
dist_regions <- dist(t(projection_matrix), method = "euclidean")
hc_regions <- hclust(dist_regions, method = "ward.D2")

# ==================== 8. 绘制主热图 ====================

ht <- Heatmap(
  projection_matrix,

  # === 基本设置 ===
  name = "Log Axon Length",
  col = col_fun2,

  # === 聚类设置 ===
  cluster_rows = hc_subtypes,      # 使用自定义聚类
  cluster_columns = hc_regions,    # 使用自定义聚类

  # 树状图外观
  row_dend_gp = gpar(col = "gray30", lwd = 0.8),
  column_dend_gp = gpar(col = "gray30", lwd = 0.8),

  # 树状图高度
  row_dend_height = unit(3, "cm"),
  column_dend_height = unit(3, "cm"),

  # === 行列标签 ===
  row_names_side = "left",
  row_names_gp = gpar(fontsize = 8),

  column_names_side = "bottom",
  column_names_gp = gpar(fontsize = 8, fontface = "bold"),
  column_names_rot = 45,  # 旋转45度

  # === 单元格设置 ===
  rect_gp = gpar(col = NA),  # 不显示单元格边框

  # === 注释条 ===
  top_annotation = ha_top,
  right_annotation = ha_right,

  # === 图例设置 ===
  heatmap_legend_param = list(
    title = "Log Axon Length",
    title_gp = gpar(fontsize = 10, fontface = "bold"),
    labels_gp = gpar(fontsize = 8),
    legend_height = unit(4, "cm"),
    legend_width = unit(1, "cm"),
    direction = "vertical"
  ),

  # === 尺寸设置 ===
  width = unit(20, "cm"),
  height = unit(16, "cm"),

  # === 标题 ===
  column_title = "Target Brain Regions",
  column_title_gp = gpar(fontsize = 12, fontface = "bold"),
  column_title_side = "bottom",

  row_title = "Neuron Subtypes",
  row_title_gp = gpar(fontsize = 12, fontface = "bold"),
  row_title_side = "left"
)

# ==================== 9. 保存图形 ====================

# 打开PDF设备
pdf("Figure2a_Projection_Heatmap.pdf", width = 16, height = 12)

# 绘制热图
draw(ht, 
     merge_legend = TRUE,
     heatmap_legend_side = "right",
     annotation_legend_side = "right")

# 关闭设备
dev.off()

cat("\n✅ 热图已保存到: Figure2a_Projection_Heatmap.pdf\n")

# ==================== 10. 创建PNG版本 (用于预览) ====================

png("Figure2a_Projection_Heatmap.png", width = 2400, height = 1800, res = 150)
draw(ht, 
     merge_legend = TRUE,
     heatmap_legend_side = "right",
     annotation_legend_side = "right")
dev.off()

cat("✅ PNG预览已保存到: Figure2a_Projection_Heatmap.png\n")

# ==================== 11. 附加分析：按Type分组的子热图 ====================

cat("\n=== 生成按Type分组的子热图 ===\n")

# 为每个Type创建单独的热图
for (t in unique(df_metadata$Type)) {

  # 筛选该Type的亚型
  subtype_idx <- which(df_metadata$Type == t)

  if (length(subtype_idx) > 1) {

    # 提取子矩阵
    sub_matrix <- projection_matrix[subtype_idx, ]
    sub_metadata <- df_metadata[subtype_idx, ]

    # 创建子热图
    sub_ht <- Heatmap(
      sub_matrix,
      name = paste0("Type_", t),
      col = col_fun2,
      cluster_rows = TRUE,
      cluster_columns = TRUE,
      row_names_side = "left",
      column_names_side = "bottom",
      column_names_rot = 45,
      top_annotation = HeatmapAnnotation(
        Soma = sub_metadata$Soma_Region,
        col = list(Soma = soma_colors),
        annotation_name_side = "left"
      ),
      column_title = paste0("Type ", t, " Projection Pattern"),
      column_title_gp = gpar(fontsize = 12, fontface = "bold"),
      width = unit(16, "cm"),
      height = unit(length(subtype_idx) * 0.8, "cm")
    )

    # 保存
    pdf(paste0("Figure2a_Type_", t, "_Heatmap.pdf"), width = 12, height = 8)
    draw(sub_ht)
    dev.off()

    cat("  ✅ Type", t, "热图已保存\n")
  }
}

cat("\n=== 所有图形生成完成! ===\n")
