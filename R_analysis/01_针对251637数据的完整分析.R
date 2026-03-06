# ============================================================================
# 针对251637数据的完整R分析脚本
# 
# 数据说明:
# - 212个神经元
# - 胞体位置: CR_Ial (右岛叶Ial区)
# - 神经元类型: ITs, ITi, CT, ITc
# - 聚类: k=9 (spearman + penalty)
# ============================================================================

# =============================================================================
# 第一部分: 环境设置和数据读取
# =============================================================================

cat("=" ,rep("=", 70), "\n", sep="")
cat("  251637数据完整分析\n")
cat("  212个神经元, 胞体在右岛叶Ial区\n")
cat("=" ,rep("=", 70), "\n\n", sep="")

# 设置工作目录 (修改为你的路径)
# setwd("/path/to/your/data")

# 检查和安装包
cat("📦 检查并安装必要的包...\n")

packages <- c("readxl", "dplyr", "tidyr", "ggplot2", "pheatmap", 
              "RColorBrewer", "viridis")

for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat("  安装包:", pkg, "\n")
        install.packages(pkg, repos = "https://cloud.r-project.org/")
        library(pkg, character.only = TRUE)
    }
}

# ComplexHeatmap (Bioconductor)
if (!require("ComplexHeatmap", quietly = TRUE)) {
    if (!require("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
    }
    BiocManager::install("ComplexHeatmap")
    library(ComplexHeatmap)
}

if (!require("circlize", quietly = TRUE)) {
    install.packages("circlize")
    library(circlize)
}

cat("✅ 所有包加载完成\n\n")

# =============================================================================
# 第二部分: 数据读取
# =============================================================================

cat("📖 读取数据...\n")

# 文件路径 (修改为你的实际路径)
file_results <- "251637_results.xlsx"
file_clustered <- "251637_results_clustered_k9_spearman_penalty.xlsx"
file_laterality <- "251637_laterality_projections.xlsx"

# 检查文件是否存在
if (!file.exists(file_results)) {
    stop("文件不存在: ", file_results, "\n请修改文件路径或确保文件在工作目录中")
}

# 读取各个Sheet
df_summary <- read_excel(file_results, sheet = "Summary")
df_hierarchy <- read_excel(file_results, sheet = "Soma_Hierarchy")
df_proj_length <- read_excel(file_results, sheet = "Projection_Length")
df_proj_strength <- read_excel(file_results, sheet = "Projection_Strength")
df_proj_length_l3 <- read_excel(file_results, sheet = "Projection_Length_L3")
df_proj_strength_l3 <- read_excel(file_results, sheet = "Projection_Strength_L3")
df_laterality <- read_excel(file_results, sheet = "Laterality")
df_terminals <- read_excel(file_results, sheet = "Terminal_Sites")
df_outliers <- read_excel(file_results, sheet = "Outliers")
df_clustered <- read_excel(file_clustered)
df_ipsi_length <- read_excel(file_laterality, sheet = "Ipsilateral_Length")

cat("✅ 数据读取完成\n")
cat("  Summary:", nrow(df_summary), "neurons x", ncol(df_summary), "columns\n")
cat("  Projection Length:", nrow(df_proj_length), "x", ncol(df_proj_length), "\n")
cat("  Brain regions:", ncol(df_proj_length) - 2, "\n")
cat("  Morphology clusters (k=9):", length(unique(df_clustered$Morph_Cluster)), "\n\n")

# =============================================================================
# 第三部分: 数据探索和统计
# =============================================================================

cat("📊 数据探索...\n\n")

# 3.1 神经元类型分布
cat("=== 神经元类型分布 ===\n")
type_table <- table(df_summary$Neuron_Type)
print(type_table)
cat("\n")

# 3.2 投射长度统计
cat("=== 总投射长度统计 (μm) ===\n")
print(summary(df_summary$Total_Length))
cat("\n")

# 3.3 按类型分组的统计
cat("=== 按神经元类型统计 ===\n")
type_stats <- df_summary %>%
    group_by(Neuron_Type) %>%
    summarise(
        n = n(),
        mean_length = mean(Total_Length),
        median_length = median(Total_Length),
        sd_length = sd(Total_Length),
        mean_terminals = mean(Terminal_Count),
        mean_ipsi_regions = mean(N_Ipsilateral)
    )
print(type_stats)
cat("\n")

# 3.4 聚类分布
cat("=== 形态学聚类分布 (k=9) ===\n")
cluster_table <- table(df_clustered$Morph_Cluster)
print(cluster_table)
cat("\n")

# 3.5 聚类与神经元类型的关系
cat("=== 聚类 vs 神经元类型 ===\n")
cluster_type_table <- df_clustered %>%
    select(NeuronID, Morph_Cluster, Neuron_Type) %>%
    table()
print(cluster_type_table)
cat("\n")

# =============================================================================
# 第四部分: 基础可视化
# =============================================================================

cat("📈 生成基础可视化...\n\n")

# 创建输出目录
output_dir <- "251637_analysis_output"
if (!dir.exists(output_dir)) {
    dir.create(output_dir)
}

# 4.1 神经元类型分布柱状图
p1 <- ggplot(df_summary, aes(x = Neuron_Type, fill = Neuron_Type)) +
    geom_bar() +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Neuron Type Distribution (n=212)",
         subtitle = "Soma location: CR_Ial (Right Insula)",
         x = "Neuron Type",
         y = "Count") +
    theme_minimal() +
    theme(legend.position = "none")

ggsave(file.path(output_dir, "01_neuron_type_distribution.pdf"), p1, width = 6, height = 4)
cat("✅ Saved: 01_neuron_type_distribution.pdf\n")

# 4.2 投射长度箱线图
p2 <- ggplot(df_summary, aes(x = Neuron_Type, y = Total_Length, fill = Neuron_Type)) +
    geom_boxplot() +
    geom_jitter(width = 0.2, alpha = 0.3) +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Total Axon Length by Neuron Type",
         x = "Neuron Type",
         y = "Total Length (μm)") +
    theme_minimal() +
    theme(legend.position = "none")

ggsave(file.path(output_dir, "02_axon_length_by_type.pdf"), p2, width = 7, height = 5)
cat("✅ Saved: 02_axon_length_by_type.pdf\n")

# 4.3 投射长度vs终端数量散点图
p3 <- ggplot(df_summary, aes(x = Total_Length, y = Terminal_Count, color = Neuron_Type)) +
    geom_point(size = 3, alpha = 0.6) +
    scale_color_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Axon Length vs Terminal Count",
         x = "Total Axon Length (μm)",
         y = "Terminal Count") +
    theme_minimal()

ggsave(file.path(output_dir, "03_length_vs_terminals.pdf"), p3, width = 8, height = 6)
cat("✅ Saved: 03_length_vs_terminals.pdf\n")

# 4.4 投射长度分布直方图
p4 <- ggplot(df_summary, aes(x = Total_Length, fill = Neuron_Type)) +
    geom_histogram(bins = 30, alpha = 0.7) +
    facet_wrap(~Neuron_Type, scales = "free_y") +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Distribution of Axon Length by Type",
         x = "Total Length (μm)",
         y = "Count") +
    theme_minimal() +
    theme(legend.position = "none")

ggsave(file.path(output_dir, "04_length_distribution.pdf"), p4, width = 10, height = 6)
cat("✅ Saved: 04_length_distribution.pdf\n")

# 4.5 聚类分布
p5 <- ggplot(df_clustered, aes(x = factor(Morph_Cluster), fill = Neuron_Type)) +
    geom_bar() +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Morphology Cluster Distribution (k=9)",
         x = "Cluster",
         y = "Count",
         fill = "Neuron Type") +
    theme_minimal()

ggsave(file.path(output_dir, "05_cluster_distribution.pdf"), p5, width = 8, height = 5)
cat("✅ Saved: 05_cluster_distribution.pdf\n")

# =============================================================================
# 第五部分: 热图分析
# =============================================================================

cat("\n🔥 生成热图...\n\n")

# 5.1 准备投射矩阵
proj_matrix <- df_proj_length %>%
    select(-NeuronID, -Neuron_Type) %>%
    as.matrix()
rownames(proj_matrix) <- df_proj_length$NeuronID

# 筛选有投射的脑区
col_sums <- colSums(proj_matrix)
proj_matrix_filtered <- proj_matrix[, col_sums > 0]

# log转换
proj_matrix_log <- log1p(proj_matrix_filtered)

cat("  Regions with projections:", ncol(proj_matrix_filtered), "\n")

# 5.2 简单热图 (pheatmap)
cat("  生成简单热图...\n")

# 选择前50个神经元和前50个脑区
pheatmap(proj_matrix_log[1:50, 1:50],
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         scale = "none",
         main = "Projection Length Heatmap (log, subset)",
         filename = file.path(output_dir, "06_heatmap_simple.pdf"),
         width = 12, height = 10)

cat("✅ Saved: 06_heatmap_simple.pdf\n")

# 5.3 按神经元类型分组的热图
cat("  生成按类型分组的热图...\n")

# 创建行注释
row_anno <- data.frame(
    Type = df_proj_length$Neuron_Type
)
rownames(row_anno) <- rownames(proj_matrix_log)

# 类型颜色
type_colors <- c("ITs" = "#4DAF4A", "ITi" = "#984EA3", 
                 "ITc" = "#377EB8", "CT" = "#FF7F00")

pheatmap(proj_matrix_log,
         cluster_rows = TRUE,
         cluster_cols = TRUE,
         annotation_row = row_anno,
         annotation_colors = list(Type = type_colors),
         show_rownames = FALSE,
         main = "Projection Heatmap by Neuron Type",
         filename = file.path(output_dir, "07_heatmap_by_type.pdf"),
         width = 14, height = 16)

cat("✅ Saved: 07_heatmap_by_type.pdf\n")

# 5.4 按聚类分组的热图 (ComplexHeatmap)
cat("  生成按聚类分组的热图...\n")

# 合并聚类信息
df_proj_clustered <- df_proj_length %>%
    left_join(df_clustered %>% select(NeuronID, Morph_Cluster), by = "NeuronID")

# 提取矩阵
proj_mat <- df_proj_clustered %>%
    select(-NeuronID, -Neuron_Type, -Morph_Cluster) %>%
    as.matrix()
rownames(proj_mat) <- df_proj_clustered$NeuronID

# 筛选有投射的脑区
col_sums <- colSums(proj_mat)
proj_mat <- proj_mat[, col_sums > 0]
proj_mat_log <- log1p(proj_mat)

# 聚类颜色
cluster_colors <- c(
    "1" = "#E41A1C", "2" = "#377EB8", "3" = "#4DAF4A",
    "4" = "#984EA3", "5" = "#FF7F00", "6" = "#FFFF33",
    "7" = "#A65628", "8" = "#F781BF", "9" = "#999999"
)

# 顶部注释
ha_top <- HeatmapAnnotation(
    Cluster = as.character(df_proj_clustered$Morph_Cluster),
    Type = df_proj_clustered$Neuron_Type,
    col = list(
        Cluster = cluster_colors,
        Type = type_colors
    ),
    annotation_name_side = "left"
)

# 创建热图
ht <- Heatmap(
    proj_mat_log,
    name = "Log Length",
    col = colorRamp2(c(0, 2, 4, 6, 8), 
                     c("#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")),
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    row_split = df_proj_clustered$Morph_Cluster,
    row_gap = unit(2, "mm"),
    top_annotation = ha_top,
    row_names_gp = gpar(fontsize = 4),
    column_names_gp = gpar(fontsize = 5),
    column_names_rot = 45,
    show_row_names = FALSE,
    column_title = "Target Brain Regions",
    row_title = "Morphology Clusters (k=9)"
)

pdf(file.path(output_dir, "08_heatmap_by_cluster_k9.pdf"), width = 16, height = 18)
draw(ht)
dev.off()

cat("✅ Saved: 08_heatmap_by_cluster_k9.pdf\n")

# =============================================================================
# 第六部分: 侧向性分析
# =============================================================================

cat("\n🧠 侧向性分析...\n\n")

# 6.1 侧向性指数分布
p6 <- ggplot(df_summary, aes(x = Laterality_Index, fill = Neuron_Type)) +
    geom_histogram(bins = 20, alpha = 0.7) +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Laterality Index Distribution",
         subtitle = "1 = Purely Ipsilateral, -1 = Purely Contralateral",
         x = "Laterality Index",
         y = "Count") +
    theme_minimal()

ggsave(file.path(output_dir, "09_laterality_index_distribution.pdf"), p6, width = 8, height = 5)
cat("✅ Saved: 09_laterality_index_distribution.pdf\n")

# 6.2 同侧vs对侧投射长度
df_laterality_long <- df_summary %>%
    select(NeuronID, Neuron_Type, 
           Ipsilateral = Total_Ipsilateral_Length,
           Contralateral = Total_Contralateral_Length) %>%
    pivot_longer(cols = c(Ipsilateral, Contralateral),
                 names_to = "Laterality",
                 values_to = "Length")

p7 <- ggplot(df_laterality_long, aes(x = Laterality, y = Length, fill = Laterality)) +
    geom_boxplot() +
    geom_jitter(width = 0.2, alpha = 0.3) +
    facet_wrap(~Neuron_Type, scales = "free_y") +
    labs(title = "Ipsilateral vs Contralateral Projection Length",
         y = "Total Length (μm)") +
    theme_minimal() +
    theme(legend.position = "none")

ggsave(file.path(output_dir, "10_ipsi_vs_contra_length.pdf"), p7, width = 10, height = 6)
cat("✅ Saved: 10_ipsi_vs_contra_length.pdf\n")

# =============================================================================
# 第七部分: 保存统计结果
# =============================================================================

cat("\n💾 保存统计结果...\n\n")

# 保存类型统计
write.csv(type_stats, file.path(output_dir, "stats_by_type.csv"), row.names = FALSE)

# 保存聚类统计
cluster_stats <- df_clustered %>%
    group_by(Morph_Cluster, Neuron_Type) %>%
    summarise(n = n(), .groups = "drop")
write.csv(cluster_stats, file.path(output_dir, "stats_cluster_type.csv"), row.names = FALSE)

cat("✅ Saved: stats_by_type.csv\n")
cat("✅ Saved: stats_cluster_type.csv\n")

# =============================================================================
# 完成
# =============================================================================

cat("\n" ,rep("=", 70), "\n", sep="")
cat("  ✅ 分析完成!\n")
cat("  输出目录:", output_dir, "\n")
cat("  生成文件:\n")
cat("    - 10个PDF图表\n")
cat("    - 2个CSV统计文件\n")
cat(rep("=", 70), "\n", sep="")
