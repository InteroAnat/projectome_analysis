# ============================================================================
# 脑岛特异性分析 - 针对251637数据
# 
# 数据特点:
# - 212个神经元
# - 胞体位置: CR_Ial (右岛叶Ial区)
# - Ial = Insula agranular anterior, lateral
# - 属于前岛叶agranular区，VENs密度较低
# ============================================================================

# =============================================================================
# 第一部分: 数据准备
# =============================================================================

cat("=" ,rep("=", 70), "\n", sep="")
cat("  脑岛特异性分析\n")
cat("  胞体位置: CR_Ial (右岛叶Ial区)\n")
cat("=" ,rep("=", 70), "\n\n", sep="")

# 加载包
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(ComplexHeatmap)
library(circlize)

# 读取数据
file_results <- "251637_results.xlsx"
file_clustered <- "251637_results_clustered_k9_spearman_penalty.xlsx"

df_summary <- read_excel(file_results, sheet = "Summary")
df_hierarchy <- read_excel(file_results, sheet = "Soma_Hierarchy")
df_proj_length <- read_excel(file_results, sheet = "Projection_Length")
df_proj_length_l3 <- read_excel(file_results, sheet = "Projection_Length_L3")
df_laterality <- read_excel(file_results, sheet = "Laterality")
df_terminals <- read_excel(file_results, sheet = "Terminal_Sites")
df_clustered <- read_excel(file_clustered)

# 创建输出目录
output_dir <- "insula_analysis_output"
if (!dir.exists(output_dir)) {
    dir.create(output_dir)
}

# =============================================================================
# 第二部分: 识别脑岛相关投射
# =============================================================================

cat("🔍 识别脑岛相关投射...\n\n")

# 提取投射矩阵
proj_matrix <- df_proj_length %>%
    select(-NeuronID, -Neuron_Type) %>%
    as.matrix()
rownames(proj_matrix) <- df_proj_length$NeuronID

# 识别脑岛相关列（包含I的列名）
insula_pattern <- "I[agl]|FI|Iai|Ial|Iam|Iapm|Iapl|Id|Ig"
insula_cols <- grep(insula_pattern, colnames(proj_matrix), 
                    value = TRUE, ignore.case = TRUE)

# 区分左右脑岛
left_insula_cols <- grep("^CL_.*I", colnames(proj_matrix), 
                         value = TRUE, ignore.case = TRUE)
right_insula_cols <- grep("^CR_.*I", colnames(proj_matrix), 
                          value = TRUE, ignore.case = TRUE)

cat("=== 脑岛区域识别 ===\n")
cat("  总脑岛区域:", length(insula_cols), "\n")
cat("  左脑岛区域:", length(left_insula_cols), "\n")
cat("  右脑岛区域:", length(right_insula_cols), "\n")
cat("  脑岛区域示例:", paste(head(insula_cols, 5), collapse = ", "), "...\n\n")

# =============================================================================
# 第三部分: 脑岛内投射分析
# =============================================================================

cat("📊 脑岛内投射分析...\n\n")

# 3.1 提取脑岛投射矩阵
insula_proj <- proj_matrix[, insula_cols, drop = FALSE]
insula_proj_log <- log1p(insula_proj)

# 3.2 计算每个神经元的脑岛内投射比例
df_insula_stats <- data.frame(
    NeuronID = rownames(proj_matrix),
    Total_Length = rowSums(proj_matrix),
    Insula_Length = rowSums(insula_proj),
    Insula_Proportion = rowSums(insula_proj) / (rowSums(proj_matrix) + 0.001)
)

# 添加神经元类型和聚类
df_insula_stats <- df_insula_stats %>%
    left_join(df_summary %>% select(NeuronID, Neuron_Type), by = "NeuronID") %>%
    left_join(df_clustered %>% select(NeuronID, Morph_Cluster), by = "NeuronID")

# 3.3 统计
cat("=== 脑岛内投射统计 ===\n")
cat("  平均脑岛投射比例:", mean(df_insula_stats$Insula_Proportion), "\n")
cat("  中位数脑岛投射比例:", median(df_insula_stats$Insula_Proportion), "\n")
cat("\n")

# 3.4 按类型分组的脑岛投射
cat("=== 按神经元类型的脑岛投射 ===\n")
insula_by_type <- df_insula_stats %>%
    group_by(Neuron_Type) %>%
    summarise(
        n = n(),
        mean_insula_prop = mean(Insula_Proportion),
        median_insula_prop = median(Insula_Proportion),
        sd_insula_prop = sd(Insula_Proportion)
    )
print(insula_by_type)
cat("\n")

# 3.5 可视化 - 脑岛投射比例分布
p1 <- ggplot(df_insula_stats, aes(x = Insula_Proportion, fill = Neuron_Type)) +
    geom_histogram(bins = 20, alpha = 0.7) +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Proportion of Projections to Insula",
         subtitle = "Soma: CR_Ial (Right Insula Ial)",
         x = "Proportion of Total Axon Length",
         y = "Count") +
    theme_minimal()

ggsave(file.path(output_dir, "01_insula_proportion_distribution.pdf"), 
       p1, width = 8, height = 5)
cat("✅ Saved: 01_insula_proportion_distribution.pdf\n")

# 3.6 按类型分组的箱线图
p2 <- ggplot(df_insula_stats, aes(x = Neuron_Type, y = Insula_Proportion, 
                                   fill = Neuron_Type)) +
    geom_boxplot() +
    geom_jitter(width = 0.2, alpha = 0.3) +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Insula Projection Proportion by Neuron Type",
         x = "Neuron Type",
         y = "Proportion") +
    theme_minimal() +
    theme(legend.position = "none")

ggsave(file.path(output_dir, "02_insula_proportion_by_type.pdf"), 
       p2, width = 7, height = 5)
cat("✅ Saved: 02_insula_proportion_by_type.pdf\n")

# =============================================================================
# 第四部分: 脑岛内连接热图
# =============================================================================

cat("\n🔥 脑岛内连接热图...\n\n")

# 合并聚类信息
df_merged <- df_proj_length %>%
    left_join(df_clustered %>% select(NeuronID, Morph_Cluster), by = "NeuronID")

# 确保顺序一致
df_merged <- df_merged[match(rownames(insula_proj_log), df_merged$NeuronID), ]

# 定义颜色
type_colors <- c("ITs" = "#4DAF4A", "ITi" = "#984EA3", 
                 "ITc" = "#377EB8", "CT" = "#FF7F00")

cluster_colors <- c(
    "1" = "#E41A1C", "2" = "#377EB8", "3" = "#4DAF4A",
    "4" = "#984EA3", "5" = "#FF7F00", "6" = "#FFFF33",
    "7" = "#A65628", "8" = "#F781BF", "9" = "#999999"
)

# 顶部注释
ha_top <- HeatmapAnnotation(
    Cluster = as.character(df_merged$Morph_Cluster),
    Type = df_merged$Neuron_Type,
    col = list(
        Cluster = cluster_colors,
        Type = type_colors
    ),
    annotation_name_side = "left"
)

# 创建热图
col_fun <- colorRamp2(c(0, 1, 2, 3, 4), 
                      c("#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"))

ht <- Heatmap(
    insula_proj_log,
    name = "Log Length",
    col = col_fun,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    row_split = df_merged$Morph_Cluster,
    row_gap = unit(2, "mm"),
    top_annotation = ha_top,
    row_names_gp = gpar(fontsize = 4),
    column_names_gp = gpar(fontsize = 6),
    column_names_rot = 45,
    show_row_names = FALSE,
    column_title = "Insula Target Regions",
    row_title = "Source Neurons (CR_Ial)",
    width = unit(12, "cm"),
    height = unit(16, "cm")
)

pdf(file.path(output_dir, "03_insula_connections_heatmap.pdf"), 
    width = 14, height = 16)
draw(ht)
dev.off()

cat("✅ Saved: 03_insula_connections_heatmap.pdf\n")

# =============================================================================
# 第五部分: 脑岛亚区投射分析
# =============================================================================

cat("\n📊 脑岛亚区投射分析...\n\n")

# 5.1 识别主要脑岛亚区
insula_subregions <- c("FI", "Iai", "Ial", "Iam", "Iapm", "Iapl", "Id", "Ig")

# 5.2 计算每个亚区的投射
subregion_proj <- data.frame(NeuronID = rownames(insula_proj))

for (sub in insula_subregions) {
    # 匹配包含该亚区的列
    cols <- grep(sub, colnames(insula_proj), value = TRUE, ignore.case = TRUE)
    if (length(cols) > 0) {
        subregion_proj[[sub]] <- rowSums(insula_proj[, cols, drop = FALSE])
    } else {
        subregion_proj[[sub]] <- 0
    }
}

# 合并类型信息
subregion_proj <- subregion_proj %>%
    left_join(df_summary %>% select(NeuronID, Neuron_Type), by = "NeuronID")

# 5.3 转换为长格式用于绘图
subregion_long <- subregion_proj %>%
    pivot_longer(cols = all_of(insula_subregions),
                 names_to = "Subregion",
                 values_to = "Length")

# 5.4 可视化 - 各亚区投射分布
p3 <- ggplot(subregion_long, aes(x = Subregion, y = Length, fill = Subregion)) +
    geom_boxplot() +
    facet_wrap(~Neuron_Type, scales = "free_y") +
    labs(title = "Projection to Insula Subregions",
         subtitle = "Source: CR_Ial neurons",
         x = "Insula Subregion",
         y = "Total Axon Length (μm)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")

ggsave(file.path(output_dir, "04_subregion_projections.pdf"), 
       p3, width = 12, height = 8)
cat("✅ Saved: 04_subregion_projections.pdf\n")

# 5.5 热图 - 亚区投射
subregion_matrix <- subregion_proj %>%
    select(-NeuronID, -Neuron_Type) %>%
    as.matrix()
rownames(subregion_matrix) <- subregion_proj$NeuronID
subregion_log <- log1p(subregion_matrix)

# 添加聚类信息
subregion_merged <- subregion_proj %>%
    left_join(df_clustered %>% select(NeuronID, Morph_Cluster), by = "NeuronID")

ha_top_sub <- HeatmapAnnotation(
    Cluster = as.character(subregion_merged$Morph_Cluster),
    Type = subregion_merged$Neuron_Type,
    col = list(Cluster = cluster_colors, Type = type_colors)
)

ht_sub <- Heatmap(
    subregion_log,
    name = "Log Length",
    col = col_fun,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    row_split = subregion_merged$Morph_Cluster,
    row_gap = unit(2, "mm"),
    top_annotation = ha_top_sub,
    row_names_gp = gpar(fontsize = 4),
    column_names_gp = gpar(fontsize = 10),
    column_title = "Insula Subregions",
    row_title = "Source Neurons",
    width = unit(10, "cm"),
    height = unit(16, "cm")
)

pdf(file.path(output_dir, "05_subregion_heatmap.pdf"), width = 12, height = 16)
draw(ht_sub)
dev.off()

cat("✅ Saved: 05_subregion_heatmap.pdf\n")

# =============================================================================
# 第六部分: 同侧vs对侧脑岛投射
# =============================================================================

cat("\n🧠 同侧vs对侧脑岛投射...\n\n")

# 6.1 提取左右脑岛投射
left_insula_proj <- proj_matrix[, left_insula_cols, drop = FALSE]
right_insula_proj <- proj_matrix[, right_insula_cols, drop = FALSE]

# 6.2 计算统计
df_lateral <- data.frame(
    NeuronID = rownames(proj_matrix),
    Left_Insula = rowSums(left_insula_proj),
    Right_Insula = rowSums(right_insula_proj),
    Total_Insula = rowSums(insula_proj)
)

df_lateral$Left_Proportion <- df_lateral$Left_Insula / (df_lateral$Total_Insula + 0.001)
df_lateral$Right_Proportion <- df_lateral$Right_Insula / (df_lateral$Total_Insula + 0.001)

# 添加类型
df_lateral <- df_lateral %>%
    left_join(df_summary %>% select(NeuronID, Neuron_Type), by = "NeuronID")

# 6.3 可视化 - 左右脑岛投射对比
df_lateral_long <- df_lateral %>%
    select(NeuronID, Neuron_Type, Left = Left_Insula, Right = Right_Insula) %>%
    pivot_longer(cols = c(Left, Right),
                 names_to = "Hemisphere",
                 values_to = "Length")

p4 <- ggplot(df_lateral_long, aes(x = Hemisphere, y = Length, fill = Hemisphere)) +
    geom_boxplot() +
    geom_jitter(width = 0.2, alpha = 0.3) +
    facet_wrap(~Neuron_Type, scales = "free_y") +
    scale_fill_manual(values = c("Left" = "#80B1D3", "Right" = "#FDB462")) +
    labs(title = "Ipsilateral vs Contralateral Insula Projections",
         subtitle = "Source: CR_Ial (Right Insula) neurons",
         y = "Axon Length (μm)") +
    theme_minimal() +
    theme(legend.position = "none")

ggsave(file.path(output_dir, "06_left_vs_right_insula.pdf"), 
       p4, width = 10, height = 6)
cat("✅ Saved: 06_left_vs_right_insula.pdf\n")

# 6.4 散点图 - 左vs右脑岛投射
p5 <- ggplot(df_lateral, aes(x = Left_Insula, y = Right_Insula, color = Neuron_Type)) +
    geom_point(size = 3, alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
    scale_color_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Left vs Right Insula Projections",
         subtitle = "Source: CR_Ial neurons (dashed line = equal)",
         x = "Left Insula Projection Length (μm)",
         y = "Right Insula Projection Length (μm)") +
    theme_minimal()

ggsave(file.path(output_dir, "07_left_vs_right_scatter.pdf"), 
       p5, width = 8, height = 6)
cat("✅ Saved: 07_left_vs_right_scatter.pdf\n")

# =============================================================================
# 第七部分: 终端位点分析
# =============================================================================

cat("\n📍 终端位点分析...\n\n")

# 7.1 统计每个神经元的终端位点
terminal_stats <- df_terminals %>%
    group_by(NeuronID, Terminal_L1, Terminal_L3) %>%
    summarise(n_sites = n(), .groups = "drop")

# 7.2 合并类型信息
terminal_stats <- terminal_stats %>%
    left_join(df_summary %>% select(NeuronID, Neuron_Type), by = "NeuronID")

# 7.3 按L1分组的终端分布
p6 <- ggplot(terminal_stats, aes(x = Terminal_L1, fill = Neuron_Type)) +
    geom_bar() +
    scale_fill_manual(values = c(
        "ITs" = "#4DAF4A", "ITi" = "#984EA3", 
        "ITc" = "#377EB8", "CT" = "#FF7F00"
    )) +
    labs(title = "Terminal Site Distribution by Brain Lobe",
         x = "Brain Lobe (L1)",
         y = "Number of Terminal Sites") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(output_dir, "08_terminal_distribution_l1.pdf"), 
       p6, width = 10, height = 6)
cat("✅ Saved: 08_terminal_distribution_l1.pdf\n")

# 7.4 脑岛内终端分布
insula_terminals <- df_terminals %>%
    filter(grepl("I[agl]|FI|Iai|Ial|Iam|Iapm|Iapl|Id|Ig", Terminal_L6, 
                 ignore.case = TRUE))

cat("  脑岛内终端位点:", nrow(insula_terminals), "/", nrow(df_terminals), "\n")

# =============================================================================
# 第八部分: 保存结果
# =============================================================================

cat("\n💾 保存统计结果...\n\n")

# 保存脑岛投射统计
write.csv(df_insula_stats, 
          file.path(output_dir, "insula_projection_stats.csv"), 
          row.names = FALSE)

# 保存亚区投射
write.csv(subregion_proj, 
          file.path(output_dir, "subregion_projections.csv"), 
          row.names = FALSE)

# 保存左右对比
write.csv(df_lateral, 
          file.path(output_dir, "lateral_insula_projections.csv"), 
          row.names = FALSE)

cat("✅ Saved: insula_projection_stats.csv\n")
cat("✅ Saved: subregion_projections.csv\n")
cat("✅ Saved: lateral_insula_projections.csv\n")

# =============================================================================
# 完成
# =============================================================================

cat("\n" ,rep("=", 70), "\n", sep="")
cat("  ✅ 脑岛特异性分析完成!\n")
cat("  输出目录:", output_dir, "\n")
cat("  生成文件:\n")
cat("    - 8个PDF图表\n")
cat("    - 3个CSV统计文件\n")
cat(rep("=", 70), "\n", sep="")
