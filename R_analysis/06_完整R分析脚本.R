# ============================================================================
# 脑岛投射分析完整R脚本
# 基于 Henry Evrard 2012/2014 脑岛分区
# 
# 使用方法:
# 1. 准备Excel数据文件 (参考 05_示例数据模板.xlsx)
# 2. 修改 file_path 变量为你的文件路径
# 3. 运行: source("06_完整R分析脚本.R")
# ============================================================================

# =============================================================================
# 第一部分：环境设置
# =============================================================================

cat("=" ,rep("=", 60), "\n", sep="")
cat("  脑岛投射分析 - R脚本\n")
cat("  基于 Henry Evrard 2012/2014 脑岛分区\n")
cat("=" ,rep("=", 60), "\n\n", sep="")

# 安装和加载包
cat("📦 检查并安装必要的包...\n")

packages <- c("ComplexHeatmap", "circlize", "RColorBrewer", "dendextend",
              "readxl", "dplyr", "tidyr", "ggplot2", "grid")

for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat("  安装包:", pkg, "\n")
        install.packages(pkg, repos = "https://cloud.r-project.org/")
        library(pkg, character.only = TRUE)
    }
}

if (!require("ComplexHeatmap", quietly = TRUE)) {
    if (!require("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
    }
    BiocManager::install("ComplexHeatmap")
    library(ComplexHeatmap)
}

cat("✅ 所有包加载完成\n\n")

# =============================================================================
# 第二部分：颜色方案
# =============================================================================

cat("🎨 设置颜色方案...\n")

# 脑岛亚区颜色（基于Evrard 2014）
INSULA_COLORS <- c(
    "FI" = "#E41A1C",      # 红色 - VENs最密集
    "Iai" = "#FF7F00",     # 橙色 - VENs存在
    "Ial" = "#FFC020",     # 黄色
    "Iam" = "#4DAF4A",     # 绿色
    "Iapm" = "#377EB8",    # 蓝色
    "Iapl" = "#984EA3",    # 紫色
    "Id" = "#999999",      # 灰色 - dysgranular
    "Ig" = "#666666"       # 深灰 - granular
)

# 细胞构筑颜色
CYTO_COLORS <- c(
    "Granular" = "#08306B",
    "Dysgranular" = "#6BAED6", 
    "Agranular" = "#C6DBEF"
)

# 半球颜色
HEMISPHERE_COLORS <- c(
    "Left" = "#FDB462",
    "Right" = "#80B1D3"
)

# 神经元类型颜色
NEURON_TYPE_COLORS <- c(
    "IT" = "#4DAF4A",
    "PT" = "#E41A1C",
    "CT" = "#FF7F00",
    "ITc" = "#984EA3",
    "ITs" = "#377EB8"
)

# 创建投射强度颜色函数
create_projection_col_fun <- function(mat) {
    colorRamp2(
        c(0, quantile(mat, 0.25, na.rm = TRUE), 
          median(mat, na.rm = TRUE),
          quantile(mat, 0.75, na.rm = TRUE), 
          max(mat, na.rm = TRUE)),
        c("#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
    )
}

cat("✅ 颜色方案设置完成\n\n")

# =============================================================================
# 第三部分：数据读取函数
# =============================================================================

read_insula_data <- function(file_path) {
    #' 读取脑岛投射数据
    
    cat("📖 读取数据:", file_path, "\n")
    
    if (!file.exists(file_path)) {
        stop("文件不存在: ", file_path)
    }
    
    # 读取各sheet
    df_projection <- read_excel(file_path, sheet = "Projection_Data")
    df_metadata <- read_excel(file_path, sheet = "Metadata")
    df_regions <- read_excel(file_path, sheet = "Region_Info")
    
    # 设置行名
    id_col <- colnames(df_projection)[1]
    df_projection <- df_projection %>% 
        column_to_rownames(var = id_col)
    
    # 识别脑岛相关列
    insula_cols <- df_regions %>% 
        filter(Is_Insula == TRUE) %>% 
        pull(Region_Name)
    
    # 检查数据
    cat("  ✓ 投射矩阵:", nrow(df_projection), "神经元 x", ncol(df_projection), "脑区\n")
    cat("  ✓ 元数据:", nrow(df_metadata), "行\n")
    cat("  ✓ 脑岛区域:", length(insula_cols), "个\n")
    cat("    ", paste(head(insula_cols, 5), collapse = ", "), "...\n")
    
    return(list(
        projection = df_projection,
        metadata = df_metadata,
        regions = df_regions,
        insula_cols = insula_cols,
        id_col = id_col
    ))
}

# =============================================================================
# 第四部分：质量检查函数
# =============================================================================

quality_check <- function(data_list) {
    #' 数据质量检查
    
    cat("\n📋 数据质量检查\n")
    cat(rep("-", 40), "\n", sep="")
    
    errors <- 0
    warnings <- 0
    
    # 检查1: Neuron_ID匹配
    proj_ids <- rownames(data_list$projection)
    meta_ids <- data_list$metadata[[data_list$id_col]]
    
    if (!all(proj_ids %in% meta_ids)) {
        cat("❌ 错误: Projection中有ID不在Metadata中\n")
        cat("   ", setdiff(proj_ids, meta_ids)[1:5], "...\n")
        errors <- errors + 1
    } else {
        cat("✅ Neuron_ID匹配正常\n")
    }
    
    # 检查2: Region_Name匹配
    proj_regions <- colnames(data_list$projection)
    region_names <- data_list$regions$Region_Name
    
    if (!all(proj_regions %in% region_names)) {
        cat("❌ 错误: Projection中有列名不在Region_Info中\n")
        errors <- errors + 1
    } else {
        cat("✅ Region_Name匹配正常\n")
    }
    
    # 检查3: 数值范围
    mat <- as.matrix(data_list$projection)
    if (any(mat < 0, na.rm = TRUE)) {
        cat("❌ 错误: 存在负值\n")
        errors <- errors + 1
    } else {
        cat("✅ 无负值\n")
    }
    
    cat("  投射强度范围: [", round(min(mat, na.rm = TRUE), 2), ", ", 
        round(max(mat, na.rm = TRUE), 2), "]\n")
    cat("  零值比例: ", round(sum(mat == 0, na.rm = TRUE) / length(mat) * 100, 1), "%\n")
    
    # 检查4: 脑岛区域
    if (length(data_list$insula_cols) == 0) {
        cat("⚠️ 警告: 未识别到脑岛区域\n")
        warnings <- warnings + 1
    } else {
        cat("✅ 识别到", length(data_list$insula_cols), "个脑岛区域\n")
    }
    
    cat(rep("-", 40), "\n", sep="")
    cat("检查结果: ", errors, "个错误, ", warnings, "个警告\n\n")
    
    return(list(errors = errors, warnings = warnings))
}

# =============================================================================
# 第五部分：从脑岛出发的分析
# =============================================================================

get_insula_soma_neurons <- function(data_list) {
    #' 获取胞体在脑岛的神经元
    
    cat("🔍 筛选胞体在脑岛的神经元...\n")
    
    # 支持多种可能的列名
    region_col <- NULL
    for (col in c("Soma_Region", "SomaRegion", "Region")) {
        if (col %in% colnames(data_list$metadata)) {
            region_col <- col
            break
        }
    }
    
    if (is.null(region_col)) {
        cat("⚠️ 警告: 未找到Soma_Region列，使用所有神经元\n")
        insula_neurons <- data_list$metadata
    } else {
        insula_neurons <- data_list$metadata %>%
            filter(tolower(!!sym(region_col)) %in% c("insula", "ins"))
    }
    
    # 提取投射数据
    common_ids <- intersect(rownames(data_list$projection), 
                           insula_neurons[[data_list$id_col]])
    insula_projection <- data_list$projection[common_ids, ]
    
    cat("✓ 找到", nrow(insula_neurons), "个胞体在脑岛的神经元\n")
    
    return(list(
        metadata = insula_neurons,
        projection = insula_projection
    ))
}

plot_insula_soma_heatmap <- function(insula_data, data_list, output_prefix) {
    #' 绘制胞体在脑岛的神经元投射热图
    
    cat("📊 绘制脑岛神经元投射热图...\n")
    
    mat <- as.matrix(insula_data$projection)
    meta <- insula_data$metadata
    
    # 确保顺序一致
    meta <- meta[match(rownames(mat), meta[[data_list$id_col]]), ]
    
    # 获取subregion列
    subregion_col <- NULL
    for (col in c("Soma_Subregion", "Subregion", "Soma_SubRegion")) {
        if (col %in% colnames(meta)) {
            subregion_col <- col
            break
        }
    }
    
    # 获取hemisphere列
    hemi_col <- NULL
    for (col in c("Soma_Hemisphere", "Hemisphere", "Soma_Hemi")) {
        if (col %in% colnames(meta)) {
            hemi_col <- col
            break
        }
    }
    
    # 获取type列
    type_col <- NULL
    for (col in c("Neuron_Type", "Type", "CellType")) {
        if (col %in% colnames(meta)) {
            type_col <- col
            break
        }
    }
    
    # 创建注释参数
    anno_params <- list()
    anno_colors <- list()
    
    if (!is.null(subregion_col)) {
        anno_params[["Insula_Subregion"]] <- meta[[subregion_col]]
        anno_colors[["Insula_Subregion"]] <- INSULA_COLORS
    }
    if (!is.null(hemi_col)) {
        anno_params[["Hemisphere"]] <- meta[[hemi_col]]
        anno_colors[["Hemisphere"]] <- HEMISPHERE_COLORS
    }
    if (!is.null(type_col)) {
        anno_params[["Neuron_Type"]] <- meta[[type_col]]
        anno_colors[["Neuron_Type"]] <- NEURON_TYPE_COLORS
    }
    
    # 创建顶部注释
    ha_top <- do.call(HeatmapAnnotation, c(anno_params, list(
        col = anno_colors,
        annotation_name_side = "left",
        annotation_name_gp = gpar(fontsize = 9, fontface = "bold"),
        show_legend = rep(TRUE, length(anno_params))
    )))
    
    # 创建热图
    col_fun <- create_projection_col_fun(mat)
    
    ht <- Heatmap(
        mat,
        name = "Log Axon Length",
        col = col_fun,
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        row_dend_gp = gpar(col = "gray40", lwd = 0.6),
        column_dend_gp = gpar(col = "gray40", lwd = 0.6),
        row_names_side = "left",
        row_names_gp = gpar(fontsize = 7),
        column_names_side = "bottom",
        column_names_gp = gpar(fontsize = 7),
        column_names_rot = 45,
        top_annotation = ha_top,
        width = unit(16, "cm"),
        height = unit(max(8, nrow(mat) * 0.35), "cm"),
        column_title = "Target Brain Regions",
        column_title_gp = gpar(fontsize = 11, fontweight = "bold"),
        row_title = "Insula Neurons",
        row_title_gp = gpar(fontsize = 11, fontweight = "bold")
    )
    
    # 保存
    output_file <- paste0(output_prefix, "_01_InsulaSoma_Heatmap.pdf")
    pdf(output_file, width = 14, height = max(10, nrow(mat) * 0.2))
    draw(ht, merge_legend = TRUE)
    dev.off()
    
    cat("✅ 热图已保存:", output_file, "\n")
    return(ht)
}

# =============================================================================
# 第六部分：从Target Region出发的分析
# =============================================================================

get_neurons_projecting_to_insula <- function(data_list, threshold = 0) {
    #' 获取投射到脑岛的神经元
    
    cat("🔍 筛选投射到脑岛的神经元...\n")
    
    insula_cols <- data_list$insula_cols
    mat <- data_list$projection
    
    # 筛选有投射的神经元
    insula_mat <- mat[, insula_cols, drop = FALSE]
    has_projection <- rowSums(insula_mat > threshold) > 0
    
    projecting_ids <- rownames(mat)[has_projection]
    projecting_neurons <- data_list$metadata %>%
        filter(!!sym(data_list$id_col) %in% projecting_ids)
    
    cat("✓ 找到", nrow(projecting_neurons), "个投射到脑岛的神经元\n")
    
    return(list(
        metadata = projecting_neurons,
        projection = mat[has_projection, ],
        insula_projection = insula_mat[has_projection, ]
    ))
}

plot_projecting_to_insula_heatmap <- function(projecting_data, data_list, 
                                               output_prefix) {
    #' 绘制投射到脑岛的神经元热图
    
    cat("📊 绘制投射到脑岛的热图...\n")
    
    mat <- as.matrix(projecting_data$insula_projection)
    meta <- projecting_data$metadata
    
    # 确保顺序一致
    meta <- meta[match(rownames(mat), meta[[data_list$id_col]]), ]
    
    # 创建简单注释
    ha_top <- HeatmapAnnotation(
        Soma_Region = meta$Soma_Region,
        col = list(Soma_Region = c("Insula" = "#E41A1C", "PFC" = "#377EB8", 
                                   "ACC" = "#4DAF4A", "Parietal" = "#984EA3")),
        annotation_name_side = "left"
    )
    
    col_fun <- create_projection_col_fun(mat)
    
    ht <- Heatmap(
        mat,
        name = "Projection to Insula",
        col = col_fun,
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        row_names_gp = gpar(fontsize = 6),
        column_names_gp = gpar(fontsize = 8),
        column_names_rot = 45,
        top_annotation = ha_top,
        column_title = "Insula Subregions",
        row_title = "Source Neurons",
        width = unit(12, "cm"),
        height = unit(max(10, nrow(mat) * 0.25), "cm")
    )
    
    output_file <- paste0(output_prefix, "_02_ProjectingToInsula_Heatmap.pdf")
    pdf(output_file, width = 12, height = max(10, nrow(mat) * 0.15))
    draw(ht, merge_legend = TRUE)
    dev.off()
    
    cat("✅ 热图已保存:", output_file, "\n")
    return(ht)
}

# =============================================================================
# 第七部分：双侧对比分析
# =============================================================================

plot_bilateral_comparison <- function(data_list, output_prefix) {
    #' 绘制左右脑岛投射对比
    
    cat("📊 绘制双侧对比热图...\n")
    
    # 识别左右脑岛列
    left_cols <- grep("^Left_I", data_list$insula_cols, value = TRUE)
    right_cols <- grep("^Right_I", data_list$insula_cols, value = TRUE)
    
    if (length(left_cols) == 0 || length(right_cols) == 0) {
        cat("⚠️ 警告: 未找到左右脑岛列，跳过双侧对比\n")
        return(NULL)
    }
    
    # 获取胞体在脑岛的神经元
    insula_neurons <- data_list$metadata %>%
        filter(tolower(Soma_Region) == "insula")
    
    common_ids <- intersect(rownames(data_list$projection), 
                           insula_neurons[[data_list$id_col]])
    
    if (length(common_ids) == 0) {
        cat("⚠️ 警告: 无胞体在脑岛的神经元\n")
        return(NULL)
    }
    
    mat <- data_list$projection[common_ids, ]
    
    # 提取左右投射
    left_proj <- mat[, left_cols, drop = FALSE]
    right_proj <- mat[, right_cols, drop = FALSE]
    
    # 标准化列名
    colnames(left_proj) <- gsub("^Left_", "", colnames(left_proj))
    colnames(right_proj) <- gsub("^Right_", "", colnames(right_proj))
    
    # 共同区域
    common_regions <- intersect(colnames(left_proj), colnames(right_proj))
    
    if (length(common_regions) == 0) {
        cat("⚠️ 警告: 无共同脑岛区域\n")
        return(NULL)
    }
    
    left_proj <- left_proj[, common_regions, drop = FALSE]
    right_proj <- right_proj[, common_regions, drop = FALSE]
    
    col_fun <- create_projection_col_fun(rbind(left_proj, right_proj))
    
    # 左半球热图
    ht_left <- Heatmap(
        as.matrix(left_proj),
        name = "Left",
        col = col_fun,
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        row_names_gp = gpar(fontsize = 6),
        column_names_gp = gpar(fontsize = 8),
        column_title = "Left Hemisphere",
        width = unit(7, "cm")
    )
    
    # 右半球热图
    ht_right <- Heatmap(
        as.matrix(right_proj),
        name = "Right",
        col = col_fun,
        cluster_rows = TRUE,
        cluster_columns = TRUE,
        row_names_gp = gpar(fontsize = 6),
        column_names_gp = gpar(fontsize = 8),
        column_title = "Right Hemisphere",
        width = unit(7, "cm")
    )
    
    output_file <- paste0(output_prefix, "_03_Bilateral_Comparison.pdf")
    pdf(output_file, width = 16, height = 10)
    draw(ht_left + ht_right)
    dev.off()
    
    cat("✅ 双侧对比热图已保存:", output_file, "\n")
    return(list(left = left_proj, right = right_proj))
}

# =============================================================================
# 第八部分：主分析流程
# =============================================================================

run_insula_analysis <- function(file_path, output_prefix = "Insula_Analysis") {
    #' 运行完整的脑岛分析
    
    cat("\n" ,rep("=", 60), "\n", sep="")
    cat("  开始脑岛投射分析\n")
    cat(rep("=", 60), "\n\n", sep="")
    
    # 1. 读取数据
    cat("【步骤1/4】读取数据\n")
    data_list <- read_insula_data(file_path)
    
    # 2. 质量检查
    cat("\n【步骤2/4】质量检查\n")
    qc <- quality_check(data_list)
    
    if (qc$errors > 0) {
        cat("❌ 数据存在错误，请先修复数据\n")
        return(NULL)
    }
    
    # 3. 从脑岛出发的分析
    cat("\n【步骤3/4】从脑岛出发的分析\n")
    insula_soma <- get_insula_soma_neurons(data_list)
    if (nrow(insula_soma$metadata) > 0) {
        plot_insula_soma_heatmap(insula_soma, data_list, output_prefix)
    }
    
    # 4. 从Target出发的分析
    cat("\n【步骤4/4】从Target Region出发的分析\n")
    projecting <- get_neurons_projecting_to_insula(data_list)
    if (nrow(projecting$metadata) > 0) {
        plot_projecting_to_insula_heatmap(projecting, data_list, output_prefix)
    }
    
    # 5. 双侧对比
    cat("\n【附加分析】双侧对比\n")
    plot_bilateral_comparison(data_list, output_prefix)
    
    cat("\n" ,rep("=", 60), "\n", sep="")
    cat("  ✅ 分析完成!\n")
    cat("  输出文件前缀:", output_prefix, "\n")
    cat(rep("=", 60), "\n", sep="")
}

# =============================================================================
# 第九部分：运行分析
# =============================================================================

# 修改这里为你的数据文件路径
file_path <- "05_示例数据模板.xlsx"

# 运行分析
run_insula_analysis(file_path, "Insula_Analysis_Output")
