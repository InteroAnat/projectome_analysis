# R语言神经科学热图绘制入门教程
# 基于Gou et al., Cell 2025 Figure 2a

---

## 📚 目录

1. [R语言基础](#1-r语言基础)
2. [热图绘制核心包介绍](#2-热图绘制核心包介绍)
3. [数据准备](#3-数据准备)
4. [完整代码解析](#4-完整代码解析)
5. [进阶技巧](#5-进阶技巧)
6. [常见问题](#6-常见问题)

---

## 1. R语言基础

### 1.1 安装R和RStudio

**Windows/Mac:**
1. 访问 https://cran.r-project.org/ 下载R
2. 访问 https://posit.co/download/rstudio-desktop/ 下载RStudio

**Linux (Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install r-base
```

### 1.2 R基础语法

```r
# 变量赋值
x <- 10        # 推荐用法
y = 20         # 也可用

# 向量
vec <- c(1, 2, 3, 4, 5)
vec2 <- 1:10

# 矩阵
mat <- matrix(1:12, nrow = 3, ncol = 4)

# 数据框 (类似Excel表格)
df <- data.frame(
  name = c("A", "B", "C"),
  value = c(10, 20, 30)
)

# 查看数据
head(df)       # 前6行
str(df)        # 数据结构
summary(df)    # 统计摘要
```

### 1.3 安装包

```r
# 从CRAN安装
install.packages("ComplexHeatmap")

# 从Bioconductor安装
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")

# 加载包
library(ComplexHeatmap)
```

---

## 2. 热图绘制核心包介绍

### 2.1 ComplexHeatmap (推荐!)

这是目前R语言中最强大的热图包，特别适合生物学数据。

**特点：**
- ✅ 支持多热图拼接
- ✅ 丰富的注释系统
- ✅ 灵活的聚类控制
- ✅ 高质量输出

**核心函数：**
- `Heatmap()` - 创建热图
- `HeatmapAnnotation()` - 创建注释条
- `draw()` - 绘制热图

### 2.2 pheatmap (简单替代)

```r
library(pheatmap)
pheatmap(matrix, 
         annotation_col = df_annotation,
         cluster_rows = TRUE,
         cluster_cols = TRUE)
```

### 2.3 ggplot2 + geom_tile (高度自定义)

```r
library(ggplot2)
ggplot(df_long, aes(x = Region, y = Subtype, fill = Value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue")
```

---

## 3. 数据准备

### 3.1 你的Excel应该包含什么？

**Sheet 1: Projection_Data (投射数据)**
```
        ACC    MCC    dlPFC   vlPFC   ...
Sub1    2.5    1.2    3.4     0.8
Sub2    0.3    2.1    1.5     3.2
Sub3    1.8    0.9    2.2     1.1
...
```

**Sheet 2: Metadata (元数据)**
```
Subtype    Type    Soma_Region    Total_Axon_Length    N_Neurons
Sub1       ITc     Area_45        15000                12
Sub2       ITs     Area_46        23000                18
Sub3       PT      Area_8A        45000                25
...
```

**Sheet 3: Region_Info (脑区信息)**
```
Region      Hemisphere      Category
ACC         Ipsilateral     Cortical
MCC         Ipsilateral     Cortical
Thal        Ipsilateral     Subcortical
...
```

### 3.2 数据读取代码

```r
library(readxl)

# 读取Excel
df_proj <- read_excel("your_data.xlsx", sheet = "Projection_Data")
df_meta <- read_excel("your_data.xlsx", sheet = "Metadata")
df_region <- read_excel("your_data.xlsx", sheet = "Region_Info")

# 转换为矩阵
projection_matrix <- as.matrix(df_proj[, -1])  # 去掉第一列(Subtype名称)
rownames(projection_matrix) <- df_proj$Subtype
```

---

## 4. 完整代码解析

### 4.1 最简单的热图 (5行代码)

```r
library(ComplexHeatmap)

# 读取数据
data <- read.csv("your_data.csv", row.names = 1)

# 绘制热图
Heatmap(data, 
        name = "Projection",
        cluster_rows = TRUE,
        cluster_columns = TRUE)
```

### 4.2 添加颜色映射

```r
library(circlize)

# 创建颜色函数
col_fun <- colorRamp2(
  c(0, 2, 5),                    # 断点值
  c("white", "yellow", "red")    # 对应颜色
)

Heatmap(data, col = col_fun)
```

### 4.3 添加顶部注释条

```r
# 创建注释数据
annotation_df <- data.frame(
  Type = c("ITc", "ITc", "ITs", "PT", "CT"),
  Soma = c("Area45", "Area46", "Area45", "Area8A", "Area9")
)

# 定义颜色
type_colors <- c("ITc" = "green", "ITs" = "purple", 
                 "PT" = "red", "CT" = "orange")

# 创建注释
ha <- HeatmapAnnotation(
  Type = annotation_df$Type,
  col = list(Type = type_colors)
)

# 绘制
Heatmap(data, top_annotation = ha)
```

### 4.4 完整的Figure 2a风格热图

```r
# 详见 plot_figure2a.R 文件
# 这里展示核心结构

library(ComplexHeatmap)
library(circlize)
library(RColorBrewer)

# 1. 读取数据
# ...

# 2. 创建颜色方案
col_fun <- colorRamp2(
  quantile(data, c(0, 0.25, 0.5, 0.75, 1)),
  c("white", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
)

# 3. 创建顶部注释
ha_top <- HeatmapAnnotation(
  Type = metadata$Type,
  Soma = metadata$Soma_Region,
  N_Neurons = anno_barplot(metadata$N_Neurons),
  col = list(Type = type_colors, Soma = soma_colors)
)

# 4. 创建热图
ht <- Heatmap(
  data,
  name = "Log Axon Length",
  col = col_fun,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  top_annotation = ha_top,
  row_names_side = "left",
  column_names_rot = 45
)

# 5. 绘制
draw(ht)
```

---

## 5. 进阶技巧

### 5.1 自定义聚类

```r
# 使用不同的聚类方法
dist_matrix <- dist(data, method = "euclidean")  # 距离计算
hc <- hclust(dist_matrix, method = "ward.D2")    # 层次聚类

# 应用到热图
Heatmap(data, cluster_rows = hc)
```

### 5.2 分割热图

```r
# 按Type分割
Heatmap(data, 
        row_split = metadata$Type,
        row_gap = unit(2, "mm"))
```

### 5.3 添加行/列标签

```r
Heatmap(data,
        row_labels = paste0("#", 1:nrow(data)),
        column_labels = colnames(data),
        row_names_gp = gpar(fontsize = 8),
        column_names_gp = gpar(fontsize = 8))
```

### 5.4 多热图拼接

```r
ht1 <- Heatmap(data1, name = "Data1")
ht2 <- Heatmap(data2, name = "Data2")

# 垂直拼接
ht1 %v% ht2

# 水平拼接
ht1 + ht2
```

### 5.5 保存高质量图片

```r
# PDF (矢量图，适合论文)
pdf("figure.pdf", width = 10, height = 8)
draw(ht)
dev.off()

# PNG (位图，适合PPT)
png("figure.png", width = 2000, height = 1600, res = 150)
draw(ht)
dev.off()

# TIFF (高分辨率，适合印刷)
tiff("figure.tiff", width = 3000, height = 2400, res = 300)
draw(ht)
dev.off()
```

---

## 6. 常见问题

### Q1: 安装ComplexHeatmap失败？

```r
# 方法1: 使用BiocManager
install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")

# 方法2: 从GitHub安装
install.packages("remotes")
remotes::install_github("jokergoo/ComplexHeatmap")
```

### Q2: 中文显示乱码？

```r
# Windows
par(family = "SimHei")

# Mac
par(family = "STHeiti")

# 或者在Heatmap中指定
Heatmap(data, 
        row_names_gp = gpar(fontfamily = "SimHei"))
```

### Q3: 数据太大导致内存不足？

```r
# 1. 使用稀疏矩阵
library(Matrix)
data_sparse <- as(data, "sparseMatrix")

# 2. 降采样
set.seed(42)
sample_idx <- sample(nrow(data), 1000)
data_sample <- data[sample_idx, ]

# 3. 分块处理
Heatmap(data, 
        show_row_names = FALSE,  # 不显示行名
        row_dend_side = "left")   # 树状图放左边
```

### Q4: 如何调整颜色范围？

```r
# 方法1: 使用分位数
quantiles <- quantile(data, c(0, 0.25, 0.5, 0.75, 1))
col_fun <- colorRamp2(quantiles, c("white", "blue", "green", "yellow", "red"))

# 方法2: 手动指定范围
col_fun <- colorRamp2(c(0, 5, 10), c("blue", "white", "red"))
```

### Q5: 如何添加统计显著性标记？

```r
# 创建标记矩阵
mark_matrix <- matrix("", nrow = nrow(data), ncol = ncol(data))
mark_matrix[p_value < 0.05] <- "*"
mark_matrix[p_value < 0.01] <- "**"

# 添加到热图
Heatmap(data,
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.text(mark_matrix[i, j], x, y)
        })
```

---

## 📖 推荐学习资源

1. **ComplexHeatmap官方文档**
   https://jokergoo.github.io/ComplexHeatmap-reference/book/

2. **R for Data Science (免费在线书)**
   https://r4ds.had.co.nz/

3. **ggplot2官方文档**
   https://ggplot2.tidyverse.org/

4. **Bioconductor教程**
   https://www.bioconductor.org/packages/release/bioc/vignettes/ComplexHeatmap/inst/doc/ComplexHeatmap.html

---

## 🎯 实践练习

### 练习1: 简单热图
```r
# 使用内置数据集
data(mtcars)
Heatmap(as.matrix(mtcars))
```

### 练习2: 添加注释
```r
# 为mtcars添加注释
annotation <- data.frame(
  cyl = as.factor(mtcars$cyl),
  gear = as.factor(mtcars$gear)
)

Heatmap(as.matrix(mtcars[, -c(2, 10, 11)]), 
        top_annotation = HeatmapAnnotation(df = annotation))
```

### 练习3: 使用你的数据
```r
# 替换为你的实际数据路径
your_data <- read_excel("your_data.xlsx")
# ... 应用本教程的代码
```

---

**祝你学习愉快！有任何问题欢迎交流。**
