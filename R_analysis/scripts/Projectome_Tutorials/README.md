# 神经投射组R语言教程系列

本文件夹包含系统学习R语言和神经投射组数据分析的完整教程。

## 学习路径 (建议按顺序)

### 基础系列 (Basics)

| 序号 | 文件名 | 内容 | 学习目标 |
|------|--------|------|----------|
| 1 | `01_R_Basics_Tutorial.Rmd` | R语言基础 | 变量、向量、矩阵、数据框、控制结构 |
| 2 | `02_Data_Manipulation_Tutorial.Rmd` | dplyr数据操作 | select, filter, mutate, group_by, 管道操作 |
| 3 | `03_ggplot2_Visualization.Rmd` | ggplot2可视化 | 散点图、柱状图、箱线图、主题定制 |
| 4 | `04_PCA_Dimensionality_Reduction.Rmd` | 主成分分析 | PCA、t-SNE、UMAP降维可视化 |
| 5 | `05_Statistical_Tests.Rmd` | 统计检验 | t检验、ANOVA、相关性、多重比较校正 |

### 进阶系列 (Advanced)

| 文件名 | 内容 | 学习目标 |
|--------|------|----------|
| `Novel_Projectome_Analysis_Tutorial.Rmd` | 创新分析方法 | Shannon熵、网络分析、随机森林、非线性降维 |
| `Projectome_Analysis_Tutorial.Rmd` | 完整分析流程 | 从数据读取到结果导出的完整工作流 |

## 使用方法

### 在RStudio中:

1. 打开任意 `.Rmd` 文件
2. 点击 "Knit" 按钮生成HTML/PDF报告
3. 或逐块运行代码学习

### 命令行:

```bash
# 渲染为HTML
Rscript -e "rmarkdown::render('01_R_Basics_Tutorial.Rmd')"

# 渲染为PDF
Rscript -e "rmarkdown::render('01_R_Basics_Tutorial.Rmd', output_format='pdf_document')"
```

## 前置要求

需要安装以下R包:

```r
install.packages(c(
  "readxl", "dplyr", "tidyr", "ggplot2",
  "ComplexHeatmap", "circlize", "pheatmap",
  "randomForest", "vegan", "igraph", "cluster",
  "factoextra", "car", "rstatix", "effectsize",
  "GGally", "corrplot", "cowplot", "viridis"
))
```

## 数据来源

教程使用 251637 号猕猴脑岛神经元投射组数据:
- 212个神经元
- 胞体位置: CR_Ial (右岛叶Ial区)
- 神经元类型: ITi, ITs, CT, PT, ITc
- 260+ 脑区投射数据

## 学习建议

1. **按顺序学习**: 基础系列(1-5) → 进阶分析
2. **动手实践**: 修改参数，观察结果变化
3. **应用到自己的数据**: 替换数据路径，重复分析
4. **查阅文档**: 使用 `?function_name` 查看帮助

## 联系与反馈

如有问题或建议，欢迎讨论！
