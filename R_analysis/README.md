# 神经投射数据热图绘制方案

> 基于 Gou et al., Cell 2025 - Figure 2a 的绘图实现

---

## 📋 项目概述

本项目提供完整的R语言和Python绘图方案，帮助你实现类似Cell文章中Figure 2a的神经投射数据热图。

### 图2a特点
- ✅ 32个神经元亚型的投射强度热图
- ✅ 顶部Type分类注释条（ITc/ITs/CT/PT/ITi）
- ✅ 层次聚类树状图
- ✅ 脑区分组（Ipsilateral/Contralateral）
- ✅ log轴突长度表示投射强度

---

## 📁 文件清单

| 文件 | 说明 |
|------|------|
| `neuron_projection_data.xlsx` | 模拟数据文件（参考格式） |
| `plot_figure2a.R` | R基础热图脚本 |
| `advanced_visualization.R` | R高级可视化脚本 |
| `plot_figure2a.py` | Python热图脚本 |
| `演示Notebook.ipynb` | Jupyter Notebook演示 |
| `快速入门指南.md` | 5分钟快速上手 |
| `R_热图绘制教程.md` | 完整R语言教程 |
| `文件清单和使用说明.md` | 详细使用说明 |

---

## 🚀 快速开始

### 方式1：R语言 (推荐用于发表)

```bash
# 1. 安装R和RStudio
# 下载: https://posit.co/download/rstudio-desktop/

# 2. 打开RStudio，运行:
source("plot_figure2a.R")
```

### 方式2：Python (无需R环境)

```bash
# 1. 安装依赖
pip install pandas matplotlib seaborn scipy openpyxl

# 2. 运行脚本
python plot_figure2a.py
```

### 方式3：Jupyter Notebook

```bash
jupyter notebook 演示Notebook.ipynb
```

---

## 📊 数据格式要求

### Excel文件结构

**Sheet 1: Projection_Data**
```
        ACC   MCC   dlPFC  vlPFC  ...
Sub1    2.5   1.2   3.4    0.8
Sub2    0.3   2.1   1.5    3.2
...
```

**Sheet 2: Metadata**
```
Subtype  Type  Soma_Region  Total_Axon_Length  N_Neurons
Sub1     ITc   Area_45      15000              12
Sub2     ITs   Area_46      23000              18
...
```

**Sheet 3: Region_Info**
```
Region  Hemisphere     Category
ACC     Ipsilateral    Cortical
Thal    Ipsilateral    Subcortical
...
```

---

## 🎨 输出示例

运行后会生成：
- `Figure2a_Main_Heatmap.pdf` - 主热图（高质量矢量图）
- `Figure2a_Main_Heatmap.png` - PNG预览
- `Figure2a_Type_XXX_Heatmap.pdf` - 各Type子热图
- `Figure2a_PCA.pdf` - PCA降维图（仅R高级版）

---

## 📖 学习路径

### 新手路线
1. 阅读 `快速入门指南.md`
2. 查看模拟数据结构
3. 运行基础脚本
4. 修改参数探索效果

### 进阶路线
1. 阅读 `R_热图绘制教程.md`
2. 运行 `advanced_visualization.R`
3. 自定义颜色和布局
4. 添加自己的注释信息

---

## 🔧 自定义选项

### 修改颜色方案

**R语言:**
```r
type_colors <- c(
  "ITc" = "#4DAF4A",
  "ITs" = "#984EA3",
  "CT"  = "#FF7F00",
  "PT"  = "#E41A1C",
  "ITi" = "#377EB8"
)
```

**Python:**
```python
type_colors = {
    'ITc': '#4DAF4A',
    'ITs': '#984EA3',
    'CT': '#FF7F00',
    'PT': '#E41A1C',
    'ITi': '#377EB8'
}
```

### 调整热图大小

**R语言:**
```r
pdf("output.pdf", width = 16, height = 12)
```

**Python:**
```python
plt.figure(figsize=(18, 14))
```

---

## ❓ 常见问题

**Q: 应该选择R还是Python？**
- 发表高质量论文图 → R (ComplexHeatmap)
- 快速查看/已有Python流程 → Python

**Q: 安装ComplexHeatmap失败？**
```r
install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")
```

**Q: 中文显示乱码？**
```r
# R
par(family = "SimHei")

# Python
plt.rcParams['font.sans-serif'] = ['SimHei']
```

---

## 📚 参考资源

- **ComplexHeatmap文档**: https://jokergoo.github.io/ComplexHeatmap-reference/book/
- **原文献**: Gou et al., Cell 2025
- **R for Data Science**: https://r4ds.had.co.nz/

---

## 📝 引用

如果你使用了本代码，请引用：
- Gou et al., Single-neuron projectomes of macaque prefrontal cortex reveal refined axon targeting and arborization, Cell 2025

---

## 📧 联系方式

如有问题或建议，欢迎交流！

---

**祝你绘图顺利！🎉**
