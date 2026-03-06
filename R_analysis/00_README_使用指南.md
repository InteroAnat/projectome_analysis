# 00_README_使用指南

> 脑岛投射分析完整方案 - 基于Henry Evrard 2012/2014分区

---

## 📚 文档列表

| 序号 | 文件名 | 说明 |
|-----|--------|------|
| 00 | `00_README_使用指南.md` | 本文件，总体使用指南 |
| 01 | `01_R语言热图绘制完整教程.md` | R语言热图绘制原理和技巧 |
| 02 | `02_脑岛投射分析方案.md` | 脑岛分析的完整方案设计 |
| 03 | `03_R代码实现_脑岛分析.md` | R代码详细实现 |
| 04 | `04_数据准备与质量控制.md` | Excel数据准备和质量检查 |
| 05 | `05_示例数据模板.xlsx` | 示例数据文件 |
| 06 | `06_完整R分析脚本.R` | 可直接运行的R脚本 |

---

## 🚀 快速开始

### 方式1：使用示例数据（推荐新手）

```bash
# 1. 确保已安装R和RStudio
# 2. 打开 06_完整R分析脚本.R
# 3. 修改文件路径（默认使用示例数据）
file_path <- "05_示例数据模板.xlsx"
# 4. 运行脚本
```

### 方式2：使用自己的数据

```bash
# 1. 阅读 04_数据准备与质量控制.md
# 2. 按照规范准备Excel文件
# 3. 修改 06_完整R分析脚本.R 中的文件路径
file_path <- "你的数据文件.xlsx"
# 4. 运行脚本
```

---

## 📊 分析内容

### 1. 从脑岛出发的分析
- ✅ 胞体在脑岛的神经元投射模式
- ✅ 按细胞构筑（Ig/Id/Ia）分组
- ✅ 按脑岛亚区（FI/Iai/Ial等）分组

### 2. 从Target Region出发的分析
- ✅ 投射到脑岛的神经元来源
- ✅ 投射模式聚类分析
- ✅ 来源区域分布

### 3. 双侧对比分析
- ✅ 左右脑岛投射对比
- ✅ 不对称性指数计算
- ✅ 并排热图展示

---

## 🧠 脑岛分区（Henry Evrard 2012/2014）

### 细胞构筑分区

| 分区 | 缩写 | Layer IV | VENs | 功能 |
|-----|------|----------|------|------|
| Granular | Ig | 清晰 | 无 | 初级内感受 |
| Dysgranular | Id | 不规则 | 无 | 多模态整合 |
| Agranular | Ia | 缺失 | 有 | 高级整合 |

### 亚区命名

| 亚区 | 全称 | VEN密度 | 位置 |
|-----|------|---------|------|
| FI | Frontoinsular cortex | 高 | 前岛叶最前端 |
| Iai | Insula agranular anterior, intermediate | 中 | 前岛叶腹侧 |
| Ial | Insula agranular anterior, lateral | 低 | 前岛叶外侧 |
| Iam | Insula agranular anterior, medial | 低 | 前岛叶内侧 |
| Iapm | Insula agranular posterior, medial | 无 | 后岛叶内侧 |
| Iapl | Insula agranular posterior, lateral | 无 | 后岛叶外侧 |

---

## 📁 Excel数据结构

### Sheet 1: Projection_Data
```
| Neuron_ID | Left_FI | Left_Iai | Left_Ial | ... | Right_Ig |
|-----------|---------|----------|----------|-----|----------|
| N01       | 0       | 15000    | 2000     | ... | 300      |
```

### Sheet 2: Metadata
```
| Neuron_ID | Soma_Hemisphere | Soma_Region | Soma_Subregion | Neuron_Type |
|-----------|-----------------|-------------|----------------|-------------|
| N01       | Left            | Insula      | Iai            | IT          |
```

### Sheet 3: Region_Info
```
| Region_Name | Hemisphere | Cytoarchitecture | Is_Insula |
|-------------|------------|------------------|-----------|
| Left_FI     | Left       | Agranular        | TRUE      |
```

---

## 🎨 输出热图

运行后会生成以下PDF文件：

| 文件名 | 说明 |
|--------|------|
| `*_01_InsulaSoma_Heatmap.pdf` | 胞体在脑岛的神经元投射热图 |
| `*_02_ProjectingToInsula_Heatmap.pdf` | 投射到脑岛的神经元热图 |
| `*_03_Bilateral_Comparison.pdf` | 左右脑岛对比热图 |

---

## 📖 学习路径

### 新手路径
1. 阅读 `00_README_使用指南.md`（本文件）
2. 运行 `06_完整R分析脚本.R`（使用示例数据）
3. 查看输出结果
4. 阅读 `01_R语言热图绘制完整教程.md` 了解原理

### 进阶路径
1. 阅读 `02_脑岛投射分析方案.md` 了解设计思路
2. 阅读 `03_R代码实现_脑岛分析.md` 学习代码细节
3. 修改代码适应自己的需求
4. 阅读 `04_数据准备与质量控制.md` 准备自己的数据

### 专家路径
1. 深入阅读Henry Evrard 2012/2014原始文献
2. 修改 `06_完整R分析脚本.R` 添加自定义分析
3. 探索更多ComplexHeatmap高级功能

---

## ⚠️ 注意事项

### 数据准备
- 确保Neuron_ID在两个Sheet中一致
- 确保Region_Name在Projection_Data和Region_Info中匹配
- 投射强度应为非负值
- 脑岛区域必须在Region_Info中标记Is_Insula = TRUE

### R环境
- 需要R 4.0或更高版本
- 需要安装ComplexHeatmap（Bioconductor包）
- 首次运行会自动安装依赖包

### 常见问题
1. **Neuron_ID不匹配**：检查两个Sheet的ID列名是否一致
2. **Region_Name不匹配**：检查列名拼写和大小写
3. **无脑岛区域识别**：检查Is_Insula列是否为TRUE/FALSE

---

## 📚 参考文献

### 核心文献
1. **Evrard et al., 2012** - Von Economo neurons in the anterior insula of the macaque monkey. *Neuron*
2. **Evrard et al., 2014** - Modular architectonic organization of the insula in the macaque monkey. *J Comp Neurol*
3. **Evrard, 2019** - Organization of the primate insular cortex. *Front Neuroanat*

### 方法学文献
4. **Gou et al., 2025** - Single-neuron projectomes of macaque prefrontal cortex. *Cell*
5. **Gu et al., 2016** - Conservation and diversification of cortical cell organization. *Science*

---

## 🔗 相关资源

- ComplexHeatmap文档: https://jokergoo.github.io/ComplexHeatmap-reference/book/
- R for Data Science: https://r4ds.had.co.nz/
- Henry Evrard实验室: https://uni-tuebingen.de/en/forschung/forschungsschwerpunkte/cin/arbeitsgruppen-alumni/arbeitsgruppen/evrard-h-functional-and-comparative-neuroanatomy/

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 |
|-----|------|---------|
| 2024-XX-XX | v1.0 | 初始版本 |

---

## 📧 联系方式

如有问题或建议，欢迎交流！

---

**祝你分析顺利！🎉**
