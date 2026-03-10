"""
Python版本：神经投射数据热图绘制
使用matplotlib和seaborn实现类似Figure 2a的效果

优点：
- 无需安装R环境
- Python生态更熟悉
- 易于集成到现有流程

缺点：
- 热图功能不如ComplexHeatmap强大
- 自定义注释较复杂
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据读取 ====================

def load_data(file_path):
    """读取Excel数据"""
    df_projection = pd.read_excel(file_path, sheet_name='Projection_Data', index_col=0)
    df_metadata = pd.read_excel(file_path, sheet_name='Metadata')
    df_regions = pd.read_excel(file_path, sheet_name='Region_Info')

    print(f"✓ 投射矩阵: {df_projection.shape}")
    print(f"✓ 元数据: {df_metadata.shape}")
    print(f"✓ 脑区信息: {df_regions.shape}")

    return df_projection, df_metadata, df_regions


# ==================== 2. 颜色方案 ====================

def get_color_schemes():
    """定义颜色方案"""
    type_colors = {
        'ITc': '#4DAF4A',
        'ITs': '#984EA3',
        'CT': '#FF7F00',
        'PT': '#E41A1C',
        'ITi': '#377EB8'
    }

    hemisphere_colors = {
        'Ipsilateral': '#FDB462',
        'Contralateral': '#80B1D3'
    }

    return type_colors, hemisphere_colors


def create_blue_cmap():
    """创建蓝色渐变颜色映射"""
    colors = ['#FFFFFF', '#C6DBEF', '#6BAED6', '#2171B5', '#08306B']
    return LinearSegmentedColormap.from_list('blue_gradient', colors)


# ==================== 3. 主热图绘制 ====================

def plot_main_heatmap(df_projection, df_metadata, df_regions, output_prefix='Figure2a'):
    """
    绘制主热图 (Figure 2a风格)

    这个函数创建一个复杂的组合图，包含：
    - 顶部树状图
    - 顶部注释条 (Type, Soma, N_Neurons, Axon_Length)
    - 主热图
    - 右侧注释 (Hemisphere)
    """

    # 确保顺序一致
    df_metadata = df_metadata.set_index('Subtype').reindex(df_projection.index)

    # 获取颜色方案
    type_colors, hemisphere_colors = get_color_schemes()
    blue_cmap = create_blue_cmap()

    # 创建图形
    fig = plt.figure(figsize=(18, 14))

    # 创建网格布局
    # [树状图, 注释条]
    # [热图,    右侧注释]

    gs = fig.add_gridspec(4, 3, 
                          width_ratios=[0.15, 4, 0.3],
                          height_ratios=[1.5, 0.8, 0.8, 4],
                          wspace=0.05, hspace=0.05)

    # === 1. 顶部树状图 (列聚类) ===
    ax_dendro_top = fig.add_subplot(gs[0, 1])

    # 计算列聚类
    col_linkage = linkage(df_projection.T.values, method='ward', metric='euclidean')

    # 绘制树状图
    with plt.rc_context({'lines.linewidth': 0.8}):
        dendro_col = dendrogram(col_linkage, ax=ax_dendro_top, 
                                color_threshold=0, above_threshold_color='gray')
    ax_dendro_top.set_xticks([])
    ax_dendro_top.set_yticks([])
    ax_dendro_top.axis('off')

    # 获取列顺序
    col_order = dendro_col['leaves']

    # === 2. 左侧树状图 (行聚类) ===
    ax_dendro_left = fig.add_subplot(gs[3, 0])

    # 计算行聚类
    row_linkage = linkage(df_projection.values, method='ward', metric='euclidean')

    # 绘制树状图
    with plt.rc_context({'lines.linewidth': 0.8}):
        dendro_row = dendrogram(row_linkage, ax=ax_dendro_left, 
                                orientation='left',
                                color_threshold=0, above_threshold_color='gray')
    ax_dendro_left.set_xticks([])
    ax_dendro_left.set_yticks([])
    ax_dendro_left.axis('off')

    # 获取行顺序
    row_order = dendro_row['leaves']

    # 重排数据
    df_ordered = df_projection.iloc[row_order, col_order]
    metadata_ordered = df_metadata.iloc[row_order]

    # === 3. 顶部注释条 - Type ===
    ax_type = fig.add_subplot(gs[1, 1])
    type_data = metadata_ordered['Type'].values
    type_color_array = np.array([[type_colors.get(t, 'gray') for t in type_data]])
    ax_type.imshow(type_color_array, aspect='auto', interpolation='nearest')
    ax_type.set_xticks([])
    ax_type.set_yticks([0])
    ax_type.set_yticklabels(['Type'], fontsize=9, fontweight='bold')
    ax_type.set_ylim(-0.5, 0.5)

    # === 4. 顶部注释条 - Soma ===
    ax_soma = fig.add_subplot(gs[2, 1])
    soma_regions = df_metadata['Soma_Region'].unique()
    soma_colors = plt.cm.Set2(np.linspace(0, 1, len(soma_regions)))
    soma_color_dict = dict(zip(soma_regions, soma_colors))
    soma_data = metadata_ordered['Soma_Region'].values
    soma_color_array = np.array([[soma_color_dict.get(s, 'gray') for s in soma_data]])
    ax_soma.imshow(soma_color_array, aspect='auto', interpolation='nearest')
    ax_soma.set_xticks([])
    ax_soma.set_yticks([0])
    ax_soma.set_yticklabels(['Soma'], fontsize=9, fontweight='bold')
    ax_soma.set_ylim(-0.5, 0.5)

    # === 5. 主热图 ===
    ax_heatmap = fig.add_subplot(gs[3, 1])

    im = ax_heatmap.imshow(df_ordered.values, 
                           aspect='auto',
                           cmap=blue_cmap,
                           interpolation='nearest')

    # 设置标签
    ax_heatmap.set_xticks(range(len(df_ordered.columns)))
    ax_heatmap.set_xticklabels(df_ordered.columns, rotation=45, ha='right', fontsize=7)
    ax_heatmap.set_yticks(range(len(df_ordered.index)))
    ax_heatmap.set_yticklabels(df_ordered.index, fontsize=7)

    # 添加标题
    ax_heatmap.set_xlabel('Target Brain Regions', fontsize=11, fontweight='bold')
    ax_heatmap.set_ylabel('Neuron Subtypes', fontsize=11, fontweight='bold')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Log Axon Length', fontsize=10, fontweight='bold')

    # === 6. 右侧注释 - Hemisphere ===
    ax_hemi = fig.add_subplot(gs[3, 2])

    # 获取每个区域的hemisphere
    region_hemi = df_regions.set_index('Region').reindex(df_ordered.columns)['Hemisphere']
    hemi_color_array = np.array([[hemisphere_colors.get(h, 'gray') for h in region_hemi]]).T
    ax_hemi.imshow(hemi_color_array, aspect='auto', interpolation='nearest')
    ax_hemi.set_xticks([0])
    ax_hemi.set_xticklabels(['Hemi'], fontsize=8, fontweight='bold', rotation=90)
    ax_hemi.set_yticks([])

    # === 7. 添加图例 ===
    legend_elements = [
        mpatches.Patch(facecolor=color, label=label) 
        for label, color in type_colors.items()
    ]
    fig.legend(handles=legend_elements, loc='upper right', 
               bbox_to_anchor=(0.98, 0.98), title='Type',
               fontsize=9, title_fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # 保存
    plt.savefig(f'{output_prefix}_Python_Main_Heatmap.pdf', 
                dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_Python_Main_Heatmap.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 主热图已保存: {output_prefix}_Python_Main_Heatmap.pdf/png")

    return fig


# ==================== 4. 使用seaborn的简化版本 ====================

def plot_simple_clustermap(df_projection, df_metadata, output_prefix='Figure2a'):
    """
    使用seaborn的clustermap (更简单但自定义性较低)
    """

    # 创建颜色映射
    type_colors, _ = get_color_schemes()

    # 为行着色 (基于Type)
    row_colors = df_metadata.set_index('Subtype')['Type'].map(type_colors)

    # 绘制聚类热图
    g = sns.clustermap(
        df_projection,
        cmap='Blues',
        row_colors=row_colors,
        figsize=(16, 12),
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        linewidths=0,
        xticklabels=True,
        yticklabels=True
    )

    # 调整标签
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7)

    # 添加标题
    g.ax_heatmap.set_xlabel('Target Brain Regions', fontsize=11, fontweight='bold')
    g.ax_heatmap.set_ylabel('Neuron Subtypes', fontsize=11, fontweight='bold')

    plt.savefig(f'{output_prefix}_Python_Simple_Heatmap.pdf', 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 简化热图已保存: {output_prefix}_Python_Simple_Heatmap.pdf")


# ==================== 5. 主程序 ====================

def main():
    """主程序"""

    print("=" * 50)
    print("  Python神经投射数据可视化")
    print("=" * 50)

    # 读取数据
    file_path = 'neuron_projection_data.xlsx'

    try:
        df_proj, df_meta, df_region = load_data(file_path)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        print("请修改代码中的 file_path 变量")
        return

    # 绘制完整版热图
    print("\n📊 绘制完整版热图...")
    plot_main_heatmap(df_proj, df_meta, df_region)

    # 绘制简化版热图
    print("\n📊 绘制简化版热图...")
    plot_simple_clustermap(df_proj, df_meta)

    print("\n" + "=" * 50)
    print("  ✅ 所有图表生成完成!")
    print("=" * 50)


if __name__ == '__main__':
    main()
