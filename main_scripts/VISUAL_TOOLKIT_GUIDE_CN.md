# Visual Toolkit 用户指南

## 神经元可视化工具包 - 中文使用手册

---

## 1. 概述

Visual Toolkit 是一个用于可视化猕猴大脑神经元数据的 Python 工具包。它支持混合分辨率数据获取：

- **高分辨率** (0.65µm): 通过 HTTP 获取体素块，用于胞体精细结构
- **低分辨率** (5.0µm): 通过 SSH 获取宽场图像，用于神经元投射整体视图

### 主要功能

| 功能 | 描述 |
|------|------|
| `Visual_toolkit.py` | 核心工具包，提供数据获取和可视化 |
| `Visual_toolkit_gui.py` | 图形界面版本，交互式操作 |
| `bulk_visual_data.py` | 批量处理脚本，处理多个神经元 |

---

## 2. 系统要求

### Python 环境
```bash
# 必需的包
pip install nibabel tifffile matplotlib paramiko numpy pandas tqdm
```

### 网络要求
- **HTTP 访问**: http://bap.cebsit.ac.cn (高分辨率数据)
- **SSH 访问**: 172.20.10.250:20007 (低分辨率数据)

---

## 3. 快速开始

### 3.1 核心工具包使用

```python
from Visual_toolkit import Visual_toolkit
import IONData as IT

# 初始化工具包
toolkit = Visual_toolkit('251637')
ion = IT.IONData()

# 获取神经元 SWC 数据
tree = ion.getRawNeuronTreeByID('251637', '003.swc')
soma_xyz = [tree.root.x, tree.root.y, tree.root.z]

# 获取高分辨率胞体块
vol_h, org_h, res_h = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
toolkit.plot_soma_block(vol_h, org_h, res_h, soma_xyz, '003.swc')

# 获取低分辨率宽场图像
vol_l, org_l, res_l = toolkit.get_low_res_widefield(soma_xyz, 
    width_um=8000, height_um=8000, depth_um=30)
toolkit.plot_widefield_context(vol_l, org_l, res_l, soma_xyz, '003.swc', 
    swc_tree=tree)

# 关闭连接
toolkit.close()
```

---

## 4. GUI 版本使用指南

### 4.1 启动 GUI

```bash
python Visual_toolkit_gui.py
```

### 4.2 界面说明

```
┌─────────────────────────────────────────────┐
│  Visual_Toolkit_gui v1.5.0                   │
├─────────────────────────────────────────────┤
│                                             │
│  样本 ID: [251637        ]                   │
│  神经元 ID: [003.swc     ]                   │
│                                             │
│  [加载并自动填充胞体坐标]                     │
│                                             │
│  高分辨率参数:                               │
│    网格半径: [1     ]                        │
│                                             │
│  低分辨率参数:                               │
│    宽度 (µm): [8000  ]                      │
│    高度 (µm): [8000  ]                      │
│    深度 (µm): [30    ]                      │
│                                             │
│  [胞体图像]  [宽场图像]  [生成两者]          │
│                                             │
│  输出目录: [...浏览...]                      │
│                                             │
│  状态: 就绪                                  │
│  [==========]                                │
│                                             │
└─────────────────────────────────────────────┘
```

### 4.3 操作步骤

1. **输入样本 ID**: 默认 `251637`
2. **输入神经元 ID**: 例如 `003.swc`
3. **点击"加载并自动填充胞体坐标"**: 自动从 SWC 文件获取胞体位置
4. **调整参数**（可选）:
   - 高分辨率网格半径: 1 = 单个块, 2 = 3×3×3 网格
   - 宽场尺寸: 默认 8000×8000 µm
5. **点击生成按钮**:
   - `胞体图像`: 仅生成高分辨率胞体块
   - `宽场图像`: 仅生成低分辨率宽场
   - `生成两者`: 先生成胞体，再生成宽场

### 4.4 输出文件

```
输出目录/
├── 251637_003.swc_SomaBlock.nii.gz      # 高分辨率体积数据
├── 251637_003.swc_SomaBlock_Plot.png    # 高分辨率 MIP 图像
├── 251637_003.swc_WideField.nii.gz      # 低分辨率体积数据
└── 251637_003.swc_WideField_Plot.png    # 低分辨率叠加图像
```

---

## 5. 批量处理指南

### 5.1 配置文件

编辑 `bulk_visual_data.py`:

```python
# ==========================================
# 配置
# ==========================================
SAMPLE_ID = '251637'  # 样本 ID

# 输入文件: 包含 NeuronID 和 Soma_Region 列的 Excel/CSV
INPUT_FILE = r'D:\projectome_analysis\main_scripts\neuron_tables\251637_INS.xlsx'

# 输出根目录
PARENT_OUTPUT_DIR = r"W:\fMOST"

# 绘图设置
GENERATE_HIGH_RES = True   # 生成高分辨率图像
GENERATE_LOW_RES = True    # 生成低分辨率图像
```

### 5.2 输入文件格式

Excel/CSV 文件需要包含以下列:

| NeuronID | Soma_Region | Soma_Phys_X | Soma_Phys_Y | Soma_Phys_Z |
|----------|-------------|-------------|-------------|-------------|
| 003.swc  | DI          | ...         | ...         | ...         |
| 004.swc  | Gu          | ...         | ...         | ...         |
| 007.swc  | CL_Ia/Id    | ...         | ...         | ...         |

**注意**: 如果没有物理坐标列，脚本会自动从 SWC 文件获取 (tree.root.x/y/z)

### 5.3 运行批量处理

```bash
python bulk_visual_data.py
```

### 5.4 输出结构

```
W:\fMOST\251637\cube_data_251637_INS_20250320\
│
├── Region_DI\                    # 按胞体区域分组
│   ├── HighRes\                  # 高分辨率数据
│   │   ├── Data\                 # NIfTI 文件
│   │   │   ├── 251637_003.swc_DI_SomaBlock.nii.gz
│   │   │   └── ...
│   │   └── Plots\                # PNG 图像
│   │       ├── 251637_003.swc_DI_SomaBlock_Plot.png
│   │       └── ...
│   └── LowRes\                   # 低分辨率数据
│       ├── Data\
│       └── Plots\
│
├── Region_Gu\                    # 其他区域
│   └── ...
│
└── Region_CL_Ia-Id\              # 区域名称自动清理
    └── ...
```

### 5.5 图像标题格式

```
251637 | 003.swc | SomaBlock
Region: DI | XYZ: (13965, 34488, 20977)
FOV: 234x234 µm | Depth: 270 µm
```

---

## 6. 高级功能

### 6.1 缓存机制

工具包自动缓存下载的数据块，避免重复下载:

```
D:\projectome_analysis\resource\cubes\251637\
├── high_res_http\     # HTTP 下载的高分辨率块
│   ├── 77\
│   │   ├── 59_147_77.tif
│   │   └── ...
│   └── ...
└── low_res_ssh\       # SSH 下载的低分辨率切片
    ├── 251637_00001_CH1_resample.tif
    └── ...
```

**注意**: GUI 和批量处理共享同一缓存目录

### 6.2 批量处理使用缓存

```python
# 创建工具包实例（自动使用共享缓存）
toolkit = Visual_toolkit(SAMPLE_ID)
```

### 6.3 自定义缓存目录（高级）

```python
# 在 Visual_toolkit.py 中修改初始化
toolkit = Visual_toolkit(SAMPLE_ID, cache_dir='/path/to/custom/cache')
```

---

## 7. 故障排除

### 7.1 HTTP 404 错误

**问题**: `Block 205_228_77 not found on server`

**原因**: Excel 中的物理坐标与 SWC 文件不匹配

**解决**: 批量脚本现在自动使用 SWC 坐标 (tree.root.x/y/z)

### 7.2 SSH 连接失败

**问题**: `[ERROR] SSH Connection Failed`

**检查**:
1. 网络连接是否正常
2. SSH 凭据是否正确:
   - Host: `172.20.10.250`
   - Port: `20007`
   - 用户名/密码是否正确

### 7.3 文件名中的斜杠问题

**问题**: 区域名称包含 `/` 导致路径错误

**解决**: 脚本自动清理:
- `CL_Ia/Id` → `CL_Ia-Id`
- 文件名和文件夹名都经过处理

### 7.4 图像质量问题

**胞体不可见**:
- 使用 MIP (最大强度投影) 代替单一切片
- 调整 `grid_radius` 获取更大视野

**对比度问题**:
- 脚本自动进行 Gamma 校正 (0.5)
- 自动调整对比度范围 (0.5% - 99.5% 百分位数)

---

## 8. 参数参考

### 8.1 Visual_toolkit 类

```python
Visual_toolkit(sample_id='251637', cache_dir=None)
```

### 8.2 高分辨率方法

```python
get_high_res_block(center_um, grid_radius=1)
```
| 参数 | 类型 | 描述 |
|------|------|------|
| `center_um` | [x, y, z] | 中心坐标 (微米) |
| `grid_radius` | int | 网格半径: 1=1块, 2=3×3×3块 |

**返回**: `(volume, origin, resolution)`

```python
plot_soma_block(volume_3d, origin, resolution, soma_coords, 
                neuron_id, suffix="SomaBlock", soma_region=None)
```

### 8.3 低分辨率方法

```python
get_low_res_widefield(center_um, width_um=10000, height_um=10000, depth_um=90)
```
| 参数 | 类型 | 描述 |
|------|------|------|
| `width_um` | int | 视野宽度 (微米) |
| `height_um` | int | 视野高度 (微米) |
| `depth_um` | int | Z 轴切片厚度 (微米) |

```python
plot_widefield_context(volume_3d, origin, resolution, soma_coords, 
                       neuron_id, swc_tree=None, soma_region=None)
```

### 8.4 数据导出

```python
export_data(volume, origin, resolution, neuron_id, 
            suffix="Volume", soma_region=None, output_dir=None)
```

---

## 9. 示例代码

### 9.1 处理单个神经元

```python
from Visual_toolkit import Visual_toolkit
import IONData as IT

# 初始化
toolkit = Visual_toolkit('251637')
ion = IT.IONData()

# 神经元 ID
neuron_id = '003.swc'
tree = ion.getRawNeuronTreeByID('251637', neuron_id)
soma_xyz = [tree.root.x, tree.root.y, tree.root.z]

print(f"胞体坐标: {soma_xyz}")

# 高分辨率
vol_h, org_h, res_h = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
toolkit.plot_soma_block(vol_h, org_h, res_h, soma_xyz, neuron_id, 
                        soma_region="DI")
toolkit.export_data(vol_h, org_h, res_h, neuron_id, 
                    suffix="SomaBlock", soma_region="DI",
                    output_dir="./output")

# 低分辨率
vol_l, org_l, res_l = toolkit.get_low_res_widefield(
    soma_xyz, width_um=8000, height_um=8000, depth_um=30)
toolkit.plot_widefield_context(vol_l, org_l, res_l, soma_xyz, neuron_id,
                               bg_intensity=2.0, swc_tree=tree, 
                               soma_region="DI")

toolkit.close()
```

### 9.2 处理特定区域的神经元

```python
import pandas as pd
from Visual_toolkit import Visual_toolkit
import IONData as IT

# 读取神经元列表
df = pd.read_excel('251637_INS.xlsx')
insula_neurons = df[df['Soma_Region'].str.contains('I', na=False)]

toolkit = Visual_toolkit('251637')
ion = IT.IONData()

for _, row in insula_neurons.iterrows():
    neuron_id = str(row['NeuronID'])
    soma_region = row['Soma_Region']
    
    tree = ion.getRawNeuronTreeByID('251637', neuron_id)
    soma_xyz = [tree.root.x, tree.root.y, tree.root.z]
    
    # 生成可视化
    vol_h, org_h, res_h = toolkit.get_high_res_block(soma_xyz, grid_radius=1)
    toolkit.plot_soma_block(vol_h, org_h, res_h, soma_xyz, neuron_id,
                            soma_region=soma_region, output_dir="./insula")

toolkit.close()
```

---

## 10. 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.5.0 | 2026-03-20 | GUI 自动加载默认神经元 |
| v1.4.0 | 2026-03-15 | 添加胞体区域支持 |
| v1.3.0 | 2026-03-10 | MIP 可视化修复 |

---

## 11. 联系与支持

如有问题，请检查:
1. 错误日志信息
2. 网络连接状态
3. 缓存目录权限

---

**文档版本**: 1.0  
**最后更新**: 2026-03-20
