# Quick Reference: Monkey Atlas Creation

## One-Page Workflow

```
NIfTI Volume → 3D Mesh → SVG Outlines → Lookup Tables → Install → Plot
```

---

## Commands

### 1. Generate Mesh

**For simple 3D atlases:**
```bash
python volume_to_mesh_mz3.py \
    --input_volume atlas.nii.gz \
    --output_path ./mesh/ \
    --out_file monkey.mz3 \
    --index_max 16 \
    --colors colors.txt
```

**For multi-hierarchy monkey atlases (ARM):**
```bash
python volume_to_mesh_mz3_MONKEY.py \
    --input_volume ARM_in_NMT_v2.1_sym.nii.gz \
    --hierarchy_level 6 \
    --output_path ./mesh/ \
    --out_file ARM_h6.mz3 \
    --colors colors.txt
```

**Note:** Output directory (`--output_path`) will be created automatically if it doesn't exist.

**Hierarchy levels (ARM):** 0=coarsest (66 regions) → 6=standard (72 regions)

### 2. Generate Colors
```python
import numpy as np
import matplotlib
cmap = matplotlib.colormaps.get_cmap('plasma')
colors = cmap(np.linspace(0, 1, 16))
rgb = (colors[:, :3] * 256).astype(int)
np.savetxt('colors.txt', rgb, fmt='%d')
```

### 3. Install Atlas
```bash
cp monkey_atlas_*.svg subcortex_visualization/data/
cp monkey_atlas_*_ordering.csv subcortex_visualization/data/
cd subcortex_visualization && pip install .
```

---

## SVG Title Format

```
{region_name}_{face}_{hemisphere}
```

**Examples:**
- `insula_lateral_L`
- `putamen_medial_R`
- `caudate_lateral_L`

---

## Lookup Table Format

```csv
region,face,plot_order,Hemisphere
insula,lateral,1,L
putamen,lateral,2,L
caudate,lateral,3,L
```

---

## Python Usage

```python
from subcortex_visualization import plot_subcortical_data

plot_subcortical_data(
    subcortex_data=df,      # DataFrame with columns: region, value, Hemisphere
    atlas='monkey_atlas',   # Your atlas name (matches filename prefix)
    hemisphere='L',         # 'L', 'R', or 'both'
    cmap='plasma',
    fill_title='Values'
)
```

---

## File Naming Convention

| File | Pattern | Example |
|------|---------|---------|
| Left SVG | `{name}_L.svg` | `monkey_atlas_L.svg` |
| Right SVG | `{name}_R.svg` | `monkey_atlas_R.svg` |
| Both SVG | `{name}_both.svg` | `monkey_atlas_both.svg` |
| Left CSV | `{name}_L_ordering.csv` | `monkey_atlas_L_ordering.csv` |
| Right CSV | `{name}_R_ordering.csv` | `monkey_atlas_R_ordering.csv` |
| Both CSV | `{name}_both_ordering.csv` | `monkey_atlas_both_ordering.csv` |

---

## Common Issues

| Problem | Solution |
|---------|----------|
| `niimath not found` | Add to PATH or use full path |
| Black plot | Check region names match exactly |
| 404 errors | Verify `sample_id` exists on server |
| Empty regions | Check NIfTI labels are correct |

---

## Required Software

1. **niimath** - https://github.com/rordenlab/niimath/releases
2. **Surf Ice** - https://github.com/neurolabusc/surf-ice
3. **Inkscape** - https://inkscape.org/

---

## Help

- Full guide: `README.md` (this folder)
- Detailed pipeline: `CUSTOM_SEGMENTATION_GUIDE.md`
- Original docs: `../custom_segmentation/README.md`
