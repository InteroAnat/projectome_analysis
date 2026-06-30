# Current handoff — projectome_analysis

**Updated:** 2026-06-30 (von Economo bulk visual + UNC low-res)

## [handoff] 当前焦点

- **Visual_toolkit** 低分辨已切到 lab UNC：`\\10.102.8.200\microscopy_data\fMOST\936-251637\251637-CH1_resample\resample_5um`（251637）；SSH 路径已失效，仅 fallback。
- **Bulk visual** red「check」von Economo 5 神经元已输出到 Downloads（高/低分辨 PNG + NIfTI，grid radius 2）。
- Soma 坐标：**raw SWC `tree.root`**（fMOST µm），与 GUI 一致；勿用 `Soma_NII_*`。

## [scope] 相关路径

- `d:\projectome_analysis\main_scripts\Visual_toolkit.py`
- `d:\projectome_analysis\main_scripts\step3.1.bulk_visual_data.py`
- `d:\projectome_analysis\main_scripts\test_low_res_print.py`
- `d:\projectome_analysis\neuron_tables\251637_von_economo_candidates.xlsx`（10）
- `d:\projectome_analysis\neuron_tables\251637_von_economo_check.xlsx`（5 check）
- 最新 bulk 输出：`%USERPROFILE%\Downloads\bulk_visual\251637\cube_data_251637_von_economo_check_20260630\`

## [verify] 验证

- `python test_low_res_print.py 007.swc` → PASS（low-res max 65534）
- `step3.1.bulk_visual_data.py` → Success 5 / Failed 0

## [next] 下一步

1. 可选：对 yellow von_economo 子集（006, 011, 026, 028, 035）跑 bulk → Downloads
2. 新 sample：扩展 `LOW_RES_SHARE_BY_SAMPLE` in `Visual_toolkit.py`
3. 重启 GUI 以加载 UNC 低分辨逻辑

## Today's daily

- **`2026-06-30.md`** — von Economo bulk visual, UNC low-res, check subset to Downloads
- **`transcript-integrated-2026-06-30.md`** — full session integrated summary
