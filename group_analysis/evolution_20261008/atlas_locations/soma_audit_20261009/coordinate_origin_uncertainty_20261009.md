# Coordinate origin remains unverified

This addendum qualifies the [current-client atlas lookup](delivery_readback_20261009.md). Agreement of cached SWC roots and portal soma coordinates validates their stored coordinate consistency under the current client rule. It does **not** establish the exported physical-coordinate origin, registration acceptance, or anatomical parcel assignment.

The locally deposited Gou et al. 2025 code [zz0atlas.jl](../../../../references/analysis-code_gou_etal_2025/monkeyrec/zz0atlas/src/zz0atlas.jl) declares NMT v2.0 CHARM/SARM at lines 207–213 and implements `pos2idx=ceil(position/resolution)` in Julia one-based indices and `idx2pos=(index-0.5)*resolution` at lines 289–294. Its SHA256 is `0d0916b4f7fb9e0effd16aef09cd70d93bcce476e1a2f247cd700411dfc2df48`. Whether the cached SWC and portal soma exports use this convention is **unknown**. The user-described registration direction is raw fMOST brain and neurons to NMT; original transforms, export/version receipts and landmark acceptance remain pending.

The [separate sensitivity output](coordinate_origin_sensitivity/provenance_and_counts.json) reverified all 2,480 exact source hashes and retained all 8,746 live identities, with missing channels unassessed. All three policies use the **same local NMT v2.1 symmetric ARM6 and LR plane**, isolating index-policy sensitivity. This does not reproduce the paper's NMT v2.0 CHARM/SARM labels. Current stored coordinates were evaluated in Float64; the original export's numeric precision is unknown.

| Policy | Zero-based index rule | Cached-root label sensitivity | Portal-coordinate label sensitivity |
|---|---|---:|---:|
| Existing client | `rint(XYZ/250)`, exact halves to even | Baseline, 1,912 assessed | Baseline, 8,409 assessed |
| Deposited edge-origin implementation | `ceil(XYZ/250)-1`, literal Julia one-based conversion | 227 labels change | 819 labels change |
| Half-open center cells | `floor(XYZ/250+0.5)` | 1 label changes | 1 label changes |

For the deposited edge-origin comparison, 1,682 cached-root voxels change. Of the 227 changed labels, 86 switch between nonbackground labels, 25 change from a label to background, and 116 from background to a label. The other 1,685 retain their label, including 179 background labels. The shared insula label-set candidate count changes from 309 to 346 among cached roots and 310 to 347 among portal coordinates. These are **policy sensitivity counts**, not adopted labels or anatomical membership changes.

All 1,912 covered roots and all 8,409 portal coordinates retain their known L/R mask side under both alternative policies. The half-open rule changes six cached-root voxel indices and thirteen portal-coordinate indices. Its single label change is `252383/064.swc`: current `CL_Tpt` to alternative `Unknown_0`. No convention was changed.

Both `251637/112.swc` and `251637/114.swc` have current-client ARM6 `Unknown_0` and alternative edge-origin `CR_Ia/Id`; both remain R under every policy. Their exact hash-bound reviewed assignments remain **112 R-IDM/layer 3** and **114 provisional R-IDM/layer 3**. The alternative lookup cannot resolve the manual fine parcel or independently accept the registration. Original manual rows and reviewed decisions remain intact.

Eight focused unit tests pass, including literal edge-origin/right-boundary behavior and rint-versus-half-open ties. No source label, coordinate convention, canonical workbook, transform, endpoint map or projection map was changed. Resolving the export convention and original transform/version lineage is required before treating either atlas lookup policy as scientifically accepted.
