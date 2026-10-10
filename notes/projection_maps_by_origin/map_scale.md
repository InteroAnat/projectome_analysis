# Meaning of the map scales

The maps use a transparent descriptive measure, not calibrated synaptic strength. Software checks support the arithmetic and source identity; unavailable raw-to-NMT transformation/export-origin evidence still limits anatomical interpretation. The prior Gao/Gou/Liu Methods review supports preserving reconstruction, spatial context and image-based terminal validation as distinct evidence. It does not validate our chosen colour percentile or glyph size.

For voxel v in a source view, let c(a,i,v) be the number of original candidate axon ends of neuron i in animal a; n(a) is the number of selected neurons in that animal/view with at least one eligible original end anywhere; A is the number of animals contributing eligible neurons. The descriptive density is:

`D(v) = [sum over a of (sum over eligible i of c(a,i,v) / n(a)) / A] / voxel_volume_mm3`

The unit is candidate ends per template mm³ per eligible reconstructed neuron, averaged equally across contributing animals. A neuron with no eligible axon end is unassessed for this measure. An eligible neuron with no end in a particular voxel contributes a known zero there. Animals without eligible neurons are not inserted as zero. Missing animals and unequal coverage therefore limit generalization. Equal animal weighting avoids letting the number of reconstructed neurons alone determine the between-animal mean; it does not correct injection/reconstruction bias or make one-animal views replicated experiments.

| Scale | Actual choice | Interpretation |
|---|---|---|
| Spatial grid | 0.25 × 0.25 × 0.25 mm in the pinned NMT MRI; affine in mm | Template voxel size, not registration precision or native microscopy resolution |
| Voxel volume | 0.015625 mm³ | Converts the eligible-neuron mean count to density; changing resolution changes local binning |
| Numerical projection map | D(v), stored as float32 | Unsmoothed descriptive candidate-end density; full voxel values are retained |
| Display colour | log10(1 + D(v)/D0), with D0 = 1 candidate end/mm³/eligible neuron | Dimensionless monotonic display transform; underlying count/density data are preserved |
| Common colour upper bound | 2.1105897426605225, corresponding to D ≈ 128 | Previous review's 99.5th percentile of positive values in the matched slices; one shared bound across current cards, not a significance threshold |
| Colour value 0 / 1 / 2 | D = 0 / 9 / 99 | Colour differences are nonlinear; use NIfTI/table values for numerical comparisons |
| Values exceeding display bound | Saturated colour | Not removed, censored or capped in numerical NIfTIs; per-view saturation is recorded in scale_display_qc.csv |
| Endpoint marker | Fixed 9 pt² occupied-voxel dot with white halo/black rim | Dot size is a visibility choice; one dot per occupied voxel, not one dot per neuron or synapse |
| Source marker and arrow | Solid 48 pt² diamond; yellow arrow with black outline | Actual displayed soma position; arbitrary glyph size and label offset convey source location, not physical soma size |
| Soma count NIfTI | Raw count of selected somata per saved source voxel | Sampling/reconstruction distribution, not an unbiased population density |
| MRI grey window | 75–920 in reference-image intensity units | Fixed contextual brightness/contrast; arbitrary MRI units, not an anatomical or projection threshold |

All target displays are single slices at X56/Y200/Z87; source locators choose the slice with most source somata per axis and record the actual cut. There is no MIP, smoothing, hemispheric mirroring, statistical t-value or p-value. Background zero is contextual space, not evidence of no anatomical connection. Dense endpoints can reflect arborization, reconstruction detail, sampling or truncation; they cannot be called synaptic density. Accepted terminal-field maps need appropriate native-image assessment. Colorbar limits, soma glyph size and arrows are explicit visualization choices; their readability alone is not scientific validation.
