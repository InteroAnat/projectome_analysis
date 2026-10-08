# ARM projection hierarchy tables — 462-neuron ledger

The single export retains all **462 exact neuron identities**, across **8 animals**, **17 ARM source groups** and **50 animal/source pairs**. Endpoint measures are computable for **429 neurons**; the other **33 remain NA**, rather than observed zero. The **52 ARM-label-0 source rows remain explicit QC strata**, not named anatomical parcels. Original source annotations and grouping are preserved as evidence.

- [Workbook: Summary, Targets, animal/group means and L1–L6 matrices](combined_arm_projection_tables_462/arm_projection_hierarchy_tables.xlsx)
- [Sparse per-neuron regional measures](combined_arm_projection_tables_462/per_neuron_regional_measures.csv)
- [Complete neuron identity, source evidence and QC ledger](combined_arm_projection_tables_462/neuron_summary.csv)
- [Official target names, indices, domains, sides and status](combined_arm_projection_tables_462/targets.csv)
- [Input, code and output hashes; parameters and completion receipt](combined_arm_projection_tables_462/export_provenance.json)
- [154 saved-NIfTI regional reconciliation checks](combined_arm_projection_tables_462/map_regional_reconciliation.csv)

Each level samples its **actual ARM volume** independently. There is no inferred parent pooling. The observed ARM indices **45 and 545 at L4 and L5 conflict with the key's level ranges**; they retain their official names and explicit `key_level_conflict` status. Label 0 and outside-reference observations remain explicit. `ActualVolumeMm3` describes the image footprint, including flagged labels; it does not establish anatomical validity.

The workbook has `Summary`, `Targets`, `Animal_Means`, `Group_Means`, and three matrices per level: `L1_EP_Count` through `L6_EP_Count`, `L1_EP_Presence` through `L6_EP_Presence`, and `L1_AxonLen_mm` through `L6_AxonLen_mm`. Common sheets include full source ARM metadata. Target headers include the stable TargetID, official full name, domain and hemisphere.

**Candidate endpoint counts** describe reconstructed type-2 axonal leaves under the reviewed builder contract, not verified biological terminals or boutons. Regional endpoint presence is a distinct-neuron indicator (`count > 0`), not summed voxel occupancy. Within each animal/source group, endpoint means and frequencies use only its eligible neurons; source-group means then weight available animals equally. Axon means use all selected neurons within each animal and equal animal weights between animals.

**Axon template length** is child-type-2 reconstruction length in reference millimetres. It is distinct from the legacy retained all-compartment length measure. Every source was rasterized once, then reduced against the six actual ARM grids. The sparse table omits rows with both measured values zero: an omitted axon value means zero; an omitted endpoint value means zero only when `EndpointEligible=True`, otherwise NA. Never sum lengths or counts across hierarchy levels.

All 154 original saved count/length NIfTIs passed per-region reconciliation using each run's original groups and denominators, before regrouping into the combined ARM ledger. Maximum absolute regional differences were approximately `1.19e-5` candidate counts and `1.15e-6` mm; checks use `rtol=1e-5, atol=1e-5` to account for saved float32 maps. Spatial outputs remain the existing NIfTIs in the [main ARM run](../../../group_analysis/evolution_20261008/arm_mapping_20261009/main), [additional endpoint run](../../../group_analysis/evolution_20261008/endpoint_maps/additional_candidates_20261009), and [additional axon run](../../../group_analysis/evolution_20261008/projection_maps/additional_candidates_20261009). This export creates no additional maps.

Software verification is descriptive. **Scientific/anatomical acceptance remains false**, including unresolved export-coordinate origin and unavailable image-terminal evidence. Twelve focused real-builder regressions passed; [test log](exporter_tests_metadata_20261009.log). Independent saved-table checks are recorded separately in this audit directory.

To reproduce from the repository root, choose a fresh output directory (existing destinations are refused):

```powershell
python -B group_analysis/scripts/export_arm_projection_tables.py `
  --endpoint-run group_analysis/evolution_20261008/arm_mapping_20261009/main/endpoints `
  --axon-run group_analysis/evolution_20261008/arm_mapping_20261009/main/axons `
  --manifest group_analysis/evolution_20261008/projection_inputs/arm_labels_20261009/combined/projection_manifest.csv `
  --additional-endpoint-run group_analysis/evolution_20261008/endpoint_maps/additional_candidates_20261009 `
  --additional-axon-run group_analysis/evolution_20261008/projection_maps/additional_candidates_20261009 `
  --output <fresh-output-directory>
```

Final combined manifest SHA256: `6bffc5ba0380db0ea0caddf85c188af7b73c7409a274aa53e588cba0272ccdd8`.
