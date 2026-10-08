This isolated variant changes user-facing labels and docstrings in the three active region-analysis modules. Production sources and existing data remain untouched.

- Regional values are **retained reconstruction lengths in the source unit**. `Total_Length` is the whole computed reconstruction-edge total. No mm conversion is inferred. Display strength means the source-unit-dependent transform `log10(retained length + 1)`.
- `Terminal_Count` describes **distinct endpoint-target regions per neuron from legacy all-compartment reconstruction leaves**. It does not count verified biological terminals or boutons. Summed per-neuron entries are labeled as entries rather than global unique regions.
- `Laterality_Index` remains the contralateral length fraction `Contra/(Ipsi+Contra)`, ranging from 0 to 1 and excluding unresolved lengths. The distinct signed contrast `(Contra-Ipsi)/(Contra+Ipsi)` ranges from -1 to 1 and is explained only; this patch does not compute it.
- CLI help specifies exact identities, absolute target dictionaries, retained source units and a fresh output destination. Existing APIs, column keys, output filenames, classification and calculations are unchanged. The existing plotting indentation and partial-missing/mixed-unit guard are preserved.

`proposed_region_terminology.patch` is the production-source proposal. `proposed_terminal_label_assertions.patch` separately updates only legacy wording expectations in the existing eight terminal-reporting regressions, retaining all numerical expectations. The first compatibility run failed because four remaining old labels had not yet been updated; its log is preserved. The completed label-only compatibility run passes all eight tests. Six independent terminology tests also pass, including actual plots, reports, CLI help and AST/schema/filename checks.

Run the isolated checks from the workspace root:

```powershell
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B notes/region_analysis_review_20261009/code_audit/terminology_variant_20261009/test_terminology_labels.py
& 'C:/Users/laika_yan/miniconda3/envs/projectome/python.exe' -B notes/region_analysis_review_20261009/code_audit/terminology_variant_20261009/staged/tests/test_terminal_reporting.py
```

`terminology_receipt.json` binds source/staged code, proposal patches, test sources and logs. These checks validate wording and preserved software contracts; they add no anatomical or biological terminal acceptance.
