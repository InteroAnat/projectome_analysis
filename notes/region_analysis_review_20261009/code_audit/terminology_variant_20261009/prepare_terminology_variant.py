"""Stage wording changes without writing production sources."""
from pathlib import Path
import hashlib,json,shutil,difflib
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
STAGE=BASE/'staged/main_scripts'
PACKAGE=STAGE/'region_analysis';PACKAGE.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'main_scripts/region_analysis').glob('*.py'):shutil.copyfile(p,PACKAGE/p.name)
for n in ('neuro_tracer.py','region_labels.py','swc_validation.py'):shutil.copyfile(ROOT/'main_scripts'/n,STAGE/n)

COMMON=[
('TERMINAL REGION DISTRIBUTION REPORT','ENDPOINT-TARGET REGION DISTRIBUTION REPORT'),
('Terminal Region Distribution','Endpoint-Target Region Distribution'),
('Total projection sites:','Total endpoint-target entries:'),
('Known sites:','Known endpoint-target entries:'),
('Unknown sites:','Unresolved endpoint-target entries:'),
('PROJECTION SITES STATISTICS (KNOWN REGIONS ONLY)','ENDPOINT-TARGET REGION COUNTS (KNOWN REGIONS ONLY)'),
('Known Projection Sites (Excluding Unknown)','Known Endpoint-Target Regions (Unresolved Excluded)'),
('Total known sites:','Total known endpoint-target entries:'),
('Mean known sites per neuron:','Mean known endpoint-target regions per neuron:'),
('Distribution of known sites per neuron:','Distribution of known endpoint-target regions per neuron:'),
('{int(sites)} site(s):','{int(sites)} endpoint-target region(s):'),
('Unknown Projection Sites (Excluded from Plot)','Unresolved Endpoint-Target Regions (Excluded from Plot)'),
('Neurons with unknown sites:','Neurons with unresolved endpoint-target regions:'),
('Total unknown sites excluded:','Total unresolved endpoint-target entries excluded:'),
('Mean unknown sites per neuron:','Mean unresolved endpoint-target regions per neuron:'),
]
changes={
'plotting.py':COMMON+[
('    needed = {"N_Ipsilateral", "N_Contralateral"}',
 '    """Plot distinct endpoint-target counts and retained reconstruction lengths.\n\n    Laterality_Index is the contralateral length fraction Contra/(Ipsi+Contra),\n    0..1, excluding unresolved lengths. It is not the signed contrast\n    (Contra-Ipsi)/(Contra+Ipsi), -1..1. Source units are never converted here.\n    """\n    needed = {"N_Ipsilateral", "N_Contralateral"}'),
('f"Site Distribution (N={total_n})"','f"Endpoint-target entries (N={total_n} neurons)"'),
('Terminal Regions','Endpoint-Target Regions'),
('Number of Neurons Projecting','Neurons with an endpoint-target region'),
('Entries are distinct target regions per neuron, not biological terminal/bouton counts.',
 'Entries are distinct endpoint-target regions per neuron from legacy all-compartment leaves; biological terminals/boutons are not verified.'),
('Projection Sites Count per Neuron','Distinct Endpoint-Target Regions per Neuron'),
('Distinct target-region','Distinct endpoint-target region'),
('Laterality Index\\n(0=ipsi, 1=contra)','Contralateral length fraction\\nContra/(Ipsi+Contra), 0 to 1'),
('Laterality Index by Type','Contralateral length fraction by type'),
('set_xlabel("Laterality Index")','set_xlabel("Contra/(Ipsi+Contra), 0 to 1")'),
('LI Distribution','Contralateral length fraction'),
('Length by Laterality','Retained reconstruction length by side'),
('axes[2].set_title("Length",','axes[2].set_title("Retained reconstruction length",'),
('Top Projections ({neuron_id})','Top retained regional lengths (source unit; {neuron_id})'),
('axes[1].set_title("Distribution")','axes[1].set_title("Share of retained reconstruction length")'),
],
'population.py':COMMON+[
('then strips prefixes for clean output.',
 'then removes hemisphere prefixes using collision-safe target labels.\n        Cortical/subcortical homonyms retain separate C_/S_ namespaces.'),
('Tuple of (ipsi_df, contra_df) with prefixes stripped.',
 'Tuple of (ipsi_df, contra_df) with collision-safe regional labels.'),
('  Unknown includes explicit unknown, outside, unmapped, absent and invalid targets.',
 '  Entries are distinct endpoint-target regions per neuron from legacy all-compartment leaves; biological terminals/boutons are not verified.\\n  Unknown includes explicit unknown, outside, unmapped, absent and invalid targets.'),
('population.py - Population-level batch neuron analysis.',
 'population.py - Population-level batch reconstruction analysis.\n\nRegional projection values are retained reconstruction lengths in Length_Unit\n(or unspecified source units). Total_Length is the whole computed edge total.\nTerminal_Count counts distinct endpoint-target regions per neuron from legacy\nleaves of all SWC compartments, not verified biological terminals or boutons.'),
('"""Neuron x region matrix in the existing atlas-voxel length unit."""','"""Neuron x region retained reconstruction lengths in the source unit."""'),
('"""Sheet 4: log10(length + 1)."""','"""Display log10(retained regional length + 1), dependent on source unit."""'),
('"""Sheet 5: Long-format terminals."""','"""Distinct endpoint-target regions from all-compartment legacy leaves.\n\n        These rows do not establish biological terminal sites or boutons.\n        """'),
('"""Sheet 6: Per-neuron laterality scalars."""','"""Per-neuron counts and lengths relative to the resolved soma side.\n\n        Laterality_Index is Contra/(Ipsi+Contra), 0..1; unresolved lengths are\n        excluded. The separate signed contrast (Contra-Ipsi)/(Contra+Ipsi)\n        ranges from -1 to 1 and is not produced by this column. A zero\n        resolved denominator remains missing.\n        """'),
('atlas voxels ({pct:.1f}%)','source units (see Length_Unit; {pct:.1f}%)'),
('LI: {row.get(\'Laterality_Index\', \'?\')}','Contra/(Ipsi+Contra), 0..1: {row.get(\'Laterality_Index\', \'?\')}'),
('  Length: {row[\'Total_Length\']:.3f}','  Total reconstruction length (source unit; see Length_Unit): {row[\'Total_Length\']:.3f}'),
('  Terminals ({row[\'Terminal_Count\']}):','  Distinct endpoint-target regions (legacy all compartments; {row[\'Terminal_Count\']}):'),
('Top projections (atlas voxels -> log10 strength):','Top retained regional lengths (source unit -> log10(length + 1)):'),
('{length:.2f} atlas voxels ->','{length:.2f} source units ->'),
('f"Projection Length by Soma Region ({stat})"','f"Total reconstruction length by soma region (source unit; display label={stat})"'),
('"Projection Length by Soma Region"','"Total reconstruction length by soma region (source unit)"'),
('Total ipsi terminal regions:','Total ipsilateral endpoint-target entries:'),
('Total contra terminal regions:','Total contralateral endpoint-target entries:'),
('Laterality Index - mean:','Contralateral length fraction [Contra/(Ipsi+Contra), 0..1] - mean:'),
('Projection Strength (Population)','Retained Regional Reconstruction Lengths (Source Units)'),
('Raw length: mean=','Retained regional length: mean='),
('{arr.max():.2f} atlas voxels','{arr.max():.2f} source units (see Length_Unit)'),
('Log strength: mean=','Display log10(length + 1): mean='),
('Total length - Mean:','Total reconstruction length - Mean:'),
('{df[\'Total_Length\'].std():.2f} atlas voxels','{df[\'Total_Length\'].std():.2f} source units (see Length_Unit)'),
('Unique terminal-region count - Mean:','Distinct endpoint-target regions (legacy all compartments) - Mean:'),
('one row per unique terminal region','one row per distinct endpoint-target region; legacy all compartments'),
('Projection Strength Sheets (split by laterality):','Projection Strength Sheets (display log10(retained length + 1), source-unit dependent):'),
('"""Report 2/3: Terminal distribution statistics."""','"""Report 2/3: Distinct endpoint-target regions from all-compartment legacy leaves."""'),
('"""Report 3/3: Projection sites count + outlier statistics."""','"""Report 3/3: Known distinct endpoint-target region counts and outliers."""'),
],
'laterality_projection_analysis.py':[
('including lengths and strengths at multiple hierarchy levels.',
 'including retained reconstruction lengths in the source unit and display\nvalues log10(length + 1). Hierarchy exports use PopulationRegionAnalysis.\nLegacy endpoint labels come from leaves of all SWC compartments; they do\nnot verify biological terminal sites or boutons.'),
('- Projection display strengths log10(length + 1)','- Display values log10(retained regional length + 1), source-unit dependent'),
('description="Laterality-based projection analysis"','description="Split retained reconstruction lengths by soma-relative side; display values are log10(length + 1) in the declared source unit"'),
('help="Input Excel or CSV file with neuron data"','help="Excel/CSV with exact neuron identities (NeuronUID or SampleID+NeuronID), soma labels and absolute target-length dictionaries"'),
('help="Output Excel file path"','help="Fresh output Excel path; existing files are protected"'),
('help="Column name for projection lengths"','help="Retained reconstruction-length dictionary column; source units are preserved and unresolved-side lengths are reported separately"'),
]}
hashes={};patch=[]
for name,pairs in changes.items():
    source=ROOT/'main_scripts/region_analysis'/name
    original=source.read_text(encoding='utf-8');text=original
    for old,new in pairs:
        if old not in text:raise RuntimeError(f'Missing wording anchor {name}: {old!r}')
        text=text.replace(old,new)
    staged=PACKAGE/name;staged.write_text(text,encoding='utf-8')
    hashes[name]={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'staged_sha256':hashlib.sha256(staged.read_bytes()).hexdigest()}
    patch.extend(difflib.unified_diff(original.splitlines(keepends=True),text.splitlines(keepends=True),fromfile='a/main_scripts/region_analysis/'+name,tofile='b/main_scripts/region_analysis/'+name))
(BASE/'proposed_region_terminology.patch').write_text(''.join(patch),encoding='utf-8')
(BASE/'terminology_receipt.json').write_text(json.dumps({'status':'staged_only','files':hashes,'scope':'User-facing strings/docstrings only; API names, data keys, file names and calculations unchanged','tests':'pending'},indent=2)+'\n',encoding='utf-8')
print('Prepared wording-only staged files:',','.join(hashes))
