"""
run_analysis.py - Full pipeline.

Priority: CSV hierarchy > ARM key
          CSV covers L3-L6 (your regions)
          ARM key covers L1-L2 (brain, cerebrum — fallback)
"""
#%%
import nibabel as nib
import pandas as pd
import os
from region_analysis import PopulationRegionAnalysis


def main(neuron_ids, sample_id="251637", show_plots=False):
    ATLAS_PATH     = r"D:\projectome_analysis\atlas\ARM_in_NMT_v2.1_sym.nii.gz"
    TABLE_PATH     = r"D:\projectome_analysis\atlas\ARM_key_all.txt"
    TEMPLATE       = r"D:\projectome_analysis\atlas\NMT_v2.1_sym\NMT_v2.1_sym\NMT_v2.1_sym_SS.nii.gz"
    ctx_HIERARCHY_CSV = r"D:\projectome_analysis\atlas\CHARM_key_table.csv"
    subctx_HIERARCHY_CSV = r"D:\projectome_analysis\atlas\SARM_key_table.csv"
    atlas_data  = nib.load(ATLAS_PATH).get_fdata()
    atlas_table = pd.read_csv(TABLE_PATH, delimiter="\t")
    template    = nib.load(TEMPLATE)
    # Validate input
    if not neuron_ids:
        print("[ERROR] No neuron IDs provided")
        return None
    
    if isinstance(neuron_ids, str):
        neuron_ids = [neuron_ids]  # Convert single ID to list
    
    pop = PopulationRegionAnalysis(
        sample_id=sample_id,
        atlas=atlas_data,
        atlas_table=atlas_table,
        template_img=template,
        arm_key_path=TABLE_PATH,        
        cortex_hierarchy_csv=ctx_HIERARCHY_CSV, 
        subcortical_hierarchy_csv=subctx_HIERARCHY_CSV,
            
        create_output_folder=True,
        show_plots=False
    )
    pop.process(neuron_id=neuron_ids, level=6)

    # Verify priority
    # pop.debug_hierarchy("CL_area_24a")
    
    output_path = pop.save_all(
        include_strength_levels=[3,6],
        generate_plots=True,
       
    )
    
    counts = pop.get_neuron_count()
    print(f"\n[COMPLETE] {output_path}")
    print(f"Processed: {counts['processed']} / {counts['total']} neurons ({counts['completion_rate']}%)")




#%%

neuron_tables = r'D:\projectome_analysis\main_scripts\neuron_tables'
ins_table = 'INS_df_v3.xlsx'
neuron_df= pd.read_excel(os.path.join(neuron_tables,ins_table))


ins_swcIDs= neuron_df['NeuronID']
ins_swcIDs= ins_swcIDs.tolist()

# %%
main(ins_swcIDs)
# %%
