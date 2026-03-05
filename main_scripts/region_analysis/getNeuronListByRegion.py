#%%
import sys, os, re
import pandas as pd
from typing import List, Union

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NEUROVIS_PATH = r'D:\projectome_analysis\neuron-vis\neuronVis'
ATLAS_PATH = r'D:\projectome_analysis\atlas\ARM_key_all.txt'
# ==============================================================================

# Setup IONData
if NEUROVIS_PATH not in sys.path:
    sys.path.append(NEUROVIS_PATH)

import IONData
_iondata = IONData.IONData()


def getNeuronListByRegion(
    sample_id: str,
    region_keywords: Union[str, List[str]],
    atlas_path: str = ATLAS_PATH,
    return_ids_only: bool = False,
    verbose: bool = True,
) -> Union[pd.DataFrame, List[str]]:
    """
    Get neurons filtered by brain region keywords.
    
    Args:
        sample_id: fMOST sample ID (e.g., '251637')
        region_keywords: Keyword(s) to search in atlas Full_Name 
                        (e.g., 'insula' or ['motor', 'cortex'])
        atlas_path: Path to ARM_key_all.txt atlas file
        return_ids_only: If True, return list of neuron IDs instead of DataFrame
        verbose: Print matching info if True
    
    Returns:
        DataFrame with filtered neurons, or list of neuron IDs if return_ids_only=True
    
    Example:
        >>> df = getNeuronListByRegion('251637', 'insula')
        >>> df = getNeuronListByRegion('251637', ['motor', 'premotor'])
        >>> ids = getNeuronListByRegion('251637', 'insula', return_ids_only=True)
    """
    # Normalize keywords to list
    if isinstance(region_keywords, str):
        region_keywords = [region_keywords]
    
    # Load atlas and extract target regions
    atlas_df = pd.read_csv(atlas_path, delimiter='\t')
    mask = atlas_df['Full_Name'].fillna('').str.contains('|'.join(region_keywords), case=False)
    roi_abbr = atlas_df.loc[mask, 'Abbreviation'].dropna().tolist()
    
    if not roi_abbr:
        if verbose:
            print(f'[WARN] No atlas regions found for: {region_keywords}')
        return [] if return_ids_only else pd.DataFrame()
    
    # Extract base names: "CL_Ial" -> "ial", "CL_Ia/Id" -> ["ia", "id"]
    base_names = set()
    for abbr in roi_abbr:
        name = abbr.replace('CL_', '').replace('CR_', '')
        base_names.update(p.strip().lower() for p in name.split('/'))
    
    if verbose:
        print(f'Atlas regions for "{region_keywords}": {len(roi_abbr)}')
        print(f'Base names: {sorted(base_names)}\n')
    
    # Load neurons
    neuron_list = _iondata.getNeuronListBySampleID(sample_id)
    if not neuron_list:
        if verbose:
            print(f'[WARN] No neurons found for sample {sample_id}')
        return [] if return_ids_only else pd.DataFrame()
    
    neurons_df = pd.DataFrame(neuron_list)
    neurons_df['region_clean'] = neurons_df['region'].str.strip().str.replace(r'[\r\n]', '', regex=True)
    
    # Match function
    def is_match(region):
        if not region:
            return False
        match = re.match(r'^([A-Za-z/]+)', region)
        if match:
            base = match.group(1).lower()
            return base in base_names or any(
                base.startswith(b) or b.startswith(base) for b in base_names
            )
        return False
    
    # Filter
    filtered = neurons_df[neurons_df['region_clean'].apply(is_match)].copy()
    
    if verbose:
        print(f'Matched: {len(filtered)} / {len(neurons_df)} neurons')
        if len(filtered) > 0:
            print(f'\nRegion breakdown:\n{filtered["region_clean"].value_counts().to_string()}')
    
    if return_ids_only:
        return filtered['name'].tolist()
    
    return filtered


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
if __name__ == '__main__':
    # Example 1: Get DataFrame
    df = getNeuronListByRegion('251637', 'insula')
    print(df[['name', 'region_clean']].head(10))
    
    print('\n' + '='*50 + '\n')
    
    # Example 2: Get just IDs
    ids = getNeuronListByRegion('251637', 'insula', return_ids_only=True)
    print(f'Neuron IDs: {ids[:10]}...')
    
    print('\n' + '='*50 + '\n')
    
    # Example 3: Multiple keywords, silent mode
    ids = getNeuronListByRegion('251637', ['motor', 'premotor'], return_ids_only=True, verbose=False)
    print(f'Motor/premotor neurons: {len(ids)}')
    
    # Example 4: Use with PopulationRegionAnalysis
    # from region_analysis.population import PopulationRegionAnalysis
    # 
    # insula_ids = getNeuronListByRegion('251637', 'insula', return_ids_only=True)
    # analysis = PopulationRegionAnalysis(sample_id='251637', atlas=atlas, atlas_table=atlas_table)
    # analysis.process(neuron_id=insula_ids)

# %%