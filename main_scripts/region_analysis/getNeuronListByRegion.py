#%%
import sys, os, re
import pandas as pd
from typing import List, Union
from region_labels import (CURATED_INSULA_LABELS, normalize_region_label,
                           normalize_portal_region_label)

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
    search_abbreviation: bool = True,
    return_ids_only: bool = False,
    verbose: bool = True,
) -> Union[pd.DataFrame, List[str]]:
    """
    Get neurons filtered by brain region keywords.
    
    Args:
        sample_id: fMOST sample ID (e.g., '251637')
        region_keywords: Keyword(s) to search in atlas Full_Name and (optionally) Abbreviation
                        (e.g., 'insula' or ['motor', 'cortex'])
        atlas_path: Path to ARM_key_all.txt atlas file
        search_abbreviation: If True, also search keywords in atlas Abbreviation (acronyms)
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
    region_keywords = [k.strip() for k in region_keywords if isinstance(k, str) and k.strip()]
    if not region_keywords:
        if verbose:
            print('[WARN] Empty region_keywords')
        return [] if return_ids_only else pd.DataFrame()
    
    # Load atlas and extract target regions
    atlas_df = pd.read_csv(atlas_path, delimiter='\t')
    keyword_regex = '|'.join(re.escape(k) for k in region_keywords)
    full_name_mask = atlas_df['Full_Name'].fillna('').str.contains(keyword_regex, case=False, regex=True)
    if search_abbreviation:
        abbr_mask = atlas_df['Abbreviation'].fillna('').str.contains(keyword_regex, case=False, regex=True)
        mask = full_name_mask | abbr_mask
    else:
        mask = full_name_mask
    roi_abbr = atlas_df.loc[mask, 'Abbreviation'].dropna().tolist()
    
    if not roi_abbr:
        if verbose:
            print(f'[WARN] No atlas regions found for: {region_keywords}')
        return [] if return_ids_only else pd.DataFrame()
    
    # Exact bases include both the combined atlas leaf and its slash members.
    base_names = set()
    for abbr in roi_abbr:
        name = normalize_region_label(abbr)
        base_names.add(name)
        base_names.update(p.strip() for p in name.split('/'))
    if any(keyword.lower() in {"insula", "insular"} for keyword in region_keywords):
        # The curated reference vocabulary is a separate candidate source.
        # Keep its complete token (e.g. IDD5); do not fuzzy-match ID prefixes.
        base_names.update(CURATED_INSULA_LABELS)
    full_names = {str(abbr).strip().upper() for abbr in roi_abbr}
    
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
    regions = neurons_df.get('region', pd.Series("", index=neurons_df.index))
    neurons_df['region_clean'] = regions.map(
        lambda value: value.strip().replace('\r', '').replace('\n', '')
        if isinstance(value, str) else "")
    
    # Match function
    def is_match(region):
        text = region.strip().upper()
        if text.startswith(("CL_", "CR_", "SL_", "SR_")):
            # Explicit atlas prefixes distinguish cortical Pi from pineal Pi.
            if text in full_names:
                return True
            head, separator, tail = text.rpartition("_")
            return bool(separator and tail.isdigit() and head in full_names)
        return normalize_portal_region_label(region, base_names) in base_names
    
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
