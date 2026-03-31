#!/usr/bin/env python3
"""
Mesh Visualisation - IDE Runner
================================
Generate 3D meshes from atlas regions or template volumes.

INSTRUCTIONS:
1. Edit the SETTINGS section below
2. Run the script (Ctrl+F5 in VS Code, or click Run)

DEBUGGING ZERO MATCHES:
- If you get 0 matches, the region might not exist at your chosen level
- Try different HIERARCHY_LEVEL values (1-6)
- Check the ARM_key_all.txt file to see valid regions at each level

Example: Extract insula regions
-------------------------------
    MODE = "atlas"
    ATLAS_REGION_NAMES = ["insula"]
    HIERARCHY_LEVEL = 6

Example: Whole brain template
-----------------------------
    MODE = "template"
"""

import os
import sys
import csv

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from subcortex_visualization.monkey_atlas_guide.volume_to_mesh_mz3_MONKEY import (
    volume_to_mesh_mz3_monkey_atlas,
    volume_to_mesh_mz3_template,
)


# =============================================================================
# SETTINGS - EDIT THESE VALUES
# =============================================================================

# ----- Mode Selection -----
# "atlas"  = Extract specific regions from labeled atlas
# "template" = Create whole-brain mesh from template
MODE = "atlas"

# ----- Atlas Settings (use when MODE = "atlas") -----
# Input files
ATLAS_VOLUME = r"D:/projectome_analysis/atlas/ARM_in_NMT_v2.1_sym.nii.gz"
ARM_KEY_FILE = r"D:/projectome_analysis/atlas/ARM_key_all.txt"

# Hierarchy level: 1-6 (ARM atlas has 6 levels)
# Level 1 = broadest regions, Level 6 = most fine-grained
HIERARCHY_LEVEL = 6

# Region selection (provide at least one selector)
# 
# ATLAS_REGION_NAMES: Substring match on Full_Name (case-insensitive)
#   Example: "insula" matches "Anterior Insula", "Posterior Insula", etc.
#   This can match MULTIPLE regions if they share the substring.
#
# ATLAS_REGION_IDS: Exact match on region ID number
#   Example: [8] matches only region with ID=8
#
# ATLAS_REGION_ABBREVS: Exact match on abbreviation (case-insensitive)
#   Example: "AINS" matches exactly "AINS"
#
# Multiple selectors are combined (OR logic) - any match is included.

ATLAS_REGION_NAMES = ["insula"]   # Substring match on full names
ATLAS_REGION_IDS = []             # Exact ID match (e.g., [8, 9])
ATLAS_REGION_ABBREVS = []         # Exact abbreviation match (e.g., ["AINS", "PINS"])

# Output
ATLAS_OUTPUT_DIR = r"D:/projectome_analysis/mesh_outputs/atlas_regions"
ATLAS_OUTPUT_FILE = f"regions_L{HIERARCHY_LEVEL}.mz3"

# Processing options
MIN_VOXELS = 50          # Minimum voxels to create mesh (higher = fewer small regions)
SMOOTH_SIGMA = 0.4       # Smoothing: 0.2 (sharp), 0.4 (balanced), 0.8+ (smooth)


# ----- Template Settings (use when MODE = "template") -----
# Input file
TEMPLATE_VOLUME = r"D:/projectome_analysis/atlas/NMT_v2.1_sym/NMT_v2.1_sym_brainmask.nii.gz"

# Threshold for binarization (0.0-1.0)
TEMPLATE_THRESHOLD = 0.5

# Output
TEMPLATE_OUTPUT_DIR = r"D:/projectome_analysis/mesh_outputs/template"
TEMPLATE_OUTPUT_FILE = "whole_brain.mz3"

# Processing options
TEMPLATE_SMOOTH_SIGMA = 0.4


# =============================================================================
# VALIDATION & DEBUGGING
# =============================================================================

def load_arm_key(key_path):
    """Load ARM key to check available regions."""
    rows = []
    with open(key_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows.append({
                    "Index": int(row["Index"]),
                    "Abbreviation": (row["Abbreviation"] or "").strip(),
                    "Full_Name": (row["Full_Name"] or "").strip(),
                    "First_Level": int(row["First_Level"]),
                    "Last_Level": int(row["Last_Level"]),
                })
            except Exception:
                continue
    return rows


def find_regions_at_level(arm_rows, level, search_term=None):
    """Find all regions available at a specific hierarchy level."""
    regions = []
    for r in arm_rows:
        if r["First_Level"] <= level <= r["Last_Level"]:
            if search_term is None or search_term.lower() in r["Full_Name"].lower():
                regions.append(r)
    return regions


def debug_region_search():
    """Debug helper to show why a search might return zero matches."""
    print("\n" + "=" * 60)
    print("DEBUG: Checking ARM key for available regions")
    print("=" * 60)
    
    arm_rows = load_arm_key(ARM_KEY_FILE)
    
    # Check all levels
    print(f"\nTotal regions in key: {len(arm_rows)}")
    
    for level in range(1, 7):
        count = sum(1 for r in arm_rows if r["First_Level"] <= level <= r["Last_Level"])
        print(f"  Level {level}: {count} regions available")
    
    # Check specific search terms
    region_names = [name.strip() for name in ATLAS_REGION_NAMES if name.strip()]
    
    if region_names:
        print(f"\n--- Searching for: {region_names} ---")
        
        for level in [1, 3, 6]:  # Check key levels
            print(f"\nLevel {level}:")
            matches = []
            for r in arm_rows:
                if r["First_Level"] <= level <= r["Last_Level"]:
                    for term in region_names:
                        if term.lower() in r["Full_Name"].lower():
                            matches.append(r)
                            break
            
            if matches:
                print(f"  ✓ Found {len(matches)} matching regions:")
                for m in matches[:5]:  # Show first 5
                    print(f"    - ID {m['Index']}: {m['Full_Name']} (abbr: {m['Abbreviation']})")
                if len(matches) > 5:
                    print(f"    ... and {len(matches) - 5} more")
            else:
                # Show similar regions at this level
                print(f"  ✗ No matches at level {level}")
                all_at_level = [r for r in arm_rows if r["First_Level"] <= level <= r["Last_Level"]]
                print(f"    Available at level {level}: {len(all_at_level)} total regions")
                if level != 6 and all_at_level:
                    print(f"    Sample: {', '.join([r['Full_Name'] for r in all_at_level[:3]])}...")


def validate_settings():
    """Check that required settings are provided."""
    errors = []
    
    if MODE not in ("atlas", "template"):
        errors.append(f"MODE must be 'atlas' or 'template', got '{MODE}'")
    
    if MODE == "atlas":
        if not os.path.exists(ATLAS_VOLUME):
            errors.append(f"ATLAS_VOLUME not found: {ATLAS_VOLUME}")
        if not os.path.exists(ARM_KEY_FILE):
            errors.append(f"ARM_KEY_FILE not found: {ARM_KEY_FILE}")
        
        # Check that at least one region selector is provided
        has_names = bool(ATLAS_REGION_NAMES)
        has_ids = bool(ATLAS_REGION_IDS)
        has_abbrevs = bool(ATLAS_REGION_ABBREVS)
        
        if not (has_names or has_ids or has_abbrevs):
            errors.append(
                "At least one region selector required. Set one of:\n"
                "  ATLAS_REGION_NAMES = ['insula']  # substring match\n"
                "  ATLAS_REGION_IDS = [8, 9]        # exact ID match\n"
                "  ATLAS_REGION_ABBREVS = ['AINS']  # exact abbreviation"
            )
    
    if MODE == "template":
        if not os.path.exists(TEMPLATE_VOLUME):
            errors.append(f"TEMPLATE_VOLUME not found: {TEMPLATE_VOLUME}")
    
    if errors:
        print("\n" + "=" * 60)
        print("CONFIGURATION ERRORS:")
        print("=" * 60)
        for i, err in enumerate(errors, 1):
            print(f"\n{i}. {err}")
        print("\n" + "=" * 60)
        print("Please fix the SETTINGS section at the top of this file.")
        print("=" * 60)
        return False
    
    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_atlas():
    """Run atlas region extraction."""
    print("\n" + "=" * 60)
    print("ATLAS MODE: Extracting labeled regions")
    print("=" * 60)
    
    # Clean up region selectors (strip whitespace, remove empty strings)
    region_names = [name.strip() for name in ATLAS_REGION_NAMES if name.strip()]
    region_ids = [int(rid) for rid in ATLAS_REGION_IDS if rid != ""]
    region_abbrevs = [abbr.strip() for abbr in ATLAS_REGION_ABBREVS if abbr.strip()]
    
    print(f"\nInput:")
    print(f"  Volume: {ATLAS_VOLUME}")
    print(f"  Key:    {ARM_KEY_FILE}")
    print(f"  Level:  {HIERARCHY_LEVEL}")
    
    if region_names:
        print(f"\nRegion name filter (substring match, case-insensitive):")
        for name in region_names:
            print(f"  - '{name}'")
        print("  Note: Matches any region containing this substring in Full_Name")
    
    if region_ids:
        print(f"\nRegion ID filter (exact match):")
        print(f"  {region_ids}")
    
    if region_abbrevs:
        print(f"\nRegion abbreviation filter (exact match, case-insensitive):")
        for abbr in region_abbrevs:
            print(f"  - '{abbr}'")
    
    # Convert empty lists to None for the function
    region_names_param = region_names if region_names else None
    region_ids_param = region_ids if region_ids else None
    region_abbrevs_param = region_abbrevs if region_abbrevs else None
    
    # Run the extraction
    result = volume_to_mesh_mz3_monkey_atlas(
        input_volume=ATLAS_VOLUME,
        hierarchy_level=HIERARCHY_LEVEL,
        output_path=ATLAS_OUTPUT_DIR,
        out_file=ATLAS_OUTPUT_FILE,
        arm_key_file=ARM_KEY_FILE,
        region_names=region_names_param,
        region_ids=region_ids_param,
        region_abbrevs=region_abbrevs_param,
        min_voxels=MIN_VOXELS,
        smooth_sigma=SMOOTH_SIGMA,
    )
    
    # Report results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if result.get("out_file"):
        print(f"\n✓ Mesh saved: {result['out_file']}")
    
    matched_count = 0
    if result.get("selection_report"):
        report = result["selection_report"]
        matched = report.get("matched_ids", [])
        unmatched = report.get("unmatched", [])
        dropped = report.get("dropped_by_level", [])
        matched_count = len(matched)
        
        print(f"\nRegions matched: {matched_count}")
        if matched:
            print(f"  IDs: {matched}")
        
        if unmatched:
            print(f"\n⚠ Not found: {unmatched}")
            print("   Check spelling or try different search terms")
        
        if dropped:
            print(f"\n⚠ Not available at level {HIERARCHY_LEVEL}: {dropped}")
            print(f"   These regions exist but not at level {HIERARCHY_LEVEL}")
            print(f"   Try a different HIERARCHY_LEVEL (1-6)")
    
    print(f"\nTotal regions processed: {result.get('total_processed', 0)}")
    
    # Debug if zero matches
    if matched_count == 0:
        print("\n" + "!" * 60)
        print("ZERO MATCHES - Running debug to help find the issue")
        print("!" * 60)
        debug_region_search()
    
    return result


def run_template():
    """Run template whole-brain mesh generation."""
    print("\n" + "=" * 60)
    print("TEMPLATE MODE: Creating whole-brain mesh")
    print("=" * 60)
    
    print(f"\nInput:")
    print(f"  Volume:    {TEMPLATE_VOLUME}")
    print(f"  Threshold: {TEMPLATE_THRESHOLD}")
    
    result = volume_to_mesh_mz3_template(
        input_volume=TEMPLATE_VOLUME,
        output_path=TEMPLATE_OUTPUT_DIR,
        out_file=TEMPLATE_OUTPUT_FILE,
        threshold_low=TEMPLATE_THRESHOLD,
        smooth_sigma=TEMPLATE_SMOOTH_SIGMA,
    )
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if result.get("out_file"):
        print(f"\n✓ Mesh saved: {result['out_file']}")
    
    print(f"  Voxels in mask: {result.get('voxel_count', 'N/A')}")
    
    return result


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("MESH VISUALISATION")
    print("=" * 60)
    print(f"\nMode: {MODE.upper()}")
    
    # Validate settings before running
    if not validate_settings():
        return None
    
    # Run based on mode
    if MODE == "atlas":
        result = run_atlas()
    else:
        result = run_template()
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")
    
    return result


if __name__ == "__main__":
    result = main()
