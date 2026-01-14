import nibabel as nib
from nilearn import plotting

    """_summary_
    """
# atlas,atlas_header=nrrd.read(r'D:\projectome_analysis\atlas\nmt_structure.nrrd')
# reso=np.diag(atlas_header['space directions'])

# charm_atlas=nib.load(r'D:\projectome_analysis\atlas\NMT_v2.0_sym\NMT_v2.0_sym\CHARM_in_NMT_v2.0_sym.nii.gz')
# sarm_atlas=nib.load(r'D:\projectome_analysis\atlas\NMT_v2.0_sym\NMT_v2.0_sym\SARM_in_NMT_v2.0_sym.nii.gz')
# charm_atlas_data=charm_atlas.get_fdata();sarm_atlas_data=sarm_atlas.get_fdata()
# sarm_atlas_data=sarm_atlas_data+300
# combined_atlas_data=charm_atlas_data+sarm_atlas_data
# combined_atlas_nii=nib.Nifti1Image(combined_atlas_data,charm_atlas.affine)
# nib.save(combined_atlas_nii,r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz')
combined_atlas_nii=nib.load(r'D:\projectome_analysis\atlas\nmt_structure_with_hiearchy.nii.gz')
atlas=combined_atlas_nii.get_fdata()
plotting.plot_anat(combined_atlas_nii.slicer[:,:,:,0,1])    