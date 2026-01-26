import nrrd
import numpy as np

# ds_volume,ds_header = nrrd.read(r'D:\projectome_analysis\atlas\251637_ch00ds.nrrd')

# ds_header['space directions']=np.diag([100,100,90])

# # producing the correct ds_volume nrrd 
# nrrd.write(r'D:\projectome_analysis\atlas\251637_ch00ds_corrected.nrrd', ds_volume, header=ds_header)

m_reso = [0.65,0.65,3]
conversion_factor = [100/0.65,100/0.65,90/3]

ds_coord= [472,269,311 ]

m_coord = np.array(ds_coord)*np.array(conversion_factor)


cube_sizes = np.array([360,360,90])
tiff_index = m_coord/cube_sizes
