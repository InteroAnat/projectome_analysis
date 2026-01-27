#change the header to correct one, with reference to D:\projectome_analysis\atlas\catalog_251637.
import nrrd
import numpy as np
reduced_volume,rv_header = nrrd.read (r'D:\projectome_analysis\atlas\251637_ch00ds.nrrd')
nrrd.write('output.nrrd', reduced_volume, header=rv_header)