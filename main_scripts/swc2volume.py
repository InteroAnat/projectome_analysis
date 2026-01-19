import navis
from PIL import Image 
import numpy as np 
import nrrd
import os,sys
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
sys.path.append(neurovis_path)
import IONData
import SwcLoader
import nibabel


import neuro_tracer as nt

def swc2volume(sampleid, name):
    iondata = IONData.IONData()
    res, path = iondata.getAnnotation()

    # Read the data back from file
    readdata, header = nrrd.read(r'D:\projectome_analysis\atlas\nmt_structure.nrrd')
    readdata[:] = 0
  
    tree = nt.neuro_tracer()
    tree.process(sampleid,name,nii_space='monkey')
    for p in tree.nodes.values():

        readdata[int(p.x_nii),int(p.y_nii),int(p.z_nii)] = 400
    
    
    project_root = os.path.dirname(os.getcwd())
    save_folder = os.path.join(project_root, 'resource','nrrd', sampleid)
    nrrd_file_path = os.path.join(save_folder, name.replace('.swc', '') + '.nrrd')
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Created directory: {save_folder}")

    nrrd.write(nrrd_file_path, readdata)
    print(f"Wrote file: {nrrd_file_path}")

if __name__ == "__main__":
    swc2volume('251637', '157.swc')
    
    
