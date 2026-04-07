import navis
from PIL import Image 
import numpy as np 
import nrrd
import os
import IONData
import SwcLoader

def swc2volume(sampleid, name):
    iondata = IONData.IONData()
    res, path = iondata.getAnnotation()

    # Read the data back from file
    readdata, header = nrrd.read(path)
    readdata[:] = 0
    tree = iondata.getNeuronTreeByID(sampleid, name)
    
    for p in tree.xyz:
        readdata[int(p[0] / 10.0), int(p[1] / 10.0), int(p[2] / 10.0)] = 400

    # Get the absolute path of the current directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Construct the full path for the NRRD folder
    nrrd_folder = os.path.join(base_dir, '..', 'resource', 'nrrd', sampleid)
    nrrd_file_path = os.path.join(nrrd_folder, name.replace('.swc', '') + '.nrrd')
    
    # Debugging: Print the constructed paths
    print(f"Base directory: {base_dir}")
    print(f"NRRD folder path: {nrrd_folder}")
    print(f"NRRD file path: {nrrd_file_path}")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(nrrd_folder):
        os.makedirs(nrrd_folder)
        print(f"Created directory: {nrrd_folder}")
    
    # Write the data to the NRRD file
    nrrd.write(nrrd_file_path, readdata)
    print(f"Wrote file: {nrrd_file_path}")

if __name__ == "__main__":
    swc2volume('211984', '045.swc')
