import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(CURRENT_FILE_PATH+"/../neuronVis")

# print(sys.path)
import json

import IONData as IONData 
import BrainRegion as BR 
import Cluster
import Visual as nv
import SwcLoader

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn  as sns
import nrrd

f=open('../resource/ceaLocallist.json', encoding='gbk')
cealist=[]
cealist = json.load(f)

# Get downsampled CEA mask ,can also get it from downsampled the annotation 
iondata = IONData.IONData()
res,path = iondata.getFileFromServer('structure_536_CEA_100um.nrrd')
ceamask,ceamaskHeader = nrrd.read(path,index_order='F')
ceamask[:,:,int(ceamask.shape[2]/2):ceamask.shape[2]]=0
print(ceamask.shape)
index=np.where(ceamask>0)
print(index)
for i in range(len(index[0])):
    ceamask[index[0][i],index[1][i],index[2][i]]=i+1

matrix_project = np.zeros([len(cealist),len(index[0])])
num_neuron_cube=np.zeros([len(index[0])])
cubeindex_neuron=np.zeros([len(cealist)])
for i in range(len(cealist)):
    neuron = cealist[i]
    print(neuron)
    neurontree = iondata.getNeuronTreeByID(neuron['sampleid'],neuron['name'])
    hemi='left'
    rootz=neurontree.root.z
    if neurontree.root.z>5700:
        hemi='right'
        rootz=11400-rootz

    rootmoduleid= ceamask[int(neurontree.root.x/100),int(neurontree.root.y/100),int(rootz/100)]
    cubeindex_neuron[i]=rootmoduleid
    if rootmoduleid>0:
        num_neuron_cube[rootmoduleid]=num_neuron_cube[rootmoduleid]+1
    for edge in neurontree.edges:
        for p in edge.data:
            tmppz=p.z
            if hemi=='right':
                tmppz=11400-tmppz
            pmoduleid = ceamask[int(p.x/100),int(p.y/100),int(tmppz/100)]
            
            if pmoduleid==0 or p.parent is None:
                continue
            tmpparentz=p.parent.z
            if hemi=='right':
                tmpparentz=11400-tmpparentz
            parentModuleid = ceamask[int(p.parent.x/100),int(p.parent.y/100),int(tmpparentz/100)]
            if parentModuleid==pmoduleid:
                length= np.sqrt(sum((np.array([p.x,p.y,tmppz])-np.array([p.parent.x,p.parent.y,tmpparentz]))**2))
                matrix_project[i,pmoduleid]=matrix_project[i,pmoduleid]+length

print(cubeindex_neuron)
# print(num_neuron_cube)
matrix_connect = np.zeros([len(index[0]),len(index[0])])

indexcube_valid=np.where(num_neuron_cube>0)
for i in range(len(index[0])):
    tmpindex=np.where(cubeindex_neuron==i+1)
    if len(tmpindex[0])>0:
        # print(len(tmpindex[0]),tmpindex,matrix_project[tmpindex,:])
        matrix_connect[i+1,:]=matrix_project[tmpindex,:].mean(axis=1).round(2)

indexcube_valid2=np.array(indexcube_valid)[0]
matrix_valid = matrix_connect[indexcube_valid2,:]
matrix_valid=matrix_valid[:,indexcube_valid2]
matrix_valid


import seaborn	as sns
sns.heatmap(matrix_valid)
import matplotlib.pyplot as plt
fig = plt.figure()

plt.show()

# neuronvis = nv.neuronVis()
# neuronvis.addNeuronByList(cealist)
# neuronvis.addRegion('CEA')

# neuronvis.render.run()
