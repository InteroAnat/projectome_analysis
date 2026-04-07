#%%
import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")
import pandas as pd
import Scene
import BrainRegion as BR 
import IONData 
iondata = IONData.IONData()

isosampleid = iondata.getSampleInfo(projectID='ISO')
len(isosampleid)

# %%
import BrainRegion as BR 
import numpy as np
br = BR.BrainRegion()
br.praseJson()

count=0
neuronRegion={}
MBneurons=[]
for sample in isosampleid:

    sampleid =sample['fMOST_id']
    neurons = iondata.getNeuronListBySampleID(str(sampleid))
    for neuron in neurons:
        prop = iondata.getNeuronPropertyByID(neuron['sampleid'],neuron['name'])
        if prop['somaregion'] not in neuronRegion.keys():
            neuronRegion[prop['somaregion']]=[]
        neuronRegion[prop['somaregion']].append(neuron)
        brproperty=BR.RegionProperty(copy.deepcopy(br))
        brproperty.setProperty(prop['projectregion'])
        regionsum = brproperty.getSumProperty('CB')
        if regionsum>0:
            MBneurons.append(neuron)

#%%
print(len(MBneurons))
for neuron in MBneurons:
    neuron['mirror']=0
Scene.createScene(neuronlist=MBneurons,filename='../resource/scene/ISO2CBneurons.nv')