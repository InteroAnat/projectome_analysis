# %% import 

import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")
sys.path.append("../neuronVis")
import pandas as pd
import Scene
import numpy as np
import BrainRegion as BR 
import IONData ,nrrd 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://matplotlib_inline.backend_inline')
%matplotlib inline
iondata = IONData.IONData()
br = BR.BrainRegion()
br.praseJson()

# %%
neuronSpcd = Scene.scene2List('../resource/scene/spcdNeurons.nv')
neuronth = Scene.scene2List('../resource/scene/th183neurons.nv')
neuronpb = Scene.scene2List('../resource/scene/pb.nv')

# %% TH region

thregion = br.getRegion('TH')
res = iondata.getStructureMask25(thregion.ID)
readdata,header = nrrd.read(res[1],index_order='F')
mins = [220,103,110]
length=[142,122,235]
maxs=[362,225,345]
thdata=readdata[mins[0]:(mins[0]+length[0]),mins[1]:(mins[1]+length[1]),mins[2]:(mins[2]+length[2])]
originTHdata = copy.deepcopy(thdata)
# thregion.pointsIn()

# %% PB region

pbregion = br.getRegion('PB')
res = iondata.getStructureMask25(pbregion.ID)
readPBdata,header = nrrd.read(res[1],index_order='F')
mins = [391,139,138]
length=[50,64,180]
maxs=[441,203,318]

pbdata=readPBdata[mins[0]:(mins[0]+length[0]),mins[1]:(mins[1]+length[1]),mins[2]:(mins[2]+length[2])]
originPBdata = copy.deepcopy(pbdata)

# %%
secondneurons={}
for secondNeuron in neuronpb:
    secondNeuronTree = iondata.getNeuronTreeByID(secondNeuron['sampleid'],secondNeuron['name'])
    secondneurons[secondNeuron['sampleid']+secondNeuron['name']]=secondNeuronTree
# %%
neuronDICT={}
for neuron in neuronSpcd:
    print(neuron)
    linkneurons=[]
    neuronTree = iondata.getNeuronTreeByID(neuron['sampleid'],neuron['name'])
    ps = neuronTree.terminals
    thdata = copy.deepcopy(originPBdata)
    for p in ps:
        if p.x/25<maxs[0] and p.y/25<maxs[1] and p.z/25<maxs[2] and p.x/25>mins[0] and p.y/25>mins[1] and p.z/25>mins[2]:
            thdata[int(p.x/25-mins[0]),int(p.y/25-mins[1]),int(p.z/25-mins[2])]= thdata[int(p.x/25-mins[0]),int(p.y/25-mins[1]),int(p.z/25-mins[2])]+1
    if np.sum(thdata)==np.sum(originPBdata):
        print('not project to TH')
        continue
    for secondNeuron in neuronpb:
        secondNeuronTree = secondneurons[secondNeuron['sampleid']+secondNeuron['name']]
        rootP = secondNeuronTree.root
        rootcoord = [int(rootP.x/25-mins[0]),int(rootP.y/25-mins[1]),int(rootP.z/25-mins[2])]
        if rootcoord[0]>0 and rootcoord[0]<length[0] and rootcoord[1]>0 and rootcoord[1]<length[1] and rootcoord[2]>0 and rootcoord[2]<length[2] :
            origindataSum = (np.sum(originPBdata[(rootcoord[0]-2):(rootcoord[0]+2),(rootcoord[1]-2):(rootcoord[1]+2),(rootcoord[2]-2):(rootcoord[2]+2)]))
            dataSum = (np.sum(thdata[(rootcoord[0]-2):(rootcoord[0]+2),(rootcoord[1]-2):(rootcoord[1]+2),(rootcoord[2]-2):(rootcoord[2]+2)]))
            if dataSum-origindataSum>1:
                print(dataSum-origindataSum)
                linkneurons.append(secondNeuron)
    if len(linkneurons)>0:
        neuronDICT[neuron['sampleid']+neuron['name']]=linkneurons

        
    
# %%
import random
linkedNeurons=[]
for key,val in neuronDICT.items():
    neuron={}
    neuron['sampleid']=key[0:6]
    neuron['name']=key[6:]
    neuron['mirror']=0
    color={'r':str(random.randint(0,255)),'g':str(random.randint(0,255)),'b':str(random.randint(0,255))}
    neuron['color']=color
    count=0
    for neu in val:
        if neu not in linkedNeurons:
            neu['color']=color
            linkedNeurons.append(neu)
            count=count+1
    if count>0:
        linkedNeurons.append(neuron)
    
print(linkedNeurons)
Scene.createScene(linkedNeurons,'../resource/scene/linkedPBNeurons.nv')


# %%
import SwcLoader
neuronTree = SwcLoader.NeuronTree()
neuronTree.readFile('../resource/210661052.swc')
import Visual as nv

neuronvis = nv.neuronVis(size=(1700,1000),renderModel=0)

neuronvis.render.setBackgroundColor((0.0,0.0,0.,1.0))

neuronvis.setLineWidth(1)
neuronvis.addRegion('STR')

neuronvis.addNeuronTree(neuronTree)

neuronvis.render.run()
# %%
