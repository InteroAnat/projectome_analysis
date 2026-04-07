import sys,copy,os
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(CURRENT_FILE_PATH+"/../neuronVis")
import json

import IONData as IONData 
import BrainRegion as BR 

import Visual as nv
import NeuronProcess
import RegionProcess
import SwcLoader
import GeometryAdapter

import numpy as np

# get neuron list
f=open('../resource/paper1figure7list.json', encoding='gbk')
paper1figure7list=[]
paper1figure7list = json.load(f)
print(paper1figure7list)

br = BR.BrainRegion()
br.praseJson()
SSbr= br.getRegion('SS')
MObr= br.getRegion('MO')
mergebr= RegionProcess.mergeRegion(SSbr,MObr)

iondata=IONData.IONData()
centroidList=[]
for neuron in paper1figure7list:
    neuronTree = iondata.getNeuronTreeByID(neuron['sampleid'], neuron['name'])
    centroid = NeuronProcess.getCentroid(neuronTree,mergebr)
    centroidList.append([np.array(neuronTree.root.xyz),centroid])

point=RegionProcess.getIntersectionFacePoints(SSbr,MObr)

centroidListtemp=copy.deepcopy(centroidList)
for centroid in centroidListtemp:
    print(centroid)
    mindist=10000
    
    for p in point:
        dist= np.sqrt(np.sum(np.square(np.array(centroid[1])-p)))
        mindist =dist if dist<mindist else mindist
    centroid.append(np.array([mindist,0,0]))
    print(centroid)

import matplotlib.pyplot as plt
temp =np.array(centroidListtemp)
x=11.4-temp[:,0,2]/1000
y=temp[:,2,0]/1000
print(x,y)

fig = plt.figure()
fig.set_size_inches(10, 4)  
ax = fig.add_subplot(1, 2, 1)  
ax.set_xlabel("soma position")
ax.set_ylabel("centroid distance")
ax.scatter(x, y)
plt.show()

# render
neuronvis = nv.neuronVis((1600,1200))
neuronvis.addNeuronByList(paper1figure7list)
# neuronvis.render.setLookAt((0,-0,10000),(0,0,0),(0,1,0))
neuronvis.render.setView('dorsal')
neuronvis.render.setLineWidth(1.5)

# neuronvis.addRegion('SS')
# neuronvis.addRegion('MO')
neuronvis.render
neuronvis.render.savepng('../resource/test3.png')
neuronvis.render.run()
