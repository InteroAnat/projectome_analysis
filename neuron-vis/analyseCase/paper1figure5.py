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
import Cluster
import Visual as nv

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn	as sns
f=open('../resource/paper1figure5list.json', encoding='gbk')
paper1figure5list=[]
paper1figure5list = json.load(f)
# print(paper1figure5list)

layer2_3list=[]
layer5list=[]
layer6list=[]
for neuron in paper1figure5list:
	if '2/3' in neuron['layer']:
		layer2_3list.append(neuron)
	if '5' in neuron['layer']:
		layer5list.append(neuron)
	if '6' in neuron['layer']:
		layer6list.append(neuron)

print(len(layer2_3list))
print(len(layer5list))
print(len(layer6list))

br = BR.BrainRegion()
br.praseJson()
df = pd.DataFrame()

iondata=IONData.IONData()

for neuron in layer6list:
	brproperty=BR.RegionProperty(copy.deepcopy(br))
	brpropertyLeft=BR.RegionProperty(copy.deepcopy(br))
	brpropertyRight=BR.RegionProperty(copy.deepcopy(br))
	prop=[]
	property = iondata.getNeuronPropertyByID(neuron['sampleid'], neuron['name'])
	brproperty.setProperty(property['projectregion'])
	brpropertyLeft.setProperty(property['projectleftregion'])
	brpropertyRight.setProperty(property['projectrightregion'])

	prop.append(property['axonlength'])

	#cortex length
	ctxsum = brproperty.getSumProperty('Isocortex')
	prop.append(ctxsum)

	#ipsi contra cortex length
	leftctxsum = brpropertyLeft.getSumProperty('Isocortex')
	rightctxsum = brpropertyRight.getSumProperty('Isocortex')

	ipsi=0
	contra=0
	if property['somapoint'][2]<5700:
		ipsi=leftctxsum
		contra=rightctxsum
	else:
		ipsi=rightctxsum
		contra=leftctxsum
	prop.append(ipsi)
	prop.append(contra)


	#motor cortex length
	mosum = brproperty.getSumProperty('MO')
	prop.append(mosum)

	#other cortex length
	prop.append(ctxsum-mosum)

	# STR
	strsum = brproperty.getSumProperty('STR')
	prop.append(strsum)

	#ipsi contra str length
	leftctxsum = brpropertyLeft.getSumProperty('STR')
	rightctxsum = brpropertyRight.getSumProperty('STR')
	ipsi=0
	contra=0
	if property['somapoint'][2]<5700:
		ipsi=leftctxsum
		contra=rightctxsum
	else:
		ipsi=rightctxsum
		contra=leftctxsum
	prop.append(ipsi)
	prop.append(contra)

	df[neuron['sampleid']+"-"+neuron['name']]=prop



df.set_index(pd.Index(['Total', 'Cortex', 'CortexIpsi', 'CortexContra','Motor','Other','STR','STRIpsi','STRContra']),inplace=True, drop = True)
df=(np.log2(df/1000.0+1))
print(df)

# cluster= Cluster.Cluster()
# cluster.fclusterDF(df)
# print(cluster.cluster)
# plt.show()

cmap= sns.dark_palette('black',reverse=True,n_colors=50)
# cmap= sns.dark_palette('blue',reverse=True,n_colors=50)
cmap2=sns.dark_palette('blue',reverse=False,n_colors=25)
cmap3=sns.dark_palette('blue',reverse=True,n_colors=25)
cmap4=sns.dark_palette('yellow',reverse=False,n_colors=50)
cmap.extend(cmap2)
cmap.extend(cmap3)
cmap.extend(cmap4)
# cmap.append(np.array([.3,.7,.6,1]))
# cmap.insert(0,np.array([.7,.7,.5,1]))

# sns.heatmap(df,cmap=cmap)


sns.clustermap(df, fmt="d",metric='euclidean',  method='average',cmap=cmap,xticklabels=True,yticklabels=True,row_cluster=False)
plt.show()

## render
# neuronvis = nv.neuronVis()
# neuronvis.addNeuronByList(layer6list[1:3])
# # neuronvis.addNeuronByID(paper1figure5list[100]['sampleid'],paper1figure5list[100]['name'])
# # neuronvis.render.setView('posterior')
# neuronvis.render.setLookAt((-10000,-10000,-10000),(0,0,0),(0,1,0))

# # # neuronvis.render.savepng('./resource/test2.png')
# neuronvis.render.run()

