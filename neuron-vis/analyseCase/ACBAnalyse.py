# %% import 

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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://matplotlib_inline.backend_inline')
%matplotlib inline
iondata = IONData.IONData()

br = BR.BrainRegion()
br.praseJson()

# %% Get all the neuron info

samples=['230012','230013','230014','230015','230016','221459']
d1samples=['230012','230013']
d2samples=['230014','230015']
ChATsamples=['230016','221459']
neurons=[]
d1neurons=[]
d2neurons=[]
ChATneurons=[]
for sample in samples:
    neurons.extend(iondata.getNeuronListBySampleID(sample))
print(len(neurons))
for sample in d1samples:
    d1neurons.extend(iondata.getNeuronListBySampleID(sample))
for sample in d2samples:
    d2neurons.extend(iondata.getNeuronListBySampleID(sample))
for sample in ChATsamples:
    ChATneurons.extend(iondata.getNeuronListBySampleID(sample))


# %% create scene
ACBneurons=[]
Otherneurons=[]
for neuron in d1neurons:
    if neuron['region']=='ACB':
        neuron['color']={'r':'255','g':'0','b':'0'}
        ACBneurons.append(neuron)
    else:
        neuron['color']={'r':'0','g':'200','b':'0'}
        Otherneurons.append(neuron)
D1ACBneurons=ACBneurons
Scene.createScene(neuronlist=ACBneurons,filename='../resource/scene/STRD1ACBneurons.nv')

ACBneurons=[]
for neuron in d2neurons:
    if neuron['region']=='ACB':
        neuron['color']={'r':'0','g':'255','b':'0'}
        ACBneurons.append(neuron)
    else:
        neuron['color']={'r':'0','g':'200','b':'0'}
        Otherneurons.append(neuron)
D2ACBneurons=ACBneurons
Scene.createScene(neuronlist=ACBneurons,filename='../resource/scene/STRD2ACBneurons.nv')

ACBneurons=[]
for neuron in ChATneurons:
    if neuron['region']=='ACB':
        neuron['color']={'r':'0','g':'0','b':'255'}
        ACBneurons.append(neuron)
    else:
        neuron['color']={'r':'0','g':'200','b':'0'}
        Otherneurons.append(neuron)
ChATACBneurons=ACBneurons
Scene.createScene(neuronlist=ACBneurons,filename='../resource/scene/STRChATACBneurons.nv')


# %% 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df = iondata.getPropertiesDF(neurons)
projectregiondf= df[df.property.str.contains('projectregion',case=False)]
projectregiondf.set_index(['property'],inplace=True)
newindex = projectregiondf.sum(1).sort_values(ascending=False).index[0:55]
projectregiondf=projectregiondf.loc[newindex,:]
data =projectregiondf.to_numpy().astype(np.float32)
data=(np.log2(data/100.0+1))
propmat=[]
indexs=[]

for neuron in neurons:
    property = iondata.getNeuronPropertyByID(neuron['sampleid'], neuron['name'])
    indexs.append(neuron['sampleid']+ neuron['name'])
    brproperty=BR.RegionProperty(copy.deepcopy(br))
    brpropertyLeft=BR.RegionProperty(copy.deepcopy(br))
    brpropertyRight=BR.RegionProperty(copy.deepcopy(br))
    brproperty.setProperty(property['projectregion'])
    brpropertyLeft.setProperty(property['projectleftregion'])
    brpropertyRight.setProperty(property['projectrightregion'])
    prop=[]

    target_regions = list(newindex)
    for targetRegion in target_regions:
        regionsum = brproperty.getSumProperty(targetRegion[14:])
        regionleft = brpropertyLeft.getSumProperty(targetRegion[14:])
        regionright = brpropertyRight.getSumProperty(targetRegion[14:])
        ipsi=0
        contra=0
        if property['somapoint'][2]<5700:
            ipsi=regionleft
            contra=regionright
        else:
            ipsi=regionright
            contra=regionleft
        prop.append(ipsi)
    propmat.append(prop)


# %%
indexs=[]
for neuron in neurons:

    indexs.append(neuron['sampleid']+ neuron['name'])
df = pd.DataFrame(propmat,index=indexs,columns=list(newindex))
d1list=[]
d1indexs=[]
d2indexs=[]
ChATindexs=[]
for neuron in D1ACBneurons:
    d1indexs.append(neuron['sampleid']+ neuron['name'])
for neuron in D2ACBneurons:
    d2indexs.append(neuron['sampleid']+ neuron['name'])
for neuron in ChATACBneurons:
    ChATindexs.append(neuron['sampleid']+ neuron['name'])
d1df = df.loc[d1indexs]
d2df = df.loc[d2indexs]
ChATdf = df.loc[ChATindexs]

data =d1df.to_numpy().astype(np.float32)
data=(np.log2(data/100.0+1))

df2 = pd.DataFrame(data, index=d1df.index, columns=d1df.columns)
df2=df2.T
sns.clustermap(df2,row_cluster=False,figsize=[5,15])


# %% ACBneruons

acbdf = iondata.getPropertiesDF(ACBneurons)
acbprojectregiondf= acbdf[acbdf.property.str.contains('projectregion',case=False)]
acbprojectregiondf.set_index(['property'],inplace=True)
acbnewindex = acbprojectregiondf.sum(1).sort_values(ascending=False).index[0:55]
acbprojectregiondf=acbprojectregiondf.loc[acbnewindex,:]
acbdata =acbprojectregiondf.to_numpy().astype(np.float32)
acbdata=(np.log2(acbdata/100.0+1))

acbdf2 = pd.DataFrame(acbdata, index=acbprojectregiondf.index, columns=acbprojectregiondf.columns)
# fig,ax = plt.subplots(figsize=(13,7))
sns.clustermap(acbdf2,row_cluster=False,figsize=[10,20])


# %% [markdown]
# Other Neurons
# %% 

otherdf = iondata.getPropertiesDF(Otherneurons)
otherprojectregiondf= otherdf[otherdf.property.str.contains('projectregion',case=False)]
otherprojectregiondf.set_index(['property'],inplace=True)
othernewindex = otherprojectregiondf.sum(1).sort_values(ascending=False).index[0:55]
otherprojectregiondf=otherprojectregiondf.loc[othernewindex,:]
otherdata =otherprojectregiondf.to_numpy().astype(np.float32)
otherdata=(np.log2(otherdata/100.0+1))

otherdf2 = pd.DataFrame(otherdata, index=otherprojectregiondf.index, columns=otherprojectregiondf.columns)
sns.clustermap(otherdf2,figsize=[50,20])


# %% render single neuron

import Visual as nv

neuronvis = nv.neuronVis(size=(1700,1000),renderModel=0)

neuronvis.render.setBackgroundColor((1.0,1.0,1.,1.0))

neuronvis.setLineWidth(1)
neuronvis.addRegion('STR')
for neuron in Otherneurons:
    neuronvis.addNeuronByID(neuron['sampleid'],neuron['name'],color=[1,0,0],somaColor=[0,1,0],somaHide=False,axonHide=False,dendriteHide=False,isLine=True)
    neuronvis.render.savepng('../resource/png/ACB/STRother/'+neuron['region']+'/'+neuron['sampleid']+neuron['name']+'.png')
    neuronvis.clear(regions=False)
neuronvis.render.closeWindow()



# %%
regionNeurons={}
regions=[]
for neuron in neurons:
    if neuron['region'] not in regionNeurons.keys():
        regions.append(neuron['region'])
        regionNeurons[neuron['region']]=[]
    regionNeurons[neuron['region']].append(neuron)
count=[]
for region in regionNeurons.keys():
    count.append(len(regionNeurons[region]))
    Scene.createScene(neuronlist=regionNeurons[region],filename='../resource/scene/STR_'+region+'neurons.nv')
plt.bar(height=count,x=regions)
plt.xticks(regions,regions, rotation=30)
plt.show()

# %%
strNeurons=[]
strRegions=['ACB','CP','LSr','OT']
for region in strRegions:
    strNeurons.extend(regionNeurons[region])

strdf = iondata.getPropertiesDF(strNeurons)
strprojectregiondf= strdf[strdf.property.str.contains('projectregion',case=False)]
strprojectregiondf.set_index(['property'],inplace=True)
strnewindex = strprojectregiondf.sum(1).sort_values(ascending=False).index[0:55]
strprojectregiondf=strprojectregiondf.loc[strnewindex,:]
strdata =strprojectregiondf.to_numpy().astype(np.float32)
strdata=(np.log2(strdata/100.0+1))

strdf2 = pd.DataFrame(strdata, index=strprojectregiondf.index, columns=strprojectregiondf.columns)
sns.clustermap(strdf2,figsize=[50,20])
# %%
