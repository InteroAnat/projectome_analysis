#%%
import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")

import IONData,Scene,SwcLoader

#%%
iondata = IONData.IONData()
neurons221246 = iondata.getNeuronListBySampleID('221246')
neurons221247 = iondata.getNeuronListBySampleID('221247')

neurons=[]
neurons = [neurons221246[0],neurons221246[1],neurons221246[2],
           neurons221246[5],neurons221246[6],neurons221246[7],
           neurons221246[9],neurons221246[11],neurons221246[12],
           neurons221246[13],neurons221246[14]]+neurons221247

#%%
import BrainRegion as BR 
import numpy as np
br = BR.BrainRegion()
br.praseJson()
target_regions=['RO','NTS','MDRN','LRN','ECU','CU','ACVII']
propmat=[]

for neuron in neurons:
    property = iondata.getNeuronPropertyByID(neuron['sampleid'], neuron['name'])
    brproperty=BR.RegionProperty(copy.deepcopy(br))
    brpropertyLeft=BR.RegionProperty(copy.deepcopy(br))
    brpropertyRight=BR.RegionProperty(copy.deepcopy(br))
    brproperty.setProperty(property['projectregion'])
    prop=[]
    for targetRegion in target_regions:
        regionsum = brproperty.getSumProperty(targetRegion)
        prop.append(regionsum)
    propmat.append(prop)
    
# %%
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
%matplotlib inline
from matplotlib import pyplot as plt
nameList = [neuron['sampleid']+neuron['name'][0:3] for neuron in neurons]
fig=plt.figure(figsize=(9,9))
ax=fig.add_subplot(111)

ax.set_yticks(range(len(nameList)))
ax.set_yticklabels(nameList)
ax.set_xticks(range(len(target_regions)))
ax.set_xticklabels(target_regions)

propmatLog = np.log2(np.array(propmat)+1)
im = ax.imshow(propmatLog, cmap='summer_r')
# plt.show()
plt.colorbar(im)
plt.savefig(os.path.abspath('../resource/svg/x.pdf'),format='pdf',bbox_inches = 'tight')#保存为.svg格式

# %%
