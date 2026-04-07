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
import Visual as nv
import seaborn as sns
import matplotlib.pyplot as plt
import Cluster
iondata = IONData.IONData()



# %%

neurons = iondata.getNeuronListBySampleID('230049')
neurons50 = iondata.getNeuronListBySampleID('230050')
neurons.extend(neurons50)

#%%

neuronvis = nv.neuronVis(size=(1700,1000),renderModel=0)

neuronvis.render.setBackgroundColor((1.0,1.0,1.,1.0))

for neuron in neurons:
    neuronvis.addNeuronByID(neuron['sampleid'],neuron['name'],color=[1,0,0],somaColor=[0,1,0],somaHide=False,axonHide=False,dendriteHide=False,isLine=True)

neuronvis.render.run()
# %%
cluster = Cluster.Cluster()
cluster.fclusterNeuronList(neurons)
plt.xticks(fontsize=12,rotation = 30, ha = 'right')
plt.show()
# %%
