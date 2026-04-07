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
import shutil
import os
#%%
iondata = IONData.IONData()

for i in range(1,20):
    if i>3:
        print('subtype: ',i)
        filename='../resource/scene/19subtype/19Subtype-'+str(i)+'.nv'
        dir = '../resource/swc_merge/subtype'+str(i)+'/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        if os.path.exists(dir):
            Scene.mergeScene(filename,dir)

        break
# %%
