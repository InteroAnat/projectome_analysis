#%%

import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")
import pandas as pd
import RenderGL 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('module://matplotlib_inline.backend_inline')
%matplotlib inline
import os

#%%
slices = os.listdir('../resource/ST_mouse/20230419')
for slice in slices:
    ## cells
    cells = pd.read_csv('../resource/ST_mouse/20230419/'+slice,sep='\t',usecols=[0,6,7,8])
    cells.set_index('cell_id',inplace=True)

    ## type ID
    typeIDfile = 'total_cell_annotation_T'+slice[12:15]+'_mouse_f001_2D_mouse1-20230119_type20230222.txt'
    print(typeIDfile)
    typeID = pd.read_csv('../resource/ST_mouse/mouse1-20230119_type20230222/'+typeIDfile,sep='\t',usecols=[1,2])
    typeID.set_index('cell_id',inplace=True)
    typeIDDict=typeID.to_dict()['cell_type_id']


# %%
