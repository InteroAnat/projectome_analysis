#%%

import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
matplotlib.use('module://matplotlib_inline.backend_inline')
%matplotlib inline

import IONData 
import Scene
iondata = IONData.IONData()




# %%
for i in range(1,8):
    print('thisocluster-'+str(i)+'.nv')


# %%
