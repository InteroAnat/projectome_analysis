# %%
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
# %%
df = pd.read_csv('../resource/qyq3_detection_coordinates.csv')
print(df)

#%% 
import nrrd
edge = nrrd.read('../resource/edge_annotation_10_2017.nrrd')
outline = nrrd.read('../resource/structure_997.nrrd')
#%%
cells = np.zeros_like(edge[0])
outline = outline[0]

#%%
plt.imshow(edge[0][200,:,:])

#%%
for index, row in df.iterrows():
    x= int(row['Atlas_X']*100)
    y= int(row['Atlas_Y']*100)
    z= int(row['Atlas_Z']*100)
    cells[x:x+3,y:y+3,z:z+3]=cells[x:x+3,y:y+3,z:z+3]+1
#%%
from skimage import filters
index = 400
img1 = filters.gaussian(cells[index,:,:]*10,sigma=15)
img1 = img1/np.max(img1)*255
for x in range(img1.shape[0]):

    for y in range(img1.shape[1]):
        if outline[index,x,y]==0:
            img1[x,y]=0

plt.imshow(img1+edge[0][index,:,:])


# %%
import GeometryAdapter

render = RenderGL.RenderGL(renderModel=0)
# render.setBackgroundColor([1,1,1])
# render.setLineWidth(4.0)
ga = GeometryAdapter.GeometryAdapter()


for index, row in df.iterrows():
    ga.geometry.addPoint([row['Atlas_X']*1000,row['Atlas_Y']*1000,row['Atlas_Z']*1000],[1.0,1.0,1.0])
    ga.geometry.addIndex(index)


ga.geometry.drawModel='points'
render.setPointSize(1)
render.addGeometry(ga.geometry)
render.setView()
# render.setClip(True)
# render.savepng('../resource/test3.png')
# render.animation(start=90)
render.run()
# %%
