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
import IONData ,volume
iondata = IONData.IONData()
#%%
import vispy.io as io
cube = volume.FNTCube()
cube.setSampleID('210726')
    # cube.setMouseCoord(6719.08,2002.89,4665.78) # cortex
    # cube.setMouseCoord(6220.28,3380.16,6409.23) # th
    # cube.setMouseCoord(6016.28,3820.16,6315.23) # th
cube.setMouseCoord(3316.73,4630.76,5811.05) # th
cube.radiu=2

    # cube.setMouseCoord(3459.95,1743.75,6370.94) #mean=137

tree = iondata.getRawNeuronTreeByID('210726','004.swc')
render = volume.VolumeRender()
render.setFNTCube(cube)
segments=[]
for edge in tree.edges[:]:
    p=[]
    for point in edge.data:
        p.append(point.xyz)
    segments.append(p)
render.drawSegments(segments)
render.volume1.clim=(20,1000)
volume.app.run()
# %%
print('help')