# In[1] import 
import volume
import matplotlib.pyplot as plt
from skimage import measure,io
from scipy import ndimage
import trimesh
import numpy as np
import IONData 
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
%matplotlib inline
# In[] render volume and soma point 
iondata = IONData.IONData()
cube = volume.FNTCube()
def SomaSegment(sampleid,neuron):
    # sampleid='210726'
    # neuron='006.swc'
    tree = iondata.getRawNeuronTreeByID(sampleid,neuron)
    tree.root
    cube.setSampleID(sampleid)
    # cube.setMouseCoord(6719.08,2002.89,4665.78) # cortex
    # cube.setMouseCoord(6220.28,3380.16,6409.23) # th
    # cube.setMouseCoord(6016.28,3820.16,6315.23) # th
    cube.setMouseCoord(tree.root.x,tree.root.y,tree.root.z) # th
    cube.radiu=2

    # cube.setMouseCoord(3459.95,1743.75,6370.94) #mean=137
        
    render = volume.VolumeRender()
    render.setFNTCube(cube)
    # render.volume1.clim=(20,1000)
    # volume.app.run()

    # %
    segments=[]
    for edge in tree.edges[:]:
        p=[]
        for point in edge.data:
            p.append(point.xyz)
        pos = cube.physicalPoint2LocalPoint(p,True)
        segments.append(pos)

    # %
    blankarray = np.zeros_like(cube.volume)
    print(cube.physicalPoint2LocalPoint([tree.root.xyz],True))
    ratio=10
    for seg in segments:
        for p in seg:
            if p[0]>ratio and p[1]>ratio and p[2]>ratio and p[0]<blankarray.shape[2]-ratio and p[1]<blankarray.shape[1]-ratio and p[2]<blankarray.shape[0]-ratio:
                blankarray[p[2]-ratio:p[2]+ratio,p[1]-ratio:p[1]+ratio,p[0]-ratio:p[0]+ratio]=25

    blankarray2=ndimage.gaussian_filter(blankarray, sigma=5)

    cube.volume[blankarray2<1.0]=0
    pos = cube.physicalPoint2LocalPoint([cube.mouseCoord],True)
    print(pos)
    x=pos[0][0]
    y=pos[0][1]
    z=pos[0][2]
    print(cube.volume.shape)

    crop = cube.volume[z-40:z+40,y-80:y+80,x-80:x+80]

    # %


    crop = ndimage.gaussian_filter(crop, sigma=1.5)
    zslice = crop[int(crop.shape[0]/2),:,:]
    print(x,y,z,zslice)
    # plt.imshow(zslice)


    # %
    verts, faces, t1, t2 = measure.marching_cubes(crop, level=160,spacing=(cube.pixelSpace[2],cube.pixelSpace[1],cube.pixelSpace[0]),method='lorensen')

    mesh = trimesh.Trimesh(vertices=verts,
                                faces=faces,
                                )
    # %
    # mesh.show()
    mesh.export("../resource/somaobj/"+sampleid+neuron+'.obj')
    return mesh,crop,tree


# %%
mesh,crop,tree =SomaSegment('210726','006.swc')
# mesh.show()

# %%
segments=[]
for edge in tree.edges[:]:
    p=[]
    for point in edge.data:
        p.append(point.xyz)
    pos = cube.physicalPoint2LocalPoint(p,True)
    segments.append(pos)

# %%
blankarray = np.zeros_like(cube.volume)
print(cube.physicalPoint2LocalPoint([tree.root.xyz],True))
ratio=10
for seg in segments:
    for p in seg:
        if p[0]>ratio and p[1]>ratio and p[2]>ratio and p[0]<blankarray.shape[2]-ratio and p[1]<blankarray.shape[1]-ratio and p[2]<blankarray.shape[0]-ratio:
            blankarray[p[2]-ratio:p[2]+ratio,p[1]-ratio:p[1]+ratio,p[0]-ratio:p[0]+ratio]=25
# print(blankarray.shape)
# %%

blankarray0=ndimage.gaussian_filter(blankarray, sigma=3)
plt.imshow(blankarray0[142,:,:])
# %%
blankarray2=ndimage.gaussian_filter(blankarray, sigma=5)
plt.imshow(blankarray2[142,:,:])
# %%

cube.volume[blankarray2<1.0]=0
plt.imshow(cube.volume[142,:,:]) 

# %%
plt.imshow(cube.volume[152,:,:]) 
# %%
