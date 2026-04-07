
# neuronVis 
![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/logo.png#pic_center)



## NeuronVis is a tool for mouse neuron analyse 

| = | = |
| ---- | ---- |
| Contact | xfwang@ion.ac.cn|
| Date | 27 April 2022 |
| Copyright |  `GFDLv1.2+ <http://www.gnu.org/licenses/fdl.html>` |
| Version | 1.0 |

neuronVis is a tool for mouse neuron analyse .

# USAGE


* git clone https://gitee.com/bigduduwxf/neuron-vis.git

* Download the resource from http://bap.cebsit.ac.cn/bap/neuronVis/resource.zip and decompressed to the directory neuronvis  

* conda env create -f env.yml

* conda activate neuronVis

### Add the path of neuronVis to sys.path
    import sys
    sys.path.append(your git repository+"/neuronVis")

## Get Data Example
* https://gitee.com/bigduduwxf/neuron-vis/blob/master/analyseCase/getDataExample.ipynb

## Get all Data in ION (only allowed in ION, some samples are not public)
    in IONData.py line 19
    //self.ip='bap.cebsit.ac.cn'
    or :
    iondata.ip='10.10.48.110'

### Get neuron list by sampleid
    from IONData import  IONData 
    import SwcLoader
    iondata =IONData()
    neuronlist=iondata.getNeuronListBySampleID('192106')

### Get region list included in a region referenced to Allen
    br = BrainRegion()
	br.praseJson()
	subbr= br.getRegion('ORB')
	subbr.annotation()
	brproperty=RegionProperty(copy.deepcopy(subbr))
	brproperty.print()
	subbr.print()
	subbrlist=br.getRegionList('grv of CBX')

### Get swc and property form iondata by sampleid and neuron name

    from IONData import  IONData 
    import SwcLoader
    iondata =IONData()
    neuronlist=iondata.getNeuronListBySampleID('192106')
    swc = iondata.getNeuronByID('192106', '001.swc')
    pro = iondata.getNeuronPropertyByID('192106', '001.swc')
    print(pro)
    
### Get property of regions with subregions like 'HY'
    import BrainRegion as BR 
    br = BR.BrainRegion()
    br.praseJson()
    br.setProperty(pro['projectregion'])
    HYProjectLength = br.getSumProperty('HY')

### Parse neuron data into a tree

    from IONData import  IONData 
    import SwcLoader
    iondata =IONData()
    neuron = SwcLoader.NeuronTree()
    neuron.readSWC(iondata.getNeuronByID('192106', '001.swc'))
    
### Render region and swc and savepng

    import neuronVis as nv

    neuronvis = nv.neuronVis()
    # neuronvis.render.setBackgroundColor((0.0,0.20,0.5,1.0))
    neuronvis.addRegion('CA2',[0.5,1.0,0.5])
    neuronvis.addNeuron('./resource/033.swc')
    neuronvis.render.savepng('resource/test.png')
    neuronvis.render.setView('posterior')
    # neuronvis.render.setLookAt()
    neuronvis.addNeuron('./resource/192092-012.swc',[1.0,1.0,0.0])
    neuronvis.addNeuronByID('192106','011.swc',[1.0,1.0,0.0])
    neuronvis.render.savepng('resource/test2.png')

    neuronvis.render
    nv.app.run()

![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/test2.png#pic_center)

### Run the neuronVis/Flatmap.py to get flatmap

![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/flatmap.png#pic_center)

![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/flatneuron.png#pic_center)

### Draw neuron somas on the flatmap
![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/Somas_on_FlattenMap.png#pic_center)
```
#position is the array of soma positions in 3D space
iondata = IONData.IONData() #wxf
res,gridpath = iondata.getFileFromServer("boundlaplace20.nrrd")
grid,header = nrrd.read(gridpath)
resRelaxation,RelaxationPath=iondata.getFileFromServer('boundlaplaceout20.nrrd')
relaxation,relaxationheader = nrrd.read(RelaxationPath)
resdv0, dv0Path = iondata.getFileFromServer('dv0.nrrd')
dv0, dv0header = nrrd.read(dv0Path)
resdv1, dv1Path = iondata.getFileFromServer('dv1.nrrd')
dv1, dv1header = nrrd.read(dv1Path)
resdv2, dv2Path = iondata.getFileFromServer('dv2.nrrd')
dv2, dv2header = nrrd.read(dv2Path)
dv0 = dv0.astype(np.float32) / 1000 - 1
dv1 = dv1.astype(np.float32) / 1000 - 1
dv2 = dv2.astype(np.float32) / 1000 - 1
position=BL.ComputeStreamlines(grid,dv0,dv1,dv2,position)
flatenPara = Flatmap.createSurfaceGraph()
flattened_position=[]
for p in position:
    flattened_position.append(Flatmap.map2Flatmap(flatenPara,np.array(p[1])*2,True))
flatmapedge=cv.imread('flatmapedge2.tif',1)
flatmap=cv.imread('flatmap.tif',2)
color = np.array([np.random.rand() for i in range(300)]).reshape(100,3)*255
for p in flattened_position:
    cv.circle(flatmapedge, (int(p[0])*2, int(p[1])*2),15, color[flatmap[int(p[1])][int(p[0])]], -1)
plt.imshow(flatmapedge)
plt.show()
```

### Laplace's equation
All the glories of Laplace's equation for measuring cortical thickness
 as initially described in:

Jones SE, Buchbinder BR, Aharon I. Three-dimensional mapping of
cortical thickness using Laplace's equation. Hum Brain Mapp. 2000
Sep;11(1):12-32.

and modified somewhat in:

Lerch JP, Carroll JB, Dorr A, Spring S, Evans AC, Hayden MR, Sled JG,
Henkelman RM. Cortical thickness measured from MRI in the YAC128 mouse
model of Huntington's disease. Neuroimage. 2008 Jun;41(2):243-51.

![laplace 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/boundlaplace.png#pic_center)

### Cluster with some neuron figure/paper1figure5.py
Apply clustermap API of package seaborn to cluster data hierarchically.

```
    seaborn.clustermap(data, *, pivot_kws=None, method='average', metric='euclidean', z_score=None, standard_scale=None, figsize=(10, 10), cbar_kws=None, row_cluster=True, col_cluster=True, row_linkage=None, col_linkage=None, row_colors=None, col_colors=None, mask=None, dendrogram_ratio=0.2, colors_ratio=0.03, cbar_pos=(0.02, 0.8, 0.05, 0.18), tree_kws=None, **kwargs)
    Parameters
        data: Rectangular data for clustering.
        method: Linkage method to use for calculating clusters.
        metric: Distance metric to use for the data.
        {row,col}_cluster: If True, cluster the {rows, columns}.
```

![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/cluster.png#pic_center)

### Save to svg
    import sys,copy,os
    sys.path.append(os.getcwd()+"/neuronVis")
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import IONData 
    import SwcLoader
    iondata =IONData.IONData()
    swc = iondata.getNeuronByID('192106', '033.swc')

    neuron = SwcLoader.NeuronTree()
    neuron.readSWC(swc)
    plt.figure(figsize=(12, 12))
    axes0 = plt.subplot(111)
    plt.axis('off')
    axes0.set_xlim(0, 10000)
    axes0.set_ylim(0, 10000)
    # plt.Circle((neuron.root.z, neuron.root.y), 300)
    for edge in neuron.edges:
        x=[]
        y=[]
        for p in edge:
            x.append(10000-p.z)
            y.append(10000-p.y)
        plt.plot(x, y,color='#FFDD44')
    # plt.figure(facecolor='gainsboro')
    plt.plot(10000-neuron.root.z,10000- neuron.root.y,'ob')


    plt.savefig(fname="./resource/neuron.svg",format="svg")
    plt.show()

![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/neuron.svg#pic_center)

Authors
=======

*XiaoFei Wang <xfwang@ion.ac.cn>  

*Shou Qiu <sqiu@ion.ac.cn> 

*Le Gao <lgao@ion.ac.cn> 

*ZhuoLei Jiao <zljiao@ion.ac.cn> 

