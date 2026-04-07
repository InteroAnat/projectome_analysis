#In[]
import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")

import IONData,Scene,SwcLoader
import NeuronProcess
from sklearn.cluster import KMeans
import numpy as np
import joblib
import pandas as pd

def loadClusterCSV(filename,cluster=None):
    neurons = pd.read_csv(filename)
    neuronsArray= neurons.to_numpy()
    neuronScene=[]
    for neuron in neuronsArray:
        # print(str(neuron[2])[0])
        neurondict={}
        neurondict['cluster'] = neuron[1]
        neurondict['sampleid'] = str(neuron[2])[0:6]
        neurondict['name']=str(neuron[2])[6:]+'.swc'
        if cluster ==neuron[1] or cluster ==None:
            neuronScene.append(neurondict)
    return neuronScene




#In[1]


import Visual as nv
import json
neuronvis = nv.neuronVis(renderModel=0)

neuronvis.render.setBackgroundColor((0.0,0.0,0.,1.0))
# neuronvis.addRegion('IB',[0.5,1.0,0.5])
# neuronvis.addRegion('MP',[0.5,.0,0.5])
# neuronvis.addRegion('P',[0.9,1.0,0.5])
# neuronvis.addRegion('MY',[0.5,1.0,0.9])
iondata = IONData.IONData()
# f=open('../resource/pfcsubtype60.json', encoding='gbk')
# neurons=[]
# neurons = json.load(f)
neurons=iondata.getNeuronListBySampleID('220978')

#neurons = loadClusterCSV('../resource/cluster_eachNeuron/cluster_eachNeuron_spcd.csv',cluster=10)
# neurons= Scene.scene2List('../resource/cluster_eachNeuron/iso.nv')
neuronvis.render.setLookAt((5000,-15000,0),(5000,0,0),(0,0,-1))

edgesForCluster=[]

for neuron in neurons[7:8]:
    print(neuron['sampleid'], neuron['name'])
    neuronT = iondata.getNeuronTreeByID(neuron['sampleid'], neuron['name'])
    neuronT.dendriteHide=True
    NeuronProcess.CalculateBranchMaxDepth(neuronT)
    mainbranch=[]
    if neuronT.rootAxonEdge is None:
        continue
    for edge in neuronT.edges:

        if edge.maxDepth>neuronT.rootAxonEdge.maxDepth*0.5 or edge.maxLength>neuronT.rootAxonEdge.maxLength*0.1:
            edge.order=0
            mainbranch.append(edge)
    for edge in mainbranch:
        for child in edge.children:
            if child.order!=0:
                NeuronProcess.OrderChildren(neuronT,child,1)
        pass
    # for child in neuronT.rootAxonEdge.children:
    # 	# if child.order!=0:
    # 		NeuronProcess.OrderChildren(neuronT,child,1)

    for edge in mainbranch:
        for child in edge.children:
            if child.order!=0:
                NeuronProcess.GetFirstOrderEdges(edgesForCluster,child)

    newneuron=SwcLoader.NeuronTree()
    if len(mainbranch):
        newneuron.edges= mainbranch
        newneuron.rootAxonEdge = neuronT.rootAxonEdge
        newneuron.root=neuronT.root
    neuronvis.addNeuronTree(newneuron,'neuronname',depthIntensity=False)
# neuronvis.render.savepng('../resource/mainbranch/cluster_'+str(neuron['cluster'])+'.png')
# neuronvis.clear()
neuronvis.render.run()

#In[2]
data = []
for edge in edgesForCluster:
    xyz=[]
    edge.children=[]
    for point in edge.data:
        xyz.append(point.xyz)
    # print(np.array(xyz).mean(axis=0))
    data.append(np.array(xyz).mean(axis=0))
cluster=5
# print(data)
estimator=KMeans(n_clusters=cluster,max_iter=1500)
res=estimator.fit_predict(data)
# 预测类别标签结果
lable_pred=estimator.labels_
# 各个类别的聚类中心值
centroids=estimator.cluster_centers_
# 聚类中心均值向量的总和
inertia=estimator.inertia_
print(lable_pred)
for i in range(len(edgesForCluster)):
    edge=edgesForCluster[i]
    edge.maxDepth=lable_pred[i]
neuronT.rootAxonEdge.maxDepth=4
newneuron=SwcLoader.NeuronTree()
if len(edgesForCluster):
    newneuron.edges= edgesForCluster
    newneuron.rootAxonEdge = neuronT.rootAxonEdge
    newneuron.root=neuronT.root
    neuronvis.addNeuronTree(newneuron,'t',depthIntensity=True)
neuronvis.render.savepng('../resource/test42.png')
# neuronvis.render.run()


#In[3]
res,path = iondata.getFileFromServer("IB_MP_P_MY.nrrd")
import nrrd
IB_MP_P_MYmap,header = nrrd.read(path)

#In[4]
clusterLength={}
for c in range(5):
    clusterLength[c]={0:0,313:0,354:0,771:0,1129:0}
for i in range(len(edgesForCluster)):
    edge=edgesForCluster[i]
    lenMap ={0:0,313:0,354:0,771:0,1129:0}
    for j in range(len(edge.data)-1):
        p = edge.data[j]
        if int(p.x/10)<IB_MP_P_MYmap.shape[0]:
            region = IB_MP_P_MYmap[int(p.x/10),int(p.y/10),int(p.z/10)]
        else:
            region=0
        length = np.linalg.norm(np.array(edge.data[j].xyz)-np.array(edge.data[j+1].xyz))
        lenMap[region]+=length
    for k,v in lenMap.items():
        clusterLength[edge.maxDepth][k]+=v

print(clusterLength)