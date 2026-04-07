import sys,copy,os
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(CURRENT_FILE_PATH+"/../neuronVis")
import nrrd
from sklearn.cluster import KMeans
# from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

import IONData
import BrainRegion
import NeuronProcess
import SwcLoader
import Scene
if __name__=="__main__":
    import Visual as nv
    import json

    def GetFirstOrderEdges(route,edgesForCluster,parentedge):
        for child in parentedge.children:
            if child.order==1 or child.order==2:
                route.append(child)
                edgesForCluster.append(child)
                child.maxDepth=1
                GetFirstOrderEdges(route,edgesForCluster,child)

        
    # neuronvis.addRegion('CA1',[0.5,1.0,0.5])
    iondata = IONData.IONData()
    br = BrainRegion.BrainRegion()
    br.praseJson()
    hybr = br.getRegion(name='HY')
    hybr.annotation()
    # neurons=iondata.getNeuronListBySampleID('211185')\
    # +iondata.getNeuronListBySampleID('211186')\
    # +iondata.getNeuronListBySampleID('211187')\
    # +iondata.getNeuronListBySampleID('211188')\
    # +iondata.getNeuronListBySampleID('211189')\
    # +iondata.getNeuronListBySampleID('211190')\
    # +iondata.getNeuronListBySampleID('211191')\
    # +iondata.getNeuronListBySampleID('211192')\
    # +iondata.getNeuronListBySampleID('211193')
    neurons= Scene.scene2List('../resource/scene/cluster5_spcd20220628_spcd.nv')
    neuronvis = nv.neuronVis(renderModel=0)
    neuronvis.render.setBackgroundColor((0.0,0.0,0.,1.0))
    neuronvis.render.setLookAt((2000.,0.0,15000),(2000,0,0),(0.0,-1.0,0.0))
    neuronvis.addRegion('HY',[0.5,1.0,0.5])
    Neurons = []
    newNeurons = []
    edgesForCluster=[]
    for neuron in neurons[0:-1]:
        neuronT = iondata.getNeuronTreeByID(neuron['sampleid'],neuron['name'])
        NeuronProcess.CalculateBranchMaxDepth(neuronT)
        Neurons.append(neuronT)
        hyedge = []
        maxorder =0
        for edge in neuronT.edges:
            terminal = edge.data[len(edge.data)-1]
            x= int(terminal.x/10)
            y=int(terminal.y/10)
            z=int(terminal.z/10)
            shape = hybr.readdata.shape
            if x>0 and x<shape[0] and y>0 and y<shape[1] and z>0 and z<shape[2] and hybr.readdata[x,y,z] and len(edge.data[len(edge.data)-1].children)==0:
                if maxorder<edge.order:
                    maxorder=edge.order
                hyedge.append(edge)
                # print(edge.data[len(edge.data)-1],edge.maxDepth,edge.order)
        route=[]
        for edge in hyedge:
            if edge.order==maxorder:
                edge.maxDepth=2
                edge.order=-1
                route.append(edge)
                parentedge = neuronT.getEdgeByTerminal(edge.data[0])
                parentedge.maxDepth=2
                parentedge.order=-1
                route.append(parentedge)
                while parentedge.data[0]!= neuronT.root:
                    parentedge = neuronT.getEdgeByTerminal(parentedge.data[0])
                    parentedge.order=-1
                    parentedge.maxDepth=2
                    for child in parentedge.children:
                        if child.order!=-1:
                            NeuronProcess.OrderChildren(neuronT,child,1)
                    children = parentedge.children
                    route.append(parentedge)
                    GetFirstOrderEdges(route,edgesForCluster,parentedge)

                break
        
        newneuron=SwcLoader.NeuronTree()
        if len(route):
            newneuron.edges= route
            newneuron.rootAxonEdge = neuronT.rootAxonEdge
            newneuron.root=neuronT.root
            newNeurons.append(newneuron)
        # print(len(route))
    for neuron in Neurons:
        neuronvis.addNeuronTree(neuron,'t')
    neuronvis.render.savepng('../resource/test40.png')
    neuronvis.clear()
    for neuron in newNeurons:
        neuronvis.addNeuronTree(neuron,'t',depthIntensity=True)
    neuronvis.render.savepng('../resource/test41.png')
    # neuronvis.render.run()

    neuronvis.clear()

    data = []
    for edge in edgesForCluster:
        xyz=[]
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
    neuronvis.render.run()