from pickle import TRUE
from matplotlib.pyplot import plot
import cv2
import SwcLoader
import IONData
from pathlib import Path
import pytorch_ssim
import torch
from torch.autograd import Variable
import numpy as np
import os
iondata =IONData.IONData()
iondata.downloadfile("http://10.10.48.110/bap/neuronVis/rmgr-ssim.exe",filename="../Resource/ssim/rmgr-ssim.exe")
def plotNeuronMPR(neuronlist,outputdir='../Resource/png',smooth=False):
    Path(outputdir).mkdir(parents=True, exist_ok=True)
    iondata =IONData.IONData()
    tree = None
    for neuronid in neuronlist:
        if tree is not None:
            del tree
        tree = iondata.getNeuronTreeByID(neuronid['sampleid'],neuronid['name'])
        print(sys.getrefcount(tree))
        filename = '../Resource/png/'+neuronid['sampleid']+'-'+neuronid['name']+'.png'
        tree.plotMPR(filename)
        # if smooth:
        #     img = cv2.imread(filename)
        #     dst1 = cv2.GaussianBlur(img, (59, 59),5)
        #     # cv2.imshow("9*9", dst1)s
        #     cv2.imwrite(filename+'.png',dst1)
        #     pass
def calculateDist(neuronlist):
    map =np.zeros([len(neuronlist),len(neuronlist)])
    for i in range(len(neuronlist)):
        neuronid = neuronlist[i]
        submap=[]
        filename = '../Resource/png/'+neuronid['sampleid']+'-'+neuronid['name']+'.png'
        npImg1 = cv2.imread(filename)
        img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
        for j in range(i,len(neuronlist)):
            neuronid2 = neuronlist[j]
            filename2 = '../Resource/png/'+neuronid2['sampleid']+'-'+neuronid2['name']+'.png'
            npImg2 = cv2.imread(filename2)
            img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0)/255.0

            if torch.cuda.is_available():
                img1 = img1.cuda()
                img2 = img2.cuda()
            img1 = Variable( img1)
            img2 = Variable( img2)
            ssim_value = pytorch_ssim.ssim(img1, img2)
            map[i,j]=ssim_value
            print(neuronid['name'],neuronid2['name'],"Initial ssim:", ssim_value)
    return map

def calculateDistByPlugin(neuronlist):
    map =np.zeros([len(neuronlist),len(neuronlist)])
    for i in range(len(neuronlist)):
        neuronid = neuronlist[i]
        submap=[]
        filename = '../Resource/png/'+neuronid['sampleid']+'-'+neuronid['name']+'.png'

        for j in range(i,len(neuronlist)):
            neuronid2 = neuronlist[j]
            filename2 = '../Resource/png/'+neuronid2['sampleid']+'-'+neuronid2['name']+'.png'
            result = os.popen('..\\Resource\\ssim\\rmgr-ssim.exe -0 '+filename+' '+filename2)
            res = result.read()
            ssim_value =0
            for line in res.splitlines():
                ssim_value=line
            
            map[i,j]=ssim_value
            print(neuronid['name'],neuronid2['name'],"Initial ssim:", ssim_value)
    return map


def getLeaf(tree,out,id=None):
    # if not tree.is_leaf():
    #     print(tree.id)
    if id is not None and tree.id==id :
        if not tree.is_leaf():
            getLeaf(tree.left,out)
            getLeaf(tree.right,out)

    elif not tree.is_leaf():
        getLeaf(tree.left,out,id)
        getLeaf(tree.right,out,id)
    else :
        if id is None:
            out.append(tree.id)

if __name__ == "__main__":
    import os,sys,json,inspect
    if hasattr(sys.modules[__name__], '__file__'):
        _file_name = __file__
    else:
        _file_name = inspect.getfile(inspect.currentframe())
    CURRENT_FILE_PATH = os.path.dirname(_file_name)

    import seaborn as sns
    from scipy.cluster.hierarchy import linkage ,fcluster,fclusterdata,dendrogram,to_tree

    import pandas as pd
    import matplotlib.pyplot as plt
    import Scene

    iondata =IONData.IONData()
    sampleid = '18712'
    neuronlist=iondata.getNeuronListBySampleID(sampleid)
    # f=open('../resource/SPCDList.json', encoding='gbk')
    # neuronlist =  json.load(f)

    # plotNeuronMPR(neuronlist,smooth=False)
    heatmap = calculateDistByPlugin(neuronlist)

    print(heatmap)
    df =pd.DataFrame(heatmap)
    df.to_csv('../resource/ssim'+sampleid+'.csv')

    df = pd.read_csv('../resource/ssim'+sampleid+'.csv')
    select_cols=df.columns[1:]
    df2= df[select_cols]
    print(df2)
    import matplotlib
    matplotlib.use('TkAgg')

    for i in range(df2.shape[0]):
        for j in range(0,i):
            df2.iloc[i,j] = df2.iloc[j,i]

    print(df2)

    linkresult = linkage(df2,method='ward',metric='euclidean')
    tree = to_tree( linkresult , rd=False )
    cluster=fcluster(linkresult,t=1.1)#这个需要先计算linkage，再出结果
    neuronarray = np.array(neuronlist)
    id_list=[neuron['sampleid']+neuron['name'] for neuron in neuronarray for key in neuron if key=='name']
    t=dendrogram(linkresult,labels=id_list)
    print('test')

    import Visual as nv
    neuronvis = nv.neuronVis()
    count=0
    # for i in range(cluster.max()):
    #     print('cluster:',count+1)
    #     clusterindex = np.where(cluster==i+1)
    #     Scene.createScene(neuronarray[clusterindex],"../resource/png/cluster"+sampleid+'-'+str(count+1)+'.nv')
    #     print(neuronarray[clusterindex])
    #     print(len(neuronarray[clusterindex]))
    #     # neuronvis.addNeuronByList(neuronarray[clusterindex],True)
    #     # neuronvis.render.setView()
    #     # neuronvis.render.savepng("../resource/png/cluster"+sampleid+'-'+str(count+1)+'.png')
    #     # neuronvis.clear()
    #     count+=1

    ids=[714,708,713,711,717,680,715,716]
    clusterlist=[]
    for id in ids:
        leaf =[]
        getLeaf(tree,leaf,id)
        clusterlist.append(leaf)

    for i in clusterlist:
        print('cluster:',count+1)
        # clusterindex = np.where(cluster==i+1)
        Scene.createScene(neuronarray[i],"../resource/png/cluster"+sampleid+'-'+str(count+1)+'.nv')
        print(neuronarray[i])
        neuronvis.addNeuronByList(neuronarray[i],True)
        neuronvis.render.setView()
        neuronvis.render.savepng("../resource/png/cluster"+sampleid+'-'+str(count+1)+'.png')
        neuronvis.clear()
        count+=1


    print('test')
    t=sns.clustermap(df2,method='ward',metric='euclidean')
    # neuronarray = np.array(neuronlist)
    # neuronlist2 = neuronarray[t.mask.columns.values.astype(np.int16)].tolist()

    # print(t.mask.columns.values)
    # import Visual as nv
    # neuronvis = nv.neuronVis()
    # neuronvis.addNeuronByList(neuronlist2[7:22])
    # neuronvis.render.run()
    