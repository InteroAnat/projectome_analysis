import requests
import os,json,time
from pathlib import Path
from SwcLoader import NeuronTree
import codecs
import pandas as pd
import sys
import os
import BrainRegion
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)

class IONData:
    def __init__(self):
        self.ip='10.10.48.110'
        # self.ip='bap.cebsit.ac.cn'
        self.localResourcePath=CURRENT_FILE_PATH+"/../resource"

    def getSoma(self,sampleid):
        url='http://'+self.ip+'/neuronbrowser/api/user/getSoma?fMOST_id='+str(sampleid)
        res=requests.get(url) #返回一个消息实体
        while res.status_code!=200:
            print("try again")
            time.sleep(1)
            res=requests.get(url) 
        return  json.loads(res.text)

    def getSampleInfo(self,fMOSTID=None,projectID=''):
        if fMOSTID==None and projectID!='':
            url='http://'+self.ip+'/neuronbrowser/api/user/getSampleInfo?project_id='+projectID
        else:
            url='http://'+self.ip+'/neuronbrowser/api/user/getSampleInfo?project_id='+projectID+'&fMOST_id='+fMOSTID

        res=requests.get(url) #返回一个消息实体
        while res.status_code!=200:
            print("try again")
            time.sleep(1)
            res=requests.get(url) 
        return  json.loads(res.text)

    def getNeuronListByProjectRegion(self,regionName):
        url='http://'+self.ip+'/neuronbrowser/api/user/projectRegion?region='+regionName
        res=requests.get(url) #返回一个消息实体
        while res.status_code!=200:
            print("try again")
            time.sleep(1)
            res=requests.get(url) 
        return  json.loads(res.text)

    def getNeuronListByTerminalRegion(self,regionName):
        url='http://'+self.ip+'/neuronbrowser/api/user/terminalRegion?region='+regionName
        res=requests.get(url) #返回一个消息实体
        while res.status_code!=200:
            print("try again")
            time.sleep(1)
            res=requests.get(url) 
        return  json.loads(res.text)


#       fuzzy search
    def getNeuronListBySomaRegion(self,regionName,fuzzy=True): # fuzzy=True 模糊搜索
        url='http://'+self.ip+'/neuronbrowser/api/user/neuronsBySoma?region='+regionName
        res=requests.get(url) #返回一个消息实体
        while res.status_code!=200:
            print("try again")
            time.sleep(1)
            res=requests.get(url) 
        if fuzzy:
            return  json.loads(res.text)
        else:
            ns= json.loads(res.text)
            neurons=[]
            for neuron in ns:
                if neuron['region']==regionName:
                    neurons.append(neuron)
            return neurons

    def getNeuronListBySomaBrainRegion(self,regionName): # 可以搜索大脑区神经元
        br = BrainRegion.BrainRegion()
        br.praseJson()
        subbrlist=br.getRegionList(regionName)
        neurons=[]
        for subregion in subbrlist:
            ns = self.getNeuronListBySomaRegion(subregion[0],False)
            if len(ns)>0:
                neurons.extend(ns)
        return neurons
    def getNeuronListByProjectBrainRegion(self,regionName): # 可搜索投射到大脑区的神经元
        br = BrainRegion.BrainRegion()
        br.praseJson()
        subbrlist=br.getRegionList(regionName)
        neurons=[]
        neuronsOut=[]
        for subregion in subbrlist:
            ns = self.getNeuronListByProjectRegion(subregion[0])
            if len(ns)>0:
                neurons.extend(ns)
                # neurons= list(set(neurons))
        df =pd.DataFrame(neurons)
        df2 = df.drop_duplicates()
        for neuron in df2.values.tolist():
            neurondict={}
            neurondict['name']=neuron[0]
            neurondict['sampleid']=neuron[1]
            neurondict['region']=neuron[2]
            neurondict['type']=neuron[3]
            neurondict['exclude']=neuron[4]
            neurondict['comment']=neuron[5]
            neuronsOut.append(neurondict)
        return neuronsOut

    def getNeuronListBySampleID(self,sampleid):
        url='http://'+self.ip+'/neuronbrowser/api/user/selectNeurons?id='+sampleid
        res=requests.get(url) #返回一个消息实体
        while res.status_code!=200:
            print("try again")
            time.sleep(1)
            res=requests.get(url) 
        return  json.loads(res.text)

    def getNeuronPropertyByID(self,sampleid,neuronid):
        pathstr=self.localResourcePath+"/json/"+sampleid+'/'
        path = Path(pathstr)
        file = Path(pathstr+neuronid+'.json')
        if  file.exists():
            with open(file) as f:
                # print('exist ',pathstr+neuronid+'.json')
                swccontext=f.read()
                f.close()
                return json.loads(swccontext)
        else:
            Path(pathstr).mkdir(parents=True, exist_ok=True)
            url='http://'+self.ip+'/neuronbrowser/api/user/neuronProperty?sampleid='+sampleid+'&&neuronid='+neuronid
            res=requests.get(url) 
            if res.status_code != 200:
                print('requests :',res.status_code,url,'try again')
                time.sleep(0.2)
                return self.getNeuronPropertyByID(sampleid,neuronid)
                
            with open(pathstr+neuronid+'.json','w+') as f:
                print('write ',pathstr+neuronid+'.json')
                swccontext=res.text[1:]
                swccontext=swccontext[:-1]
                swccontext = codecs.decode(swccontext,'unicode_escape')
                f.write(swccontext)
                f.close()
                return json.loads(swccontext)
        pass
    def getPropertiesDF(self,neurons,savetocsv=False):
        propertyJson={}
        for neuron in neurons:
            propertyJson[neuron['sampleid']+'-'+neuron['name']]=self.getNeuronPropertyByID(neuron['sampleid'], neuron['name'])
        
        projectmat = pd.DataFrame()
        columns = []
        for neuron,property in propertyJson.items():
            items = property.items()
            neuronpropertydict = {}
            columns.append(neuron)
            for key, value in items:
                if type(value) ==dict:
                    for subkey, subvalue in value.items():
                        neuronpropertydict[str(key)+"."+str(subkey)] = subvalue
                elif type(value) ==list:
                    for i in value:
                        neuronpropertydict[str(key)+"."+str(value.index(i))] = str(i)
                elif str(key)!='somaregion':
                    neuronpropertydict[str(key)] = value

            neuronproperty = pd.DataFrame.from_dict(neuronpropertydict, orient='index').sort_index()
            projectmat = pd.concat([projectmat, neuronproperty], axis=1)
        # print(columns)
        projectmat.sort_index(axis = 0, ascending = True)
        projectmat.fillna(0, inplace = True)
        projectmat.index.name='property'
        projectmat.columns = columns
        projectmat=projectmat.reset_index()

        if savetocsv:
            projectmat.to_csv(self.localResourcePath+"/test.csv")
        return projectmat
        # pass

    def getNeuronsBySampleID(self,sampleid):
        neuronlist = self.getNeuronListBySampleID(sampleid)
        neuronswc=[]
        for neuron in neuronlist:
            neuronswc.append(self.getNeuronByID(sampleid,neuron['name']))
        return neuronswc

    def getNeuronTreeBySampleID(self,sampleid):
        neuronlist = self.getNeuronListBySampleID(sampleid)
        neurontrees=[]
        for neuron in neuronlist:
            neurontrees.append(self.getNeuronTreeByID(sampleid,neuron['name']))
        return neurontrees
    
    def getNeuronByID(self,sampleid,neuronid):
        pathstr=self.localResourcePath+"/swc/"+sampleid+'/'
        path = Path(pathstr)
        file = Path(pathstr+neuronid)
        if  file.exists():
            with open(file) as f:
                print('exist ',pathstr+neuronid)
                swccontext=f.read()
                f.close()
                return swccontext
        else:
            Path(pathstr).mkdir(parents=True, exist_ok=True)
            url='http://'+self.ip+'/neuronbrowser/api/user/getSWC?sampleid='+sampleid+'&&neuronid='+neuronid
            res=requests.get(url) 
            if res.status_code != 200 or res.text=='':
                print('requests :',res.status_code,url,'try again')
                time.sleep(0.2)
                return self.getNeuronByID(sampleid,neuronid)
                
            with open(pathstr+neuronid,'w+') as f:
                print('write ',pathstr+neuronid)
                f.write(res.text)
                f.close()
                return res.text
        pass
    def getNeuronTreeByID(self,sampleid,neuronid):
        swc = self.getNeuronByID(sampleid,neuronid)
        tree = NeuronTree()
        tree.readSWC(swc)
        return tree
    def getRawNeuronTreeByID(self,sampleid,neuronid):
        info = self.getSampleInfo(sampleid)
        url = 'http://10.10.31.31/swc/newswc/'+info[0]['project_id']+'/'+sampleid+'/swc_raw/'+neuronid
        filename = '../resource/swc_raw/'+sampleid+'/'+neuronid
        self.downloadfile(url,filename)
        tree = NeuronTree()
        tree.readFile(filename)
        return tree
    def getRawNeuronTreeBySampleID(self,sampleid):
     
        neuronlist = self.getNeuronListBySampleID(sampleid)
        rawneurontrees=[]
        for neuron in neuronlist:
            rawneurontrees.append(self.getRawNeuronTreeByID(sampleid,neuron['name']))
        return rawneurontrees
    def downloadfile(self,url,filename=None):
        if(not filename):                         #如果参数没有指定文件名
            filename=os.path.basename(url)          #取用url的尾巴为文件名
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        leng=1
        while(leng==1):
            torrent=requests.get(url)
            leng=len(list(torrent.iter_content(1024)))  #下载区块数
            if(leng==1):                                #如果是1 就是空文件 重新下载
                print(filename,' download failed and will download again',url)
                time.sleep(1)
            else:
                pass
            print(filename,' downloaded')
        with open(filename,'wb') as f:                
            for chunk in torrent.iter_content(1024):    #防止文件过大，以1024为单位一段段写入
                f.write(chunk)
        

    def getAnnotation(self):
        res,filepath=self.getFileFromServer('annotation_10_2017.nrrd')
        return res,filepath

    def getFileFromServer(self,filename):
        pathstr=self.localResourcePath+"/"+filename
        filepath = Path(pathstr)
        if  filepath.exists():
            return True,pathstr
        else:
            url='http://'+self.ip+'/bap/neuronVis/'+filename
            self.downloadfile(url,pathstr)
            return True,pathstr

    def getStructureMask(self,id):
        pathstr=self.localResourcePath+"/strcture/"
        filepathstr=pathstr+"structure_"+str(id)+".nrrd"
        path = Path(pathstr)
        path.mkdir(parents=True, exist_ok=True)
        filepath=Path(filepathstr)
        if  filepath.exists():
            return [True,filepathstr]
        else:
            url='http://'+self.ip+'/bap/neuronVis/allen-structure/'+"structure_"+str(id)+".nrrd"
            self.downloadfile(url,filepathstr)
            return [True,filepathstr]
    def getStructureMask25(self,id):
        pathstr=self.localResourcePath+"/strcture25/"
        filepathstr=pathstr+"structure_"+str(id)+".nrrd"
        path = Path(pathstr)
        path.mkdir(parents=True, exist_ok=True)
        filepath=Path(filepathstr)
        if  filepath.exists():
            return [True,filepathstr]
        else:
            url='http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_masks/structure_masks_25/'+"structure_"+str(id)+".nrrd"
            self.downloadfile(url,filepathstr)
            return [True,filepathstr]

if __name__=="__main__":
    from scipy.cluster.hierarchy import linkage ,fcluster,fclusterdata,dendrogram
    from scipy import spatial

    import numpy as np
    from scipy.spatial import Delaunay
    import matplotlib.pyplot as plt

    iondata =IONData()
    # neuronlist=iondata.getNeuronListBySampleID('000002')
    # neuronlist=iondata.getNeuronListBySomaRegion('AI')
    info = iondata.getSampleInfo('221043')
    neuronlist=iondata.getNeuronListBySampleID('221043')
    iondata.getNeuronByID(neuronlist[0]['sampleid'],neuronlist[0]['name'])
    # print(AINeuronlist)
    # terminalPAGNeuronlist=iondata.getNeuronListByTerminalRegion('PAG')
    # projectPAGNeuronlist=iondata.getNeuronListByProjectRegion('SO')
    # print(projectPAGNeuronlist)


    # distance = spatial.distance.pdist(matData.astype(float))
    # linkresult = linkage(distance,method='average',metric='euclidean')
    # fcluster(linkresult,t=0.99,criterion='inconsistent',depth=2,R=None,monocrit=None)#这个需要先计算linkage，再出结果
    # print(linkresult)
    # dendrogram(linkresult,labels=prodf.columns[1:])
    # plt.xticks(fontsize=12,rotation = 30, ha = 'right')
    # plt.show()
    # # g= sns.clustermap(prodf)
    # swc = iondata.getNeuronByID('192106', '001.swc')
    # pro = iondata.getNeuronPropertyByID('192106', '001.swc')
    # iondata.getAnnotation()
    # loader = NeuronTree()
    # loader.readSWC(swc)
    # print(pro)
    pass