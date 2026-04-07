import sys,copy,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")
import IONData as IONData 
import BrainRegion as BR 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import SwcLoader
import BoundLaplace
import Flatmap
import nrrd
import BoundLaplace as BL
from sklearn.mixture import GaussianMixture
import random
from sklearn.cluster import KMeans
import cv2 as cv
import joblib
from tqdm.notebook import tqdm as tool_bar
###################################### I am a beautiful  seperation line! ##########################################
class HieracyAnalysis():
    def __init__(self,
                 region_resolution=50,
                 target=['MOs','ACAd','ACAv','AId','AIv','FRP','PL','ILA','ORBm','ORBl','ORBvl'],
                 layerName=['L1','L2/3','L5','L6'],
                 layer=np.array([120, 400, 850, 1200]),
                 flatmap_path='flatten_map.npy',
                 
                ):
        self.target = target
        self.layerName = layerName
        iondata =IONData.IONData()
        self.iondata = iondata
        br = BR.BrainRegion()
        br.praseJson()
        flatmap=np.load(flatmap_path)
        flatmap_bound = flatmap.shape
        j = []
        for i in flatmap.reshape(-1,):
            if i not in j:
                j.append(i)
        region2id = {}
        id2region = {}
        regions = []
        for i in j:
            region = br.getRegion(id=i)
            if region!=None:
                id2region[i] = region.name.split('1')[0]
                region2id[region.name.split('1')[0]] = i
                regions.append(region.name.split('1')[0])
        if target=='all':
            target = regions.copy()
        self.layer = layer/20
        res,gridpath = iondata.getFileFromServer("boundlaplace20.nrrd")
        self.grid,header = nrrd.read(gridpath)
        resRelaxation,RelaxationPath=iondata.getFileFromServer('boundlaplaceout20.nrrd')
        relaxation,relaxationheader = nrrd.read(RelaxationPath)
        resdv0, dv0Path = iondata.getFileFromServer('dv0.nrrd')
        dv0, dv0header = nrrd.read(dv0Path)
        resdv1, dv1Path = iondata.getFileFromServer('dv1.nrrd')
        dv1, dv1header = nrrd.read(dv1Path)
        resdv2, dv2Path = iondata.getFileFromServer('dv2.nrrd')
        dv2, dv2header = nrrd.read(dv2Path)
        self.dv0 = dv0.astype(np.float32) / 1000 - 1
        self.dv1 = dv1.astype(np.float32) / 1000 - 1
        self.dv2 = dv2.astype(np.float32) / 1000 - 1
        self.flatenPara = Flatmap.createSurfaceGraph()
        grid_nums = np.array([region_resolution,region_resolution])
        grid_size = np.array(flatmap.shape)/grid_nums
        custom_map = np.zeros(flatmap.shape)
        for i in range(flatmap.shape[0]):
            for j in range(flatmap.shape[1]):
                mark_1 = i//grid_size[0]
                if mark_1==flatmap.shape[0]:
                    mark_1 = mark_1-1
                mark_2 = j//grid_size[1]
                if mark_2==flatmap.shape[1]:
                    mark_2 = mark_2-1
                mark = mark_1*flatmap.shape[1]+mark_2
                custom_map[i,j] = mark+1
        for i in target:
            custom_map[flatmap==region2id[i]] = -custom_map[flatmap==region2id[i]]
        self.custom_map = np.where(custom_map<0,-custom_map,0)*(flatmap>0).astype(float)
        self.custom_regions = []
        for i in self.custom_map.flatten():
            if i !=0 and i not in self.custom_regions:
                self.custom_regions.append(i)
        self.custom_regions_num = len(self.custom_regions)
    def generate_data(self,sample_rate=-1,neuronlist=[]):
        if len(neuronlist)==0:
            neuronlist = []
            for source_region in self.target:
                neuronlist += self.iondata.getNeuronListBySomaRegion(source_region)
        random.shuffle(neuronlist)
        X = np.zeros((self.custom_regions_num,self.custom_regions_num,len(self.layerName)))
        X_num = np.zeros((self.custom_regions_num))
        i = 0
        e = 0
        if sample_rate==-1:
            sample_rate = 2
        else:
            sample_rate = sample_rate/len(neuronlist)
        for i in tool_bar(range(len(neuronlist))):
            if np.random.rand()>sample_rate:
                continue
            if neuronlist[i]['sampleid']=='000001':
                continue
            e = 0
            while e<3: 
                try:
                    tmp = self.collect_information(self.iondata.getNeuronTreeByID(neuronlist[i]['sampleid'],neuronlist[i]['name']))
                    if tmp!=[]:
                        X_num[self.custom_regions.index(int(tmp[0][0][0]))] += 1
                        for j in tmp[1].astype(int):
                            X[self.custom_regions.index(int(tmp[0][0][0])),self.custom_regions.index(j[0]),j[1]] += 1
                    e = 3
                except:
                    e += 1
        self.X = X
        self.X_num = X_num
    def load_data(self,path_1,path_2):
        self.X = np.load(path_1)
        self.X_num = np.load(path_2)
    def claculate_flatten_position(self,x):
        return BoundLaplace.ComputeStreamlines(self.grid,self.dv0,self.dv1,self.dv2,copy.deepcopy(x))
    def mirror_points(self,X):
        Y = []
        for i in X:
            if i.xyz[2] > 5700:
                Y.append([i.xyz[0],i.xyz[1],11400-i.xyz[2],1])
            else:
                Y.append([i.xyz[0],i.xyz[1],i.xyz[2],0])
        Y = np.array(Y)
        return Y
    def calculate_point_region_and_layer(self,X):
        X = X.copy()
        X[:,:3] = X[:,:3]/20
        L = []
        #X = X[X[:,0]<660-1]
        Y = self.claculate_flatten_position(X)
        Z = []
        for i in range(len(X)):
            if Y[i][0]!=0:
                Z.append(Y[i][1])
                L.append([np.argmin(abs(self.layer-Y[i][0])),X[i][3]])
        if Z==[]:
            return []
        X = np.hstack([np.array(Z),np.array(L)])
        Y = []
        for i in range(len(X)):
            p = Flatmap.map2Flatmap(self.flatenPara,X[i][:3]*2,True)
            if p is None or p==[]:
                continue
            region = self.custom_map[int(p[1])][int(p[0])]
            if region>0:
                Y.append([region,X[i][3],X[i][4]])
        Y = np.array(Y)
        return Y
    def collect_information(self,x):
        root_information = self.calculate_point_region_and_layer(self.mirror_points([x.root]))
        terminals_information = self.calculate_point_region_and_layer(self.mirror_points(x.terminals))
        if root_information!=[] and terminals_information!=[]:
            if 1:
                return [root_information,terminals_information[~(terminals_information[:,0]==root_information[0][0])*(terminals_information[:,2]==root_information[0][2])]]
            else:
                return [root_information,terminals_information]
        else:
            return []
    def update_reward(self,reward_pattern,h=[]):
        max_features = int(np.max(reward_pattern)+1)
        conf = 1-abs(np.mean(reward_pattern[reward_pattern!=-1]))
        max_score = [-2,[],[]]
        tmp_reward = reward_pattern*1
        for i in range(2**max_features):
            tmp_reward = reward_pattern*0
            tmp = i
            pattern = []
            for j in range(max_features):
                tmp_reward[reward_pattern==j] = 1 if tmp%2 == 1 else -1
                pattern.append(1 if tmp%2 == 1 else -1)
                tmp = tmp//2
            if len(h)>0:
                score = h.reshape(-1,1)
            else:
                score = (np.mean(tmp_reward.transpose()-tmp_reward,axis=-1)).reshape(-1,1)
            score = np.mean(tmp_reward.transpose()*(score-score.transpose()))
            if score>max_score[0]:
                max_score[0] = score
                max_score[1] = tmp_reward*1
                max_score[2] = pattern
        return max_score[1]
    def claculate_hierachical_score(self,x,reward,reward_pattern,h,epochs=1000,if_generate_reward=False,if_update_reward=False):
        if if_generate_reward:
            reward = self.update_reward(reward_pattern)
            reward *= self.X_num.reshape(-1,1)>0
        #better way, normalization along one direction
        _in = np.sum(x,axis=0).reshape(-1,1)
        _out = np.sum(x,axis=1).reshape(-1,1)
        for e in range(epochs):
            #better way, h += lr*(np.sum(np.nan_to_num(x.transpose()*(h+reward.transpose()))+np.nan_to_num(x*(h-reward)),axis=-1)/2-regularization*2*np.sum(h)), when regularization=1/N, it equals to mean.
            h = np.sum(np.nan_to_num(x.transpose()/_in*(h+reward.transpose()))+np.nan_to_num(x/_out*(h-reward)),axis=-1)/2
            h -= np.mean(h)
            if if_update_reward:
                reward = self.update_reward(reward_pattern,h)
                reward *= self.X_num.reshape(-1,1)>0
        return h
    def process_data(self,mode=0,min_num=3,iteration=1000,vmin=-1,vmax=1,power=1,
                     FF_FB_table = np.array([
                                    [0.15,0.3,0.15,0.42],
                                    [0.15,0.32,0.37,0.15],
                                    [0.42,0.32,0.15,0.13],
                                    [0,0.08,0.35,0.37],
                                    [0.3,0,0.15,0.6],
                                    [0.32,0.15,0.33,0.16]
                                                       ]),
                     FF_FB_seperate_index = 3,
                     if_generate_reward=False,if_update_reward=False,clusters=0,if_draw=False,save_path='tmp'):
        '''
            mode: whether remove data below threshold min_num
            min_num: minimum neurons to count as a pattern
            vmin/vmax: score drawing range
            power: power function, nonlinear transformation, 1 no influence, <1 more attention on small value, >1 vise versa
            FF_FB_table: predefined feedforward/feedback mapping
            FF_FB_seperate_index: an index to indicate [0-index) as FF, others as FB
            if_generate_reward: whether reinitialize feedforward/feedback mapping 
            if_update_reward: whether update feedforward/feedback mapping during iterations
            clusters: >0 clustering and generating feedforward/feedback mapping
                      =0 use predifiend feedforward/feedback mapping FF_FB_table
                      <0 use projection strength as reward instead of feedforward/feedback patterns
            if_draw: draw results or not
        '''
        norm = self.X_num.reshape(-1,1,1)
        X_norm = self.X/np.where(norm==0,1,norm)
        reward = np.sum(X_norm,axis=-1)
        reward_pattern = np.sum(X_norm,axis=-1)
        norm = np.linalg.norm(X_norm,axis=-1).reshape(X_norm.shape[0],-1,1)
        X_norm /= np.where(norm==0,1,norm)
        if clusters >0:
            tmp = X_norm.reshape(-1,X_norm.shape[-1])
            reward_pattern = np.sum(tmp,axis=-1)*0-1
            reward_pattern[np.sum(tmp,axis=-1)!=0] = KMeans(n_clusters=clusters,random_state=9).fit_predict(tmp[np.sum(tmp,axis=-1)!=0])
            reward_pattern = reward_pattern.reshape(X_norm.shape[0],-1)
            reward_pattern = (reward_pattern+1)*(self.X_num.reshape(-1,1)>min_num)-1
        elif clusters==0:
            FF_FB_table = np.array(FF_FB_table)
            norm = np.linalg.norm(FF_FB_table,axis=-1).reshape(-1,1)
            FF_FB_table = FF_FB_table/np.where(norm==0,1,norm)
            reward_pattern = np.argmax(np.dot(X_norm,FF_FB_table.transpose()),axis=-1)
            reward_pattern = (reward_pattern+1)*(self.X_num.reshape(-1,1)>min_num)-1
            reward = np.where(reward_pattern<FF_FB_seperate_index,1,-1)
            reward *= self.X_num.reshape(-1,1)>min_num
        else:
            reward = np.mean(X_norm,axis=-1)
            reward *= self.X_num.reshape(-1,1)>min_num
        if mode==0:
            h_X = np.power(np.sum(self.X,axis=-1),power)
        else:
            h_X = np.power(np.sum(self.X,axis=-1),power)*(self.X_num.reshape(-1,1)>min_num)
        h = self.claculate_hierachical_score(h_X,reward,reward_pattern,np.zeros((h_X.shape[0])),iteration,if_generate_reward=if_generate_reward,if_update_reward=if_update_reward)
        Y = np.zeros(self.custom_map.shape)
        for i in range(len(self.custom_regions)):
            Y[self.custom_map==self.custom_regions[i]] = h[i]
        if if_draw:
            plt.matshow(Y,cmap='bwr',vmin=vmin,vmax=vmax)
            plt.colorbar()
            plt.title('hieracical map')
            plt.savefig(save_path,dpi=600,format="pdf")
            plt.show()
        return Y
    def save(self,path_1='X',path_2='X_num'):
        np.save(path_1,self.X)
        np.save(path_2,self.X_num)
    def get_hierachical_score(self,computing_mode=0,if_draw=True,save_path='tmp'):
        '''
            computing_mode: 0 original paper method
                            1 original paper method with iterated updating reward
                            2 self clustering
                            3 self clustering and updating
                            4 custom method
        '''
        if computing_mode==0:
            return self.process_data(mode=0,min_num=3,iteration=100,vmin=-0.4,vmax=0.4,power=1,if_generate_reward=False,if_update_reward=False,clusters=0,if_draw=if_draw,save_path=save_path)
        if computing_mode==1:
            return self.process_data(mode=0,min_num=3,iteration=100,vmin=-0.4,vmax=0.4,power=1,if_generate_reward=False,if_update_reward=True,clusters=0,if_draw=if_draw,save_path=save_path)
        if computing_mode==2:
            return self.process_data(mode=0,min_num=3,iteration=100,vmin=-0.1,vmax=0.1,power=1,if_generate_reward=True,if_update_reward=False,clusters=6,if_draw=if_draw,save_path=save_path)
        if computing_mode==3:
            return self.process_data(mode=0,min_num=3,iteration=100,vmin=-0.1,vmax=0.1,power=1,if_generate_reward=True,if_update_reward=True,clusters=6,if_draw=if_draw,save_path=save_path)
        if computing_mode==4:
            return self.process_data(mode=0,min_num=0,iteration=100,vmin=-0.1,vmax=0.1,power=1,if_generate_reward=False,if_update_reward=False,clusters=-1,if_draw=if_draw,save_path=save_path)
############################## I am the brother of the beautiful  seperation line! ##################################
if __name__=="__main__":
    matplotlib.use('TkAgg')
#########################very suggested using pre-generated flatten_map, if already generated, just skip this part####################
    flatenPara = joblib.load('../resource/flatenPara.pkl')
    Flatmap.createFlatmap(flatenPara)
    grid,header = nrrd.read('../resource/flatmap.nrrd')
    tmp = []
    for i in grid.flatten():
        if i not in tmp:
            tmp.append(i)
    len(tmp)
    def f(x,r=1):
        y = x.copy()
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i,j]!=0:
                    continue
                tmp = np.array([i-r,i+r+1,j-r,j+r+1])
                tmp = np.where(tmp<0,0,tmp)
                tmp2 = {}
                for k in x[tmp[0]:tmp[1],tmp[2]:tmp[3]].flatten():
                    if k not in tmp2.keys():
                        tmp2[k] = 0
                    tmp2[k] += 1
                c = 0
                tmp = 0
                for k in tmp2.keys():
                    if tmp2[k]>tmp:
                        c = k
                        tmp = tmp2[k]
                y[i,j] = c
        return y
    y = grid.transpose()
    for i in range(200):
        y = f(y,1)
    np.save('../resource/flatten_map',y)
#####################################################################################################################################
    model = HieracyAnalysis(flatmap_path='../resource/flatten_map.npy')
    model.load_data('../resource/X.npy','../resource/X_num.npy')
    _ = model.get_hierachical_score(0)
