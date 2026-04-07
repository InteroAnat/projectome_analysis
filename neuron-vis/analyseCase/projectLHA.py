import sys,os,inspect
if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(CURRENT_FILE_PATH+"/../neuronVis")
# sys.path.append(r"E:\jupyter\dataanalyse\anlaysmodel\neuron-vis\neuronVis")
import IONData
import pickle
import Scene

##1.获取ISO和PFC项目sampleidlist->allsamplelist
iondata =IONData.IONData()
isosampleid = iondata.getSampleInfo(projectID='ISO')#获取ISO项目sampleid
pfcsampleid = iondata.getSampleInfo(projectID='PFC')#获取PFC项目sampleid
isosamplelist=[]
pfcsamplelist=[]
for i in range(len(isosampleid)):
    isosamplelist.append(isosampleid[i]['fMOST_id'])
for i in range(len(pfcsampleid)):
    pfcsamplelist.append(pfcsampleid[i]['fMOST_id'])
allsamplelist = isosamplelist + pfcsamplelist
print(len(allsamplelist))

##2.获取ISO和pfc项目中投射到LHA的所有neuron编号
projectLHANeuronlist=iondata.getNeuronListByProjectRegion('LHA')#获取投射到LHA的所有神经元编号
projectLHAlist_ISO_PFC=[]
for i in projectLHANeuronlist:
    if int(i['sampleid']) in allsamplelist:
        projectLHAlist_ISO_PFC.append(i)

##3.保存pkl
with open('../resource/projectLHAlist_ISO_PFC.pkl', 'wb') as f:
    pickle.dump(projectLHAlist_ISO_PFC, f)
##保存Scene
Scene.createScene(projectLHAlist_ISO_PFC, '../resource/ISO_PFC_ProjectLHA.nv')