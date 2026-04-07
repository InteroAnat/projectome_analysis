import sys,copy,os,inspect


if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
sys.path.append(os.getcwd()+"/../neuronVis")

import joblib
import SwcLoader,IONData
import Visual as nv

def loadEdges(edgesOrigin):
    newneuron=SwcLoader.NeuronTree()
    edges=[]
    for branch in edgesOrigin:
        edge =  SwcLoader.Edge()
        for point in branch[0:-1:10]:
            edge.addPoint(SwcLoader.Point(point))
        if len(edge.data):
            edge.addPoint(SwcLoader.Point(branch[-1]))
            edges.append(edge)
        else:
            edge.addPoint(SwcLoader.Point(branch[0]))
            edge.addPoint(SwcLoader.Point(branch[-1]))
            edges.append(edge)

    newneuron.edges= edges
    newneuron.rootAxonEdge = edges[0]
    newneuron.root=edges[0].data[0]
    return newneuron

def plotNeuron(filename):
    edgesOrigin = joblib.load(filename)
    return loadEdges(edgesOrigin)

def plotPartNeuron(filename,regions):
    neuronpart = joblib.load(filename)
    newneurons=[]
    print(neuronpart.keys())
    for region in regions:
        edgesOrigin = neuronpart[region]['branches_df'][region]
        newneuron=loadEdges(edgesOrigin)
        newneuron.width=3
        newneurons.append(newneuron)
    return newneurons

iondata = IONData.IONData()
iondata.getFileFromServer('202271_126.pkl')
iondata.getFileFromServer('202271_126(1).pkl')
newneurons = plotPartNeuron('../resource/202271_126.pkl',['ENT','ACA'])


neuronvis = nv.neuronVis(renderModel=0)
neuronvis.render.setBackgroundColor((1.0,1.0,1.0,1.0))
neuronvis.render.setLookAt((0.,0.0,15000),(0,0,0),(0.0,-1.0,0.0))
# neuronvis.addRegion('HY',[0.5,1.0,0.5])
for neuron in newneurons:
    neuronvis.addNeuronTree(neuron,'t',color=[1.0,0,0])
neuronvis.addNeuronTree(plotNeuron('../resource/202271_126(1).pkl'),color=[0.5,0.5,0.5])
# neuronvis.clear(root=True,neurons=False,regions=False)
neuronvis.render.run()
# %%
