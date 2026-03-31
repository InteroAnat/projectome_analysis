import os,sys ,random
import platform as pf

neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)

from Render import *
from RenderMacaqueGL import *
from vispy import app,gloo,io
from SwcLoader import *
from GeometryAdapter import GeometryAdapter 
from IONData import IONData
import NeuronProcess

if hasattr(sys.modules[__name__], '__file__'):
	_file_name = __file__
else:
	_file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)



class neuronVis:
	def __init__(self,size=(800,600),renderModel=0,near=10000,far=30000,species='monkey'):
		if pf.system()=='Windows' or pf.system()=='Linux':
			self.render = RenderGL(size=size,renderModel=renderModel,near=near,far=far,species=species)
		elif pf.system()=='Darwin':
			self.render = Render(size=size)

		self.regionID2Name={}
		self.regionName2ID={}
		self.Region()
		self.neurons={}
		self.regions={}
		self.regions['root']=self.render.rootMesh
		self.neuronWidth=1
	def setLineWidth(self,width=1):
		self.neuronWidth=width
		self.render.setLineWidth(width)

	def addNeuronByList(self,neurons,color=None,somaColor=None,axonColor=None,dendriteColor=None,mirrorToRight=False,somaHide=False,axonHide=False,dendriteHide=False,isLine=True,highLight=False,somaRadius=200):
		for neuron in neurons:
			color0=color
			if color is None:
				color0=[random.random(),random.random(),random.random()]
			self.addNeuronByID(neuron['sampleid'],neuron['name'],color0,somaColor,axonColor,dendriteColor,mirrorToRight,somaHide=somaHide,axonHide=axonHide,dendriteHide=dendriteHide,isLine=isLine,highLight=highLight,somaRadius=somaRadius)

	def addNeuronByID(self,sampleid,neuronid,color=[1.0,1.0,1.0],somaColor=None,axonColor=None,dendriteColor=None,mirrorToRight=False,somaHide=False,axonHide=False,dendriteHide=False,isLine=True,highLight=False,somaRadius=200):
		ga = GeometryAdapter(self.render.species)
		ga.isLine=isLine
		ga.width = self.neuronWidth
		ga.axonColor = axonColor
		ga.dendriteColor = dendriteColor
		ga.somaColor=somaColor
		ga.setMirrorToRight(mirrorToRight)
		ga.readByID(sampleid,neuronid,somaHide,axonHide,dendriteHide,somaRadius)
		ga.geometry.name=sampleid+'-'+neuronid
		ga.geometry.highLight=highLight
		self.neurons[ga.geometry.name]=ga.geometry
		self.render.addGeometry(ga.geometry,color,highLight)
	
	def addNeuron(self,name='name',color=[1.0,1.0,1.0],somaColor=None,axonColor=None,dendriteColor=None,mirrorToRight=False,somaHide=False,axonHide=False,dendriteHide=False,isLine=True,highLight=False):
		ga = GeometryAdapter(self.render.species)
		
		ga.width = self.neuronWidth
		ga.isLine=isLine
		ga.axonColor = axonColor
		ga.dendriteColor = dendriteColor
		ga.somaColor=somaColor
		ga.setMirrorToRight(mirrorToRight)
		ga.readFile(name,somaHide,axonHide,dendriteHide)
		ga.geometry.name=name
		ga.geometry.color=color
		ga.geometry.highLight=highLight
		self.neurons[name]=ga.geometry
		self.render.addGeometry(ga.geometry,color,highLight)
	def addNeuronTree(self,neuronTree,name='name',color=[1.0,1.0,1.0],somaColor=None,axonColor=None,dendriteColor=None,mirrorToRight=False,depthIntensity=False,somaHide=False,axonHide=False,dendriteHide=False,isLine=True,highLight=False):
		neuronTree.somaHide = somaHide
		neuronTree.axonHide = axonHide
		neuronTree.dendriteHide = dendriteHide
		ga = GeometryAdapter()
		ga.width = self.neuronWidth
		ga.isLine=isLine
		ga.axonColor = axonColor
		ga.dendriteColor = dendriteColor
		ga.somaColor=somaColor
		ga.setMirrorToRight(mirrorToRight)
		ga.depthIntensity=depthIntensity
		ga.parse(neuronTree)
		ga.geometry.name=name
		ga.geometry.color=color
		ga.geometry.highLight=highLight
		self.neurons[name]=ga.geometry
		self.render.addGeometry(ga.geometry,color,highLight)

	def addRegion(self,name=None,color=[1.0,1.0,1.0],regionFileName=None):
		iondata = IONData()
		if regionFileName is None and name is not None:
			response,regionFileName=iondata.getFileFromServer('allobj/'+str(self.regionName2ID[name][0])+'.obj')
		# regionFileName = CURRENT_FILE_PATH+'/../resource/allobj/'+str(self.regionName2ID[name][0])+'.obj'
		if regionFileName is None:
			return
		geo  = self.render.loadGeometry(regionFileName)
		geo.name=name
		self.regions[name]=geo
		self.render.addGeometry(geo,color)

	def clear(self,root=False,neurons=True,regions=True):
		if root:
			self.regions['root'].Geometry.setHide(True)
		if neurons:
			# del self.neurons
			removeneuron=[]
			for neuronhandle in self.render.geometries:
				if neuronhandle.Geometry.type!=-1:
					removeneuron.append(neuronhandle)
			for neuron in removeneuron:
				self.render.geometries.remove(neuron)
			removeneuron=[]
			# for name,neuron in self.neurons.items():
			# 	del neuron
		if regions:
			for name,region in self.regions.items():
				if name!='root':
					region.setHide(True)


	def Region(self):
		if len(self.regionID2Name)==0:
			lines=[]
			iondata = IONData()
			regionFileName=iondata.getFileFromServer('annot.txt')
			with open(regionFileName[1], 'r') as file_to_read:
			#   while True:
				lines = file_to_read.readlines()
			for line in lines:
				linetmp=line[:-1].split(":")
				self.regionID2Name[int(linetmp[0])]=[linetmp[1],linetmp[2]]
				self.regionName2ID[linetmp[2]]=[int(linetmp[0]),linetmp[1]]




if __name__=="__main__":
	import Visual as nv
	import json
	import SwcLoader as nt
	import Scene
	neuronvis = nv.neuronVis(species='monkey')

	neuronvis.render.setBackgroundColor((1.0,1.0,1.,1.0))
	# neuronvis.addRegion('CLA',[0.5,1.0,0.5])
	# neuronvis.addRegion( color=[0.5, 1.0, 0.5],regionFileName="D:/neuron-vis/resource/allobj/Layer3.obj")
	iondata = IONData()
	neuronlist = iondata.getNeuronListBySampleID('251637')
	# f=open('../resource/pfcsubtype60.json', encoding='gbk')
	# neurons=[]
	# neurons = json.load(f)
	# neurons=iondata.getNeuronListBySomaRegion('PB')
	# print(neurons)

	# Scene.createScene(neurons,"../resource/test.nv")
	# neuronvis.render.setLookAt((5000,0,15000),(5000,0,0),(0,-1,0))
	# tree= nt.NeuronTree()
	# tree.readFile('../resource/swc_merge/subtype6/merge/3-0merge.swc')
 
	neuronvis.setLineWidth(2)
	# neuronvis.addNeuronByID('17234','015.swc',somaColor=[0,1,0],somaHide=False,axonHide=False,dendriteHide=False,isLine=True)
	# neuronvis.addNeuronTree(tree,color=[1,0,0],isLine=False)
	# tree= nt.NeuronTree()
	# tree.readFile('../resource/swc_merge/2-1merge.swc')
	# neuronvis.addNeuronTree(tree,color=[0,1,0],isLine=False)
	# tree= nt.NeuronTree()
	# tree.readFile('../resource/swc_merge/2-2merge.swc')
	# neuronvis.addNeuronTree(tree,color=[0,0,1],isLine=False)

	# neuronvis.addNeuronByList(neuronlist[156:157],mirrorToRight=False,isLine=True,somaRadius=500)
	# neuronvis.addNeuronByList(neuronlist[69:70],color=[0,1,0],mirrorToRight=True,isLine=False,highLight=True)

	# neuronvis.render.savepng('../resource/testwxf.png')
	# neuronvis.clear(root=True,neurons=False)
	bg_neurons = [
		'001.swc', '007.swc', '008.swc', '013.swc', '014.swc', '045.swc', '046.swc', 
		'049.swc', '051.swc', '052.swc', '054.swc', '057.swc', '061.swc', '062.swc', 
		'083.swc', '084.swc', '085.swc', '087.swc', '097.swc', '098.swc', '103.swc', 
		'104.swc', '122.swc', '124.swc', '128.swc', '131.swc', '134.swc', '135.swc', 
		'136.swc', '149.swc', '152.swc', '157.swc', '392.swc', '398.swc', '399.swc', 
		'400.swc', '401.swc', '402.swc', '403.swc', '404.swc', '405.swc', '409.swc', 
		'414.swc', '415.swc', '417.swc', '419.swc', '420.swc', '425.swc', '430.swc', 
		'431.swc', '433.swc', '434.swc', '437.swc', '442.swc', '445.swc', '447.swc', 
		'450.swc', '452.swc', '457.swc', '463.swc', '465.swc', '466.swc', '472.swc', 
		'474.swc', '475.swc', '477.swc', '478.swc', '479.swc', '480.swc', '481.swc', 
		'482.swc', '483.swc', '484.swc', '489.swc', '490.swc', '491.swc', '492.swc', 
		'493.swc'
	]
	sample_id = '251637'
	color_by_type = {
    'ITs': [1, 0.5, 0],    # Orange (48 neurons)
    'ITi': [0, 0.5, 1],    # Light blue (20 neurons)
    'CT':  [1, 0, 1],      # Magenta (5 neurons)
    'PT':  [0, 1, 0]       # Green (5 neurons)
	}

	ITs_BG = {'001.swc', '007.swc', '013.swc', '045.swc', '052.swc', '054.swc', '061.swc', 
			'083.swc', '084.swc', '085.swc', '087.swc', '097.swc', '098.swc', '103.swc', 
			'104.swc', '122.swc', '124.swc', '128.swc', '131.swc', '392.swc', '399.swc', 
			'401.swc', '402.swc', '405.swc', '409.swc', '430.swc', '431.swc', '433.swc', 
			'437.swc', '442.swc', '465.swc', '472.swc', '474.swc', '475.swc', '477.swc', 
			'478.swc', '482.swc', '484.swc', '490.swc', '491.swc', '492.swc', '493.swc'}

	ITi_BG = {'049.swc', '051.swc', '062.swc', '135.swc', '136.swc', '403.swc', '404.swc', 
			'415.swc', '417.swc', '419.swc', '420.swc', '425.swc', '452.swc', '457.swc', 
			'466.swc', '479.swc', '480.swc', '481.swc', '483.swc'}

	CT_BG = {'008.swc', '398.swc', '400.swc', '434.swc', '463.swc'}
	PT_BG = {'014.swc', '134.swc', '157.swc', '414.swc', '489.swc'}

	for swc_name in bg_neurons:
		if swc_name in ITs_BG:
			color = color_by_type['ITs']
		elif swc_name in ITi_BG:
			color = color_by_type['ITi']
		elif swc_name in CT_BG:
			color = color_by_type['CT']
		else:
			color = color_by_type['PT']
    
		neuronvis.addNeuronByID(sample_id, swc_name,color=color ,somaHide=False, axonHide=False, dendriteHide=False, isLine=True)
	
	neuronvis.addRegion(
		name='MyRegion',  # Any name you want
		color=[0.3, 0.7, 1.0],  # Light blue
		regionFileName=r"C:\Users\binbi\Downloads\NeuronView (1)\data\allobj\macaqueallobj\striatum (Str).obj"
	)

	neuronvis.render.setView('anterior')
	# neuronvis.render.animation(90*0)
	neuronvis.render.run()
	# neuronvis.render.closeWindow()
 
 
 
 
 	# could use getneuronlist or pd operation to filter specific neuron groups.
  
	pass