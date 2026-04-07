import os,sys ,random
import platform as pf
from Render import *
from RenderGL import *
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

	neuronvis.addNeuronByList(neuronlist[156:157],mirrorToRight=False,isLine=True,somaRadius=500)
	# neuronvis.addNeuronByList(neuronlist[69:70],color=[0,1,0],mirrorToRight=True,isLine=False,highLight=True)

	# neuronvis.render.savepng('../resource/testwxf.png')
	# neuronvis.clear(root=True,neurons=False)
	neuronvis.render.setView()
	# neuronvis.render.animation(90*0)
	neuronvis.render.run()
	# neuronvis.render.closeWindow()
	pass