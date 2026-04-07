import sys,os
import numpy as np
from vispy.geometry import MeshData,create_sphere
from SwcLoader import *

from IONData  import IONData
import Node
import glm

colormap=[
[1,0,0],[0,1,0],[0,0,1],
[0,1,1],[1,0,1],
[0.5,0.5,0.5],[0,0,0],
[0.5,1,0],[1,0.5,0],[1,1,0],[0,0.5,1]
]

if hasattr(sys.modules[__name__], '__file__'):
    _file_name = __file__
else:
    _file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)
class GeometryAdapter:
	def __init__(self,species='mouse'):
		self.geometry = Node.Geometry()
		self.mirrorToRight=False
		self.depthIntensity=False
		self.somaColor=None
		self.axonColor=None
		self.dendriteColor=None
		self.width=1
		self.isLine=True
		self.species=species
		self.brainWidth=11400
		if species=='monkey':
			self.brainWidth=60000

	def computeCylinderVertexs(self,pt0,  pt1, radius):
		vts=[]
		nors=[]
		dir = glm.normalize(glm.vec3(pt0[0],pt0[1],pt0[2])-glm.vec3(pt1[0],pt1[1],pt1[2]))
		# xx=0
		# yy=0
		# zz=0
		# if abs(dir[2]>0.001):
		# 	xx = pt0[0]+10
		# 	yy=pt0[1]
		# 	zz=-(dir[0]*(xx - pt0[0]) + dir[1]*(yy - pt0[1])) / dir[2] + pt0[2]
		# elif dir[0]!=0:
		# 	yy=pt0[1]+10
		# 	zz=pt0[2]
		# 	xx= -(dir[1]*(yy - pt0[1]) + dir[2]*(zz - pt0[2])) / dir[0] + pt0[0]
		# else:
		# 	xx=pt0[0]+10
		# 	zz=pt0[2]
		# 	yy=-(dir[0]*(xx - pt0[0]) + dir[2]*(zz - pt0[2])) / dir[1] + pt0[1]
		
		# planeDir= glm.normalize(glm.vec4(xx - pt0[0], yy - pt0[1], zz - pt0[2],0))
		planeDir = glm.normalize(glm.vec4(glm.cross(dir,glm.normalize(glm.vec3(pt0[0],pt0[1],pt0[2]))),0.0))
		edges=6
		for k in range(edges):
			mt =glm.mat4()
			mt = glm.rotate(mt,glm.radians(k*360./edges),dir)
			newrotDir = glm.normalize(planeDir * mt)
			newPt0 = glm.vec3(pt0[0],pt0[1],pt0[2])+glm.vec3(newrotDir[0],newrotDir[1],newrotDir[2])*radius
			newPt1 = glm.vec3(pt1[0],pt1[1],pt1[2])+glm.vec3(newrotDir[0],newrotDir[1],newrotDir[2])*radius
			nor0=glm.normalize(newPt0-glm.vec3(pt0[0],pt0[1],pt0[2]))
			vts.append(newPt0)
			vts.append(newPt1)
		vts.append(vts[0])
		vts.append(vts[1])
		return vts
		pass
	def setMirrorToRight(self,mirrorToRight):
		self.mirrorToRight=mirrorToRight

	def readFile(self,filename,somaHide=False,axonHide=False,dendriteHide=False):
		self.geometry.name=filename
		neuron = NeuronTree()
		neuron.readFile(filename)
		neuron.width=self.width
		neuron.somaHide = somaHide
		neuron.axonHide = axonHide
		neuron.dendriteHide = dendriteHide
		self.parse(neuron)

	def readByID(self,sampleid,neuronid,somaHide=False,axonHide=False,dendriteHide=False,somaRadius=200):
		self.geometry.name=sampleid+'-'+neuronid
		iondata =IONData()
		neuron = iondata.getNeuronTreeByID(sampleid, neuronid)
		neuron.somaRadius=somaRadius
		neuron.width=self.width
		neuron.somaHide = somaHide
		neuron.axonHide = axonHide
		neuron.dendriteHide = dendriteHide
		self.parse(neuron)
		pass

	def parse(self,neuron):
		mirror=False
		if  self.mirrorToRight:
			mirror=True
		meshdata = create_sphere(radius=neuron.somaRadius)
		rootxyz = neuron.root.xyz.copy()
		if mirror:
			rootxyz=self.mirrorFun(rootxyz)
			
		self.geometry.vertex=(meshdata.get_vertices()+rootxyz)
		self.geometry.index=meshdata.get_faces()
		self.geometry.normal=meshdata.get_vertex_normals()
		self.geometry.drawModel='triangles'
		self.geometry.type=0
		self.geometry.hide = neuron.somaHide
		self.geometry.width = neuron.width
		self.geometry.name='soma'
		if self.somaColor is not None:
			self.geometry.uniformColor = self.somaColor

		axon = Node.Geometry()
		axon.hide = neuron.axonHide
		axon.type=1
		axon.width = neuron.width
		axon.name='axon'
		if self.axonColor is not None:
			axon.uniformColor=self.axonColor
		self.geometry.addChild(axon)
		dendrite = Node.Geometry()
		
		dendrite.hide = neuron.dendriteHide
		dendrite.type=1
		dendrite.name='dendrite'
		dendrite.width = neuron.width

		if self.dendriteColor is not None:
			dendrite.uniformColor=self.dendriteColor
		self.geometry.addChild(dendrite)
		if self.isLine:
			axon.drawModel='lines'
			dendrite.drawModel='lines'
			self.createAxon(axon,neuron,mirror,self.depthIntensity)
			self.createDendrite(dendrite,neuron,mirror)
		else:
			axon.drawModel='triangles'
			dendrite.drawModel='triangles'
			self.createAxonTri(axon,neuron,mirror,self.depthIntensity)
			self.createDendriteTri(dendrite,neuron,mirror)
	def createDendriteTri(self,dendrite,neuron,mirror):
		start=0
		for edge in neuron.edges:
			edgestart=0
			if edge.data[1].type!=3:
				continue
			fromP=None
			pre_vts=None
			v=[]
			for index in range(len(edge.data)):
				point = edge.data[index]
				fromP = point
				toP=point
								

				if index>0:
					fromP=edge.data[index-1]
					fromPxyz = fromP.xyz.copy()
					toPxyz = toP.xyz.copy()
					if mirror:
						fromPxyz=self.mirrorFun(fromPxyz)
						toPxyz=self.mirrorFun(toPxyz)
					vts =self.computeCylinderVertexs(fromPxyz, toPxyz, dendrite.width*4000/256)
					if pre_vts and len(pre_vts)==len(vts):
						for i in range(7):
							vts[i*2]=pre_vts[i*2+1]
					fromP=toP
					for j in range(len(vts)):
						v.append(vts[j])
						dendrite.addPoint((vts[j][0],vts[j][1],vts[j][2]))
					pre_vts=vts
			offset = len(dendrite.vertex) -len(v)
			r=6
			for i in range(len(edge.data)-1):
				sum=i*(r+1)*2
				for j in range(r):
					dendrite.addIndex(0+j*2+sum+offset)
					dendrite.addIndex(1+j*2+sum+offset)
					dendrite.addIndex(2+j*2+sum+offset)
					dendrite.addIndex(1+j*2+sum+offset)
					dendrite.addIndex(3+j*2+sum+offset)
					dendrite.addIndex(2+j*2+sum+offset)
		pass
	def createDendrite(self,dendrite,neuron,mirror):
		start=0

		for edge in neuron.edges:
			edgestart=0
			if edge.data[1].type!=3:
				continue
			for index in range(len(edge.data)):
				point = edge.data[index]
				prepoint = point
				if index>0:
					prepoint=edge.data[index-1]
				normal = [prepoint.x-point.x,prepoint.y-point.y,prepoint.z-point.z]
				if normal[0]==0 and normal[1]==0 and normal[2]==0:
					normal=[0,0,1]

				xyz = point.xyz.copy()
				if mirror:
					xyz=self.mirrorFun(xyz)
				dendrite.addPoint(xyz)
				dendrite.addType(point.type)
				dendrite.addNormal(normal)
				if edgestart>0:
					dendrite.addIndex(start-1)
					dendrite.addIndex(start)
				start=start+1
				edgestart=edgestart+1
			# if len(point.children)>0:
			# 	for child in point.children:
			# 		xyz = child.xyz.copy()
			# 		if mirror:
			# 			xyz[2]=11400-xyz[2]
			# 		dendrite.addPoint(xyz)
			# 		dendrite.addIndex(start-1)
			# 		dendrite.addIndex(start)
			# 		start=start+1

	def createAxonTri(self,axon,neuron,mirror,depthIntensity=False):
		if neuron.rootAxonEdge is None:
			return
		start=0

		for edge in neuron.edges:
			edgestart=0
			# if edge.data[0].type!=1 and edge.data[0].type!=2 and edge.data[0].type!=0 and edge.data[0].type!=12 and edge.data[0].type!=10 and edge.data[0].type!=11:
			if edge.data[0].type==3:
				continue
			fromP=None
			pre_vts=None
			v=[]
			for index in range(len(edge.data)):
				point = edge.data[index]
				fromP = point
				toP=point

				if index>0:
					fromP=edge.data[index-1]
					fromPxyz = fromP.xyz.copy()
					toPxyz = toP.xyz.copy()
					if mirror:
						fromPxyz=self.mirrorFun(fromPxyz)
						toPxyz=self.mirrorFun(toPxyz)
					vts =self.computeCylinderVertexs(fromPxyz, toPxyz, axon.width*4000/256)
					# vts =self.computeCylinderVertexs(fromPxyz, toPxyz, ((fromP.ratio+toP.ratio)/2.0+0.5) *4000/256)
					if pre_vts and len(pre_vts)==len(vts):
						for i in range(7):
							vts[i*2]=pre_vts[i*2+1]
					fromP=toP
					for j in range(len(vts)):
						v.append(vts[j])
						axon.addPoint((vts[j][0],vts[j][1],vts[j][2]))
					pre_vts=vts
			offset = len(axon.vertex) -len(v)
			r=6
			for i in range(len(edge.data)-1):
				sum=i*(r+1)*2
				for j in range(r):
					axon.addIndex(0+j*2+sum+offset)
					axon.addIndex(1+j*2+sum+offset)
					axon.addIndex(2+j*2+sum+offset)
					axon.addIndex(1+j*2+sum+offset)
					axon.addIndex(3+j*2+sum+offset)
					axon.addIndex(2+j*2+sum+offset)
		pass
	def mirrorFun(self,coord):
		result=coord
		if self.species=='mouse':
			result[2]=self.brainWidth-coord[2]
		else:
			result[0]=self.brainWidth-coord[0]
		return result
	def createAxon(self,axon,neuron,mirror,depthIntensity=False):
		if neuron.rootAxonEdge is None:
			return
		start=0

		for edge in neuron.edges:
			edgestart=0
			# if edge.data[0].type!=1 and edge.data[0].type!=2 and edge.data[0].type!=0 and edge.data[0].type!=12 and edge.data[0].type!=10 and edge.data[0].type!=11:
			if edge.data[0].type==3:
				continue
			index=0
			points = edge.data[::10]
			points.append(edge.data[len(edge.data)-1])
			for point in points:
				prepoint = point
				if index>0:
					prepoint=points[(index-1)]
				normal = [prepoint.x-point.x,prepoint.y-point.y,prepoint.z-point.z]
				if normal[0]==0 and normal[1]==0 and normal[2]==0:
					normal=[0,0,1]
				xyz = point.xyz.copy()
				if mirror:
					xyz=self.mirrorFun(xyz)
				if depthIntensity:
					intensity = float(edge.maxDepth)/neuron.rootAxonEdge.maxDepth
					# axon.addPoint(xyz,color=[intensity,intensity,intensity])
					axon.addPoint(xyz,color=colormap[edge.maxDepth%11])
					axon.addType(point.type)
				else:
					axon.addPoint(xyz)
					axon.addType(point.type)
				axon.addNormal(normal)
				if edgestart>0:
					axon.addIndex(start-1)
					axon.addIndex(start)
				start=start+1
				edgestart=edgestart+1
				index+=1


			# if len(point.children)>0:
			# 	for child in point.children:
			# 		xyz = child.xyz.copy()
			# 		if mirror:
			# 			xyz[2]=11400-xyz[2]
			# 		if depthIntensity:
			# 			intensity = float(edge.maxDepth)/neuron.rootAxonEdge.maxDepth
			# 			# axon.addPoint(xyz,color=[intensity,intensity,intensity])
			# 			axon.addPoint(xyz,color=colormap[edge.maxDepth%11])
			# 		else:
			# 			axon.addPoint(xyz)
			# 		axon.addIndex(start-1)
			# 		axon.addIndex(start)
			# 		start=start+1

if __name__=="__main__":
	geo = GeometryAdapter()
	geo.readFile(CURRENT_FILE_PATH+"/../resource/033.swc")
	geo.readByID('192106','012.swc')

	pass