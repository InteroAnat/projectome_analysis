from cmath import pi
import SwcLoader
import numpy as np
import math
import BrainRegion
from sklearn.neighbors import KernelDensity
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt

def GetFirstOrderEdges(edgesForCluster,parentedge):
    for child in parentedge.children:
        if child.order!=0:
            edge = SwcLoader.Edge()
            edgesForCluster.append(edge)
            for p in child.data:
                edge.addPoint(SwcLoader.Point([p.index,p.type,p.x,p.y,p.z,p.ratio,p.parentIndex]))
            child.maxDepth=1
            GetFirstOrderEdges(edgesForCluster,child)
			
def OrderChildren(neuronTree,edge,parentOrder,currentOrder=-1):

	order = parentOrder
	if currentOrder>0:
		edge.order=currentOrder
	else:
		if edge.angle>45:
			order+=1
		else:
			order=parentOrder
		edge.order=order
	
	tranverseEdge=[]
	maxlength=0
	children = edge.children
	for child in children:
		if maxlength<len(child.data):
			maxlength = len(child.data)
	for edge0 in children:
		if len(edge0.data)>maxlength/10:
			tranverseEdge.append(edge0)
		else:
			OrderChildren(neuronTree,edge0,order,order+1)
	if len(tranverseEdge)==1:
		OrderChildren(neuronTree,tranverseEdge[0],order,order)
	else:
		for edge0 in tranverseEdge:
			OrderChildren(neuronTree,edge0,order)


	# print(head,"not in edges")

# by branch angle
def CalculateBranchOrder(neuronTree):
	for edge in neuronTree.edges:
		if edge.data[0].type==1:
			edge.order=0
			continue
		#current angle
		currentAngle=0
		vector = None
		if len(edge.data)>1:
			vector = np.array(edge.data[1].xyz)-np.array(edge.data[0].xyz)
			if len(edge.data)>2:
				vector = np.array(edge.data[int(len(edge.data)/2)].xyz) - np.array(edge.data[0].xyz)
		elif len(edge.data[0].children)>0:
			vector = np.array(edge.data[0].children[0].xyz) - np.array(edge.data[0].xyz)

		parentVector=None
		if edge.data[0].parent is not None and edge.data[0].parent.parent is not None:
			parentVector = np.array(edge.data[0].parent.xyz) - np.array(edge.data[0].parent.parent.xyz)

		if vector is not None and parentVector is not None:
			angle= math.acos(np.dot(vector/np.linalg.norm(vector),parentVector/np.linalg.norm(parentVector)))
			angle *= 180/pi
			edge.angle=angle
	
	# neuronTree.root
	for edge in neuronTree.edges:
		if  neuronTree.root ==edge.data[0]:
			OrderChildren(neuronTree,edge,0)
	for i in range(len(neuronTree.edges)):
		if neuronTree.maxOrder<neuronTree.edges[i].order:
			neuronTree.maxOrder=neuronTree.edges[i].order
		if neuronTree.edges[i].order==-1:
			print(neuronTree.edges[i])
	# print("done")

def DepthChildren(edge):
	maxDepth=-1
	maxLength=len(edge.data)
	if len(edge.children)==0:
		maxDepth=0

	else:
		for child in edge.children:
			depth,length = DepthChildren(child)
			length+=len(edge.data)
			if child.order>edge.order:
				depth+=1
			if maxDepth<depth:
				maxDepth=depth
			if maxLength<length:
				maxLength =length
	edge.maxDepth=maxDepth
	edge.maxLength=maxLength
	return maxDepth,maxLength

def CalculateBranchMaxDepth(neuronTree):
	CalculateBranchOrder(neuronTree)
	branchcount=1
	if neuronTree.rootAxonEdge:
		depth,length = DepthChildren(neuronTree.rootAxonEdge)
		branchcount+=depth
		if neuronTree.rootAxonEdge.maxDepth==0:
			neuronTree.rootAxonEdge.maxDepth=1
	pass

def UpSampleFromNeuronTree(neuronTree,space=1.0,outputname=None):
	neuron=SwcLoader.NeuronTree()
	maxindex=0
	for p in neuronTree.points:
		if maxindex<p.index:
			maxindex=p.index
	# print('maxindex:',maxindex)
	sum=0
	for edge in neuronTree.edges:
		newEdge=None
		start =np.array(edge.data[0].xyz)
		count=0
		# edgelength=0
		for point in edge.data:
			end=np.array(point.xyz)
			if start.tolist()!=end.tolist():
				sl=end-start
				pd = np.sqrt(np.sum(np.square(sl)))
				# edgelength=edgelength+pd
				sp = np.arange(0.001,pd,space)
				sp=np.append(sp,values=np.array(pd))/pd
				updata = np.transpose([sp])*sl
				updata=(updata+np.transpose(np.repeat(np.transpose([start]),len(sp),1)))
				# print(updata)
				if newEdge is None:
					newEdge=updata[:len(updata)]
				else:
					newEdge=np.concatenate((newEdge,updata[:len(updata)-2]))
				# print('newEdge',newEdge)
				start=updata[len(updata)-2]
				# break
		if newEdge is not None:
			start = SwcLoader.Point([edge.data[0].index,edge.data[0].type,newEdge[0][0],newEdge[0][1],newEdge[0][2],edge.data[0].ratio,edge.data[0].parentIndex])
			if start.parentIndex==-1:
				neuron.root=start
				
			edgep=SwcLoader.Edge(start)
			for p in newEdge[1:len(newEdge)-1]:
				maxindex=maxindex+1
				point=SwcLoader.Point([maxindex,edge.data[0].type,p[0],p[1],p[2],edge.data[0].ratio,edgep.data[len(edgep.data)-1].index])
				# print(point)
				edgep.data[len(edgep.data)-1].children.append(point)
				edgep.addPoint(point)
			tile = SwcLoader.Point([edge.data[len(edge.data)-1].index,edge.data[len(edge.data)-1].type,newEdge[len(newEdge)-1][0],newEdge[len(newEdge)-1][1],newEdge[len(newEdge)-1][2],edge.data[len(edge.data)-1].ratio,edge.data[len(edge.data)-1].parentIndex])
			edgep.data[len(edgep.data)-1].children.append(tile)
			tile.parentIndex=edgep.data[len(edgep.data)-1].index
			tile.parent=edgep.data[len(edgep.data)-1]
			
			neuron.edges.append(edgep)
	
	for i in range(len(neuronTree.edges)):
		startold = neuronTree.edges[i].data[0]
		startnew = neuron.edges[i].data[0]
		for j in range(len(neuronTree.edges)):
			if startold.parent==neuronTree.edges[j].data[len(neuronTree.edges[j].data)-2]:
				startnew.parent = neuron.edges[j].data[len(neuron.edges[j].data)-2]
				startnew.parentIndex=neuron.edges[j].data[len(neuron.edges[j].data)-2].index
		pass
	points =[]
	phead=[]
	ptile=[]
	for edge in neuron.edges:
		if edge.data[0].index not in phead:
			phead.append(edge.data[0].index)
			points.append(edge.data[0])
		for i in range(1,len(edge.data)-1):
			points.append(edge.data[i])
		if edge.data[len(edge.data)-1].index not in ptile:
			ptile.append(edge.data[len(edge.data)-1].index)
			points.append(edge.data[len(edge.data)-1])
	neuron.points=points
	if outputname is not None:
		output = outputname
		o = open(output,"w")
		for file in neuron.points:
			o.write(str(file.index)+' '+str(file.type)+' '+str(file.x)+' '+str(file.y)+' '+str(file.z)+' '+str(file.ratio)+' '+str(file.parentIndex)+'\n')
		o.close()
	return neuron

def getCentroid(neuronTree,brainRegion=None):
	newneuron=UpSampleFromNeuronTree(neuronTree)
	if brainRegion is not None:
		points = brainRegion.pointsIn(newneuron.points)

	centroid=np.array([0,0,0])
	for p in points:
		centroid[0]=centroid[0]+p.x
		centroid[1]=centroid[1]+p.y
		centroid[2]=centroid[2]+p.z
	centroid=centroid/len(points)
	return centroid
	
def densityEstimation3D(neuronTree,offset=[2000,2000,3000],step=75,size=(60,60,60)):
	points = np.array(neuronTree.xyz)
	print(points)
	kde = KernelDensity(bandwidth=50)
	kde.fit(points)
	p = np.zeros(size)
	tp = np.where(p==0)
	tp = np.hstack([tp[0].reshape(-1,1), 
					tp[1].reshape(-1,1),
					tp[2].reshape(-1,1)
				   ])*step+np.array(offset).reshape(1,-1)
	print(tp)
	p = np.exp(kde.score_samples(tp)).reshape(size[0],size[1],size[2], order="C")
	p = p / np.sum(p)
	
	if False:
		fig, ax = plt.subplots(1,1, figsize=(7,7))
		ax.imshow(np.transpose(np.max(p, axis=2)), origin='lower', cmap='Reds', alpha=0.75, aspect='equal')
		plt.show()
	return p

def getOverlap(neuronDensity1,neuronDensity2):
	res = np.zeros(neuronDensity1.shape)
	# Calculate the frequency under curve
	lab = np.where(neuronDensity1>=neuronDensity2)
	res[lab] = neuronDensity2[lab]
	lab = np.where(neuronDensity1<neuronDensity2)
	res[lab] = neuronDensity1[lab]
	return np.sum(res)

if __name__ == "__main__":
	import IONData
	import SwcLoader
	import Render
	import GeometryAdapter
	import BrainRegion
	from vispy import app
	
	# br = BrainRegion.BrainRegion()
	# br.praseJson()
	# SSbr= br.getRegion('SS')
	
	iondata=IONData.IONData()
	
	neuron = iondata.getNeuronTreeByID('200313', '083.swc')
	CalculateBranchMaxDepth(neuron)
	neuron349 = iondata.getNeuronTreeByID('sample', 'AA0349.swc')

	p = np.array(neuron.xyz)
	rangemin=np.array([np.min(p[:,0]),np.min(p[:,1]),np.min(p[:,2])])-50
	rangemax=np.array([np.max(p[:,0]),np.max(p[:,1]),np.max(p[:,2])])+50
	width=rangemax-rangemin
	step=np.max(width)/30
	size=(int(width[0]/step),int(width[1]/step),int(width[2]/step))

	# density = densityEstimation3D(neuron,rangemin.tolist(),step,size)
	# density349 = densityEstimation3D(neuron349,rangemin.tolist(),step,size)
	# overlap = getOverlap(density,density349)
	newneuron=UpSampleFromNeuronTree(neuron,2,'../resource/test2.swc')
	print("")
	# centroid = getCentroid(neuron,SSbr)
