from vispy.geometry import MeshData

class Node:
	def __init__(self):
		self.children=[]

	def addChild(self,node):
		self.children.append(node)

class Geometry(Node):
	def __init__(self):
		super(Geometry,self).__init__()
		#lines for neurons
		#triangles for region and soma
		self.name=''
		self.type=-1
		self.drawModel='triangles'
		self.vertex=[]
		self.normal=[]
		self.types=[]
		self.index=[]
		self.bound=[]
		self.color=[]
		self.texcoord=[]
		self.uniformColor=None
		self.hide=False
		self.width=1.0
		self.textures=[]
		self.highLight=False

		pass

	def addPoint(self,point,color=None,texcoord=None):
		self.vertex.append(point)
		if color:
			self.color.append(color)
		if texcoord:
			self.texcoord.append(texcoord)
	def addType(self,type):
		self.types.append(type)
	def addIndex(self,index):
		self.index.append(index)
	def addNormal(self,normal):
		self.normal.append(normal)
	def setHide(self,_hide):
		self.hide=_hide
		for child in self.children:
			child.setHide(_hide)
	def addTexture(self,typename,image):
		self.textures.append({typename:image})
		pass
if __name__=="__main__":
	soma = Geometry()
	axon = Geometry()
	soma.addChild(axon)

	pass