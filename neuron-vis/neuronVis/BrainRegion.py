import json,sys,os,copy
import IONData
import nrrd
import numpy as np
if hasattr(sys.modules[__name__], '__file__'):
	_file_name = __file__
else:
	_file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)

class BrainRegion:
	mid_z = 570
	def __init__(self):
		self.json=None
		self.children=[]
		self.ID=''
		self.resulation=10
		self.parentID=''
		self.name=''
		self.property=0
		self.readdata=None
		self.jsonobj=None
		# self.praseJson(self.jsonobj)

	def praseJson(self,jsonobj=None):
		if jsonobj is None:
			with open(CURRENT_FILE_PATH+"/../resource/1.json") as jsonfile:
				self.json=json.load(jsonfile)['msg']
				self.ID=self.json[0]['id']
				self.parentID=self.json[0]['parent_structure_id']
				self.name=self.json[0]['acronym']
				self.childrenjson=self.json[0]['children']
				self.st_level=self.json[0]['st_level']
			self.parse(self.childrenjson)
			br =BrainRegion()
			br.name='unknow'
			br.ID=0
			self.children.append(br)
		else:
			self.ID=jsonobj['id']
			self.parentID=jsonobj['parent_structure_id']
			self.name=jsonobj['acronym']
			if jsonobj['acronym']=='fiber tracts':
				self.name = 'fibertracts'
			self.childrenjson=jsonobj['children']
			self.st_level=jsonobj['st_level']
			pass

			self.parse(self.childrenjson)
		# br =BrainRegion()
		# br.name='unknow'
		# br.ID=0
		# self.children.append(br)
	def annotation(self):
		iondata=IONData.IONData()
	
		if self.readdata  is None:
			res=[]
			# if self.ID==997 or self.name=='root':
			# 	res = iondata.getAnnotation()
			if self.resulation==10:
				res = iondata.getStructureMask(self.ID)
			else :
				res = iondata.getStructureMask25(self.ID)
			if res[0]:
				print("loading nrrd ... ")
				self.readdata,self.header = nrrd.read(res[1],index_order='F')
				# self.readdata=self.readdata[:,:,0:self.mid_z] #only left brain 
				print("loaded only left brain %s"%res[1])
	
	def pointsIn(self,points):
		pointsin=[]
		self.annotation()
		for point in points:
			z=int(point.z/10) if self.mid_z>int(point.z/10) else self.mid_z*2-1-int(point.z/10)
			if self.readdata[int(point.x/10),int(point.y/10),z]:
				pointsin.append(point)
		return pointsin



	def parse(self,json=None):
		for child in self.childrenjson:
			br =BrainRegion()
			br.praseJson(child)
			self.children.append(br)


	def getRegion(self,name=None,id=None):
		
		if name is not None:
			res=None
			if self.name==name:
				# self.print()
				res=self
			else:
				for child in self.children:
					res = child.getRegion(name)
					if res:
						break
			return res
		elif id is not None:
			res=None
			if self.ID==id:
				# self.print()
				res=self
			else:
				for child in self.children:
					res = child.getRegion(name=None,id=id)
					if res:
						break
			return res

	def getList(self,res):
		for child in self.children:
			res.append([child.name,child.ID])
			child.getList(res)

	def getRegionList(self,name=None,id=None):
		if name is not None:
			br = self.getRegion(name)
			res=[]
			res.append([br.name,br.ID])
			br.getList(res)
			return res
		elif id is not None:
			br = self.getRegion(name=None,id=id)
			res=[]
			res.append([br.name,br.ID])
			br.getList(res)

	def getPropertyList(self,res):
		for child in self.children:
			if child.property>0:
				res.append([child.name,child.ID,child.property])
			child.getPropertyList(res)

	def getRegionPropertyList(self,name='root'):
		br = self.getRegion(name)
		res=[]
		if br.property>0:
			res.append([br.name,br.ID,br.property])
		
		br.getPropertyList(res)
		if name=='root':
			br =self.getRegion('unknow')
			if [br.name,br.ID,br.property] not in res:
				res.append([br.name,br.ID,br.property])
		return res
	def getSumProperty(self,name='root'):
		plist=self.getRegionPropertyList(name)
		psum=0
		for subregion in plist:
			psum=psum+subregion[2]
		return psum

	def print(self):
		print(self)
		for child in self.children:
			child.print()

	def __str__(self):
		return str(self.ID)+" "+self.name+" "+str(self.property)
	def merge(self,BrainRegion):

		self.readdata[BrainRegion.readdata>0]=BrainRegion.readdata[BrainRegion.readdata>0]
		pass

class RegionProperty(BrainRegion):
	def __init__(self,region=None):
		super(RegionProperty,self).__init__()
		self.children=region.children
		self.ID=region.ID
		self.name=region.name


	def setProperty(self,properties):
		# self.print()
		for region,prop in properties.items():
			regionname=''
			regionid=0
			if isinstance(region,str):
				if self.getRegion(region) is None:
					pass#print(region)
				else:
					self.getRegion(region).property=prop
			if isinstance(region,int):
				if self.getRegion(id=region) is None:
					pass#print(region)
				else:
					self.getRegion(id=region).property=prop

if __name__=="__main__":
	dict_temp = {}

	# 打开文本文件
	file = open('../resource/region.txt','r')

	# 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
	file.readline()
	for line in file.readlines():
		line = line.strip()
		splitLine = line.split('\t')

		dict_temp[splitLine[2]] = splitLine[0]

	# 依旧是关闭文件
	file.close()

	#  可以打印出来瞅瞅
	print(dict_temp)

	br = BrainRegion()
	br.resulation=25
	br.praseJson()
	subbr= br.getRegion(id=997)
	subbr.annotation()
	subbr.readdata=subbr.readdata* int(dict_temp[subbr.name])
	subregion=['TH','Isocortex','STR','STRd','STRv','VPL','PO']

	for region in subregion:

		thbr= br.getRegion(name=region)
		thbr.resulation=25
		thbr.annotation()
		thbr.readdata=thbr.readdata*int(dict_temp[thbr.name])
		subbr.merge(thbr)

	
	


	nrrd.write('../resource/strcture25/merge_25.nrrd',subbr.readdata,subbr.header)

	brproperty=RegionProperty(copy.deepcopy(subbr))
	brproperty.print()
	subbr.print()
	# brproperty=subbr
	# print(brproperty)

	subbrlist=br.getRegionList('grv of CBX')
	# print(subbrlist)
	pass
