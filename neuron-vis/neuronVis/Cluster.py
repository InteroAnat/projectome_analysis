from scipy.cluster.hierarchy import linkage ,fcluster,fclusterdata,dendrogram
from scipy import spatial

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from IONData  import IONData

class Cluster:
	def __init__(self):
		pass
	def fclusterNeuronList(self,neuronlist):
		iondata =IONData()
		prodf = iondata.getPropertiesDF(neuronlist,True)
		self.fclusterDF(prodf)

	def fclusterDF(self,dataframe):
		matData =  dataframe.values[:,1:].transpose()
		distance = spatial.distance.pdist(matData.astype(float))
		self.linkresult = linkage(distance,method='average',metric='euclidean')
		print(self.linkresult)
		self.cluster=fcluster(self.linkresult,t=0.99,criterion='inconsistent',depth=3,R=None,monocrit=None)#这个需要先计算linkage，再出结果
		dendrogram(self.linkresult,labels=dataframe.columns[1:])


if __name__=="__main__":
	iondata =IONData()
	neuronlist=iondata.getNeuronListBySampleID('210013')
	cluster= Cluster()
	cluster.fclusterNeuronList(neuronlist)
	plt.xticks(fontsize=12,rotation = 30, ha = 'right')
	plt.show()