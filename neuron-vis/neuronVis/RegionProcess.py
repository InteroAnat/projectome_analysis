import BrainRegion
import numpy as np
from scipy.ndimage import binary_dilation
def resampleAnnotation(brainRegion,downsample=10):
	mask = brainRegion.readdata[::downsample,::downsample,::downsample] # True or False
	xs,ys,zs = np.where(mask)
	bbox=[[0,0],[0,0],[0,0]]
	try:
		bbox[0][0] = np.min(xs)-3
	except:
		bbox[0][0]=0
	try:
		bbox[1][0] = np.min(ys)-3
	except:
		bbox[1][0]=0
	try:
		bbox[2][0] = np.min(zs)-3
	except:
		bbox[2][0]=0
	bbox[0][1] = (np.max(xs)+3) if (np.max(xs)+3)<mask.shape[0] else mask.shape[0]
	bbox[1][1] = (np.max(ys)+3) if (np.max(ys)+3)<mask.shape[1] else mask.shape[1]
	bbox[2][1] = (np.max(zs)+3) if (np.max(zs)+3)<mask.shape[2] else mask.shape[2]

	return mask,bbox,downsample # only left brain 

def cropAnnotation(mask,bbox):
	crop_mask=mask[	bbox[0][0]:bbox[0][1],
					bbox[1][0]:bbox[1][1],
					bbox[2][0]:bbox[2][1]]
	return crop_mask,bbox

def dialation(cropMask,shape,bbox,structure=None,iterations=1):
	
	diatedCropMask=binary_dilation(cropMask,structure,iterations=iterations)
	diatedMask = np.zeros(shape)

	diatedMask[	bbox[0][0]:bbox[0][1],
				bbox[1][0]:bbox[1][1],
				bbox[2][0]:bbox[2][1]]=diatedCropMask
	return diatedMask

def mergeRegion(br1,br2):
	br1.annotation()
	br2.annotation()
	br=BrainRegion.BrainRegion()
	br.name=br1.name+br2.name
	br.readdata=(br1.readdata==1) | (br2.readdata==1)
	return br

def getIntersectionFacePoints(br1,br2):
	br1.annotation()
	br2.annotation()
	mask1,bbox1,sample=resampleAnnotation(br1)
	crop_mask1,bbox1=cropAnnotation(mask1,bbox1)
	diatedMask1=dialation(crop_mask1,br1.readdata.shape,bbox1)

	mask2,bbox2,sample=resampleAnnotation(br2)
	crop_mask2,bbox2=cropAnnotation(mask2,bbox2)
	diatedMask2=dialation(crop_mask2,br2.readdata.shape,bbox2)
	
	intersection=(diatedMask1==1) & (diatedMask2==1)
	xs,ys,zs = np.where(intersection)
	points = []
	for x,y,z in zip(xs,ys,zs): 
		points.append([x*100,y*100,z*100])
	return points


if __name__=="__main__":
	import Render,GeometryAdapter
	from vispy import app
	br = BrainRegion.BrainRegion()
	br.praseJson()
	SSbr= br.getRegion('SS')
	SSbr.annotation()
	MObr= br.getRegion('MO')
	MObr.annotation()
	point=getIntersectionFacePoints(SSbr,MObr)
	print(point)

	win = Render.Render()
	ga = GeometryAdapter.GeometryAdapter()
	index=0
	for p in point:
		ga.geometry.addPoint(p)
		ga.geometry.addIndex(index)
		index=index+1
	ga.geometry.drawModel='points'
	win.addGeometry(ga.geometry)
	# win.show()
	# win
	app.run()
	pass