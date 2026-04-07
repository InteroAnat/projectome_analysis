
import matplotlib.pyplot as plt
from skimage import draw, io
import matplotlib
import BoundLaplace
import copy
import Flatmap
import BoundLaplace
import nrrd
import numpy as np
import IONData
# matplotlib.use('module://matplotlib_inline.backend_inline')
# %matplotlib inline
layer = [120, 400, 570, 850, 1200] # in micron, SS results employing ../neuronVis/LayerThickness.py
layerName = ['L1','L2/3','L4','L5','L6']
def getStreamLine():



    iondata = IONData.IONData()

    flatenPara=Flatmap.createSurfaceGraph()

    res,gridpath = iondata.getFileFromServer("boundlaplace20.nrrd")
    grid,header = nrrd.read(gridpath)

    resRelaxation,RelaxationPath=iondata.getFileFromServer('boundlaplaceout20.nrrd')
    relaxation,relaxationheader = nrrd.read(RelaxationPath)

    resdv0,dv0Path=iondata.getFileFromServer('dv0.nrrd')
    dv0,dv0header = nrrd.read(dv0Path)
    resdv1,dv1Path=iondata.getFileFromServer('dv1.nrrd')
    dv1,dv1header = nrrd.read(dv1Path)
    resdv2,dv2Path=iondata.getFileFromServer('dv2.nrrd')
    dv2,dv2header = nrrd.read(dv2Path)

    dv0=dv0.astype(np.float32)/1000-1
    dv1=dv1.astype(np.float32)/1000-1
    dv2=dv2.astype(np.float32)/1000-1

    res,depth20path = iondata.getFileFromServer("depth20.nrrd")
    res,outP1path = iondata.getFileFromServer("outP1.nrrd")
    res,outP2path = iondata.getFileFromServer("outP2.nrrd")
    res,outP3path = iondata.getFileFromServer("outP3.nrrd")
    res,inP1path = iondata.getFileFromServer("inP1.nrrd")
    res,inP2path = iondata.getFileFromServer("inP2.nrrd")
    res,inP3path = iondata.getFileFromServer("inP3.nrrd")
    depth20,header = nrrd.read(depth20path)
    outp1,header = nrrd.read(outP1path)
    outp2,header = nrrd.read(outP2path)
    outp3,header = nrrd.read(outP3path)
    inp1,header = nrrd.read(inP1path)
    inp2,header = nrrd.read(inP2path)
    inp3,header = nrrd.read(inP3path)
    return grid,dv0,dv1,dv2,flatenPara,depth20/500.0,outp1/100.0,outp2/100.0,outp3/100.0,inp1/100.0,inp2/100.0,inp3/100.0

def flatneuron(neurontree,grid,dv0,dv1,dv2,flatenPara,img=None,sum=False,layerlabel=False,depth0=None,outp1=None,outp2=None,outp3=None,inp1=None,inp2=None,inp3=None):

    index=0
    newedgesh=[]
    newedgesv=[]
    minx =99999
    maxx =0
    print(0)
    for p in neurontree.points:
        if p.z<5700:
            p.xyz[2]=11400-p.xyz[2]
    print(1)
    
    for edge in neurontree.edges:
        # print(index)
        index+=1
        points=[]
        for point in edge.data[0:-1:20]:
            points.append([point.xyz[0]/20,point.xyz[1]/20,point.xyz[2]/20])
        point=edge.data[-1]
        if point.xyz[0]/20<grid.shape[0] and  point.xyz[1]/20<grid.shape[1] and  point.xyz[2]/20<grid.shape[2]:
            points.append([point.xyz[0]/20,point.xyz[1]/20,point.xyz[2]/20])
        out=BoundLaplace.ComputeStreamlines(grid,dv0,dv1,
                                        dv2,copy.deepcopy(points),depth=depth0,outp1=outp1,outp2=outp2,outp3=outp3,inp1=inp1,inp2=inp2,inp3=inp3)
        
        newedgeh = []
        newedgev = []
        
        ii = 0
        for p in out:
            depth = p[0]
            if depth != 0:
                if minx>points[ii][0]:
                    minx = points[ii][0]
                if maxx<points[ii][0]:
                    maxx = points[ii][0]
                newedgeh.append([points[ii][0],depth,points[ii][2]])
                newedgev.append(points[ii])
            ii += 1
        if len(newedgeh)>0:
            newedgesh.append(newedgeh)
            newedgesv.append(newedgev)
    print(2)

    if maxx-minx>0:
        plt.subplots(figsize=((maxx-minx)/20,7))
        ax = plt.subplot(111)  
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.invert_yaxis()
        plt.xlim(minx,maxx)
        for edge in newedgesh:
            
            if len(edge):
                ax.plot(np.array(edge)[:,0],np.array(edge)[:,1],ls='-',linewidth=0.5,c='k',zorder=0) 
        if layerlabel:
            for i in range(len(layer)):
                plt.axhline(y=layer[i]/20,ls="-",c="black", linewidth=0.5)
                plt.text(minx-18,layer[i]/20,layerName[i],fontsize=20)
        plt.yticks([])
        plt.xticks([])
    if img is None:
        img = io.imread(r'../resource/flatmapedge.png')
    # fig,ax = plt.subplots(figsize=(16,17))
    plt.close()
    print(3)
    outpx,outpxHeader = nrrd.read("../resource/outpx.nrrd",index_order='F')
    outpy,outpyHeader = nrrd.read("../resource/outpy.nrrd",index_order='F')
    for edge in newedgesv:
        prePoint = None
        for p in edge:
            p2d = Flatmap.map2FlatmapIndex(flatenPara,np.array(p)*2,outpx,outpy)
            # p2d = Flatmap.map2Flatmap(flatenPara,np.array(p)*2,True)
            if prePoint is not None and len(p2d)==2 and p2d[0]>0 and p2d[1]>0 and prePoint[0]>0 and prePoint[1]>0:
                lenp = abs(float(prePoint[0])-float(p2d[0]))+abs(float(prePoint[1])-float(p2d[1]))
                if lenp>20:
                    print(p2d,prePoint,lenp)
                rr,cc = draw.line(int(prePoint[1]),int(prePoint[0]),int(p2d[1]),int(p2d[0]))
                if sum:
                    img[rr,cc] = img[rr,cc]+1
                else:
                    img[rr,cc] = [0,0,255,255]
                pass
            if len(p2d)==2:
                prePoint =p2d
    print(4)

    # io.imshow(img)
    # for edge in neurontree.edges:
    #     points=[]
    #     for point in edge.data:
    #         points.append([point.xyz[0]/20,point.xyz[1]/20,point.xyz[2]/20])
    #     ax.plot(np.array(points)[:,0],np.array(points)[:,1],ls='-',c='k',zorder=0)   
    # plt.show()


if __name__ == '__main__':
    import IONData 

    grid,dv0,dv1,dv2,flatenPara = getStreamLine()

    iondata =IONData.IONData()

    fig,ax = plt.subplots(figsize=(11,7))
    
    neurontree = iondata.getNeuronTreeByID('210098','004.swc')


    flatneuron(neurontree,grid,dv0,dv1,dv2,flatenPara)
    plt.savefig('../resource/thsample/flatmap/test.png', format='png', dpi=300)