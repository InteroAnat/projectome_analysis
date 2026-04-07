import sys,os
from skimage import measure,io
import IONData
import nrrd
import trimesh
import networkx as nx
import numpy as np
import joblib

anchor_center = [615,409, 364]
def createSurfaceGraph():
    flatenPara=[]
    if not os.path.exists('../resource/rtflatenPara.pkl'):
        print('step1: Get the mask, boundary, annotation file from server')

        iondata = IONData.IONData()
        resISOCLA,isoclaPath=iondata.getFileFromServer('rtmask.nrrd')
        resBoundary,boundaryPath=iondata.getFileFromServer('rtboundary10.nrrd')
        if resISOCLA and resBoundary:
            print(isoclaPath,boundaryPath,'downloaded!')


        mask_combined,CLAHeader = nrrd.read(isoclaPath,index_order='F')
        boundary,boundaryHeader = nrrd.read(boundaryPath,index_order='F')

        [ROW, COL, SLICE] = mask_combined.shape

        print('step2: Get the surface form mask')
        verts, faces, t1, t2 = measure.marching_cubes(mask_combined, 0,method='lorensen')

        mesh = trimesh.Trimesh(vertices=verts,
                            faces=faces,
                            )
        x,y,z=np.where(boundary==1)
        boundaryvert = np.hstack((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)))
        verts = mesh.vertices.astype(np.uint64)
        faces = mesh.faces

        print('step3: Get superficial part from the surface')
        vertsflat=verts[:,0]+verts[:,1]*ROW+verts[:,2]*ROW*COL
        boundaryvertflat=boundaryvert[:,0]+boundaryvert[:,1]*ROW+boundaryvert[:,2]*ROW*COL
        # vertsback = np.array([(vertsflat%ROW).astype(np.Uint64),((vertsflat/ROW).astype(np.Uint64)%COL).astype(np.Uint64),((vertsflat/COL).astype(np.Uint64)/ROW).astype(np.Uint64)])
        flag_vertex_superficial=np.in1d(vertsflat,boundaryvertflat)
        faceflag=flag_vertex_superficial[mesh.faces]
        flag_face_superficial=sum(faceflag.T)==3

        index=faces[flag_face_superficial==1,:]

        print('step4: Calculate the shortest path to anchor_center ')
        indexx=index[:,0];
        indexy=index[:,1];
        indexz=index[:,2];

        coordx=verts[indexx,:]
        coordy=verts[indexy,:]
        coordz=verts[indexz,:]

        squarexy = (coordx - coordy)**2;
        sqrtxy=np.sqrt(sum(squarexy.T))

        squareyz = (coordy - coordz)**2;
        sqrtyz=np.sqrt(sum(squareyz.T))

        squarexz = (coordx - coordz)**2;
        sqrtxz=np.sqrt(sum(squarexz.T))

        G=nx.Graph()
        x=np.hstack([indexx,indexy,indexz,indexy,indexz,indexx])
        y=np.hstack([indexy,indexz,indexx,indexx,indexy,indexz])
        dis=np.hstack([sqrtxy,sqrtyz,sqrtxz,sqrtxy,sqrtyz,sqrtxz])
        list=np.vstack([x,y,dis]).T
        G.add_weighted_edges_from(list)
        # anchor point [649,57,349]
        anchor_center_index = np.where(np.all(verts==anchor_center,axis=1))[0]
        length=nx.single_source_shortest_path_length(G,anchor_center_index[0])

        flatenPara=[verts,flag_vertex_superficial,length]

        joblib.dump(flatenPara, '../resource/rtflatenPara.pkl')
    else:
        flatenPara = joblib.load('../resource/rtflatenPara.pkl')
    return flatenPara

def map2Flatmap(flatenPara,point,findNearest=False):
    if point[0]==0 and point[1]==0 and point[2]==0:
        return [0,0]
    verts = flatenPara[0]
    flag_vertex_superficial = flatenPara[1]
    length = flatenPara[2]

    
    iVertex =-1
    index = np.where((verts == point).all(axis=1))[0]
    
    if index.size<1:
        # print("not in boundary")
        if not findNearest:
            return
        else:
            dist = np.sum((verts-point)**2,axis=1)
            iVertex = np.argmin(dist)
            pass
    else:
        iVertex =index[0]
    if flag_vertex_superficial[iVertex]==0:
        print("not in up boundary")
    tmpRow=verts[iVertex,0]
    tmpCol=verts[iVertex,1]
    tmpSlice=verts[iVertex,2]
    d = np.sqrt((anchor_center[0] - tmpRow)**2 + (anchor_center[1] - tmpCol)**2);
    p2d=[]
    if d == 0:
        p2d= [400, 400]
        return p2d
    # print(iVertex)
    if iVertex not in length or length[iVertex] > 1000:
        return p2d
    tmpCoordinate = [length[iVertex]/d*(tmpCol-anchor_center[1]), length[iVertex]/d*(tmpRow -anchor_center[0])]
    p2d = [tmpCoordinate[0] + 400, tmpCoordinate[1] + 400]
    return p2d
 
def createFlatmap(flatenPara):
    iondata = IONData.IONData()
    res,annotationPath= iondata.getAnnotation()
    annotation,annoHeader = nrrd.read(annotationPath,index_order='F')

    verts = flatenPara[0]
    flag_vertex_superficial = flatenPara[1]
    length = flatenPara[2]

    print('step5: Start flaten with anchor_center')
    
    coordinate_flatmap = np.zeros([len(verts),2])
    image_flatmap = np.zeros([800, 800])
    for iVertex in range(len(verts)):
        if flag_vertex_superficial[iVertex]==0:
            continue
        tmpRow=verts[iVertex,0]
        tmpCol=verts[iVertex,1]
        tmpSlice=verts[iVertex,2]
        d = np.sqrt((anchor_center[0] - tmpRow)**2 + (anchor_center[1] - tmpCol)**2);
        if d == 0:
            coordinate_flatmap[iVertex, :] = [0, 0];
            continue;
        # print(iVertex)
        if iVertex not in length or length[iVertex] > 1000:
            continue;
        tmpCoordinate = [length[iVertex]/d*(tmpCol-anchor_center[1]), length[iVertex]/d*(tmpRow -anchor_center[0])]

        coordinate_flatmap[iVertex, :] = [tmpCoordinate[0] + 400, tmpCoordinate[1] + 400]
        # print(tmpRow,tmpCol,tmpSlice)
        # print(coordinate_flatmap[iVertex, 1], coordinate_flatmap[iVertex, 0])
        image_flatmap[int(coordinate_flatmap[iVertex, 1]), int(coordinate_flatmap[iVertex, 0])] =annotation[tmpRow, tmpCol, tmpSlice]
    nrrd.write('../resource/rtflatmap.nrrd',image_flatmap.T)
    print('write the flatmap to ../resource/rtflatmap.nrrd')

if __name__=="__main__":
    import numpy as np
    import BoundLaplace
    import json

# In[0] load points
    point=[]
    pointjson=[]
    with open("../resource/newDataall.json") as jsonfile:
        pointjson=json.load(jsonfile)
        # print(json[0])
        index=0
        for p in pointjson:
            z=float(p['L'])*50+5700/20
            if z>285:
                z=570-z
            point.append([-50*float(p['B'])+5472/20,50*float(p['D']),z])
            # ga.geometry.addPoint([-1000*float(p['B'])+5472,1000*float(p['D']),float(p['L'])*1000+5700],[random.random()+0.5,random.random()+0.5,random.random()+0.5])

# In[1] create surface and flatmap

    flatenPara=createSurfaceGraph()

    createFlatmap(flatenPara)

#In[2] test map2Flatmap
    # p2d = map2Flatmap(flatenPara,np.array([1024,228,3000]))
    # print(p2d)

#In[3] get data for streamline
    iondata = IONData.IONData()

    res,gridpath = iondata.getFileFromServer("rtboundaryLaplace-10.nrrd")
    grid,header = nrrd.read(gridpath)



    resdv0,dv0Path=iondata.getFileFromServer('rtdv0.nrrd')
    dv0,dv0header = nrrd.read(dv0Path)
    resdv1,dv1Path=iondata.getFileFromServer('rtdv1.nrrd')
    dv1,dv1header = nrrd.read(dv1Path)
    resdv2,dv2Path=iondata.getFileFromServer('rtdv2.nrrd')
    dv2,dv2header = nrrd.read(dv2Path)

    dv0=dv0.astype(np.float32)/1000-1
    dv1=dv1.astype(np.float32)/1000-1
    dv2=dv2.astype(np.float32)/1000-1
    
    ######################## dv can also be computed like this
    # resRelaxation,RelaxationPath=iondata.getFileFromServer('boundlaplaceout20.nrrd')
    # relaxation,relaxationheader = nrrd.read(RelaxationPath)
    # relaxation = grid.copy().astype(np.float32)
    # dv0 = np.zeros_like(relaxation)
    # dv1 = np.zeros_like(relaxation)
    # dv2 = np.zeros_like(relaxation)
    # BoundLaplace.computeGradients(grid,relaxation,dv0,dv1,dv2)

#In[4] compute streamlines
    out=BoundLaplace.ComputeStreamlines(grid,dv0,dv1,dv2,point)
    index=0
    for p in out:
        p2d = map2Flatmap(flatenPara,np.array(p[1])*2,True)
        print("point:",index,p2d)
        pointjson[index]['flatmap']=p2d
        index+=1
    with open("../resource/injectout.json","w") as f:
        json.dump(pointjson,f)
    print("test")
    
