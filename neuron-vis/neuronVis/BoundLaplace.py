import nrrd
import IONData
import numpy as np



def LaplaceStep(grid,out):
    gridsize = grid.shape
    convergence = 0.0
    oldvalue=0
    tmpvalue=0
    newvalue=0
    thirdboundary = 19.9

    for v0 in range(gridsize[0]):
        for v1 in range(gridsize[1]):
            for v2 in range(gridsize[2]):
                gridvoxel=grid[v0,v1,v2]
                if gridvoxel > 0 and gridvoxel < 10:
                    tmpvalue=0.0
                    oldvalue = out[v0,v1,v2]
                    counter=0.0
                    if grid[v0+1,v1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += out[v0+1,v1,v2]
                    if grid[v0-1,v1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += out[v0-1,v1,v2]
                    if grid[v0,v1+1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += out[v0,v1+1,v2]
                    if grid[v0,v1-1,v2] < thirdboundary:
                        counter += 1
                        tmpvalue += out[v0,v1-1,v2]
                    if grid[v0,v1,v2+1] < thirdboundary :
                        counter += 1
                        tmpvalue += out[v0,v1,v2+1]
                    if grid[v0,v1,v2-1] < thirdboundary :
                        counter += 1
                        tmpvalue += out[v0,v1,v2-1]
                    tmpvalue = tmpvalue / counter
                    oldvalue = out[v0,v1,v2]
                    # over-relaxation:
                    newvalue = oldvalue + 1.666 * ( tmpvalue - oldvalue )
                    if newvalue < 0 or newvalue > 10:
                        out[v0,v1,v2] = tmpvalue
                    else:
                        out[v0,v1,v2] = newvalue
                    convergence += abs(oldvalue - out[v0,v1,v2])
    return convergence

def computeGradients(grid,relaxed,dv0,dv1,dv2):
    nv0=grid.shape[0]
    nv1=grid.shape[1]
    nv2=grid.shape[2]
    if nv2>1:
        for v0 in range(1,nv0-1):
            for v1 in range(1,nv1-1):
                for v2 in range(1,nv2-1):
                    if grid[v0,v1,v2] > 0 and grid[v0,v1,v2] < 10:
                        d0 = relaxed[v0+1,v1,v2] - relaxed[v0-1,v1,v2]
                        d1 = relaxed[v0,v1+1,v2] - relaxed[v0,v1-1,v2]
                        d2 = relaxed[v0,v1,v2+1] - relaxed[v0,v1,v2-1]
                        
                        mag = np.sqrt( (d0*d0) + (d1*d1) + (d2*d2) )
                        if mag == 0:
                            mag = mag + 1e-7
                        dv0[v0,v1,v2] = d0 / mag
                        dv1[v0,v1,v2] = d1 / mag
                        dv2[v0,v1,v2] = d2 / mag

def BoundLaplace(grid,out,iteration=100,minconvergence=0.00001):

    normalize_factor = 1.0 / LaplaceStep(grid,out)
    for i in range(iteration):
            # call a single relaxation step and multiple output by the
            # normalization factor for convergence checking
            convergence = LaplaceStep(grid, out) * normalize_factor
            print("iteration", i, ":", convergence)
            if convergence<minconvergence:
                break;

def FastTrilinearInterpolant(image,point):
    i =int(point[0])
    j =int(point[1])
    k =int(point[2])
    # fill the coefficients
    coefs=[0,0,0,0,0,0,0,0]
    coefs[0] = image[ i  , j  , k   ]
    coefs[1] = image[ i  , j  , k+1 ]
    coefs[2] = image[ i  , j+1, k   ]
    coefs[3] = image[ i  , j+1, k+1 ]
    coefs[4] = image[ i+1, j  , k   ]
    coefs[5] = image[ i+1, j  , k+1 ]
    coefs[6] = image[ i+1, j+1, k   ]
    coefs[7] = image[ i+1, j+1, k+1 ]

    u=point[0]-i
    v=point[1]-j
    w=point[2]-k
    
    # get the four differences in the u direction
    du00 = int(coefs[4]) - int(coefs[0])
    du01 = int(coefs[5]) - int(coefs[1])
    du10 = int(coefs[6]) - int(coefs[2])
    du11 = int(coefs[7]) - int(coefs[3])

    # reduce to a 2D problem by interpolating in the u direction
    c00 = coefs[0] + u * du00
    c01 = coefs[1] + u * du01
    c10 = coefs[2] + u * du10
    c11 = coefs[3] + u * du11

    # get the two differences in the v direction for the 2D problem
    dv0 = c10 - c00
    dv1 = c11 - c01

    # reduce 2D to a 1D problem by interpolating in the v direction
    c0 = c00 + v * dv0
    c1 = c01 + v * dv1

    # get the 1 difference in the w direction for the 1D problem
    dw = c1 - c0

    # interpolate in 1D to get the value
    value = c0 + w * dw
    return value

def CreateStreamline(grid,dv0,dv1,dv2,v0,v1,v2,h=0.1):
    h_negative = h * -1
    oldpoint = [v0,v1,v2]
    point=[0,0,0]
    stream_length=0
    counter=0
    grid_position = FastTrilinearInterpolant(grid, oldpoint)
    outerPoint=oldpoint
    while grid_position<9.99:
        outerPoint = oldpoint
        newv0 = FastTrilinearInterpolant(dv0, oldpoint)
        newv1 = FastTrilinearInterpolant(dv1, oldpoint)
        newv2 = FastTrilinearInterpolant(dv2, oldpoint)
        mag = newv0*newv0 + newv1*newv1 + newv2*newv2
        if mag < 1.0e-6:
            grid_position=10
        else:
            pass
            point[0] = oldpoint[0] + newv0 * h
            point[1] = oldpoint[1] + newv1 * h
            point[2] = oldpoint[2] + newv2 * h

            stream_length = stream_length + np.sqrt( (point[0] - oldpoint[0])*(point[0] - oldpoint[0]) + (point[1] - oldpoint[1])*(point[1] - oldpoint[1]) + (point[2]-oldpoint[2])*(point[2]-oldpoint[2]) )

            grid_position = FastTrilinearInterpolant(grid, point)
            oldpoint[0] = point[0]
            oldpoint[1] = point[1]
            oldpoint[2] = point[2]
            counter=counter+1
            real_line_distance = np.sqrt ( (newv0-v0)*(newv0-v0) + 
                                        (newv1-v1)*(newv1-v1) + 
                                        (newv2-v2)*(newv2-v2) )
            if stream_length > (4.0*real_line_distance):
                grid_position = 10.0

    oldpoint = [v0,v1,v2]
    real_line_distance = 0
    counter=0
    stream_lengthtwo=0
    innerPoint =oldpoint
    while grid_position > 0.01:
        innerPoint =oldpoint
        newv0 = FastTrilinearInterpolant(dv0, oldpoint)
        newv1 = FastTrilinearInterpolant(dv1, oldpoint)
        newv2 = FastTrilinearInterpolant(dv2, oldpoint)
        mag = newv0*newv0 + newv1*newv1 + newv2*newv2
        if mag < 1.0e-6:
            grid_position=0.0
        else:
            point[0] = oldpoint[0] + newv0 * h_negative
            point[1] = oldpoint[1] + newv1 * h_negative
            point[2] = oldpoint[2] + newv2 * h_negative

            stream_lengthtwo = stream_lengthtwo + np.sqrt( (point[0] - oldpoint[0])*(point[0] - oldpoint[0]) + (point[1] - oldpoint[1])*(point[1] - oldpoint[1]) + (point[2]-oldpoint[2])*(point[2]-oldpoint[2]) )
            grid_position = FastTrilinearInterpolant(grid, point)
            #print "INSIDE:", point[0], point[1], point[2], grid_position, counter
            counter = counter + 1
            oldpoint[0] = point[0]
            oldpoint[1] = point[1]
            oldpoint[2] = point[2]


            real_line_distance = np.sqrt ( (newv0-v0)*(newv0-v0) + 
                                        (newv1-v1)*(newv1-v1) + 
                                        (newv2-v2)*(newv2-v2) )
            if stream_lengthtwo > (4.0*real_line_distance):
                grid_position = 0.0
    return stream_length,outerPoint,innerPoint

# h=0.1
def ComputeStreamlineFromOutPandInP(point,depth20=None,outp1=None,outp2=None,outp3=None,inp1=None,inp2=None,inp3=None): 

    depth = FastTrilinearInterpolant(depth20, point)
    outx = FastTrilinearInterpolant(outp1, point)
    outy = FastTrilinearInterpolant(outp2, point)
    outz = FastTrilinearInterpolant(outp3, [point[0],point[1],570-point[2]])
    inx = FastTrilinearInterpolant(inp1, point)
    iny = FastTrilinearInterpolant(inp2, point)
    inz = FastTrilinearInterpolant(inp3, point)

    # depth = depth20[ int(point[0]),int(point[1]),int(point[2])]
    # outx = outp1[ int(point[0]),int(point[1]),int(point[2])]
    # outy = outp2[ int(point[0]),int(point[1]),int(point[2])]
    # outz = outp3[ int(point[0]),int(point[1]),int(570-point[2])]
    # inx = inp1[ int(point[0]),int(point[1]),int(point[2])]
    # iny = inp2[ int(point[0]),int(point[1]),int(point[2])]
    # inz = inp3[ int(point[0]),int(point[1]),int(570-point[2])]
    return depth,[outx,outy,outz],[inx,iny,inz]

def ComputeStreamlines(grid,dv0,dv1,dv2,points,h=0.1,depth=None,outp1=None,outp2=None,outp3=None,inp1=None,inp2=None,inp3=None):
    out=[]
    for point in points:
        grid_position = FastTrilinearInterpolant(grid, point) 
        if grid_position!=5:
            # print("not in the boundary range")
            # print(point)
            out.append([0,[0,0,0],[0,0,0]])
            continue
        if depth is not None and outp1 is not None and outp2 is not None and outp3 is not None  and inp1 is not None and inp2 is not None and inp3 is not None:
            length,outerpoint,innerpoint = ComputeStreamlineFromOutPandInP(point,depth20=depth,outp1=outp1,outp2=outp2,outp3=outp3,inp1=inp1,inp2=inp2,inp3=inp3)
        else:
            length,outerpoint,innerpoint = CreateStreamline(grid, dv0, dv1, dv2,point[0],point[1],point[2], h)
        out.append([length,outerpoint,innerpoint ])
    return out

def ComputeAllStreamlines(grid,out,op,ip,dv0,dv1,dv2,h=0.1):
    nv0=grid.shape[0]
    nv1=grid.shape[1]
    nv2=grid.shape[2]
    for v0 in range(nv0):
        print('slice: ',v0)
        for v1 in range(nv1):
            for v2 in range(nv2):
                if grid[v0,v1,v2] > 0 and grid[v0,v1,v2] < 10:
                    d,o,i= CreateStreamline(grid, dv0, dv1, dv2,
                                                v0,v1,v2, h)
                    # print(d,o,i)
                    try:
                        out[v0,v1,v2]=d
                        op[v0,v1,v2,:]=o
                        ip[v0,v1,v2,:] =i
                    except ValueError:
                        print(d,o,i)
                    # print(out[v0,v1,v2])
                else:
                    out[v0,v1,v2] = 0






if __name__=="__main__":
    from PIL import Image
    import numpy as np
    iondata = IONData.IONData()
    res,gridpath = iondata.getFileFromServer("boundlaplace100.nrrd")
    grid,header = nrrd.read(gridpath)
    relaxation = grid.copy().astype(np.float32)
    BoundLaplace(grid,relaxation,10)

    dv0 = np.zeros_like(relaxation)
    dv1 = np.zeros_like(relaxation)
    dv2 = np.zeros_like(relaxation)
    computeGradients(grid,relaxation,dv0,dv1,dv2)

    out = np.zeros_like(grid).astype(np.float32)
    op = np.zeros_like(grid).astype(np.float32)
    ip = np.zeros_like(grid).astype(np.float32)
    ComputeAllStreamlines(grid,out,op,ip,dv0,dv1,dv2)
    nrrd.write('../resource/depthMap.nrrd',out)
    nrrd.write('../resource/outPointMap.nrrd',out)
    nrrd.write('../resource/inPointMap.nrrd',out)

    im50=grid[50,30:40,20:30]*3000
    img50 = Image.fromarray(im50)

    im50=relaxation[50,:,:].astype(np.float32)*20
    img50 = Image.fromarray(im50)

    img50.show()
    pass