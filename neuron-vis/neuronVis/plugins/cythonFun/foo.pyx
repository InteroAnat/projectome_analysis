# example.pyx
import numpy as np
cimport cython
cimport numpy as cnp
cnp.import_array()
cpdef foo(cnp.ndarray[double, ndim=2] A,cnp.ndarray[double, ndim=2] B):
    cdef long long a = 0, b = 1
    cdef long index
    cdef cnp.ndarray[double, ndim=2] c,d
    cdef cnp.ndarray[double, ndim=1] dist
    cdef cnp.ndarray[int, ndim=1] iVertex
    cdef float mindist=9999
    iVertex = np.empty(B.shape[0], dtype=np.int)
    for j in range(B.shape[0]):
        point =B[j]
        c=A-point
        d=c*c
        dist = sum(d.T,0)
        #iVertex
        index=0
        mindist=9999
        iVertex[j]=-1

        for i in range(dist.shape[0]):
            if mindist>dist[i]:
                mindist=dist[i]
                index=i
        iVertex[j]=index
    return iVertex