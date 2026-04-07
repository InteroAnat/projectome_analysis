# %%
from ast import While
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
class Point:

    def __init__(self,point):
        self.parentIndex=-1
        self.children=[]
        self.parent=None
        self.index=int(point[0])
        self.type=int(point[1])
        self.x=point[2]
        self.y=point[3]
        self.z=point[4]
        self.ratio=point[5]
        self.parentIndex=int(point[6])
        self.children=[]
        self.xyz=[np.float32(point[2]),np.float32(point[3]),np.float32(point[4])]
    def __str__(self):
        
        return str(self.index)+","+str(self.type)+","+str(self.x)+","+\
        str(self.y)+","+str(self.z)+","+str(self.ratio)+","+str(self.parentIndex)
        
class Edge:
    def __init__(self,point=None):
        if point is None:
            self.data=[]
        else:
            self.data=[point]
        self.order=-1
        self.angle=0
        self.maxDepth=-1
        self.maxLength=-1
        self.children=[]
        self.parent=None
        self.len=0
    def addPoint(self,point):
        self.data.append(point)
    def getLength(self):
        if self.len==0:
            for i in range(len(self.data)-1):
                self.len+=np.linalg.norm(np.array(self.data[i].xyz)-np.array(self.data[i+1].xyz))
        return self.len
        
class NeuronTree:
    def __init__(self):
        self.root=None
        self.edges=[]
        self.points=[]
        self.width=1
        self.branchs=[]
        self.terminals=[]
        self.swc={}
        self.xyz=[]
        self.somaRadius=200
        self.somaHide=False
        self.axonHide=False
        self.dendriteHide=False
        self.maxOrder=-1
        self.rootAxonEdge=None
        self.dendrites=[]
 
    def readSWC(self,swc):
        swclines=swc.split('\n')
        for line in swclines:
            if len(line)==0 or line[0] == '#'or line[0] == '\r':
                continue
            p=list(map(float,(line.split())))
            self.swc[int(p[0])]=Point(p)
        # self.swc.pop()
        self.parseswc()

    def readFile(self,filename='',spliter=' '):

        with open(filename, "r") as f:
            text_lines = f.read()
            swclines=text_lines.split('\n')
            for line in swclines:
                if len(line)==0 or line[0] == '#':
                    continue
                p=list(map(float,(line.split(spliter))))
                self.swc[int(p[0])]=Point(p)
        self.parseswc()

    def parseswc(self):
        for key,point in self.swc.items():
            self.xyz.append(point.xyz)
            self.points.append(point)
            if point.parentIndex==-1:
                if self.root is None:
                    self.root=point
        self.getEdges()
        for edge in self.edges:
            if self.rootAxonEdge is None and (edge.data[0]==self.root or edge.data[0].type==3)and (edge.data[1].type!=3):
                self.rootAxonEdge=edge
            if len(edge.data[len(edge.data)-1].children)==0:
                self.terminals.append(edge.data[len(edge.data)-1])
        for edge in self.edges:
            for tail in edge.data[len(edge.data)-1].children:
                for edge0 in self.edges:
                    if tail==edge0.data[1]:
                        edge.children.append(edge0)
                        edge0.parent = edge
    def getDendrite(self):
        if len(self.dendrites)==0:
            for edge in self.edges:
                if edge.data[1].type!=3:
                    continue
                self.dendrites.append(edge)
        return self.dendrites
        
    def getEdges(self):
        for index,point in self.swc.items():
            if point.parentIndex!=-1  and self.swc.get(point.parentIndex) is not None:
                point.parent=self.swc[int(point.parentIndex)]
                point.parent.children.append(point)
            pass
        start = self.swc[list(self.swc.keys())[0]]
        if len(start.children):
            lastp=None
            self.getEdge(start)

    def getEdge(self,start):
        for p in start.children:
            edge=Edge(start)
            edgestart=p
            while len(edgestart.children)==1:
                edge.addPoint(edgestart)
                #del self.swc[edgestart.index]
                edgestart=edgestart.children[0]
            edge.addPoint(edgestart)
            self.edges.append(edge)
            lastp=edgestart
            self.getEdge(lastp)
        pass
    def getEdgeByTerminal(self,terminal):
        for edge in self.edges:
            if edge.data[len(edge.data)-1]==terminal:
                return edge
        pass
    def print(self):
        level='-';
        print(level,self.root.index)
        self.printChildren(self.root,level)
        
        
    def printChildren(self,point,level):
        level=level+'---'
        for child in point.children:
            print(level,child.index)
            self.printChildren(child,level)
            
    def printEdges(self):
        for edge in self.edges:
            print('==============')
            for p in edge:
                print (p.index);

    def plotMPR(self,filename=None):
        dpi = 100
        fig = plt.figure(figsize=(1200/dpi, 400/dpi), dpi=dpi)
        
        axes0 = plt.subplot(131)
        plt.axis('off')
        axes0.set_xlim(0, 11400)
        axes0.set_ylim(0, 10000)
        fig.patch.set_facecolor('#000000')
        # plt.Circle((neuron.root.z, neuron.root.y), 300)
        mirror = False
        if self.root.z<5700:
            mirror=True
        for edge in self.edges:
            x=[]
            y=[]
            for p in edge.data:
                z= p.z
                if mirror:
                    z= 11400-p.z
                x.append(11400-z)
                y.append(10000-p.y)
            plt.plot(x, y,'g-',color='#FFFFFF')
        # plt.figure(facecolor='gainsboro')
        rootz = self.root.z
        if mirror:
            rootz= 11400-self.root.z
        plt.plot(10000-rootz,10000- self.root.y,'g,')

        axes0 =  plt.subplot(132)
        axes0.set_axis_off()
        axes0.set_xlim(0, 16200)
        axes0.set_ylim(0, 16200)
        fig.patch.set_facecolor('#000000')
        # plt.Circle((neuron.root.z, neuron.root.y), 300)
        for edge in self.edges:
            x=[]
            y=[]
            for p in edge.data:
                x.append(16200-p.x)
                y.append(10000-p.y)
            plt.plot(x, y,'g-',color='#FFFFFF')
        # plt.figure(facecolor='gainsboro')
        plt.plot(10000-self.root.x,10000- self.root.y,'g,')

        axes0 = plt.subplot(133)
        plt.axis('off')
        axes0.set_xlim(0, 16200)
        axes0.set_ylim(0, 16200)
        fig.patch.set_facecolor('#000000')
        # plt.Circle((neuron.root.z, neuron.root.y), 300)
        for edge in self.edges:
            x=[]
            y=[]
            for p in edge.data:
                z= p.z
                if mirror:
                    z= 11400-p.z
                x.append(16200-p.x)
                y.append(11400-z)
            plt.plot(x, y,'g-',color='#FFFFFF')
        # plt.figure(facecolor='gainsboro')
        plt.plot(10000-self.root.x,10000- rootz,'g,')

        fname = filename
        if filename is None:
            fname="../resource/neuron.png"
        plt.savefig(fname=fname,format="png",bbox_inches='tight')
        fig.clf()
        plt.clf()
        plt.cla()
        plt.close('all')
        plt.close('fig')
        gc.collect()
        pass
# %%  
if __name__ == "__main__":
    import os,sys
    if hasattr(sys.modules[__name__], '__file__'):
        _file_name = __file__
    else:
        _file_name = inspect.getfile(inspect.currentframe())
    CURRENT_FILE_PATH = os.path.dirname(_file_name)
    print(CURRENT_FILE_PATH)
    print(_file_name)
# %%
    tree = NeuronTree()
    # tree.readFile("../resource/swc/17109/17109_1701_x8048_y22277.semi_r.swc")
    
    father_path=os.path.dirname(CURRENT_FILE_PATH)
    # path_to_load=os.path.join(father_path+"/resource/swc/18715/025.swc")
    
    tree.readFile(father_path+"\\resource\\swc\\192106\\001.swc")
    tree.plotMPR()
    print(len(tree.edges))
    # for edge in tree.edges:
    #     print(len(edge.data))
        # for p in edge.data:
        #     print(p)

# %%
