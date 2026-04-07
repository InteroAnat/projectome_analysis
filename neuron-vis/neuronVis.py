import Render
from vispy import app,gloo,io
import swcloader
import GeometryAdapter as GA
class neuronVis:
    def __init__(self):
        self.render = Render.Render()
        self.regionID2Name={}
        self.regionName2ID={}
        self.Region()

    def addNeuron(self,name,color=[1.0,1.0,1.0]):

        geo = GA.GeometryAdapter(name)
        self.render.addLines([name,geo],color)
        pass
    def addRegion(self,name,color=[1.0,1.0,1.0]):
        regionFileName = './resource/allobj/'+str(self.regionName2ID[name][0])+'.obj'
        self.render.addGeometry([name,self.render.loadGeometry(regionFileName)],color)
        pass
    def clear(self,root=False,neurons=True,regions=True):
        if regions:
            if 'root' in self.render.regions:
                rootregion=self.render.regions['root']
                self.render.regions={}
                self.render.regions['root']=rootregion
            else:
                self.render.regions={}
        if root and 'root' in self.render.regions:
            del self.render.regions['root']
        pass
    def Region(self):
        if len(self.regionID2Name)==0:
            lines=[]
            with open("./resource/annot.txt", 'r') as file_to_read:
            #   while True:
                lines = file_to_read.readlines()
            for line in lines:
                linetmp=line[:-1].split(":")
                self.regionID2Name[int(linetmp[0])]=[linetmp[1],linetmp[2]]
                self.regionName2ID[linetmp[2]]=[int(linetmp[0]),linetmp[1]]



if __name__=="__main__":
    import neuronVis as nv
    neuronvis = nv.neuronVis()
    # neuronvis.render.setBackgroundColor((0.0,0.20,0.5,1.0))
    neuronvis.addRegion('CA2',[0.5,1.0,0.5])
    neuronvis.addNeuron('./resource/033.swc')
    neuronvis.render.savepng('resource/test.png')
    # neuronvis.render.setView('anterior')
    neuronvis.render.setLookAt()
    neuronvis.addNeuron('./resource/192092-012.swc',[1.0,1.0,0.0])
    neuronvis.render.savepng('resource/test2.png')

    neuronvis.render
    nv.app.run()
    pass