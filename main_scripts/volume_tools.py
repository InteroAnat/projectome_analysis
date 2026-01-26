#%%
import ctypes
import time,os
import tempfile
import numpy as np
import urllib.request
import sys
from itertools import cycle
from vispy import app, scene, io
from vispy.color import get_colormaps, BaseColormap
from vispy.visuals.transforms import STTransform
from vispy.util import logger, Frozen
from vispy import gloo 
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)
import IONData as IT
    
import neuro_tracer as nt
ll = ctypes.cdll.LoadLibrary
current_path =neurovis_path
dllpath = os.path.join(current_path, "./plugins/fnt_libpy.dll")

print(dllpath)
imageLoader = ctypes.WinDLL(dllpath)
imageLoader.create.restype = ctypes.POINTER(ctypes.c_int)
class FNTCube:


    def __init__(self) -> None:
        self.sampleid='221473'
        self.server='http://bap.cebsit.ac.cn/'
        self.path='monkeydata'
        self.imageCoord=[]
        self.mouseCoord=[]
        self.cubeIndex=[]
        self.pixelSpace=[0.65,0.65,3]
        self.radiu=3
        self.cubesize=[]
        self.imagesize=[]
        self.volume = None
        pass
    # ADD THIS IMPORT AT THE TOP OF YOUR FILE
    import tifffile 

    def getVolumeFromIndex(self, x, y, z):
        
        idx_x = int(x // self.cubesize[0])
        idx_y = int(y // self.cubesize[1])
        idx_z = int(z // self.cubesize[2]) 
      
        filename = f"{idx_x}_{idx_y}_{idx_z}.tif"
        
        # The folder is the Z-Index
        url = f"{self.server}/{self.path}/{self.sampleid}/cube/{idx_z}/{filename}"
        
 

        try:
            response = urllib.request.urlopen(url, timeout=20)
            buf = response.read()
        except Exception as e:
            print(f"Failed: {url} | {e}")
            # Return empty black box on failure
            return np.zeros((self.cubesize[2], self.cubesize[1], self.cubesize[0]), dtype=np.uint16)

        # 3. READ TIFF
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as fp:
            fp.write(buf)
            fp.close()
            try:
                import tifffile
                self.volume = tifffile.imread(fp.name)
                
                # OPTIONAL: Flip axes if the image looks rotated
                # self.volume = np.transpose(self.volume, (2, 1, 0))
                
            except Exception as e:
                print(f"TIFF Error: {e}")
                self.volume = np.zeros((self.cubesize[2], self.cubesize[1], self.cubesize[0]), dtype=np.uint16)
            
            os.remove(fp.name)
            return self.volume
            
    def getVolume(self):

        rangei = self.radiu if self.radiu<4 else 3

        sj = (self.radiu-1)*2+1
        si = (rangei-1)*2+1
        sk = (rangei-1)*2+1
        volume = np.zeros((self.cubesize[2]*si,self.cubesize[1]*sj,self.cubesize[0]*sk),dtype=np.uint16)
        progress =0
        self.startIndex=[self.cubeIndex[0]+(1-rangei)*self.cubesize[0],self.cubeIndex[1]+(1-self.radiu)*self.cubesize[1],self.cubeIndex[2]+(1-rangei)*self.cubesize[2]]
        for i in range(1-rangei,rangei):
            for j in range(1-self.radiu,self.radiu):
                for k in range(1-rangei,rangei):
                    vol = self.getVolumeFromIndex(self.cubeIndex[0]+i*self.cubesize[0],self.cubeIndex[1]+j*self.cubesize[1],self.cubeIndex[2]+k*self.cubesize[2])
                    volume[(rangei-1+k)*self.cubesize[2]:(rangei+k)*self.cubesize[2],
                    (self.radiu-1+j)*self.cubesize[1]:(self.radiu+j)*self.cubesize[1],
                    (rangei-1+i)*self.cubesize[0]:(rangei+i)*self.cubesize[0]] = vol
                    progress+=1
                    print('finished '+str(progress)+'/'+str(si*sj*sk))
        self.volume=volume
        return volume
        
    def phraseCatalog(self):
        print("--- Parsing Catalog ---")
        
        # 1. Download the catalog text
        url = f"{self.server}/{self.path}/{self.sampleid}/catalog"
        try:
            response = urllib.request.urlopen(url, timeout=5)
            content = bytes.decode(response.read())
            # print(content) # Uncomment to debug
        except Exception as e:
            print(f"Error downloading catalog: {e}")
            # Fallback to the values visible in your text snippet
            self.pixelSpace = [0.65, 0.65, 3.0]
            self.cubesize = [360, 360, 90]
            return

        # 2. Parse the specific sections
        # We prioritize [CH00TIFFM] because you are using TIFFs
        lines = content.splitlines()
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Detect Section Header
            if line.startswith("["):
                current_section = line
                continue
            
            # We only care about the TIFF section
            if current_section == "[CH00TIFFM]":
                
                # Parse Resolution (direction)
                # Format: direction=0.65 0 0 0 0.65 0 0 0 3
                if line.startswith("direction="):
                    parts = line.split('=')[1].strip().split()
                    # The matrix is [X 0 0, 0 Y 0, 0 0 Z]. We take indices 0, 4, 8.
                    res_x = float(parts[0])
                    res_y = float(parts[4])
                    res_z = float(parts[8])
                    self.pixelSpace = [res_x, res_y, res_z]
                    print(f"Found Resolution: {self.pixelSpace}")

                # Parse Block Size (cubesize)
                # Format: cubesize=360 360 90
                elif line.startswith("cubesize="):
                    parts = line.split('=')[1].strip().split()
                    self.cubesize = [int(parts[0]), int(parts[1]), int(parts[2])]
                    print(f"Found CubeSize: {self.cubesize}")

                # Parse Total Image Size
                elif line.startswith("size="):
                    parts = line.split('=')[1].strip().split()
                    self.imagesize = [int(parts[0]), int(parts[1]), int(parts[2])]
    def setSampleID(self,sampleid):
        self.sampleid = sampleid
        self.phraseCatalog()

    def setMouseCoord(self,x,y,z):
        self.mouseCoord=[x,y,z]
        self.imageCoord = [int(x/self.pixelSpace[0]),int(y/self.pixelSpace[1]),int(z/self.pixelSpace[2])]
        self.cubeIndex = [self.imageCoord[0]//self.cubesize[0]*self.cubesize[0],self.imageCoord[1]//self.cubesize[1]*self.cubesize[1],self.imageCoord[2]//self.cubesize[2]*self.cubesize[2]]
        pass
    def setRadiu(self,radiu=1):
        self.radiu = radiu
        pass

    def physicalPoint2LocalPoint(self,physicalPoints,isImageCoord=False):
        shape = self.volume.shape
        pos=[]
        for point in physicalPoints:
            x=point[0]
            y=point[1]
            z=point[2]
            imageCoord = [int(x/self.pixelSpace[0]),int(y/self.pixelSpace[1]),int(z/self.pixelSpace[2])]
            # imageCoord = cube.imageCoord

            localPoint = [imageCoord[0] - self.startIndex[0],imageCoord[1] - self.startIndex[1],imageCoord[2] - self.startIndex[2]]
            pointtemp = [(localPoint[0]-shape[2]/2)*self.pixelSpace[0],(localPoint[1]-shape[1]/2)*self.pixelSpace[1],(localPoint[2]-shape[0]/2)*self.pixelSpace[2]]
            if isImageCoord:
                pos.append(localPoint)
            else:
                pos.append(pointtemp)
        return pos
    
    def save_to_nrrd(self, filename):
                import nrrd
                # if any(self.volume): return
                vol = self.volume.astype(np.int16)
                vol = np.transpose(vol,(2,1,0))
                # header = {'spacings': [sp[2], sp[1], sp[0]], 'units': ['microns']*3} # Z, Y, X
                
                output_dir = r'../resource/nrrds/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                print(f"Saving {filename}...")
                nrrd.write(os.path.join(output_dir,filename), vol)
        
        #%%    
# class VolumeRender(app.Canvas):

#     def __init__(self,size=(800,600)) -> None:
#         # app.Canvas.__init__(self, size=size, title='Neuron Render',keys='interactive')
#         self.Canvas = scene.SceneCanvas(keys='interactive',title='FNTCube Render', size=(800, 600), show=True)
#         # self.Canvas.events.connect(self.on_mouse_move)
#         self.Canvas.events.mouse_move.connect(self.on_mouse_move)
#         self.Canvas.events.key_press.connect(self.on_key_press)
#         pass
#     def setClim(self,clim):
#         self.volume1.clim=clim
#     def CubeImageCoord2GlobalPhysicalCoord(self,cubeImageCoord):
#         #check input in cube volume
#         shape = self.volume1._vol_shape
#         if cubeImageCoord[0]>=shape[0] or cubeImageCoord[1]>=shape[1] or cubeImageCoord[2]>=shape[2] :
#             return
#         pass
#     def getVoxelValue(self,points):
#         shape = self.volume1._vol_shape
#         # shape = [shape[2]*cube.pixelSpace[0]/2,shape[1]*cube.pixelSpace[1]/2,shape[0]/2]
        
#         self.point = scene.visuals.Markers(parent=self.view.scene,size=0.2,scaling=True)
#         pos = []
#         values=[]
#         for point in points:
#             x=point[0]
#             y=point[1]
#             z=point[2]
#             imageCoord = [int(x/self.cube.pixelSpace[0]),int(y/self.cube.pixelSpace[1]),int(z/self.cube.pixelSpace[2])]
#             localPoint = [imageCoord[0] - self.cube.startIndex[0],imageCoord[1] - self.cube.startIndex[1],imageCoord[2] - self.cube.startIndex[2]]
#             value = self.volume1._last_data[localPoint[2]][localPoint[1]][localPoint[0]]
#             values.append(value)
#         return values

#     def drawSegments(self,segments):
#         shape = self.volume1._vol_shape
#         # shape = [shape[2]*cube.pixelSpace[0]/2,shape[1]*cube.pixelSpace[1]/2,shape[0]/2]
        
#         self.point = scene.visuals.Line(parent=self.view.scene,connect='segments')
#         pos = []
#         # pos.append(lineSegments[0])
#         for lineSegments in segments:
#             for i in range(len(lineSegments)):
#                 point = lineSegments[i]
#                 x=point[0]
#                 y=point[1]
#                 z=point[2]
#                 imageCoord = [int(x/self.cube.pixelSpace[0]),int(y/self.cube.pixelSpace[1]),int(z/self.cube.pixelSpace[2])]
#                 # imageCoord = cube.imageCoord

#                 localPoint = [imageCoord[0] - self.cube.startIndex[0],imageCoord[1] - self.cube.startIndex[1],imageCoord[2] - self.cube.startIndex[2]]
#                 pointtemp = [(localPoint[0]-shape[2]/2)*self.cube.pixelSpace[0],(localPoint[1]-shape[1]/2)*self.cube.pixelSpace[1],(localPoint[2]-shape[0]/2)*self.cube.pixelSpace[2]]
#                 if i==0 or i==len(lineSegments)-1:
#                     pos.append(pointtemp)
#                     continue
#                 pos.append(pointtemp)
#                 pos.append(pointtemp)
#         # pos.append(lineSegments[len(lineSegments)-1])
        
#         self.point.set_data(pos=np.array(pos))
#         s = STTransform(translate=self.cam3.center)
#         affine = s.as_matrix()
#         self.point.transform = affine
#         self.point.set_gl_state(depth_test=False)
#         self.point.antialias = True
#         pass
#     def drawLines(self,lineSegments):
#         shape = self.volume1._vol_shape
#         # shape = [shape[2]*cube.pixelSpace[0]/2,shape[1]*cube.pixelSpace[1]/2,shape[0]/2]
        
#         self.point = scene.visuals.Line(parent=self.view.scene,connect='segments')
#         pos = []
#         # pos.append(lineSegments[0])
#         for i in range(len(lineSegments)):
#             point = lineSegments[i]
#             x=point[0]
#             y=point[1]
#             z=point[2]
#             imageCoord = [int(x/self.cube.pixelSpace[0]),int(y/self.cube.pixelSpace[1]),int(z/self.cube.pixelSpace[2])]
#             # imageCoord = cube.imageCoord

#             localPoint = [imageCoord[0] - self.cube.startIndex[0],imageCoord[1] - self.cube.startIndex[1],imageCoord[2] - self.cube.startIndex[2]]
#             pointtemp = [(localPoint[0]-shape[2]/2)*self.cube.pixelSpace[0],(localPoint[1]-shape[1]/2)*self.cube.pixelSpace[1],(localPoint[2]-shape[0]/2)*self.cube.pixelSpace[2]]
#             if i==0 or i==len(lineSegments)-1:
#                 pos.append(pointtemp)
#                 continue
#             pos.append(pointtemp)
#             pos.append(pointtemp)
#         # pos.append(lineSegments[len(lineSegments)-1])
        
#         self.point.set_data(pos=np.array(pos))
#         s = STTransform(translate=self.cam3.center)
#         affine = s.as_matrix()
#         self.point.transform = affine
#         self.point.set_gl_state(depth_test=False)
#         self.point.antialias = True


    


#     def drawPoints(self,points):
#         self.point = scene.visuals.Markers(parent=self.view.scene,size=0.02,scaling=False)
#         pos=self.cube.physicalPoint2LocalPoint(points)
#         self.point.set_data(pos=np.array(pos))
#         s = STTransform(translate=self.cam3.center)
#         affine = s.as_matrix()
#         self.point.transform = affine
#         self.point.set_gl_state(depth_test=False)
#         self.point.antialias = True
#         pass

#     def setFNTCube(self,cube):
#         self.cube=cube
#         # vol1 = np.load(io.load_data_file('brain/mri.npz'))['data']
#         vol1 =cube.getVolume()
#         # vol1[vol1>vol1.mean()*20]=vol1.mean()*20
#         # print(vol1.min(),vol1.mean(),vol1.max())
#         # Prepare canvas
#         self.Canvas.measure_fps()

#         # Set up a viewbox to display the image with interactive pan/zoom
#         self.view = self.Canvas.central_widget.add_view()

#         # Create the volume visuals, only one is visible
#         self.volume1 = scene.visuals.Volume(vol1, parent=self.view.scene, threshold=0.025)
#         self.volume1.transform = scene.STTransform(translate=(0,0,0),scale=(cube.pixelSpace[0],cube.pixelSpace[1],cube.pixelSpace[2], 1))
#         # self.volume1.transform = scene.STTransform(translate=(256*((cube.radiu-1)*2+1)*cube.pixelSpace[0],256*((cube.radiu-1)*2+1)*cube.pixelSpace[0],0),scale=(cube.pixelSpace[0],cube.pixelSpace[1],cube.pixelSpace[2], 1))

#         # Create three cameras (Fly, Turntable and Arcball)
#         fov = 60.
#         self.cam1 = scene.cameras.FlyCamera(parent=self.view.scene, fov=fov, name='Fly')
#         self.cam2 = scene.cameras.TurntableCamera(parent=self.view.scene, fov=fov,
#                                             name='Turntable')
#         self.cam3 = scene.cameras.ArcballCamera(parent=self.view.scene, fov=fov, name='Arcball')
#         self.view.camera = self.cam3  # Select Arcball at first
#         self.cam3.center = [self.cam3.center[0]*cube.pixelSpace[0],self.cam3.center[1]*cube.pixelSpace[1],self.cam3.center[2]]

#         # # Create an XYZAxis visual
#         # self.axis = scene.visuals.XYZAxis(parent=self.view.scene)
#         # # s = STTransform(translate=(128*((cube.radiu-1)*2+1),128*((cube.radiu-1)*2+1),((cube.radiu-1)*2+1)*45), scale=(10, 10, 10, 1))
#         # s = STTransform(translate=self.cam3.center, scale=(10, 10, 10, 1))
#         # affine = s.as_matrix()
#         # self.axis.transform = affine
#         # self.axis.set_gl_state(depth_test=False)
#         # self.axis.antialias = True
#         shape = self.volume1._vol_shape
#         shape = [shape[2]*cube.pixelSpace[0]/2,shape[1]*cube.pixelSpace[1]/2,shape[0]/2]
#         self.point = scene.visuals.Line(parent=self.view.scene,pos=np.array([[-shape[2],-shape[1],-shape[0]],[shape[2],-shape[1],-shape[0]],[shape[2],shape[1],-shape[0]],[shape[2],shape[1],shape[0]]]))
#         s = STTransform(translate=self.cam3.center)
#         affine = s.as_matrix()
#         self.point.transform = affine
#         self.point.set_gl_state(depth_test=False)
#         self.point.antialias = True
        
#         self.drawPoints([cube.mouseCoord])
#         print(self.getVoxelValue([cube.mouseCoord]))

#         # create colormaps that work well for translucent and additive volume rendering
#         class TransFire(BaseColormap):
#             glsl_map = """
#             vec4 translucent_fire(float t) {
#                 return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
#             }
#             """


#         class TransGrays(BaseColormap):
#             glsl_map = """
#             vec4 translucent_grays(float t) {
#                 return vec4(t*2, t*2, t*2, t*0.05);
#             }
#             """

#         # Setup colormap iterators
#         self.opaque_cmaps = cycle(get_colormaps())
#         self.translucent_cmaps = cycle([TransFire(), TransGrays()])
#         self.opaque_cmap = next(self.opaque_cmaps)
#         self.translucent_cmap = next(self.translucent_cmaps)

#         self.interp_methods = cycle(self.volume1.interpolation_methods)
#         interp = next(self.interp_methods)
#         self.Canvas.show()
#         # self.show()


#     # Implement axis connection with cam2
#     # @canvas.events.mouse_move.connect
#     # @app.Canvas.events.key_press.connect
#     def on_mouse_move(self,event):
#         if event.button == 1 and event.is_dragging:
#             return
#             self.axis.transform.reset()
#             rot, x, y, z = self.cam3._quaternion.get_axis_angle()
#             self.axis.transform.rotate(180 * rot / np.pi, (x, z, y))

#             # self.axis.transform.rotate(self.cam2.roll, (0, 0, 1))
#             # self.axis.transform.rotate(self.cam2.elevation, (1, 0, 0))
#             # self.axis.transform.rotate(self.cam2.azimuth, (0, 1, 0))

#             # self.axis.transform.scale((50, 50, 0.001))
#             # self.axis.transform.translate((50., 50.))
#             self.axis.transform.scale((50, 50, 1, 0.5))
#             self.axis.transform.translate((90*((1-1)*2+1),90*((1-1)*2+1),0))
            
#             self.axis.visible = True
#             self.axis.update()


#     # Implement key presses
#     # @Canvas.events.key_press.connect
#     def on_key_press(self,event):
#         self.opaque_cmap, self.translucent_cmap
#         if event.text == '1':
#             cam_toggle = {self.cam1: self.cam2, self.cam2: self.cam3, self.cam3: self.cam1}
#             self.view.camera = cam_toggle.get(self.view.camera, self.cam3)
#             print(self.view.camera.name + ' camera')
#             if self.view.camera is self.cam2:
#                 self.axis.visible = True
#             else:
#                 self.axis.visible = False
#         elif event.text == '2':
#             methods = ['mip', 'translucent', 'iso', 'additive']
#             method = methods[(methods.index(self.volume1.method) + 1) % 4]
#             print("Volume render method: %s" % method)
#             cmap = self.opaque_cmap if method in ['mip', 'iso'] else self.translucent_cmap
#             self.volume1.method = method
#             self.volume1.cmap = cmap

#         elif event.text == '3':
#             self.volume1.visible = not self.volume1.visible
#             # volume2.visible = not volume1.visible
#         elif event.text == '4':
#             if self.volume1.method in ['mip', 'iso']:
#                 cmap = self.opaque_cmap = next(self.opaque_cmaps)
#             else:
#                 cmap = self.translucent_cmap = next(self.translucent_cmaps)
#             self.volume1.cmap = cmap
#             # volume2.cmap = cmap
#         elif event.text == '5':
#             interp = next(self.interp_methods)
#             self.volume1.interpolation = interp
#             # volume2.interpolation = interp
#             print(f"Interpolation method: {interp}")
#         elif event.text == '6':
#             self.volume1.clim=(self.volume1._texture._clim[0]+10,self.volume1._texture._clim[1]+10)
#             print(self.volume1.clim)
#         elif event.text == '7':
#             self.volume1.clim=(self.volume1._texture._clim[0]-10,self.volume1._texture._clim[1]-10)
#             print(self.volume1.clim)
#         elif event.text == '8':
#             self.volume1.clim=(self.volume1._texture._clim[0]-10,self.volume1._texture._clim[1]+10)
#             print(self.volume1.clim)
#         elif event.text == '9':
#             self.volume1.clim=(self.volume1._texture._clim[0]+10,self.volume1._texture._clim[1]-10)
#             print(self.volume1.clim)
#             # self.cam3.set_range()
#         elif event.text == '0':
#             self.cam1.set_range()
#             self.cam3.set_range()
#         elif event.text != '' and event.text in '[]':
#             s = -0.025 if event.text == '[' else 0.025
#             self.volume1.threshold += s
#             # volume2.threshold += s
#             th = self.volume1.threshold if self.volume1.visible else self.volume1.threshold
#             print("Isosurface threshold: %0.3f" % th)
   
#%%
# if __name__ == '__main__':
#     import vispy.io as io
#     cube = FNTCube()
#     cube.setSampleID('251637')
#     cube.setMouseCoord(14299.08,35213.89,20665.78) # cortex
#     # cube.setMouseCoord(6220.28,3380.16,6409.23) # th
#     # cube.setMouseCoord(6016.28,3820.16,6315.23) # th
#     # cube.setMouseCoord(3316.73,4630.76,5811.05) # th
#     cube.radiu=1
#     volume =cube.getVolume()
#     # cube.setMouseCoord(3459.95,1743.75,6370.94) #mean=137
    
#     render = VolumeRender()
#     render.setFNTCube(cube)
#     render.volume1.clim=(20,1000)
#     app.run()
#%%

sampleID='251637'
neuronID='003.swc'
# neuron = nt.neuro_tracer()
# neuron = neuron.process(sampleID,neuronID,nii_space='monkey')

# raw_neuron.
# %%
# import SwcLoader
    # tree.readFile("../resource/swc/17109/17109_1701_x8048_y22277.semi_r.swc")
iondata=IT.IONData()
raw_neuron_tree = iondata.getRawNeuronTreeByID(sampleID,neuronID)
x,y,z=raw_neuron_tree.root.xyz

if __name__ == '__main__':
    import vispy.io as io
    cube = FNTCube()
    cube.setSampleID('251637')
    cube.setMouseCoord(x,y,z) # cortex
    # cube.setMouseCoord(6220.28,3380.16,6409.23) # th
    # cube.setMouseCoord(6016.28,3820.16,6315.23) # th
    # cube.setMouseCoord(3316.73,4630.76,5811.05) # th
    cube.radiu=1
    volume =cube.getVolume()
    # cube.setMouseCoord(3459.95,1743.75,6370.94) #mean=137
    volume
    # render = VolumeRender()
    # render.setFNTCube(cube)
    # render.volume1.clim=(20,1000)
    # app.run()


# tree = NeuronTree()
# tree.readFile(CURRENT_FILE_PATH+"/../resource/swc/251637/002.swc")


# %%
# def SomaSegment(sampleid,neuron):
#     # sampleid='210726'
#     # neuron='006.swc'
#     # tree = SwcLoader.NeuronTree()
#     # tree.readFile(r'D:\projectome_analysis\resource\swc_raw\251637\001.swc')
#     tree = iondata.getRawNeuronTreeByID(sampleid,neuron)
#     tree.root
#     cube.setSampleID(sampleid)
#     # cube.setMouseCoord(6719.08,2002.89,4665.78) # cortex
#     # cube.setMouseCoord(6220.28,3380.16,6409.23) # th
#     # cube.setMouseCoord(6016.28,3820.16,6315.23) # th
#     cube.setMouseCoord(tree.root.x,tree.root.y,tree.root.z) # th
#     cube.radiu=1

#     # cube.setMouseCoord(3459.95,1743.75,6370.94) #mean=137
        
#     # The above code snippet is written in Python and appears to be setting up a volume rendering
#     # process using a library or module named `volume`. Here is a breakdown of the code:
#     render = volume.VolumeRender()
#     render.setFNTCube(cube)
#     render.volume1.clim=(20,1000)
#     volume.app.run()
