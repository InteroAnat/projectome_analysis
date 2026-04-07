import numpy as np
import glm
import os,sys,random
from PIL import Image
# from IPython import get_ipython
from vispy import app, gloo, visuals,io

from vispy.gloo import gl,Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate,scale,frustum,ortho
from vispy.geometry import create_cube,MeshData

from GeometryAdapter import *
import Node

if hasattr(sys.modules[__name__], '__file__'):
	_file_name = __file__
else:
	_file_name = inspect.getfile(inspect.currentframe())
CURRENT_FILE_PATH = os.path.dirname(_file_name)


# app.use_app('jupyter_rfb')
# print(vispy.sys_info())

# def type_of_script():
# 	try:
# 		ipy_str = str(type(get_ipython()))
# 		if 'zmqshell' in ipy_str:
# 			return 'jupyter'
# 		if 'terminal' in ipy_str:
# 			return 'ipython'
# 	except:
# 		return 'terminal'
# if type_of_script()=='jupyter':
# 	app.use_app('jupyter_rfb')
# else:
# 	app.use_app('pyqt5')

neuronVertShader = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 position;
uniform vec3 color;
varying vec4 v_color;

void main (void) {
	v_color =vec4(color,1.0);
	vec4 pos  = projection * view * model * vec4(position,1.0);
	gl_Position=pos;
}
"""

neuronFragShader = """
varying vec4 v_color;
void main()
{
	gl_FragColor = v_color;
}
"""
colormap=[
[1,0,0],[0,1,0],[0,0,1],
[1,1,0],[0,1,1],[1,0,1],
[0.5,0.5,0.5],[0,0,0],
[0.5,1,0],[1,0.5,0],[0,0.5,1]
]
class GeometryHandle():
	def __init__(self):
		self.program=None
		self.index=None
	def setGeometry(self,geometry):
		self.Geometry = geometry
		pass
class Render(app.Canvas):


	def __init__(self,size=(800, 600)):
		self.vertShader="""
					uniform mat4 model;
					uniform mat4 view;
					uniform mat4 projection;
					attribute vec3 position;
					attribute vec3 normal;
					attribute vec3 color;
					varying vec4 v_color;
					varying vec3 tnorm;
					varying vec3 view_direction;
					void main()
					{
						vec4 pos_in_camera = view*model * vec4(position, 1.0);
						v_color =vec4(color,0.5);
						gl_Position =projection*pos_in_camera;
						tnorm = normalize((view*model*vec4(normal.xyz, 0.0)).xyz);
						view_direction = normalize(pos_in_camera.xyz);

					}

					"""
		self.fragShder="""
					varying vec4 v_color;
					varying vec3 tnorm;
					varying vec3 view_direction;
					void main()
					{
						float edginess = 1.0 - abs(dot(tnorm, view_direction));
						float opacity = clamp(edginess - 0.30, 0.0, 0.5);
						// Darken compartment at the very edge\n          
						float blackness = pow(edginess, 4.0) - 0.3;
						vec3 c = mix(v_color.xyz*0.8, vec3(0, 0, 0), blackness);
						gl_FragColor = vec4(c, opacity);

					   // gl_FragColor =v_color;//vec4(1.0,0.0,0.0,1.0);
					}
					"""

		app.Canvas.__init__(self, size=size, title='Neuron Render',
				keys='interactive')
		gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
		self.saveimg=False
		self.phi =0
		self.theta =0
		self.scale=1;
		self.rotate=np.eye(4,k=0)
		self.xaxis=[0,0,1,0]
		self.yaxis=[0,1,0,0]

		self.rootMesh = self.loadGeometry(CURRENT_FILE_PATH+"/../resource/allobj/997.obj")
		self.rootMesh.name='root'
		bounds=self.rootMesh.bound
		self.center = glm.vec3(bounds[0][1]+bounds[0][0],bounds[1][1]+bounds[1][0],bounds[2][1]+bounds[2][0])/2.0
		# self.view=glm.transpose(glm.lookAt(glm.vec3(0.,0.0,15000.0),glm.vec3(0,0,0),glm.vec3(0.0,-1.0,0)))
		self.setView('anterior')
		self.geometries=[]

		program=  Program(self.vertShader, self.fragShder)
		self.model = translate(-self.center)
		program['model'] = self.model
		program['view'] = self.view
		program['position'] =  self.rootMesh.vertex
		program['normal'] =   self.rootMesh.normal
		program['color'] =   [1.0,1.0,1.0]
		self.rootindex = IndexBuffer(self.rootMesh.index)
		gh =GeometryHandle()
		gh.setGeometry(self.rootMesh)
		gh.program = program
		gh.index=self.rootindex
		self.geometries.append(gh)
		# self.geometries[self.rootMesh]=[program,self.rootindex]

		gloo.set_state(clear_color=(0.0, 0.0, 0.0, 1.00),blend=True,blend_func=('src_alpha', 'one_minus_src_alpha'),  depth_test=False)
		gloo.set_state(samples=4)
		self.activate_zoom()
		self.prepos=[0,0]
		self.show()
		self.colorIndex = int(random.random()*10)
		pass
	def setBackgroundColor(self,color):
		gloo.set_state(clear_color=color)

	def setLookAt(self,eye=(0.,0.0,15000),center=(0,0,0),up=(1,0,0)):
		self.view=glm.transpose(glm.lookAt(eye,center,up))

		pass
	def setView(self,type='dorsal'):
		if type=='dorsal':
			self.xaxis=[0,0,1,0]
			self.yaxis=[1,0,0,0]
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,15000,0.0),glm.vec3(0,0,0),glm.vec3(1.0,0.0,0.0)))
		if type=='vontral':
			self.xaxis=[0,0,1,0]
			self.yaxis=[1,0,0,0]
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,-15000,0.0),glm.vec3(0,0,0),glm.vec3(1.0,0.0,0.0)))
		if type=='left':
			self.xaxis=[0,1,0,0]
			self.yaxis=[1,0,0,0]
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,0.0,15000),glm.vec3(0,0,0),glm.vec3(1.0,0.0,0.0)))
		if type=='right':
			self.xaxis=[0,1,0,0]
			self.yaxis=[1,0,0,0]
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,0.0,-15000),glm.vec3(0,0,0),glm.vec3(1.0,0.0,0.0)))
		if type=='anterior':
			self.xaxis=[0,0,1,0]
			self.yaxis=[0,1,0,0]
			self.view=glm.transpose(glm.lookAt(glm.vec3(-15000.,0.0,0),glm.vec3(0,0,0),glm.vec3(0.0,1.0,0.0)))
		if type=='posterior':
			self.xaxis=[0,0,1,0]
			self.yaxis=[0,1,0,0]
			self.view=glm.transpose(glm.lookAt(glm.vec3(15000.,0.0,-0),glm.vec3(0,0,0),glm.vec3(0.0,1.0,0.0)))

		pass
	def addGeometry(self,geometry,color=None):
		if color is None:
			color=colormap[self.colorIndex]
			self.colorIndex=(self.colorIndex+1)%len(colormap)
		program=None
		index = IndexBuffer(geometry.index)

		if geometry.drawModel=='triangles' and geometry.type==-1:
			program=  Program(self.vertShader, self.fragShder)
			program['normal'] =   geometry.normal
		else:
			program=  Program(neuronVertShader, neuronFragShader)
		program['model'] = self.model
		program['view'] = self.view
		program['position'] =  geometry.vertex
		program['color'] =   color
		program['projection'] = self.projection
		gh =GeometryHandle()
		gh.setGeometry(geometry)
		gh.program = program
		gh.index=index
		self.geometries.append(gh)
		for geo in geometry.children:
			self.addGeometry(geo,color)


	# def addLines(self,lines,color=[1.0,1.0,1.0]):
	# 	program=  Program(neuronVertShader, neuronFragShader)
	# 	program['model'] = self.model
	# 	program['view'] = self.view
	# 	program['projection'] = self.projection
	# 	program['position'] =  lines.vertex
	# 	program['color'] =   color
	# 	gloo.gl.glLineWidth(1.0)
	# 	self.neurons[lines.name]=[program,IndexBuffer(lines.index)]

	def setLineWidth(self,width):
		gloo.gl.glLineWidth(width)

	def loadGeometry(self,filename):
		V, I, normals, nothing = io.read_mesh(filename)
		mesh=MeshData(V,I,None)
		geo=Node.Geometry()
		geo.name=filename
		geo.drawModel='triangles'
		geo.vertex=mesh.get_vertices()
		geo.bound=mesh.get_bounds().copy()
		geo.normal=mesh.get_vertex_normals()
		geo.index=mesh.get_faces()
		# print(mesh.get_bounds())
		return geo

	def savepng(self,filename):
		self.saveimg=True
		self.filename=filename;
		self.update()

	def on_draw(self, event):

		gloo.clear(color=True, depth=True)
		
		for gh in self.geometries:
			gh.program['view'] = self.view
			if not gh.Geometry.hide:
				if gh.Geometry.drawModel=='triangles' and gh.Geometry.type==-1:
					gloo.set_state(depth_test=False)
				else:
					gloo.set_state(depth_test=True)
				gh.program.draw(gh.Geometry.drawModel, gh.index) # draw the colorbar

		if self.saveimg:
			print("save "+self.filename)
			np_array = gloo.wrappers.read_pixels()
			np_arr2=np_array.copy()
			np_arr2.flags.writeable = True
			np_arr2[:,:,3]=255
			img = Image.fromarray(np_arr2)
			img=img.resize((int(np_arr2.shape[1]/2),int(np_arr2.shape[0]/2)),resample=1)
			io.image.imsave(self.filename,np.array(img))
			self.saveimg=False

	def on_resize(self, event):
		self.activate_zoom()
		self.update()

	def activate_zoom(self):
		gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
		self.projection = perspective(60.0, self.size[0] / float(self.size[1]),1.0, 150000.0)
		r=self.physical_size[0]/ float(self.physical_size[1])
		width=7000
		if r<1:
			self.projection = ortho(-width, width, width*r,-width*r, -150000, 150000)
		else:
			self.projection = ortho(-width*r, width*r, width,-width, -150000, 150000)

		for gh in self.geometries:
			gh.program['projection'] = self.projection

	def on_mouse_move(self, event):
		if event.button==1:
			self.phi = -self.prepos[1]+event.pos[1]
			self.theta =  -self.prepos[0]+event.pos[0]
			translatemat =translate(self.center)

			rotatex = rotate(-self.phi/1.0, (self.xaxis[0], self.xaxis[1], self.xaxis[2]))
			rotatey = np.dot(rotate(self.theta/1.0, (self.yaxis[0], self.yaxis[1], self.yaxis[2])),rotatex)
			self.rotate = np.dot(self.rotate,rotatey)
			translatemat2 =translate(-self.center)

			scalemat = scale((self.scale,self.scale,self.scale))
			tempmodel = np.dot(np.dot(translatemat2,self.rotate),scalemat)
			self.model=tempmodel

			for gh in self.geometries:
				gh.program['model'] = self.model

			self.update()
		self.prepos=event.pos

		pass

	def on_mouse_wheel(self,event):
		translatemat2 =translate(-self.center)
		if event._delta[1]>=1.0 or event._delta[1]<=-1.0:
			self.scale=self.scale*(1+event._delta[1]/10)
		else:
			self.scale=self.scale*(1+event._delta[1])

		scalemat = scale((self.scale,self.scale,self.scale))
		self.model = np.dot(np.dot(translatemat2,self.rotate),scalemat)
		for gh in self.geometries:
			gh.program['model'] = self.model

		self.update()
		pass
	def run(self):
		app.run()

if __name__=="__main__":
	win = Render()
	ga = GeometryAdapter()
	ga.geometry.addPoint([10000,10000,10000])
	ga.geometry.addPoint([5000,5000,5000])
	ga.geometry.addIndex(0)
	ga.geometry.addIndex(1)
	ga.geometry.drawModel='lines'
	win.addGeometry(ga.geometry)
	# win.show()
	# win
	app.run()
	# pass