import numpy as np
import glm
from PIL import Image
from IPython import get_ipython
from vispy import app, gloo, visuals,io

from vispy.gloo import gl,Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate,scale,frustum,perspective,ortho
from vispy.geometry import create_cube,MeshData


# app.use_app('jupyter_rfb')
# print(vispy.sys_info())

def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'
if type_of_script()=='jupyter':
	app.use_app('jupyter_rfb')
else:
	app.use_app('pyqt5')

neuronVertShader = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 position;
attribute vec3 color;
varying vec4 v_color;

void main (void) {
	v_color =vec4(color,0.5);
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

class Render(app.Canvas):


	def __init__(self):
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

		app.Canvas.__init__(self, size=(800, 600), title='Neuron Render',
				keys='interactive')
		gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])

		self.phi =0
		self.theta =0
		self.scale=1;

		self.rootMesh = self.loadGeometry("./resource/allobj/997.obj")

		bounds=self.rootMesh.get_bounds()
		self.center = glm.vec3(bounds[0][1]+bounds[0][0],bounds[1][1]+bounds[1][0],bounds[2][1]+bounds[2][0])/2.0
		# self.view=glm.transpose(glm.lookAt(glm.vec3(0.,0.0,15000.0),glm.vec3(0,0,0),glm.vec3(0.0,-1.0,0)))
		self.setView('anterior')
		self.regions={}
		self.neurons={}
		program=  Program(self.vertShader, self.fragShder)
		self.model = translate(-self.center)
		program['model'] = self.model
		program['view'] = self.view
		program['position'] =  self.rootMesh.get_vertices().tolist()
		program['normal'] =   self.rootMesh.get_vertex_normals().tolist()
		program['color'] =   [1.0,1.0,1.0]
		self.rootindex = IndexBuffer(self.rootMesh.get_faces())
		self.regions['root']=[program,self.rootindex]

		gloo.set_state(clear_color=(0.0, 0.0, 0.0, 1.00),blend=True,sample_coverage=0.0, blend_func=('src_alpha', 'one_minus_src_alpha'),  depth_test=False)

		self.activate_zoom()
		self.prepos=[0,0]
		self.show()
		pass
	def setBackgroundColor(self,color):
		gloo.set_state(clear_color=color)

	def setLookAt(self,eye=(0.,0.0,15000),center=(0,0,0),up=(-1,0,0)):
		self.view=glm.transpose(glm.lookAt(eye,center,up))

		pass
	def setView(self,type='dorsal'):
		if type=='dorsal':
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,15000,0.0),glm.vec3(0,0,0),glm.vec3(-1.0,0.0,0.0)))
		if type=='vontral':
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,-15000,0.0),glm.vec3(0,0,0),glm.vec3(-1.0,0.0,0.0)))
		if type=='left':
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,0.0,15000),glm.vec3(0,0,0),glm.vec3(-1.0,0.0,0.0)))
		if type=='right':
			self.view=glm.transpose(glm.lookAt(glm.vec3(0.,0.0,-15000),glm.vec3(0,0,0),glm.vec3(-1.0,0.0,0.0)))
		if type=='anterior':
			self.view=glm.transpose(glm.lookAt(glm.vec3(-15000.,0.0,0),glm.vec3(0,0,0),glm.vec3(0.0,-1.0,0.0)))
		if type=='posterior':
			self.view=glm.transpose(glm.lookAt(glm.vec3(15000.,0.0,-0),glm.vec3(0,0,0),glm.vec3(0.0,-1.0,0.0)))

		pass
	def addGeometry(self,geometry,color=[1.0,1.0,1.0]):
		program=  Program(self.vertShader, self.fragShder)
		program['model'] = self.model
		program['view'] = self.view
		program['position'] =  geometry[1].get_vertices().tolist()
		program['normal'] =   geometry[1].get_vertex_normals().tolist()
		program['color'] =   color
		program['projection'] = self.projection
		index = IndexBuffer(geometry[1].get_faces())
		self.regions[geometry[0]]=[program,index]

	def addLines(self,lines,color=[1.0,1.0,1.0]):
		program=  Program(neuronVertShader, neuronFragShader)
		program['model'] = self.model
		program['view'] = self.view
		program['projection'] = self.projection
		program['position'] =  lines[1].xyz
		program['color'] =   color
		gloo.gl.glLineWidth(2.0)
		self.neurons[lines[0]]=[program,IndexBuffer(lines[1].index)]
		
	def setLineWidth(self,width):
		gloo.gl.glLineWidth(width)

	def loadGeometry(self,filename):
		V, I, normals, nothing = io.read_mesh(filename)

		return MeshData(V,I,None)

	def savepng(self,filename):
		gloo.clear(color=True, depth=True)
		for name,value in self.regions.items():
			value[0]['view'] = self.view
			value[0].draw('triangles', value[1]) # draw the colorbar
		for name,value in self.neurons.items():
			value[0]['view'] = self.view
			value[0].draw('lines',value[1]) # draw the colorbar
		np_array = gloo.wrappers.read_pixels()
		np_arr2=np_array.copy()

		np_arr2.flags.writeable = True
		np_arr2[:,:,3]=255
		img = Image.fromarray(np_arr2)
		img=img.resize((int(np_arr2.shape[1]/2),int(np_arr2.shape[0]/2)),resample=1)
		io.image.imsave(filename,np.array(img))

	def on_draw(self, event):

		gloo.clear(color=True, depth=True)
		for name,value in self.regions.items():
			value[0]['view'] = self.view
			value[0].draw('triangles', value[1]) # draw the colorbar

		for name,value in self.neurons.items():
			value[0]['view'] = self.view
			value[0].draw('lines',value[1]) # draw the colorbar

	def on_resize(self, event):
		self.activate_zoom()
		self.update()

	def activate_zoom(self):
		gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
		self.projection = perspective(60.0, self.size[0] / float(self.size[1]),1.0, 150000.0)

		for name,region in self.regions.items():
			region[0]['projection'] = self.projection
		for name,neuron in self.neurons.items():
			neuron[0]['projection'] = self.projection
	def on_mouse_move(self, event):
		if event.button==1:
			self.phi += -self.prepos[1]+event.pos[1]
			self.theta +=  -self.prepos[0]+event.pos[0]
			translatemat =translate(self.center)
			rotatex = rotate(self.theta/1.0, (1, 0, 0))
			rotatey = np.dot(rotate(self.phi/1.0, (0, 1, 0)),rotatex)
			translatemat2 =translate(-self.center)

			scalemat = scale((self.scale,self.scale,self.scale))
			self.model = np.dot(np.dot(translatemat2,rotatey),scalemat)

			for name,region in self.regions.items():
				region[0]['model'] = self.model
			for name,neuron in self.neurons.items():
				neuron[0]['model'] = self.model
			self.update()
		self.prepos=event.pos

		pass

	def on_mouse_wheel(self,event):
		# print(event._delta)

		rotatex = rotate(self.theta/1.0, (1, 0, 0))
		rotatey = np.dot(rotate(self.phi/1.0, (0, 1, 0)),rotatex)
		translatemat2 =translate(-self.center)
		if event._delta[1]>0:
			self.scale=self.scale*1.1
		else:
			self.scale=self.scale*0.9
		scalemat = scale((self.scale,self.scale,self.scale))
		self.model = np.dot(np.dot(translatemat2,rotatey),scalemat)
		for name,region in self.regions.items():
			region[0]['model'] = self.model
		for name,neuron in self.neurons.items():
			neuron[0]['model'] = self.model
		self.update()
		pass

if __name__=="__main__":
	win = Render()
	# win.show()
	# win
	app.run()
	# pass