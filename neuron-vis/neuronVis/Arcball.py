import glm
import math
import glfw
class Arcball:
	def __init__(self,window_width, window_height, roll_speed = 1.0, x_axis = True, y_axis = True) -> None:
		self.windowWidth =window_width
		self.windowHeight = window_height
		self.mouseEvent=0
		self.rollSpeed=roll_speed
		self.angle=0.0
		self.prevPos=None
		self.currPos=None
		self.camAxis=glm.vec3(0.0, 1.0, 0.0)
		self.camAxisPre=glm.vec3(0.0, 0.0, 0.0)
		self.xAxis=x_axis
		self.yAxis=y_axis
		pass
	def resize(self,w,h):
		self.windowWidth=w
		self.windowHeight=h
	def toScreenCoord( self, x, y ):
		coord=glm.vec3(0.0)
		
		if self.xAxis:
			coord.x =  -(2 * x - self.windowWidth ) / self.windowWidth
		
		if self.yAxis:
			coord.y = (2 * y -self.windowHeight) / self.windowHeight

		
		coord.x = glm.clamp( coord.x, -1.0, 1.0 )
		coord.y = glm.clamp( coord.y, -1.0, 1.0 )
		
		length_squared = coord.x * coord.x + coord.y * coord.y
		if length_squared <= 1.0 :
			coord.z = math.sqrt( 1.0 - length_squared )
		else:
			coord = glm.normalize( coord )
		
		return coord

	def mouseButtonCallback(self,  window,  button,  action,  mods ):
		self.mouseEvent = ( action == glfw.PRESS and button == glfw.MOUSE_BUTTON_LEFT )

	def cursorCallback( self,window, x, y ):
		if  self.mouseEvent == 0 :
			return
		elif self.mouseEvent == 1:
			self.prevPos = self.toScreenCoord( x, y )
			self.mouseEvent  = 2
		self.currPos  = self.toScreenCoord( x, y )
		self.angle    = math.acos( min(1.0, glm.dot(self.prevPos, self.currPos) ))
		if self.angle !=0:
			self.camAxis  = glm.cross( self.prevPos, self.currPos )


	def createViewRotationMatrix(self):
		return glm.rotate( glm.degrees(self.angle) * self.rollSpeed, self.camAxis )

	def createModelRotationMatrix( self, view_matrix ):
		axis =glm.mat4( view_matrix) * self.camAxis
		return glm.rotate( glm.degrees(self.angle) * self.rollSpeed, axis )