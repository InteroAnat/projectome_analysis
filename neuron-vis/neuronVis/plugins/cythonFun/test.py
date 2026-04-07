import pyximport 
pyximport.install()
# help(pyximport)
# import sys,os
# from importlib.machinery import ExtensionFileLoader
# # pyd_path就是my_module.pyd所在的绝对路径
# pyd_path = os.path.abspath("C:/Users/xfwang/Documents/workspace/neuron-vis/neuronVis/plugins/cythonFun/foo.pyd")
# foo = ExtensionFileLoader('foo', pyd_path).load_module()
# import ctypes

# foo = ctypes.cdll.LoadLibrary(r"C:/Users/xfwang/Documents/workspace/neuron-vis/neuronVis/plugins/cythonFun/build/lib.win-amd64-3.7/foo.pyd")
# sys.path.append("C:/Users/xfwang/Documents/workspace/neuron-vis/neuronVis/plugins/cythonFun/")
import foo
import numpy as np
# example.py
def fib(A,B):
    c=A-B
    d=c*c
    dist = np.sum(d,axis=1)
    iVertex = np.argmin(dist)
    return iVertex
    
a = np.random.random((300000,3))
b=[0.5,0.5,0.5]
print(a)


# print(foo.foo(a,1))
print(foo.foo(a,b))
