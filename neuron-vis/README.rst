
# neuronVis


----------------------------------------------------------------
 visualization and analyse library for mouse neuron based on vispy 
----------------------------------------------------------------

* Author: XiaoFei WANG
* Contact: xfwang@ion.ac.cn
* Date: 27 April 2022
* Copyright: `GFDLv1.2+ <http://www.gnu.org/licenses/fdl.html>`_
* Version: 1.0

neuronVis is a tool for mouse neuron visualization which is based on vispy(gloo).

USAGE
======

git clone https://gitee.com/bigduduwxf/neuron-vis.git

Download the resource from http://10.10.31.25/source/neuronVis/resource.rar and decompressed to the directory neuronvis 

<<<<<<< HEAD
### Get swc and property form iondata by sampleid and neuron name




    iondata =IONData()
    neuronlist=iondata.getNeuronListBySampleID('192106')
    swc = iondata.getNeuronByID('192106', '001.swc')
    pro = iondata.getNeuronPropertyByID('192106', '001.swc')
    loader = swcloader.NeuronTree()
    loader.readSWC(swc)
    print(pro)
    
### Render region and swc


=======
.. code:: bash
>>>>>>> 4d92fd3 (line width)



    import neuronVis as nv

    neuronvis = nv.neuronVis()
    # neuronvis.render.setBackgroundColor((0.0,0.20,0.5,1.0))
    neuronvis.addRegion('CA2',[0.5,1.0,0.5])
    neuronvis.addNeuron('./resource/033.swc')
    neuronvis.render.savepng('resource/test.png')
    neuronvis.render.setView('posterior')
    # neuronvis.render.setLookAt()
    neuronvis.addNeuron('./resource/192092-012.swc',[1.0,1.0,0.0])
    neuronvis.addNeuronByID('192106','011.swc',[1.0,1.0,0.0])
    neuronvis.render.savepng('resource/test2.png')

    neuronvis.render
    nv.app.run()

![test2 截图](https://gitee.com/bigduduwxf/neuron-vis/raw/master/resource/test2.png)


Authors
=======

*XiaoFei Wang <xfwang@ion.ac.cn>

