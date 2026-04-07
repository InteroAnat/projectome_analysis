import json,time
import random
from pathlib import Path
import subprocess
import IONData
import shutil
import json
import os
import pandas as pd
def execute_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    output = output.decode('utf-8')
    error = error.decode('utf-8')
    
    if output:
        print("Output:")
        print(output)
    
    if error:
        print("Error:")
        print(error)
def createScene(neuronlist,filename,color=None):
    # f=open('../resource/template.nv', encoding='gbk')

    # templatescene =  json.load(f)

    test={}
    test["Comment"]="Undefined"
    test["Datetime"]=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    neurons=[]
    for neuron in neuronlist:
        neu={}
        if color is None and 'color' not in neuron.keys():
            neu["color"]={"r":str(random.randint(0, 255)),"g":str(random.randint(0, 255)),"b":str(random.randint(0, 255))}
        elif 'color' in neuron.keys():
            neu["color"]=neuron['color']
            # neu["color"]={"r":str(neuron['color'][0]),"g":str(neuron['color'][1]),"b":str(neuron['color'][2])}

        else:
            neu["color"]=color
        if 'mirror' not in neuron.keys():
            neu['mirror']=0 # 0 or 1 
        else:
            neu['mirror']=neuron['mirror'] # 0 or 1 
        neu["name"]=neuron['name']
        neu["sampleid"]=neuron['sampleid']
        if 'soma' in neuron.keys():
            neu["soma"]=neuron['soma']
        neurons.append(neu)
    test["Neurons"]=neurons
    with open (filename,'w') as f:
        json.dump(test,f)
    
def scene2List(filename):
    f=open(filename, encoding='gbk')

    scene = json.load(f)
    neurons=scene['Neurons']
    return neurons
def swc2fnt(filename):

    command='fnt-from-swc.exe '+filename+' '+ filename+'.fnt'
    print(command)
    execute_command(command)
def joinScene(filename):

    neurons = scene2List(filename)

    for neuron in neurons:
        name = '../resource/swc/'+neuron['sampleid']+'/'+neuron['name']
        swc2fnt(name)
    command='fnt-join.exe '
    for neuron in neurons:
        name = '../resource/swc/'+neuron['sampleid']+'/'+neuron['name']
        fntname=name+'.fnt'
        command+=' '+fntname
    command+=' -o '+filename+'join.fnt'
    print(command)
    execute_command(command)
def decimate(filename,dist=160):
    command='fnt-decimate.exe -d '+str(dist)+' -a '+str(dist)+' '+filename+' '+ filename+'decimate.fnt'
    print(command)
    execute_command(command)

def copy(filename,dst):
    shutil.copy(filename,dst)
def updateRadius(filename,r=0):
    file=Path(filename)
    if  file.exists():

            swccontext=''
            with open(file) as f:
                swccontext=f.read()
                f.close()
            with open(file,'w+') as f:
                swclines=swccontext.split('\n')
                for line in swclines:
                    if len(line)==0 or line[0] == '#'or line[0] == '\r':
                        continue
                    p=list(map(float,(line.split())))
                    p[5]='0.0'
                    line=str(int(p[0]))+' '+str(int(p[1]))+' '+str(p[2])+' '+str(p[3])+' '+str(p[4])+' '+str(p[5])+' '+str(int(p[6]))+'\n'
                    f.write(line)
                f.close()   
    pass
def mergeScene(filename,dir='D:/projectome_analysis/neuron-vis/resource/swc_merge/subtype/'):
    neurons = scene2List(filename)[1:50]
    iondata =IONData.IONData()
    if not os.path.exists(dir):
            os.makedirs(dir)
    for neuron in neurons:
        
        iondata.getNeuronByID(sampleid=neuron['sampleid'],neuronid=neuron['name'])
        srcname = r'D:/projectome_analysis/neuron-vis/resource/swc/'+neuron['sampleid']+'/'+neuron['name']
        dstname = dir+neuron['sampleid']+'-'+neuron['name']
        result = Path(dstname)
        if not result.exists():
            copy(srcname,dstname)
            updateRadius(dstname)
        name = dir+neuron['sampleid']+'-'+neuron['name']
        result = Path(name+'.fnt')
        if not result.exists():
            swc2fnt(name)
            decimate(name+'.fnt',)
    groupLength=int(len(neurons)/2)
    group1=neurons[0:groupLength]
    group2=neurons[groupLength:2*groupLength]
    output = '../resource/swc_merge/'
    outputpath = dir+'merge/'
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    for i in range(groupLength):
        command='fnt-merge.exe '
        neuron1= group1[i]
        neuron2= group2[i]
        name1 = dir+neuron1['sampleid']+'-'+neuron1['name']
        fntname1=name1+'.fntdecimate.fnt'
        name2 = dir+neuron2['sampleid']+'-'+neuron2['name']
        fntname2=name2+'.fntdecimate.fnt'
        command=command+' '+fntname1+' '+fntname2
        outfile=outputpath+str(i)+'merge.fnt'
        command+=' -o '+ outfile
        print(command)
        if not os.path.exists(outfile):
            execute_command(command)
    level=1
    while(int(groupLength/2)>0):
        groupLength =int(groupLength/2)
        for i in range(groupLength):
            print(str(level)+'-'+str(i))
            command='fnt-merge.exe '
            tag=str(level-1)+'-'
            if level==1:
                tag=''
            name1 = outputpath+tag+str(i*2)+'merge.fnt'
            name2 = outputpath+tag+str(i*2+1)+'merge.fnt'
            outfile=outputpath+str(level)+'-'+str(i)+'merge.fnt'
            command=command+' '+name1+' '+name2+' -o '+ outfile
            print(command)
            if not os.path.exists(outfile):
                result = execute_command(command)
                if result is None:
                    decimate(name1,400*pow(2,level))
                    decimate(name2,400*pow(2,level))
                    name1= outputpath+tag+str(i*2)+'merge.fntdecimate.fnt'
                    name2= outputpath+tag+str(i*2+1)+'merge.fntdecimate.fnt'
                    command='fnt-merge.exe '
                    command=command+' '+name1+' '+name2+' -o '+ outfile
                    print(command)
                    result = execute_command(command)

        level+=1
if __name__=="__main__":


    mergeScene('test.nv')
    
    pass
# %%
