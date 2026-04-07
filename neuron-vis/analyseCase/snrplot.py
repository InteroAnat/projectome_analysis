#%%

from pathlib import Path

directory = '../resource/SNR'
sampleneurons={}
allneurons =[]
path = Path(directory)
for subpath in path.iterdir():
    # print(subpath)
    if subpath.is_dir():
        neurons=[]

        for file in subpath.iterdir():
            print(file)
            neurons.append(file)
            allneurons.append(file)
        sampleneurons[subpath]=neurons


# %%  sample
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

samplepair={}
for sample,neurons in sampleneurons.items():
    print(sample)
    allneurondf=[]
    for neuron in neurons:
        df = pd.read_csv(neuron,sep=' ',header=None,usecols=range(9))
        df.columns=['x','y','z','dis','fore','back','sd','sbr','snr']
        allneurondf.append(df)
# 输出txt文档只有前面九列是有意义的，分别是
# 采样点的x坐标，y坐标，z坐标，采样点到胞体的距离，
# 数据块的前景信号强度，背景信号强度，背景信号的标准差，信背比，信噪比

    sampley=[]
    neurondftmp=pd.DataFrame()
    for neurondf in allneurondf:
        if len(neurondftmp)==0:
            neurondftmp=neurondf
        else:
            neurondftmp = pd.concat([neurondftmp,neurondf],ignore_index=True)
    dist=np.array(neurondftmp['dis'])
    foreS=np.array(neurondftmp['fore'])
    backS=np.array(neurondftmp['back'])
    maxdis = round(dist.max()/1000)
    snr=np.array(neurondftmp[neurondftmp['back']>0]['snr'])
    print(snr.mean())

    samplex=np.linspace(0,maxdis,maxdis*2+1)
    print(samplex)
    for x in samplex:

        dftmp = neurondftmp[neurondftmp['dis']>x*1000]
        dftmp = dftmp[dftmp['dis']<(x+0.5)*1000]
        sampley.append(dftmp['snr'].median())
    
    # sampley.append(0)
    print(len(sampley))
    samplex=samplex[np.array(sampley)>0]+0.25
    sampley=np.array(sampley)[np.array(sampley)>0]
    fig,ax=plt.subplots()
    ax.set_title(str(sample)[-6:]+' SNR')
    ax.set_yscale('log')
    # plt.plot(samplex,sampley)
    l1=ax.scatter(dist/1000,(foreS),s=1,label='Signal')
    l2=ax.scatter(dist/1000,(backS),s=1,label='Background')
    plt.xlabel('Distance from the soma(mm)')
    plt.ylabel('Grayscale value')
    # plt.legend()

    ax2 = ax.twinx()
    
    plt.ylim(0.01, 100000)
    samplepair[sample]=[samplex,sampley]
    l3,=ax2.plot(samplex,sampley,'r',label='Median SNR')
    ax2.set_yscale('log')
    g = [l1,l2,l3]
    
    labels = [l.get_label() for l in g]
    plt.legend(g,labels)

    plt.savefig('../resource/svg/snr'+str(sample)[-6:]+'.pdf',format='pdf')
    plt.show()
    # break
fig,ax=plt.subplots()
plt.ylim(0.1, 10000)
plt.xlabel('Distance from the soma(mm)')
plt.ylabel('SNR')
for sample,s in samplepair.items():
    l3,=ax.plot(s[0],s[1],label=str(sample)[-6:])
    ax.set_yscale('log')
plt.legend()


plt.savefig('../resource/svg/snrallsample.pdf',format='pdf')
plt.show()

# %%  neuron
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



samplepair={}

for neuron in allneurons:
    print(neuron)
    allneurondf=[]
    
    df = pd.read_csv(neuron,sep=' ',header=None,usecols=range(9))
    df.columns=['x','y','z','dis','fore','back','sd','sbr','snr']
    allneurondf.append(df)
# 输出txt文档只有前面九列是有意义的，分别是
# 采样点的x坐标，y坐标，z坐标，采样点到胞体的距离，
# 数据块的前景信号强度，背景信号强度，背景信号的标准差，信背比，信噪比

    sampley=[]
    neurondftmp=pd.DataFrame()
    for neurondf in allneurondf:
        if len(neurondftmp)==0:
            neurondftmp=neurondf
        else:
            neurondftmp = pd.concat([neurondftmp,neurondf],ignore_index=True)
    dist=np.array(neurondftmp['dis'])
    foreS=np.array(neurondftmp['fore'])
    backS=np.array(neurondftmp['back'])
    maxdis = round(dist.max()/1000)

    samplex=np.linspace(0,maxdis,maxdis*2+1)
    print(samplex)
    for x in samplex:

        dftmp = neurondftmp[neurondftmp['dis']>x*1000]
        dftmp = dftmp[dftmp['dis']<(x+0.5)*1000]
        sampley.append(dftmp['snr'].median())
    snrdf =neurondftmp[neurondftmp['snr']>0.1]['snr']
    snrdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    snrdf.dropna(inplace=True)
    # sampley.append(0)
    print(len(sampley))
    samplex=samplex[np.array(sampley)>0]+0.25
    sampley=np.array(sampley)[np.array(sampley)>0]
    fig,ax=plt.subplots()
    ax.set_title(str(neuron)[-14:-8]+str(neuron)[-7:-4]+' SNR')
    ax.set_yscale('log')
    # plt.plot(samplex,sampley)
    l1=ax.scatter(dist/1000,(foreS),s=1,label='Signal')
    l2=ax.scatter(dist/1000,(backS),s=1,label='Background')
    plt.xlabel('Distance from the soma(mm)')
    plt.ylabel('Grayscale value')
    # plt.legend()

    ax2 = ax.twinx()
    
    plt.ylim(0.01, 100000)
    samplepair[str(neuron)[-14:-8]+str(neuron)[-7:-4]]=[samplex,sampley,snrdf]
    l3,=ax2.plot(samplex,sampley,'r',label='Median SNR')
    ax2.set_yscale('log')
    g = [l1,l2,l3]
    
    labels = [l.get_label() for l in g]
    plt.legend(g,labels)

    plt.savefig('../resource/svg/snr'+str(neuron)[-14:-8]+str(neuron)[-7:-4]+'.pdf',format='pdf')
    plt.show()
    # break

# %%
fig,ax=plt.subplots()
plt.ylim(0.1, 10000)
plt.xlabel('Distance from the soma(mm)')
plt.ylabel('SNR')
snrdata=[]
snrlabels=[]
for neuron,s in samplepair.items():
    l3,=ax.plot(s[0],s[1],label=str(neuron))
    snrdata.append(s[1])
    snrlabels.append(neuron)
    ax.set_yscale('log')
# ax.boxplot(snrdata,labels=snrlabels,sym='')
# plt.xticks(rotation=45)
plt.legend()


plt.savefig('../resource/svg/snrallneurons.pdf',format='pdf')
plt.show()

# %%
