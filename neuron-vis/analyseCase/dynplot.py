#%%
import pandas as pd


# 加载xlsx文件
Nmur2df = pd.read_excel('../resource\Table 7 Projection intensity of brain areas.xlsx',sheet_name='Nmur2')
Pvalbdf = pd.read_excel('../resource\Table 7 Projection intensity of brain areas.xlsx',sheet_name='Pvalb')
Qrfprdf = pd.read_excel('../resource\Table 7 Projection intensity of brain areas.xlsx',sheet_name='Qrfpr')
Sncgdf = pd.read_excel('../resource\Table 7 Projection intensity of brain areas.xlsx',sheet_name='Sncg')
credf={'Nmur2':Nmur2df,'Pvalb':Pvalbdf,'Qrfpr':Qrfprdf,'Sncg':Sncgdf}
# %%
regions=['Acs5','ACVII','AMB','CU','ECU','FOTU','GR','IC','IntG','IO','IRN','LC','LRN','MARN','MDRN','MEV',
'MT','MY-sen','NLL','NR','NTB','NTS','Pa5','PARN','PAS','PB','PC5','PGRN','PIL','PP','PPY','PSV','RM','RO','RPA',
'RR','SLD','SOC','SUT','TRN','TT','VeCB','VII','VTA','x','ZI']

credf={'Nmur2':Nmur2df,'Pvalb':Pvalbdf,'Qrfpr':Qrfprdf,'Sncg':Sncgdf}
credf2={}
for crename,df in credf.items():
    df2 = df.T
    df2.columns=df2.iloc[0]
    df2=df2.drop(df2.iloc[0].name)
    df2 = df2[regions]
    credf[crename]=df2
    credict={}
    for region  in regions:
        rvalues=0
        lvalues=0
        rcount=0
        lcount=0
        regionsize=0
        if df2[region].iloc[3,0]=='Left (Contralateral)':
            lvalues = df2[region].iloc[0,0]
            lcount = df2[region].iloc[1,0]
        else:
            rvalues = df2[region].iloc[0,0]
            rcount = df2[region].iloc[1,0]
        if df2[region].iloc[3,1]=='Left (Contralateral)':
            lvalues = df2[region].iloc[0,1]
            lcount = df2[region].iloc[1,1]
        else:
            rvalues = df2[region].iloc[0,1]
            rcount = df2[region].iloc[1,1]


        regionsize = df2[region].iloc[2,0]


        print(rvalues,rcount,regionsize)
        credict[region]=[rvalues,rcount,lvalues,lcount,rvalues+lvalues,rcount+lcount,regionsize]
    credf2[crename]=credict

# %%

import matplotlib.pyplot as plt
x=[]
y=[]
sizer=[]
sizel=[]
size=[]
colors=[]
colorsl=[]
colorsr=[]
indexx=0
for crename ,credict in credf2.items():
    indexy=0
    dftemp = pd.DataFrame(credict)
    print(dftemp)
    for region in regions:
        x.append(indexx)
        y.append(len(regions)-indexy-1)
        sizer.append(credict[region][1]/dftemp.iloc[5].max()/credict[region][6])
        sizel.append(credict[region][3]/dftemp.iloc[5].max()/credict[region][6])
        size.append(credict[region][5]/dftemp.iloc[5].max()/credict[region][6])
        colorsr.append(credict[region][0]/dftemp.iloc[4].max()/credict[region][6])
        colorsl.append(credict[region][2]/dftemp.iloc[4].max()/credict[region][6])
        colors.append(credict[region][4]/dftemp.iloc[4].max()/credict[region][6])
        indexy=indexy+1
    indexx=indexx+1

# %%
import numpy as np
size = np.array(size)
sizel = np.array(sizel)
sizer = np.array(sizer)
colors = np.array(colors)
colorsl = np.array(colorsl)
colorsr = np.array(colorsr)
normmax=1#size.max()
size=size/normmax*1
sizel=sizel/normmax*1
sizer=sizer/normmax*1
size[size<10]=0
sizel[sizel<10]=0
sizer[sizer<10]=0
colors =colors/colors.max()
colorsl =colorsl/colorsl.max()
colorsr =colorsr/colorsr.max()
print(size)
print(sizel)
print(sizer)

s1= plt.scatter(4,1,s=100/5*1,zorder=2,c='black')
s2= plt.scatter(4,1,s=100/5*2,zorder=2,c='black')
s3= plt.scatter(4,1,s=100/5*3,zorder=2,c='black')
s4= plt.scatter(4,1,s=100/5*4,zorder=2,c='black')
s5= plt.scatter(4,1,s=100,zorder=2,c='black')
fig,ax=plt.subplots(figsize=(4,15))
cmap = plt.cm.Blues

plt.scatter(x, y,s=size,c=colors,cmap=cmap)
plt.colorbar()
plt.yticks(range(len(regions)),list(reversed(regions)))
plt.xticks(range(0,4),list(credf.keys()),rotation=45)
ax.set_title('Total')

plt.legend((s1,s2,s3,s4,s5),('1','2','3','4','5') ,loc = 'best')
plt.savefig('../resource/svg/dyntotle.pdf',format='pdf')
plt.show()

fig,ax=plt.subplots(figsize=(4,15))
plt.scatter(x, y,s=sizel,c=colorsl,cmap=cmap)
plt.colorbar()
plt.yticks(range(len(regions)),list(reversed(regions)))
plt.xticks(range(0,4),list(credf.keys()),rotation=45)
ax.set_title('Left')
plt.legend((s1,s2,s3,s4,s5),('1','2','3','4','5') ,loc = 'best')

plt.savefig('../resource/svg/dynleft.pdf',format='pdf')
plt.show()

fig,ax=plt.subplots(figsize=(4,15))
plt.scatter(x, y,s=sizer,c=colorsr,cmap=cmap)
plt.colorbar()
plt.yticks(range(len(regions)),list(reversed(regions)))
plt.xticks(range(0,4),list(credf.keys()),rotation=45)
ax.set_title('Right')

plt.legend((s1,s2,s3,s4,s5),('1','2','3','4','5') ,loc = 'best')

plt.savefig('../resource/svg/dynright.pdf',format='pdf')
plt.show()
# %%
