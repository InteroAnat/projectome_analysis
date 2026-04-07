# %%
import nrrd
imdata,iminfo = nrrd.read('./resource/boundlaplace20_v2.nrrd')

# %%
imdata.shape
for i in range(286,570):
    imdata[:,:,i]=imdata[:,:,570-i]
    
# %%
nrrd.write('./resource/boundlaplace20_v3.nrrd',imdata)
