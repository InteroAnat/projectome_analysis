import shutil
import subprocess
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
def swc2fnt(filename):

    command='fnt-from-swc.exe '+filename+' '+ filename+'.fnt'
    print(command)
    execute_command(command)
    
def region_swc2fnt(data_frame,region):
     swc_names=data_frame.loc[data_frame['Soma_Region']==region]['NeuronID'].tolist()
     sample_id=data_frame.loc[data_frame['Soma_Region']==region]['SampleID'].tolist()[0]
     swc_address='../resource/swc_raw/'+sample_id
     os.makedirs(swc_address+'/'+region,exist_ok=True)
     for swc_index in swc_names:
         swc_file=swc_address+'/'+swc_index
         
         swc2fnt(swc_file)
         shutil.move(swc_file+'.fnt',swc_address+'/'+region+'/')

def swc_fnt_tracer

#%%
swc2fnt('../resource/swc_raw/251637/157.swc')
    
# %%
