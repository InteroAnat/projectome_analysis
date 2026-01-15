#%%
import shutil
import subprocess
import neuro_tracer as nt
import sys
neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)
import IONData


#%%
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

def swc_fnt_tracer(sample_id,neuron_id):
    swc_file_address ='../resource/swc_raw/'+sample_id+'/'+neuron_id
    print('working on',swc_file_address)
    fnt_file_address=swc_file_address+'.fnt'
    print(fnt_file_address)
    catalog_address = f"http://bap.cebsit.ac.cn/monkeydata/{sample_id}/catalog"
    iondata=IONData.IONData()
    iondata.getRawNeuronTreeByID(sample_id,neuron_id)
    swc2fnt(swc_file_address)
    with open(fnt_file_address) as file:
        lines= file.readlines()
        
    lines[1] = catalog_address + '\n'
    with open(fnt_file_address, 'w') as file:
        file.writelines(lines)
    command = 'fnt-tracer '+fnt_file_address
    execute_command(command)
#%%
swc2fnt('../resource/swc_raw/251637/157.swc')
    
# %%
with open('../resource/swc_raw/251637/157.swc', 'r') as file:
    # Read the content of the file
    content = file.read()
    content.
    # Process the content as needed
    print(content)
# %%
def read_specific_lines(file_path, line_numbers):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract specific lines
    selected_lines = [lines[i-1] for i in line_numbers if i-1 < len(lines)]
    
    return selected_lines

read_specific_lines("../resource/swc_raw/251637/157.swc.fnt",[0])
# %%
sample_id=251637
catalog_address = f"10.10.48.110/monkey_data/{sample_id}/catalog"
print(catalog_address)
# %%
swc_fnt_tracer("251637",'001.swc')

# %%   
iondata=IONData.IONData() 
iondata.getannotation()


# %%
sample_id='22'
catalog_address = f"10.10.48.110/monkey_data/{sample_id}/catalog"

# %%
catalog_address