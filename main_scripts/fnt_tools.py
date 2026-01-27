"""
fnt_tools.py - FNT (Functional Neuroanatomy Toolbox) Utilities

Version: 2.1.0 (2026-01-27)
Author: [Your Name]

Features:
- SWC to FNT conversion helpers
- Direct FNT file opening with neuron/sample IDs
- FNT tool integration

See CHANGELOG.md for detailed version history.
"""

#%%
import shutil
import subprocess
import neuro_tracer as nt
import sys
import os

neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
if neurovis_path not in sys.path:
    sys.path.append(neurovis_path)
import IONData
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
    
def fnt_decimate (swc_filename):
    command = 'fnt-decimate'+ ' -d 5000 -a 5000 '+ swc_filename + ' ' + swc_filename+'.decimate.fnt'
    execute_command (command)


def fnt_join (fnt_filenames, output_filename):
    command = 'fnt-join'+' '+ fnt_filenames + ' -o ' + output_filename
    execute_command(command)
    
def update_fnt_names_in_directory(target_directory):
    """
    Scans a specific directory for *.decimate.fnt files and updates 
    their last line to match the filename.
    
    Args:
        target_directory (str): Path to the folder containing .fnt files.
    """
    
    # 1. Validate Directory
    if not os.path.exists(target_directory):
        print(f"Error: Directory not found: {target_directory}")
        return

    print(f"--- Updating FNT names ---")
    print(f"Target: {target_directory}")
    
    # 2. Find files
    files = [f for f in os.listdir(target_directory) if f.endswith('.decimate.fnt')]
    
    if not files:
        print("No .decimate.fnt files found.")
        return

    print(f"Found {len(files)} files to check.")
    updated_count = 0

    # 3. Process Loop
    for filename in files:
        file_path = os.path.join(target_directory, filename)
        
        # Extract Neuron Name
        # Logic: "156.swc.decimate.fnt" -> "156.swc"
        neuron_name = filename.replace('.decimate.fnt', '')
        
        try:
            # Read
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                continue

            # Prepare new footer
            # Format: "0 <NeuronName>" (Standard FNT requirement)
            new_last_line = f"0 {neuron_name}\n"
            
            # Check and Update
            # (We use strip() comparison to ignore whitespace diffs, then write clean version)
            if lines[-1].strip() != new_last_line.strip():
                lines[-1] = new_last_line
                
                # Write Back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                updated_count += 1
                
        except Exception as e:
            print(f"  [Error] {filename}: {e}")

    print(f"Done. Updated {updated_count} files.\n")



#%%
# swc2fnt('../resource/swc_raw/251637/157.swc')
    



#%%
swc_fnt_tracer("251637",'003.swc')

#%%
# swc2fnt('../resource/swc_raw/251637/156.swc')

# fnt_decimate('../resource/swc_raw/251637/156.swc.fnt')
# # %%
# fnt_join('D:\projectome_analysis\main_scripts\processed_neurons\251637\fnt_processed\*.fnt','joined_fnt.fnt')
# # %%
# $fnt_component_dir/fnt-join $data_dir/*.decimate.fnt -o "$home_dir/fnt_decimate_join.fnt"
update_fnt_names_in_directory(r'D:\projectome_analysis\main_scripts\processed_neurons\251637\fnt_processed')
# %%
# workflow
# 1. fnt