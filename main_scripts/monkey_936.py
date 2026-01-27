#%% import necessary libraries

import sys,copy,os,inspect

neurovis_path = os.path.abspath(r'D:\projectome_analysis\neuron-vis\neuronVis')
sys.path.append(neurovis_path)

import IONData 
iondata = IONData.IONData()

from collections import defaultdict


from pathlib import Path
import matplotlib

import matplotlib.pyplot as plt


import numpy as np
import nibabel as nib  # parse the NII data
from collections import deque
import pandas as pd
from typing import Tuple
from io import StringIO

import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.use('TkAgg') 
from scipy.cluster.hierarchy import dendrogram, linkage

import requests
from bs4 import BeautifulSoup
#%%

neuronlist=iondata.getNeuronListBySampleID('251637')

neurontable = pd.DataFrame(neuronlist)


neurontable['region'].value_counts(dropna=False)
neurontable['region'] = neurontable['region'].str.replace('\r', '',regex=False)
# Replace empty strings with 'unknown'
neurontable['region'] = neurontable['region'].replace('', 'unknown', regex=False)


# Get the distribution
region_distribution = neurontable['region'].value_counts(dropna=False)
print(region_distribution)
# Plot the distribution
plt.figure(figsize=(10, 6))
bars=region_distribution.plot(kind='bar', rot=45)
for bar in bars.patches:
    y_value = bar.get_height()
    x_value = bar.get_x() + bar.get_width() / 2
    # Annotate the bar with its value
    plt.text(x_value, y_value, f'{y_value}', ha='center', va='bottom', fontsize=10)
plt.title('Distribution of Neuron Regions for Monkey 936')
plt.xlabel('Region')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('../figures/neuron_region_distribution_936.png', dpi=300)

plt.show()
#%%
neuronRaw = iondata.getRawNeuronTreeBySampleID('251637')

# %%
import difflib
import os

# Path to the files
original_file_path = '../resource/swc_raw/251637/002.swc '
modified_file_path = '../resource/swc_raw/251637/002r.swc '

# Check if the files exist
if not os.path.exists(original_file_path):
    print(f"File not found: {original_file_path}")
    exit()

if not os.path.exists(modified_file_path):
    print(f"File not found: {modified_file_path}")
    exit()

# Read the contents of the files
with open(original_file_path, 'r') as f1, open(modified_file_path, 'r') as f2:
    original_lines = f1.readlines()
    modified_lines = f2.readlines()

# Generate unified diff
diff = difflib.unified_diff(original_lines, modified_lines, fromfile='original_file.txt', tofile='modified_file.txt', lineterm='')

# Print the differences
for line in diff:
    print(line, end='')

# Print the current working directory to verify
print("\nCurrent Working Directory:", os.getcwd())


# %%
