def same_lines_unordered(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        set1 = set(line.strip() for line in f1)
        set2 = set(line.strip() for line in f2)
    return set1 == set2
result = same_lines_unordered(r'D:\projectome_analysis\251637\001.swc.fnt.decimate.fnt', r'D:\projectome_analysis\main_scripts\processed_neurons\251637\001.swc.fnt.decimate.fnt')
print("Differences:\n", result)
#%%


# %%
