"""Additional original graph passage and native optical planes; no boutons claimed."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("build",HERE/"build_expanded_native_review.py")
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
m=b.helper
record=next(r for r in json.loads((HERE/"assessment_inputs.json").read_text()) if r["NeuronUID"]=="252790::037.swc")
n=np.asarray(m.parse_swc((m.ROOT/record["native_swc"]["path"]).read_text()),float)
w=np.asarray(m.parse_swc((m.ROOT/record["atlas_swc"]["path"]).read_text()),float)
a=np.asarray(m.nib.load(m.ATLAS).dataobj)[...,0,5]
by,children,leaves=m.graph_information(n)
labels=m.atlas_labels(w,a)
options=[]
for nodes in m.target_components(n,labels,229):
    st=set(nodes)
    if st & leaves or any(sum(c in st for c in children[x])>=2 for x in nodes):
        continue
    covered=[x for x in nodes if m.cube_path("252790",m.cube_index(by[x][2:5])).is_file()]
    if covered:
        options.append((nodes,covered))
nodes,covered=sorted(options,key=lambda p:(-len(p[0]),min(p[0])))[0]
selected=covered[len(covered)//2]
st=set(nodes)
start=next(x for x in nodes if int(by[x][6]) not in st)
exits=[(x,c) for x in nodes for c in children[x] if c not in st]
if selected in leaves or len(children[selected])!=1 or not exits:
    raise ValueError("Expected original internal passage node with continuing graph")
point=by[selected][2:5]
cube=m.cube_index(point)
origin=np.asarray(cube)*m.BLOCK
image_path=m.cube_path("252790",cube)
image=tifffile.imread(image_path)
centre=np.floor((point-origin)/m.SPACING+.5).astype(int)
x0,x1=max(0,centre[0]-77),min(360,centre[0]+78)
y0,y1=max(0,centre[1]-77),min(360,centre[1]+78)
planes=list(range(max(0,centre[2]-4),min(90,centre[2]+5)))
lo,hi=np.percentile(image[planes,y0:y1,x0:x1],[1,99.7])
fig,axes=plt.subplots(4,3,figsize=(15,16),layout="constrained")
for ax,(x,y) in zip(axes[0],[(0,1),(0,2),(1,2)]):
    edges=[(int(by[z][6]),z) for z in nodes if int(by[z][6]) in st]
    ax.add_collection(LineCollection([np.asarray([by[u][2:5],by[v][2:5]])[:,[x,y]] for u,v in edges],colors="#536b8e"))
    coords=np.asarray([by[z][2:5] for z in nodes])
    ax.scatter(coords[:,x],coords[:,y],s=6,color="#536b8e")
    ax.scatter(point[x],point[y],facecolors="none",edgecolors="#ff654f",s=60)
    for u,v in [(int(by[start][6]),start)]+exits:
        ax.plot([by[u][2+x],by[v][2+x]],[by[u][2+y],by[v][2+y]],"--",color="#d9761e")
    ax.autoscale();ax.set_aspect("equal",adjustable="datalim")
    ax.set_xlabel(f"Native {'XYZ'[x]} (nominal µm)")
    ax.set_ylabel(f"Native {'XYZ'[y]} (nominal µm)")
    ax.set_title("Original entry/exit links dashed; no graph leaves",fontsize=10)
for ax,z in zip(axes[1:].flat,planes):
    ax.imshow(image[z,y0:y1,x0:x1],origin="lower",cmap="gray",vmin=lo,vmax=max(lo+1,hi),
              extent=[origin[0]+(x0-.5)*.65,origin[0]+(x1-.5)*.65,origin[1]+(y0-.5)*.65,origin[1]+(y1-.5)*.65])
    ax.scatter(point[0],point[1],s=45,facecolors="none",edgecolors="#ff654f",linewidths=.8)
    ax.set_title(f"Actual XY plane {z}; native Z={origin[2]+z*3:.1f} µm",fontsize=10)
    ax.set_xlabel("Native X (nominal µm)");ax.set_ylabel("Native Y (nominal µm)")
for ax in list(axes[1:].flat)[len(planes):]:
    ax.set_axis_off()
fig.suptitle(f"252790::037.swc | original internal PASSAGE node {selected}\n"
             +f"Declared ARM6 target: CL_granular_insula (229); {len(nodes)} nodes, zero original leaves, zero section branches\n"
             +f"Ring marks node XY only; node Z={point[2]:.2f} nominal µm. Passage does not exclude en passant boutons.",fontsize=12)
path=HERE/"review_panels/252790_037_internal_passage_optical_planes.png"
if path.exists():
    raise FileExistsError(path)
fig.savefig(path,dpi=140,bbox_inches="tight");plt.close(fig)
m.write_json(HERE/"passage_inputs.json",{
    "NeuronUID":"252790::037.swc","selected_internal_node_id":selected,
    "node_ids":sorted(nodes),"entry_link":[int(by[start][6]),start],"exit_links":exits,
    "full_graph_leaves":[],"section_branch_ids":[],"selected_original_children":children[selected],
    "selection_rule":"Longest target-local no-leaf/no-branch original component with cached nodes, lowest start-ID tie; middle covered node in original traversal",
    "native_swc":record["native_swc"],"atlas_swc":record["atlas_swc"],
    "source_image":{"path":m.relative(image_path),"sha256":m.sha(image_path),"cube_xyz":list(cube),"plane_indices":planes},
    "panel":{"path":m.relative(path),"sha256":m.sha(path)},
    "code":{"path":m.relative(Path(__file__)),"sha256":m.sha(Path(__file__))},
    "image_review_status":"awaiting_actual_inspection","bouton_synapse_state":"unassessed"})
print("passage",selected,"nodes",len(nodes),"entry",int(by[start][6]),start,"exit",exits)
