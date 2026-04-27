"""Probe IONData getSampleInfo and SWC URLs to debug bulk_visual fetch."""
import sys, os, requests
sys.path.insert(0, r"D:\projectome_analysis\neuron-vis\neuronVis")
import IONData

ion = IONData.IONData()
for sid in ["251637", "251730", "252383", "252384", "252385"]:
    info = ion.getSampleInfo(sid)
    if info:
        proj = info[0].get("project_id")
        print(f"{sid}: project_id={proj}")
        keys = list(info[0].keys())
        print(f"  available keys: {keys}")

        nlist = ion.getNeuronListBySampleID(sid)
        if nlist:
            n0 = nlist[0]
            print(f"  first neuron entry: {n0}")
            nid = n0.get("name")
            url = f"http://10.10.31.31/swc/newswc/{proj}/{sid}/swc_raw/{nid}"
            try:
                r = requests.get(url, timeout=10)
                print(f"  raw URL status={r.status_code}, content_len={len(r.text)}")
                if r.status_code == 200:
                    print(f"  preview: {r.text[:120]!r}")
            except Exception as e:
                print(f"  raw URL ERR: {e}")
            # Also try the non-raw / processed URL via getNeuronByID API
            try:
                txt = ion.getNeuronByID(sid, nid)
                print(f"  getNeuronByID len={len(txt) if txt else 0}, "
                      f"preview={(txt[:120] if txt else '')!r}")
            except Exception as e:
                print(f"  getNeuronByID ERR: {e}")
    else:
        print(f"{sid}: no info")
    print()
