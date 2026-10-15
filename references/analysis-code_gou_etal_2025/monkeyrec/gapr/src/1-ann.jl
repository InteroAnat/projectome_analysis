#***Template /data/dat/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz
#***Surf  /data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/atlases/CHARM/Level_4/CHARM_4.V4.k234.gii
function _get_nmt_data(pos)
    global __nmt_charm_sarm_dat
    if !@isdefined(__nmt_charm_sarm_dat) || __nmt_charm_sarm_dat === nothing
        @info "loading NMT img"
        charm = niread("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/CHARM_in_NMT_v2.0_sym.nii.gz")
        sarm = niread("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/SARM_in_NMT_v2.0_sym.nii.gz")
        __nmt_charm_sarm_dat = (charm, sarm)
    end
    pos = (pos[1], pos[2], pos[3])
    ii = map(v -> round(Int64, v / 250.0 + 0.5), pos)
    ss = size(__nmt_charm_sarm_dat[1])
    for d in 1:3
        (ii[d] >= 1 && ii[d] <= ss[d]) || return (0, (zeros(Int16, 6)...,))
    end
    cc = (__nmt_charm_sarm_dat[1].raw[ii..., 1, 1:6]...,)
    ss = (__nmt_charm_sarm_dat[2].raw[ii..., 1, 1:6]...,)
    ccc = maximum(cc)
    sss = maximum(ss)
    if ccc == 0
        sss != 0 || return (0, ss)
        return (2, ss)
    end
    sss == 0 || println("double ann $ss/$cc")
    (1, cc)
end
function _get_nmt_annot()
    global __nmt_charm_sarm_tbl
    function get_table(fn)
        res = Dict{Int64,Tuple{String,String,Int64,Int64}}()
        tab = readdlm(fn, '\t', header=true)[1]
        for i in 1:size(tab)[1]
            id = tab[i, 1]
            sym = replace(tab[i, 2], "/" => "|")
            ful = tab[i, 3]
            f = tab[i, 4]
            t = tab[i, 5]
            res[id] = (sym, ful, f, t)
        end
        res
    end
    if !(@isdefined __nmt_charm_sarm_tbl) || __nmt_charm_sarm_tbl === nothing
        @info "loading NMT annot"
        charm = get_table("/data/dat/NMT_v2.0_sym/tables_CHARM/CHARM_key_all.txt")
        sarm = get_table("/data/dat/NMT_v2.0_sym/tables_SARM/SARM_key_all.txt")
        __nmt_charm_sarm_tbl = (charm, sarm)
    end
    __nmt_charm_sarm_tbl
end

function get_nmt_hier_int(id::Tuple{Int64,NTuple{6,Int16}})
    id[1] != 0 || return "/"
    ann = _get_nmt_annot()[id[1]]
    stk = []
    push!(stk, "")
    push!(stk, ["CHARM", "SARM"][id[1]])
    for i = 1:6
        if i > 1 && id[2][i] == id[2][i-1]
            continue
        end
        push!(stk, ann[id[2][i]][1])
    end
    join(stk, "/")
end
function get_nmt_hier(pos)
    id = _get_nmt_data(pos)
    get_nmt_hier_int(id)
end

function get_allen_tree(; rev=true)
    global _allen_ids_rev
    if !(@isdefined _allen_ids_rev) || _allen_ids_rev === nothing
        anns = _get_allen_annot()
        ids = collect(keys(anns))
        ids = ids[sortperm(map(v -> anns[v].order, ids), rev=true)]
        _allen_ids_rev = map(id -> get_allen_hier_int(id), ids)
    end
    deepcopy(_allen_ids_rev)
end

function get_ann_hier_par(ann_id)
    i = findlast('/', ann_id)
    par_id = "/"
    if i != 1
        par_id = ann_id[1:(i-1)]
    end
    par_id
end

function select_brain_areas(sel, bas)
    res = []
    for ba in bas
        hit = 0
        for s in sel
            if typeof(s) <: AbstractString
                if ba == s
                    hit += 1
                end
            else
                if startswith(s[1])(ba) &&
                   length(findall('/', ba[length(s[1]):end])) < s[2]
                    hit += 1
                end
            end
        end
        if hit > 0
            push!(res, ba)
        end
    end
    sort(res)
end

####################################
#function select_brain_areas(sel_bas)
#ann=_get_allen_annot()
#sel_bas=map(sel_bas) do ba
#if typeof(ba)<:AbstractString
#for i in eachindex(ann)
#if ann[i][3]==ba
#ba=(get_brain_area_hier(i), 1)
#break
#end
#end
#end
#ba
#end
#ann_ids=allen_ids_rev()
#ids=filter(ann_ids) do id
#ba=get_brain_area_hier(id)
#matches=map(sel_bas) do (bas, lvl)
#m=match(Regex("^$bas(|/.*)\$"), ba)
#m!==nothing || return false
#length(findall('/', m.captures[1]))<lvl || return false
#true
#end
#length(findall(matches))>0
#end
#map(reverse(ids)) do id
#Symbol(ann[id][3])
#end
#end

###############################
###############################################
###################################################################################################
#function get_brain_area_abbr(id::Int64)
#id!=0 || return "none"
#ann=_get_allen_annot()
#ann[id][3]
#end
#function Base.getindex(h::ProjHelper, sym::Symbol, s::Symbol)
#global __symbol_to_id
#if !(@isdefined __symbol_to_id) || __symbol_to_id===nothing
#t=Dict{Symbol, Int64}()
#for (id, d) in pairs(Main._get_allen_annot())
#t[Symbol(d[3])]=id
#end
#__symbol_to_id=t
#end
#Base.getindex(h, __symbol_to_id[sym], s)
#end
