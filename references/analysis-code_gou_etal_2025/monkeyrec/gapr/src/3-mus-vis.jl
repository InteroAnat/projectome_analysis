Make.@target xform_swcs_mus_vis () -> begin
	indir="/data/dat/tracing_VIS"
	outdir=force_dir("/data/tmp/swcs-xform-mus-vis") do tmpdir
		fns=readdir(indir)
		newid=1
		ds_qroot=Set(["17100", "17099"])
		for (idx, fn) in pairs(fns)
			m=match(r"^(\d+|MouseLight|Peng)-(\d+).swc$", fn)
			m!==nothing || error("fn: $fn")
			ds=m.captures[1]
			nm=m.captures[2]
			id2newid=Dict()
			function get_id(i::Int64)
				r=get!(id2newid, i, newid)
				if r==newid
					newid+=1
				end
				r
			end
			qroot=in(ds, ds_qroot)
			roots=Set()
			vals=[]
			open(joinpath(indir, fn), "r") do ii
				for l in eachline(ii)
					length(l)>0 || continue
					l[1]!='#' || continue
					(id, t, pos, r, par)=parse_swc_line(l)
					id=get_id(id)
					pos=pos[[3, 2, 1]]
					if par==-1
						push!(roots, (id, pos))
					else
						par=get_id(par)
					end
					if qroot
						t=2
					end
					push!(vals, (id, t, pos, r, par))
				end
			end
			length(roots)==1 || error("dup root")
			ord=string(idx-1, pad=3)
			open(joinpath(tmpdir, "$ds-1009-$ord@root$nm-good.swc"), "w") do oo
				(rootid, rootpos)=first(roots)
				flip=(rootpos[1]>456*25/2)
				for (id, t, pos, r, par) in vals
					(x, y, z)=pos
					if flip
						x=456*25-x
					end
					write(oo, format_swc_line(id, t, (x,y,z), r, par))
				end
				write(oo, format_gapr_annot(rootid, "root", "nm"))
			end
		end
	end
	df=map(readdir(outdir)) do fn
		match(r"^\w+.*\.swc$", fn)!==nothing || error("e")
		(fn=fn, dir=outdir, path=joinpath(outdir, fn))
	end
	sort(DataFrame(df), :fn)
end

Make.@target neurons_summary_mus_vis (xform_swcs_mus_vis) -> begin
	swcs=deepcopy(xform_swcs_mus_vis)
	roi=Set([])
	df=map(swcs.fn, swcs.path) do fn, fnf
		m=match(r"^(\w+)-(\d+)-(\d+)@(.*)\.swc(\.gz)?$", fn)
		m!==nothing || error("wrong fn $fn")
		ds=m.captures[1]
		stg=parse(Int64, m.captures[2])-1000
		ord=parse(Int64, m.captures[3])
		pos=get_soma_pos(fnf)
		ba=get_allen_hier(pos)
		hier=split(ba, "/")
		for i=1:length(hier)
			if in(hier[i], roi)
				hier=hier[1:i]
				break
			end
		end
		ba=join(hier, "/")
		len_tot,lens_typ=get_neurite_len(fnf)
		len_axon=get(lens_typ, 2, 0.0)
		qc=check_neuron_qc(fn)>1
		dst="VIS"
		(fn=fn,
		 ds=ds,
		 dst=dst,
		 stg=stg,
		 ord=ord,
		 soma_ba=ba,
		 soma_pos=pos,
		 qc=qc,
		 mm_tot=len_tot/1000,
		 mm_axon=len_axon/1000,
		 )
	end
	sort(DataFrame(df), :fn)
end

Make.@target cluster_vis_calc (xform_swcs_mus_vis, neurons_summary_mus_vis) -> begin
	swcs=deepcopy(xform_swcs_mus_vis)
	nrns=deepcopy(neurons_summary_mus_vis)
	swcs=decimate_swcs_helper(swcs, "/tmp/swcs-xform-small-mus-vis")
	call_fnt_dist_helper(swcs, nrns, npar=16) do row
		##row.dst=="VMH" || return false
		@assert row.stg%10==9
		@assert row.ord<999000
		true
	end
end

Make.@target projections_mus_vis (xform_swcs_mus_vis) -> begin
	swcs=deepcopy(xform_swcs_mus_vis)
	projs=map(swcs.fn, swcs.path) do fn, fn_path
		strength=Dict{String,NTuple{2, Float64}}()
		nodes=Dict{Int64, NTuple{3, Float64}}()
		open(fn_path, "r") do io
			for l in eachline(io)
				(!isempty(l) && l[1]!='#') || continue
				(id, typ, pos, rad, p)=parse_swc_line(l)
				nodes[id]=pos
				p!=-1 || continue
				typ==2 || continue
				s=norm(pos.-nodes[p])
				s=pos[1]<456*25/2 ? (s, 0.0) : (0.0, s)
				anns=get_allen_hier(pos)
				strength[anns]=get(strength, anns, (0.0, 0.0)).+s
			end
		end
		(fn=fn, projraw=strength)
	end

	ann_ids=get_allen_tree(rev=true)
	projs=map(projs) do row
		hh=My.ProjHelper()
		for ann_id in ann_ids
			v=get(row.projraw, ann_id, (0.0, 0.0))
			v!=(0.0, 0.0) || continue
			hh.dat[ann_id]=v./1000
			ann_id!="/" || continue
			par_id=get_ann_hier_par(ann_id)
			row.projraw[par_id]=get(row.projraw, par_id, (0.0, 0.0)).+v
		end
		(fn=row.fn, mm_proj=hh)
	end
	DataFrame(projs)
end

Make.@target cluster_mus_vis (cluster_vis_calc, neurons_summary_mus_vis) -> begin
	outdir="/data/tmp/swcs-xform-clustered-mus-vis"
	swcs, mat=deepcopy(cluster_vis_calc)
	nrns=deepcopy(neurons_summary_mus_vis)
	leftjoin!(swcs, nrns, on=:fn)
	seli=map(r->match(r"@q?root\d+", r.fn)!==nothing, eachrow(swcs))
	swcs=swcs[seli, :]
	mat=mat[seli, seli]

	h=hclust(mat, linkage=:ward, branchorder=:barjoseph)
	cls=cutree(h, k=8)
	cls=invperm(unique(cls[h.order]))[cls]
	insertcols!(swcs, :cls=>cls)

	swcs=remap_groups_alt(swcs, outdir)
	(swcs,h)
end

Make.@target pick_brain_area_mus_vis (cluster_mus_vis, projections_mus_vis) -> begin
	swcs=deepcopy(cluster_mus_vis)[1]
	projs=deepcopy(projections_mus_vis)
	swcs2=leftjoin(swcs, projs, on=:fn)
	anns=get_allen_tree(rev=true)
	bas=map(anns) do ba
		quant=quantile(map(v->v.mm_proj[ba, :all], eachrow(swcs2)), [0.7, 0.8, 0.9, 0.95, 0.98])
		(ba=ba, q7=quant[1], q8=quant[2], q9=quant[3], q95=quant[4], q98=quant[5])
	end
	sel_bas=[
		 "/root/grey/BS/HB/P",
		 "/root/grey/BS/IB/TH/DORpm",
		 "/root/grey/BS/IB/TH/DORsm",
		 "/root/grey/BS/MB/MBmot",
		 "/root/grey/BS/MB/MBsen",
		 "/root/grey/CH/CNU/STR/STRd",
		 "/root/grey/CH/CTX/CTXpl/HPF",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/ACA",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/PTLp/VISa",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/PTLp/VISrl",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/RSP",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/SS/SSp",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/VIS/VISal",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/VIS/VISam",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/VIS/VISl",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/VIS/VISp",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/VIS/VISpl",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/VIS/VISpm",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/VIS/VISpor",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/AUD",
		 "/root/grey/CH/CTX/CTXpl/Isocortex/MO",
		]
	#sel_bas=select_brain_areas(sel_bas, get_allen_tree(rev=true))
	sort(map(v->(v.q98, v.ba), bas),rev=true)[1:60], sel_bas
end

Make.@target cluster_plot_mus_vis (cluster_mus_vis, projections_mus_vis, pick_brain_area_mus_vis) -> begin
	swcs,h=deepcopy(cluster_mus_vis)
	projs=deepcopy(projections_mus_vis)
	sel_bas=deepcopy(pick_brain_area_mus_vis)[2]
	clustering_plot_func(swcs, h, projs, sel_bas, "extra/clustering-mus-vis.tex", reorder=true)
end

Make.@target regression_data_mus_vis (neurons_summary_mus_vis, projections_mus_vis, cluster_mus_vis, pick_brain_area_mus_vis) -> begin
	swcs=deepcopy(neurons_summary_mus_vis)
	projs=deepcopy(projections_mus_vis)
	cls=deepcopy(cluster_mus_vis)[1][:, [:fn, :cls]]
	sel_bas=deepcopy(pick_brain_area_mus_vis)[2]
	leftjoin!(swcs, cls, on=:fn)
	leftjoin!(swcs, projs, on=:fn)
	insertcols!(swcs, :soma_x=>map(v->v[1], swcs.soma_pos))
	insertcols!(swcs, :soma_y=>map(v->v[2], swcs.soma_pos))
	insertcols!(swcs, :soma_z=>map(v->v[3], swcs.soma_pos))
	insertcols!(swcs, :mm_dend=>(swcs.mm_tot.-swcs.mm_axon))
	swcs=dropmissing(swcs)
	for col in [:cls, :ds, :dst, :soma_ba]
		transform!(swcs, col => categorical => col)
	end
	sel_cols=[:fn, :ds, :dst,
		  :cls,
		  :mm_axon,
		  :mm_dend,
		  :mm_tot,
		  :qc,
		  :soma_ba,
		  :soma_x,
		  :soma_y,
		  :soma_z,
		  #*pl_
		  #*pr_
		  #*pt_
		  ]

	push!(sel_bas, "/root")
	for id in sel_bas
		sym=split(id, "/")[end]
		p=map(eachrow(swcs)) do row
			row.mm_proj[id, :both]
		end
		#quantile(map(v->sum(v), p), 0.99)>=0.01 || continue
		new_cols=[Symbol("pp_l_"*sym)=>map(v->v[1], p),
			  Symbol("pp_r_"*sym)=>map(v->v[2], p),
			  Symbol("pp_t_"*sym)=>map(v->sum(v), p),
			  ]
		insertcols!(swcs, new_cols...)
		for p in new_cols
			push!(sel_cols, p.first)
		end
	end
	removed=setdiff(Set(names(swcs)), Set(String.(sel_cols)))
	@info "removed $removed"
	swcs[:, sel_cols]
end

##########################
###############################################
#########################################################################
######################################
#########################
#####################################
#######################################
################################
############################################################################
############################################################################
############################################################################
################
	#################################################################
######################
#######################################################
###############################################################
########################################################
###############################################################
