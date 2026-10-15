
Make.@target neurons_summary_mac (xform_swcs_mac) -> begin
	swcs=deepcopy(xform_swcs_mac)
	roi=Set([])
	df=map(swcs.fn, swcs.path) do fn, fnf
		m=match(r"^(\d+)-(\d+)-(\d+)@(.*)\.swc(\.gz)?$", fn)
		m!==nothing || error("wrong fn $fn")
		pos=get_soma_pos(fnf)
		ba=get_nmt_hier(pos)
		hier=split(ba, "/")
		for i=1:length(hier)
			if in(hier[i], roi)
				hier=hier[1:i]
				break
			end
		end
		ba=join(hier, "/")
		ds=m.captures[1]
		stg=parse(Int64, m.captures[2])-1000
		ord=parse(Int64, m.captures[3])
		len_tot,lens_typ=get_neurite_len(fnf)
		len_axon=get(lens_typ, 2, 0.0)
		qc=check_neuron_qc(fn)>1
		dst="MISC"
		soma_st=0
		if match(r"@qroot", fn)!==nothing
			soma_st=1
		elseif match(r"@root", fn)!==nothing
			soma_st=2
		end
		@assert soma_st!=0
		(fn=fn,
		 ds=ds,
		 dst=dst,
		 stg=stg,
		 ord=ord,
		 soma_ba=ba,
		 soma_pos=pos,
		 soma_st=soma_st,
		 qc=qc,
		 mm_tot=len_tot/1000,
		 mm_axon=len_axon/1000,
		 )
	end
	df=sort(DataFrame(df), :fn)
	somas=df.soma_pos[df.soma_st.==2]
	inj_site=[median(map(v->v[i], somas)) for i in 1:3]
	mdist=maximum(map(p->norm(p.-inj_site), somas))
	@info "$inj_site: $mdist"
	insertcols!(df, :dinj=>map(p->norm(p.-inj_site)/mdist, df.soma_pos))
	df
end

Make.@target cluster_mac_calc (xform_swcs_mac, neurons_summary_mac) -> begin
	swcs=deepcopy(xform_swcs_mac)
	swcs=decimate_swcs_helper(swcs, "/tmp/swcs-xform-small-macaca")
	call_fnt_dist_helper(swcs, deepcopy(neurons_summary_mac), npar=16) do row
		##row.dst=="VMH" || return false
		@assert row.stg%10==9
		@assert row.ord<999000
		true
	end
end

Make.@target projections_mac (xform_swcs_mac) -> begin
	swcs=deepcopy(xform_swcs_mac)
	all_anns=Set()
	projs=map(swcs.fn, swcs.path) do fn, fn_path
		strength=Dict{String,NTuple{2, Float64}}()
		nodes=Dict{Int64, NTuple{3, Float64}}()
		open(fn_path, "r") do io
			while !eof(io)
				l=readline(io)
				(!isempty(l) && l[1]!='#') || continue
				(id, typ, pos, rad, p)=parse_swc_line(l)
				nodes[id]=pos
				p!=-1 || continue
				typ==2 || continue
				s=norm(pos.-nodes[p])
				s=pos[1]<256*250/2 ? (s, 0.0) : (0.0, s)
				anns=get_nmt_hier(pos)
				strength[anns]=get(strength, anns, (0.0, 0.0)).+s
				pars=get_ann_hier_par(anns)
				push!(all_anns, anns)
				while pars!="/"
					push!(all_anns, pars)
					pars=get_ann_hier_par(pars)
				end
			end
		end
		(fn=fn, path=fn_path, projraw=strength)
	end
	projs=map(projs) do row
		hh=My.ProjHelper()
		ann_ids=reverse(sort(collect(all_anns)))
		for ann_id in ann_ids
			v=get(row.projraw, ann_id, (0.0, 0.0))
			v!=(0.0, 0.0) || continue
			hh.dat[ann_id]=v./1000
			ann_id!="/" || continue
			par_id=get_ann_hier_par(ann_id)
			row.projraw[par_id]=get(row.projraw, par_id, (0.0, 0.0)).+v
		end
		(fn=row.fn, path=row.path, mm_proj=hh)
	end
	DataFrame(projs), sort(unique(collect(all_anns)))
end

# XXX filter crippled ones
Make.@target cluster_mac (cluster_mac_calc, neurons_summary_mac, guess_itptct_mac) -> begin
	outdir="/data/tmp/swcs-xform-clustered-macaca"
	swcs, mat=deepcopy(cluster_mac_calc)
	nrns=deepcopy(neurons_summary_mac)
	leftjoin!(swcs, nrns, on=:fn)
	seli=map(eachrow(swcs)) do row
		match(r"@q?root\d+", row.fn)!==nothing || return false
		row.mm_axon>=0.3 || return false
		row.qc || return false
		(row.soma_st<2 && row.dinj>1.05) && return false
		true
	end
	swcs=swcs[seli, :]
	mat=mat[seli, seli]

	delta=.5*maximum(mat)
	leftjoin!(swcs, deepcopy(guess_itptct_mac), on=:fn)
	for r in eachrow(swcs)
		if r.typ2!=""; r.typ=r.typ2; end
		#if r.typ==""; r.typ=r.typ2; end
	end
	lines=[]
	if true
		for i in axes(mat, 1), j in axes(mat, 2)
			if swcs[i, :typ]!=swcs[j, :typ]
				push!(lines, (swcs[i, :fn], swcs[j, :fn], swcs[i, :typ], swcs[j, :typ], mat[i,j]))
				mat[i, j]+=delta
			end
		end
	end
	open("/tmp/debug-dist-mat.txt", "w") do io
		for (fni, fnj, typi, typj, d) in lines
			println(io, "$d $typi/$fni $typj/$fnj")
		end
	end

	h=hclust(mat, linkage=:ward, branchorder=:barjoseph)
	cls=cutree(h, k=8)
	cls=invperm(unique(cls[h.order]))[cls]
	insertcols!(swcs, :cls=>cls)

	swcs=remap_groups_alt(swcs, outdir)
	(swcs[:, [:fn, :cls, :typ]],h)
end

Make.@target guess_bas_mac () -> begin
	hits=Set()
	for x in 0:250:250*250, y in 0:250:250*300, z in 0:250:250*200
		res=get_nmt_hier((x,y,z))
		if contains("post_TE")(res)
			for dx in -500:250:500, dy in -500:250:500, dz in -500:250:500
				n=get_nmt_hier((x+dx, y+dy, z+dz))
				push!(hits, n)
			end
		end
	end
	hits
end

Make.@target pick_brain_area_mac (cluster_mac, projections_mac) -> begin
	swcs=deepcopy(cluster_mac)[1]
	projs, anns=deepcopy(projections_mac)
	swcs2=leftjoin(swcs, projs, on=:fn)
	bas=map(anns) do ba
		quant=quantile(map(v->v.mm_proj[ba, :all], eachrow(swcs2)), [0.7, 0.8, 0.9, 0.95, 0.99])
		(ba=ba, q7=quant[1], q8=quant[2], q9=quant[3], q95=quant[4], q99=quant[5])
	end
	sel_bas=[
		 # from literature
		 "/CHARM/Occipital/V1",
		 "/CHARM/Occipital/V2-V4/V2-V3/V2",
		 "/CHARM/Occipital/V2-V4/V2-V3/V3",
		 "/CHARM/Occipital/V2-V4/V2-V3/V3/V3d|V3A/V3A",
		 "/CHARM/Parietal/SPL/vm_IPS/fundus_IPS/PIP",
		 "/CHARM/Temporal/STG|STSd/STGr|STSd/STSd/TPO", # PO???
		 # DP???
		 "/CHARM/Parietal/SPL/vm_IPS/fundus_IPS/VIP",
		 "/CHARM/Parietal/IPL/lat_IPS/LIP/LIPd",
		 "/CHARM/Parietal/IPL/lat_IPS/LIP/LIPv",
		 "/CHARM/Parietal/IPL/MST",
		 "/CHARM/Occipital/MT",
		 # V4t???
		 "/CHARM/Temporal/ITC/STSf/FST",
		 "/CHARM/Temporal/ITC/TEO",
		 "/CHARM/Temporal/ITC/TE/gyral_TE/post_TE", #TEp
		 "/CHARM/Temporal/ITC/TE/gyral_TE/ant_TE", # TEa
		 "/CHARM/Temporal/ITC/TE/TE_in_STSv/TEm",

		 # extended literature
		 "/CHARM/Occipital/V2-V4/V4",
		 "/CHARM/Parietal/SPL/vm_IPS/MIP",
		 "/CHARM/Temporal/floor_of_ls/Ins|Pi/Ins",
		 "/CHARM/Temporal/floor_of_ls/Ins|Pi/Pi",
		 "/CHARM/Temporal/ITC/STSf/ant_STSf",

		 # subcort
		 "/SARM/di/Thal/PThal/Pul/IPul",
		 "/SARM/di/Thal/PThal/Pul/LPul",
		 "/SARM/tel/BG/Str/DStr/Cd",
		 "/SARM/tel/BG/Str/DStr/Pu",
		 "/SARM/tel/LVPal/LPal/Cl",
		 "/SARM/di/Thal/PThal/LP",
		 "/SARM/mes/Mid/DMid/Co/SC",
		 "/SARM/met/Pons",
		 #
		 ##################
		 # 
		 #"/SARM/di/Thal",
		 #"/SARM/di/Thal/PThal",
		 #"/SARM/di/Thal/PThal/Pul",
		 #"/SARM/tel/BG/Str",
		 #"/SARM/tel/BG/Str/DStr",
		 #"/SARM/tel/LVPal/LPal",
		 # prior
		 "/CHARM/Frontal/lat_PFC/area_8A", #FEF
		 "/CHARM/Frontal/lat_PFC/dlPFC/rostral_dlPFC/area_46d",
		 "/CHARM/Parietal/IPL/area_7_in_IPL/area_7a|b/area_7a",
		 "/CHARM/Temporal/ITC/TE/gyral_TE/post_TE/TEpd",
		 ]
	others=[
		#never
		#"LPons",
		#"R",
		#"TEa",
		#"Amy",
		#"MR",
		#"PO",
		#"VThal",
		#"PNThal",
		#"DR",
		#"DPons",
		#"area_7m",
		#"area_8B",
		#"area_9d",
		#"LC-Me5",
		"",
		"8l",
		"8m",
		"LGi",
		"LIT",
		"MFT",
		"nBM",
		"PBr",
		"STPc",
		"STPr",
	       ]
	for x in others
		x=="" && continue
		cond=endswith("/"*x)
		mat=filter(cond, anns)
		if length(mat)==1
			if !in(mat[1], sel_bas)
				@info "mat $x $(mat[1])"
				push!(sel_bas, mat[1])
			else
				@info "dup $x"
			end
		else
			@info "nomatch $x"
		end

	end
	sort(map(v->(v.q99, v.ba), bas),rev=true)[1:60], sel_bas
end

Make.@target cluster_plot_mac (cluster_mac, projections_mac, pick_brain_area_mac) -> begin
	swcs,h=deepcopy(cluster_mac)
	projs,bas=deepcopy(projections_mac)
	sel_bas=deepcopy(pick_brain_area_mac)[2]
	clustering_plot_func(swcs, h, projs, sel_bas, "extra/clustering-macaca.tex"; reorder=true)
end

Make.@target pick_macaca_neurons (xform_swcs_mac, neurons_summary_mac)->begin
	outdir="/data/tmp/swcs-random-pick-macaca"
	swcs=deepcopy(xform_swcs_mac)
	nrns=deepcopy(neurons_summary_mac)
	leftjoin!(swcs, nrns, on=:fn)
	insertcols!(swcs, :cls=>0)
	for row in eachrow(swcs)
		if row.soma_st==1
			row.cls=(row.qc ? 2 : -1)
		elseif row.soma_st==2
			row.cls=(row.qc ? 1 : 0)
		else
			row.cls=-1
		end
		@assert row.stg%10==9
		@assert row.ord<999000
	end
	swcs=filter(v->v.cls>=0, swcs)

	remap_groups_alt(swcs[:, [:fn, :cls, :path]], outdir)
end

include("2-arbor.jl")

Make.@target soma_depth_mac (neurons_summary_mac, get_depth_mac_fit_nurbs_v4) -> begin
	nrns=deepcopy(neurons_summary_mac)
	surf,offset2,offset3=deepcopy(get_depth_mac_fit_nurbs_v4)
	surf2=nurbs_change(surf, shift=offset2)
	surf3=nurbs_change(surf, shift=offset3.-offset2.+surf.P)
	surf4=nurbs_change(surf, shift=offset3)
	nrns=filter(r->r.soma_st==2, nrns)
	raw_pos=nrns.soma_pos
	raw_pos=[raw_pos[j][i] for i in 1:3, j in axes(raw_pos, 1)]
	insertcols!(nrns, :soma_uvw=>(NaN,NaN,NaN), :soma_uvw_loss=>NaN,
		    :soma_xyz=>nrns.soma_pos, :soma_xyz_fix=>(NaN,NaN,NaN))

	function get_pts(surf2, surf, uvwvec)
		pt1=nurbs_surf_points(surf, uvwvec[1:2, :])
		pt2=nurbs_surf_points(surf2, uvwvec[1:2, :])
		pt1+(pt2-pt1).*reshape(uvwvec[3, :], 1, :)
	end
	function get_dist(pt, tgt)
		reshape(sum((pt.-tgt).^2, dims=1), :)
	end
	ref_uvw=rand(Float64, 3, 1000_000)
	ref_tree=KDTree(get_pts(surf, surf2, ref_uvw))
	ref_idx=knn(ref_tree, raw_pos, 4)[1]

	@showprogress "locating soma" for row in eachrow(nrns)
		tgt=raw_pos[:, rownumber(row)]
		opt=Flux.Optimise.AdaGrad()
		uvwvec=ref_uvw[:, ref_idx[rownumber(row)]]
		params=Flux.Params([uvwvec])
		best_uvw,best_loss=nothing,nothing
		for i in 1:200
			loss,grads=Flux.withgradient(params) do
				pt=get_pts(surf, surf2, uvwvec)
				dist=get_dist(pt, tgt)
				j=argmin(dist)
				best_uvw=(uvwvec[:, j]..., )
				best_loss=dist[j]
				mean(dist)
			end
			best_loss<1e-6 && break
			Flux.update!(opt, params, grads)
		end
		row.soma_uvw=best_uvw
		row.soma_uvw_loss=best_loss
		row.soma_xyz_fix=(get_pts(surf3, surf4, reshape([best_uvw...], :, 1))...,)
	end
	nrns[:, [:fn, :soma_uvw, :soma_uvw_loss, :soma_xyz, :soma_xyz_fix]]
end

Make.@target soma_depth_mac_disp (soma_depth_mac, get_depth_mac_labels, meshes_for_mac) -> begin
	nrns=deepcopy(soma_depth_mac)
	labels=deepcopy(get_depth_mac_labels)
	mshv4=deepcopy(meshes_for_mac)[1]

	label(self, i, p)="$(nrns.soma_xyz[i]) $(nrns.soma_uvw[i])"
	fig=GLMakie.Figure()
	GLMakie.scatter(fig[1,1], nrns.soma_xyz, markersize=.7, color=[p[3] for p in nrns.soma_uvw],
			inspector_label=label)
	for (l, pts) in pairs(labels)
		GLMakie.scatter!(fig[1,1], pts, color=:black, markersize=.1)
	end
	GLMakie.scatter(fig[1,2], nrns.soma_xyz_fix, markersize=.7, color=[p[3] for p in nrns.soma_uvw],
			inspector_label=label)
	GLMakie.mesh!(fig[1,2], mshv4, color=(:gray, .3), transparency=true, inspectable=false)
	display(fig)
	GLMakie.DataInspector(fig)
	quantile(nrns.soma_uvw_loss, [.5, .7, .9, 1])
end

Make.@target arbor_depth_mac_v2 (xform_swcs_mac, get_depth_mac_fit_nurbs_v2, get_depth_mac_labels) -> begin
	nrns=deepcopy(xform_swcs_mac)
	surf,offset=deepcopy(get_depth_mac_fit_nurbs_v2)[[:surf1, :off2]]
	surf2=nurbs_change(surf, shift=offset)
	labels=deepcopy(get_depth_mac_labels)
#########################################################################
##################################################################
###########################################################
	function get_pts(uvwvec)
		pt1=nurbs_surf_points(surf, uvwvec[1:2, :])
		pt2=nurbs_surf_points(surf2, uvwvec[1:2, :])
		pt1+(pt2-pt1).*reshape(uvwvec[3, :], 1, :)
	end
	function get_dist(pt, tgt)
		reshape(sum((pt.-tgt).^2, dims=1), :)
	end
	ref_uvw=rand(Float64, 3, 1000000)
	ref_xyz=get_pts(ref_uvw)
	ref_tree=KDTree(ref_xyz)
	display(GLMakie.scatter(ref_xyz[:, 1:500], markersize=.7, color=ref_uvw[3,1:500]))
	for (l, pts) in pairs(labels)
		GLMakie.scatter!(pts, color=:black, markersize=.1)
	end

	hits=[]

	@showprogress for row in eachrow(nrns)
		nodes=[]
		open(row.path, "r") do io
			for l in eachline(io)
				length(l)==0 && continue
				l[1]=='#' && continue
				(id,typ,pos,rad,par)=parse_swc_line(l)
				push!(nodes, (id,typ,pos,par))
			end
		end
		pts=reduce(hcat, [[n[3]...] for n in nodes])
		ind,dist=knn(ref_tree, pts, 1)
		ind=reduce(vcat,ind)
		dist=reduce(vcat,dist)
		sel=findall(dist.<10)
		@info "n: $(length(sel)) dist: $(minimum(dist))"
		for k in sel
			push!(hits, (nodes[k][3], (ref_uvw[:, ind[k]]...,)))
		end

	#####################################
	########################################
	###################################
	#############################
	##################################
	#######################
	#####################
	#=
		tgt=raw_pos[:, rownumber(row)]
		opt=Flux.Optimise.AdaGrad()
		uvwvec=ref_uvw[:, ref_idx[rownumber(row)]]
		params=Flux.Params([uvwvec])
		best_uvw,best_loss=nothing,nothing
		for i in 1:200
			loss,grads=Flux.withgradient(params) do
				pt=get_pts(uvwvec)
				dist=get_dist(pt, tgt)
				j=argmin(dist)
				best_uvw=(uvwvec[:, j]..., )
				best_loss=dist[j]
				mean(dist)
			end
			best_loss<1e-6 && break
			Flux.update!(opt, params, grads)
		end
		row.soma_uvw=best_uvw
		row.soma_uvw_loss=best_loss
		=#
		#rownumber(row)>30 && break
	end

	GLMakie.scatter!([hits[i][1][j] for j in 1:3, i in axes(hits, 1)], markersize=.7, color=[h[2][3] for h in hits])
	hits
	#nrns[:, [:fn, :soma_uvw, :soma_uvw_loss]]
end

# XXX use raw_length???
Make.@target fig_length_dist_mac (neurons_summary_mac, cluster_mac)->begin
	nrns=deepcopy(neurons_summary_mac)
	swcs=deepcopy(cluster_mac)[1]
	leftjoin!(nrns, swcs, on=:fn)
	filter!(r->r.cls!==missing, nrns)
	rename!(nrns, :mm_tot=>:len)

	xticks=[0,1,2,3]
	xticks=(xticks, map(v->10^v, xticks))
	p=groupedhist(log10.(nrns[:, :len]),
		      group=["Quasi-neurons","Neurons"][nrns[:, :soma_st]],
		      extra_kwargs=Dict(:subplot=>Dict(
						       "scale only axis"=>true,
						       )),
		      bins=range(0, 3, length = 20),
		      legend_font_pointsize=5,
		      legend=:topright,
		      ylim=(0, 50),
		      tickfontsize=5,
		      yguide="Count",
		      xguide=tex("Neurite length (\\si{mm})"),
		      guidefontsize=6,
		      xticks=xticks,
		      grid=false,
		      linewidth=.5,
		      size=(190,130),
		      bar_position=:stack,
		      )
	n1=count(v->v==2, nrns.soma_st)
	n2=count(v->v==1, nrns.soma_st)
	annotate!(p, [(0.0, 30.0, ("Projection ``neurons'' (N=$n1+$n2)", 6, :left))])
	savefig(p, "extra/length-dist-macaca.tex")
	PGFPlotsX.push_preamble!(p.o.the_plot, "\\usepackage{siunitx}")
	p
end

Make.@target fig_soma_depth_mac (soma_depth_mac, cluster_mac) -> begin
	depth=deepcopy(soma_depth_mac)
	swcs=deepcopy(cluster_mac)[1]
	leftjoin!(depth, swcs, on=:fn)
	filter!(r->r.cls!==missing, depth)
	clsa, clsb=extrema(depth.cls)

	idxmap=Dict(["#$i"=>Int[] for i in clsa:clsb])
	for (i,c) in pairs(depth.cls)
		push!(idxmap["#$c"], i)
	end
	uvals, vvals, wvals=[[p[i] for p in depth.soma_uvw] for i in 1:3]
	p=violin(depth.cls, wvals,
		 grid=false,
		 xticks=(clsa:clsb, map(c->tex("\\#$c"), clsa:clsb)),
		 xguide="Cluster",
		 yguide=tex("Depth"),
		 legend=:none,
		 tickfontsize=5,
		 guidefontsize=6,
		 size=(200,160),
		 xlim=(clsa-.5, clsb+.5),
		 linewidth=0,
		 group=idxmap,
		 )
	p=scatter!(p, depth.cls.+rand(size(depth, 1))./3 .-1/6, wvals,
		   markersize=.5,
		   )
	savefig(p, "extra/soma-depth-macaca.tex")

	q=scatter(uvals, vvals, group=idxmap,
		  xguide="U coord",
		  yguide="V coord",
		  tickfontsize=5,
		  guidefontsize=6,
		  size=(200,160),
		  markersize=2.0,
		  markerstrokewidth=0,
		  legend_font_pointsize=5,
		  )
	savefig(q, "extra/soma-uv-macaca.tex")
	q
end

Make.@target regression_data_mac (neurons_summary_mac, projections_mac, cluster_mac, pick_brain_area_mac, soma_depth_mac) -> begin
	swcs=deepcopy(neurons_summary_mac)
	projs, ann_ids=deepcopy(projections_mac)
	cls=deepcopy(cluster_mac)[1][:, [:fn, :cls]]
	bas=deepcopy(pick_brain_area_mac)[2]
	depth=deepcopy(soma_depth_mac)
	leftjoin!(swcs, cls, on=:fn)
	leftjoin!(swcs, projs, on=:fn)
	insertcols!(swcs, :soma_x=>map(v->v[1], swcs.soma_pos))
	insertcols!(swcs, :soma_y=>map(v->v[2], swcs.soma_pos))
	insertcols!(swcs, :soma_z=>map(v->v[3], swcs.soma_pos))
	insertcols!(swcs, :mm_dend=>(swcs.mm_tot.-swcs.mm_axon))
	for row in eachrow(swcs)
		row.qc || (row.cls=0)
		if row.soma_st==1
		elseif row.soma_st==2
		else
			row.cls=0
		end
	end
	swcs=dropmissing(swcs)

	for col in [:cls, :ds, :dst, :soma_ba]
		transform!(swcs, col => categorical => col)
		#swcs[:, col]=categorical(swcs[:, col])
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

	leftjoin!(swcs, depth, on=:fn)
	for i in 1:3
		sym=(:soma_u, :soma_v, :soma_w)[i]
		vals=map(v->v===missing ? missing : v[i], swcs.soma_uvw)
		insertcols!(swcs, sym=>vals)
		push!(sel_cols, sym)
	end
	for id in bas
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
	filter!(swcs) do r
		r.qc || return false
		r.soma_st==2 || r.dinj<=1.05 || return false
		true
	end
	@info "removed $removed"
	swcs[:, sel_cols]
end

Make.@target fig_coprojection_mac (regression_data_mac) -> begin
	swcs=deepcopy(regression_data_mac)
	projs=filter(startswith("pp_l_"), names(swcs))
	to_name(s)=replace(s[6:end], "_"=>"")
	open("extra/coproj-macaca.tex", "w") do io
		write(io, "\\begin{tikzpicture}\n");
		#write(io, "\\usetikzlibrary {graphs}
		write(io, "\\graph [spring layout] { \n");
		edges=[]
		for i in 1:length(projs)
			p=projs[i]
			write(io, to_name(p), i==length(projs) ? "" : ",")
			for j in i+1:length(projs)
				q=projs[j]
				xp=swcs[:, p]
				xq=swcs[:, q]
				mp=maximum(xp)
				mq=maximum(xq)
				selp=findall(xp.>mp/10)
				selq=findall(xq.>mq/10)
				selpq=intersect(selp, selq)
				r=2*length(selpq)/(length(selp)+length(selq))
				if r>.03
					push!(edges, (to_name(p), to_name(q), round(r, digits=2)))
				end
			end
		end
		for (p, q, r) in edges
			sty=""
			if r>.5
				sty="line width=2pt"
			elseif r>.1
				sty="line width=1pt"
			end
			write(io, ", $p --[$sty] $q\n")
		end
		write(io, "};");
		write(io, "\\end{tikzpicture}\n");
	end
end

Make.@target fiber_bundle_topo_select (neurons_summary_mac, xform_swcs_mac) -> begin
	nrns=deepcopy(neurons_summary_mac)
	swcs=deepcopy(xform_swcs_mac)
	leftjoin!(nrns, swcs[:, [:fn, :path]], on=:fn)
	function pred(cent, peri)
		d=norm(peri.-cent)
		function(pos)
			norm(pos.-cent)<d*1.3
		end
	end
	pred1=pred((16913.1, 21150.7, 29062.5), (17341.6, 21375.3, 29567.6))
	pred2=pred((18264.9, 26301.7, 25622.3), (18380.6, 26242.3, 24999.4))
	pred3=pred((32674.5, 30006, 26389.6), (32692.7, 29323.1, 26564.5))
	function firstnd(fn, par, id2pos)
		x=nothing
		while true
			(p, pp)=id2pos[par]
			pp==-1 && break
			x,brk=fn(p)
			brk && break
			par=pp
		end
		x
	end
	res=pmap(eachrow(nrns)) do row
		!row.qc && return missing
		row.soma_st<2 && row.dinj>1.05 && return missing
		hits=Any[nothing for k in 1:3]
		id2pos=Dict()
		open(row.path, "r") do io
			for l in eachline(io)
				isempty(l) && continue
				l[1]=='#' && continue
				(id,typ,pos,rad,par)=parse_swc_line(l)
				id2pos[id]=(pos, par)
				for k in 1:3
					if hits[k]===nothing
						if (pred1,pred2,pred3)[k](pos)
							hits[k]=id
						end
					end
				end
			end
		end
		hits[1]===nothing && return missing
		cls=0
		x=0
		if hits[2]===nothing
			if hits[3]===nothing
				return missing
			end
			cls=1
			x=hits[3]
		else
			if hits[3]!==nothing
				@assert false
			end
			cls=2
			x=hits[2]
		end
		par=hits[1]
		cur=id2pos[par][1]
		nrm=firstnd(par, id2pos) do p
			dd=p.-cur
			l=norm(dd)
			nrm=dd./l
			nrm, l>1000
		end
		dval=sum(nrm.*cur)
		(fn=row.fn, cls=cls, nrm=nrm, dval=dval, nodes=id2pos, nd=x)
	end
	res=collect(skipmissing(res))
	nrm=[mean([v.nrm[k] for v in res]) for k in 1:3]
	nrm=nrm./norm(nrm)
	dval=mean([v.dval for v in res])
	@info "nrm: $nrm; dval: $dval"
	dval1=sum((14865.1, 20571.7, 32370.2).*nrm)
	dval2=sum((15960.5, 21151.7, 29926.3).*nrm)
	dval3=sum((17904.6, 23112.8, 27354.1).*nrm)
	res=pmap(res) do row
		p1=firstnd(row.nd, row.nodes) do p
			d=sum(p.*nrm)
			p, d>dval1
		end
		p2=firstnd(row.nd, row.nodes) do p
			d=sum(p.*nrm)
			p, d>dval2
		end
		p3=firstnd(row.nd, row.nodes) do p
			d=sum(p.*nrm)
			p, d>dval3
		end
		(row..., p1=p1, p2=p2, p3=p3)
	end
	leftjoin!(swcs, DataFrame(res), on=:fn)
	dropmissing!(swcs)
	swcs=remap_groups_alt(swcs, "/data/tmp/swcs-fiber-topo-mac")
	swcs.nrm.=[nrm]
	swcs[:, [:fn, :cls, :nrm, :p1, :p2, :p3]]
end

Make.@target validate_itptct_mac (fiber_bundle_topo_select, guess_itptct_mac, cluster_mac) -> begin
	nrns=transform(deepcopy(fiber_bundle_topo_select), :cls=>:cls2)
	typs=deepcopy(guess_itptct_mac)
	leftjoin!(nrns, typs, on=:fn)
	clu=transform(deepcopy(cluster_mac)[1], :cls=>:cls1)[:, [:fn, :cls1]]
	leftjoin!(nrns, clu, on=:fn)
	sort(nrns[:, [:fn, :cls1, :cls2, :typ, :typ2]], [:cls2, :cls1])
end

Make.@target fig_fiber_bundle_topo (fiber_bundle_topo_select, soma_depth_mac) -> begin
	nrns=deepcopy(fiber_bundle_topo_select)
	depth=deepcopy(soma_depth_mac)
	leftjoin!(nrns, depth, on=:fn)
	nrns2=dropmissing(nrns)

	uvals,vvals,wvals=[[v[i] for v in nrns2.soma_uvw] for i in 1:3]
	p=scatter(wvals, vvals, group=["contra", "ipsi"][nrns2.cls],
		  #markershape=[:circle,:cross][nrns2.cls],
		  xguide="depth",
		  yguide="V coord",
		  size=(480,360),
		  )
	savefig(p, "extra/fiber-bundle-soma-macaca.tex")
	function norm01(x)
		a,b=extrema(x)
		(x.-a)./(b-a)
	end
	p2=map([:p1, :p2, :p3]) do sec
		n=[nrns.nrm[1]...]
		u=[1, 0, 0]
		v=cross(n, u)
		u=u./norm(u)
		v=v./norm(v)
		uvals=[sum(p.*u) for p in nrns[:, sec]]
		vvals=[sum(p.*v) for p in nrns[:, sec]]
		uvals=norm01(uvals)
		vvals=norm01(vvals)
		p=scatter(uvals, vvals, group=["contra", "ipsi"][nrns.cls],
			  #markershape=[:circle,:cross][nrns2.cls],
			  xguide="U coord",
			  yguide="V coord",
			  marker=(0.5, [:cross :circle], 2),
			  markercolor=:white,
			  size=(480,360),
			  )
		savefig(p, "extra/fiber-bundle-sect$sec-macaca.tex")
		p
	end
	plot(p2...)
end
##################
####################

Make.@target regression_mac (regression_data_mac) -> begin
	df=deepcopy(regression_data_mac)
	targets=filter(v->startswith("pp_")(v), names(df))
	res=pmap(targets) do tgt
		df2=insertcols(df, :proj=>df[:, tgt])
		#res=glm(@formula(proj~1+soma_ba+soma_x+soma_y+soma_z+cls), df2, Normal())
		res=glm(@formula(proj~cls+soma_u+soma_v+soma_w), df2, Normal())
		tgt=>res
	end
	return Dict(res)
	hits=[]
	for (ba, m) in res
		ints=confint(m)
		lvl=length(findall('/', ba))
		(lvl>=3&&lvl<6) || continue
		for i=1:size(ints)[1]
			a=ints[i, 1]
			b=ints[i, 2]
			if a>1
				push!(hits, (ba, i-1))
			end
			if b<-1
				error("asdf")
				push!(hits, (ba, i-1))
			end
		end
	end
	hits
end
##########################################################################
##########################################################################################
###################################################################################################

#===
#convexhull?
#
#union 4187.3853f0
#
*AP
RadiusedCSGUnion

Descartes.jl 2021 J * *
GeometryTypes.AbstractMesh
Mesh
GeometryTypes.AbstractMesh
??union working



function main7()
	#sel_bas=[ "MT", "V4", "MST", "7a_(Opt/PG)", "V3A", "V1", "V2", "FST", ]
	for ba in sel_bas
		sym=Symbol("p_"*ba)
		insertcols!(swcs, sym=>0.0)
		for row in eachrow(swcs)
			row[sym]=get(row.proj, ba, 0.0)
		end
	end
	for row in eachrow(swcs)
	end
	swcs, DataFrame(bas), h
end

===#
#
#
#
#
# top brain areas
# (142.42156217563036, "/CHARM")
# (103.98892550558863, "/CHARM/Occipital")
# (81.25714115415079, "/CHARM/Occipital/V2-V4")
# (54.242278410546454, "/SARM")
# (46.66570918684298, "/CHARM/Occipital/V2-V4/V2-V3")
# (39.85226181774491, "/CHARM/Temporal")
# (39.8266649922561, "/CHARM/Temporal/ITC")
# (38.694678894817606, "/CHARM/Occipital/V2-V4/V2-V3/V2/clearly_V2")
# (38.694678894817606, "/CHARM/Occipital/V2-V4/V2-V3/V2")
# (35.77664854187505, "/CHARM/Occipital/V2-V4/V4")
# (26.328461432813288, "/CHARM/Occipital/V2-V4/V4/V4d")
# (14.983470554710237, "/CHARM/Occipital/MT")
# (14.722976118642089, "/CHARM/Occipital/V1")
# (10.65264742870875, "/CHARM/Parietal")
# (8.850838129796612, "/CHARM/Temporal/ITC/TE")
# (5.057534713882722, "/CHARM/Temporal/ITC/TE/gyral_TE/post_TE")
# (5.057534713882722, "/CHARM/Temporal/ITC/TE/gyral_TE")
# (4.9221818741417245, "/CHARM/Temporal/ITC/STSf")
# (4.270582910586465, "/CHARM/Occipital/V2-V4/V2-V3/V3")
# (3.9861741505091945, "/CHARM/Temporal/ITC/STSf/FST")
# (2.8109748439827507, "/CHARM/Occipital/V2-V4/V2-V3/V3/V3d|V3A")
# (2.5543400420524796, "/CHARM/Parietal/SPL/vm_IPS")
# (2.5543400420524796, "/CHARM/Parietal/SPL")
# (2.435451850796648, "/CHARM/Occipital/V2-V4/V2-V3/V3/V3d|V3A/V3A")
# (2.1892578209245035, "/CHARM/Temporal/ITC/TE/gyral_TE/post_TE/TEpv")
# (1.10141816261955, "/CHARM/Parietal/IPL")
# (22.202134588439904, "/SARM/di/Thal/PThal")
# (22.202134588439904, "/SARM/di/Thal")
# (22.202134588439904, "/SARM/di")
# (19.037849304725693, "/SARM/di/Thal/PThal/Pul")
# (15.712570606469008, "/SARM/di/Thal/PThal/Pul/LPul")
# (9.252445126579143, "/SARM/tel")
# (2.5338666756469017, "/SARM/tel/BG/Str/DStr/Cd/CdT")
# (2.5338666756469017, "/SARM/tel/BG/Str/DStr/Cd")
# (2.5338666756469017, "/SARM/tel/BG/Str/DStr")
# (2.5338666756469017, "/SARM/tel/BG/Str")
# (2.5338666756469017, "/SARM/tel/BG")

