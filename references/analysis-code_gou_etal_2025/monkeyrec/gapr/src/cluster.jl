
#==
Make.@target decimate_swcs (xform_swcs_mirror) -> begin
	indir=deepcopy(xform_swcs_mirror)
	name2pos=Dict()
	for f in readdir(indir)
		m=match(r"^.*\.swc$", f)
		m!==nothing || continue
		pos=get_soma_pos(joinpath(indir, f))
		name2pos[f]=pos
	end

	tmpdir="/data/tmp/prev/spl"
	idx2name=Dict()
	dist=function(a, b)
		c=a.-b
		sqrt(sum(c.*c))
	end
	for f in readdir(tmpdir)
		m=match(r"^(\d+)\.fnt\.swc$", f)
		m!==nothing || continue
		idx=parse(Int, m.captures[1])
		pos=get_soma_pos(joinpath(tmpdir, f))
		k0=nothing
		d0=Inf
		for k in keys(name2pos)
			v=name2pos[k]
			d=dist(v, pos)
			if d<d0
				d0=d
				k0=k
			end
		end
		!haskey(idx2name, idx) || error("dup")
		idx2name[idx]=k0
		delete!(name2pos, k0)
	end
	ii=collect(keys(idx2name))
	println(sort(ii))
	println(length(ii))
	sort(ii)==0:length(ii)-1 || error("wrong")
	DataFrame(:fn=>map(i->idx2name[i], sort(ii)))
end
==#

##################################################

##############################
#################################
###################################


function remap_groups(indir, outdir, fns, cls)
	rm("$outdir", force=true, recursive=true)
	rm("$outdir.temp", force=true, recursive=true)
	mkdir("$outdir.temp")
	fns=pmap(1:length(fns)) do i
		fn=fns[i]
		v=readdlm(joinpath(indir, fn), ' ', comments=true)
		idadd=(10*i+cls[i])*100000000
		v=map(1:size(v)[1]) do j
			(Int64(v[j, 1])%100000000+idadd, Int(v[j, 2]), v[j, 3], v[j, 4], v[j, 5], v[j, 6], v[j, 7]!=-1 ? Int64(v[j, 7])%100000000+idadd : -1)
		end
		rr=v[1][1]
		m=match(r"^.*@(.*)\.swc(\.gz)?", fn)
		m=m.captures[1]
		cc=cls[i]
		fn="$cc-$fn"
		open(joinpath("$outdir.temp", fn), "w") do io
			writedlm(io, v, ' ')
			write(io, "#Gapr!$rr@root=$m\n")
		end
		fn
	end
	touch(joinpath("$outdir.temp", "dir"))
	mv("$outdir.temp", outdir)
	fns
end


Make.@target select_brain_area (cluster_hy_calc) -> begin
	fns, mat, h, cls, proj_mat1, proj_mat2=deepcopy(run_cluster)
	ann_ids=allen_ids_rev()
	projmax=max.(maximum(proj_mat1, dims=1), maximum(proj_mat2, dims=1))
	projmax=maximum(projmax)
	bounds=(projmax^(1/16)).^(0:16)
	#return bounds
	function get_cls(val)
		for i=1:length(bounds)
			if val<bounds[i]
				return i
			end
		end
		return length(bounds)+1
	end
	df=map(1:length(ann_ids)) do j
		cls2=map((u,v)->(get_cls(u)*100+get_cls(v)), proj_mat1[:, j], proj_mat2[:, j])
		mi=mutualinfo(cls,cls2)
		(j=j, ba=get_brain_area_hier(ann_ids[j]), mi=mi)
	end
	df=DataFrame(df)
end

####################
####################################################################################
	########################################################################################################################
	#########################
	#####################
	########################
	##########################
###########################################
#############################

#############################################################
##########################################################################################
##########################################################################################
#=
function xxxxxxxxxxx()

	#return
	#fns=fns[cls.==3]
	cls=map(fns) do fn
		m=match(r"^\d+-(\d+)-.*.swc", fn)
		m!==nothing || error("wrong name")
		ds=m.captures[1]
		if in(ds, ["17391", "18133"])
			return 1
		elseif in(ds, ["192095", "192096"])
			return 2
		else
			error("not possible")
		end
	end
	remap_groups("/tmp/swcs-xform-clustered", "/tmp/swcs-xform-clustered2", fns, cls)
###################################
	
	#############################
############################################
################################################
##############################################
		#
		#
		#
		#Axon, VMH, HY, valid, ALL/HY/HY/VMH/VMH, gbgbgb, axon ident, axon ident

###############################################################
###########################################################################

	# selection!!! XXX
	if false
		# *1 *2 long
		# .3 .4 .5 local
		sel=collect(1:2)
		seli=filter(v->in(cls[v], sel), collect(1:length(cls)))
	elseif false
		seli=filter(i->match(r"-17391-", fns[i])===nothing, 1:length(fns))
	else
			#if ba!="/root/grey/BS/IB/HY/MEZ/VMH"
				#return false
			#end
			println(nn_axon, '\t', fns[i])
			return true
		end
	end
	fns[seli], mat[seli, seli]
	fns
end
=#


Make.@target run_cluster3 (cluster_hy_calc) -> begin
	fns, mat=deepcopy(run_cluster)
	indir="/tmp/swcs-xform-clustered"
	outdir="/tmp/swcs-xform-clustered2"
	h=hclust(mat, linkage=:complete, branchorder=:barjoseph)
	cls=cutree(h, k=6)
	fns=remap_groups(indir, outdir, fns, cls)
	fns,cls,h,strengths
end

Make.@target run_cluster4 (run_cluster3) -> begin
	indir="/tmp/swcs-xform-clustered2"
	fns,cls,h,strengths=deepcopy(run_cluster3)

	#ann_ids=allen_ids_rev()
	begin
		a=map(ann_ids) do id
			get_brain_area_hier(id)=="/root/grey/CH/CTX"
		end
		a=findfirst(a)
		a=fns[sortperm(mat[:, a])[end]]
		println(a)
	end

	#h=hclust(1 .-cor(mat'), linkage=:complete, branchorder=:barjoseph)
	#cls=cutree(h, k=6)
	#println(h.heights)

	df=map(1:length(fns)) do i
		pos=get_soma_pos(joinpath(indir, fns[i]))
		id=_get_allen_data(pos)
		ba=get_brain_area_hier(id)
		(c=cls[i], ba=id, fn=fns[i], bas=ba)
	end
	df=DataFrame(df)

	p1=plot([0,1], color=:transparent, label=tex("{}"))
	ordered_cls=unique(cls[h.order])
	for i=1:length(ordered_cls)
		c=ordered_cls[i]
		l,r=extrema(findall(map(i->cls[i]==c, h.order)))
		vspan!(p1, [l-.5, r+.5],
		       label=tex("#$i"),
		       color=palette(:auto)[c],
		       )
	end
	plot!(p1, h,
	      extra_kwargs=Dict(:subplot=>Dict(
					       "scale only axis"=>true,
					       )),
	      label="",
	      linewidth=.5,
	      framestyle=:none,
	      ylim=(0, maximum(h.heights)*1.05),
	      xlim=(0, length(h.order)+1),
	      )
	plot!(p1,
	      legend_title="Clusters",
	      legend=:inline,
	      legend_column=2,
	      #legend=:inside,
	      #legend=:legend,
	      legend_title_font_pointsize=7,
	      legend_font_pointsize=6,
	      legend_foreground_color=:transparent,
	      )
	#####################################################
	#####################################################
	#####################################################
	#####################################################
	######################################################
	sumy=sum(mat, dims=1)[1, :]
	maxy=maximum(mat, dims=1)[1, :]
	stdy=std(mat, dims=1)[1, :]

	sel_ids=sumy.>0.001
	sumy=sumy[sel_ids]
	stdy=stdy[sel_ids]
	maxy=maxy[sel_ids]
	mat=mat[:, sel_ids]
	ann_ids=ann_ids[sel_ids]
	#h2o=hclust(1 .-cor(mat), linkage=:single, branchorder=:barjoseph).order
	h2o=sortperm(map(v->get_brain_area_hier(v), ann_ids))
	exclude=[997,8,343,1129,1065,313,323, 1097]
	sely=map(h2o) do i
		a=ann_ids[i]
		#XXX
		maxy[i]>5.100 && !in(a, exclude)
		#sumy[i]>800 && !in(a, exclude)
		#stdy[i]>2 && !in(a, exclude)
	end
	sel_bas=[
		 #"/root/fiber tracts",
		 #"/root/fiber tracts/cm",
		 #"/root/fiber tracts/mfbs",
		 "/root/grey/BS/HB/MY",
		 "/root/grey/BS/HB/P",
		 "/root/grey/BS/IB/HY",
		 #"/root/grey/BS/IB/HY/LZ",
		 #"/root/grey/BS/IB/HY/MEZ",
		 #"/root/grey/BS/IB/HY/PVR",
		 #"/root/grey/BS/IB/HY/PVZ",
		 "/root/grey/BS/IB/TH",
		 #"/root/grey/BS/IB/TH/DORpm",
		 #"/root/grey/BS/IB/TH/DORsm",
		 "/root/grey/BS/MB",
		 #"/root/grey/BS/MB/MBmot",
		 #"/root/grey/BS/MB/MBsen",
		 #"/root/grey/CB",
		 #"/root/grey/CB/CBX",
		 #"/root/grey/CH",
		 #"/root/grey/CH/CNU",
		 #"/root/grey/CH/CNU/PAL",
		 #"/root/grey/CH/CNU/STR",
		 #"/root/grey/CH/CTX",
		 #"/root/grey/CH/CTX/CTXpl",
		 #"/root/grey/CH/CTX/CTXpl/HPF",
		 #"/root/grey/CH/CTX/CTXpl/Isocortex",
		 #"/root/grey/CH/CTX/CTXpl/OLF",
		 #"/root/grey/CH/CTX/CTXsp",
		 ]
	sely=map(h2o) do i
		a=ann_ids[i]
		in(get_brain_area_hier(a), sel_bas)
	end
	sely=h2o[sely]
	println(map(v->get_brain_area_hier(v), ann_ids[sely]))

	p2=heatmap(mat[h.order, sely]',
		   fillcolor=cgrad(:OrRd_9),
		   extra_kwargs=Dict(:subplot=>Dict(
						    "scale only axis"=>true,
						    "every colorbar/.append style"=>"height=2mm,width=20mm,at={(1.03,-0.01)},anchor=north west,xlabel style={font={{\\fontsize{7 pt}{0pt}\\selectfont\\bfseries}},anchor=base west,at={(1.0,0)}},xticklabel pos={lower}",
						    "xlabel style"=>"{anchor=west, at={{(ticklabel cs:0.0)}}}",
						    )),
		   ymirror=true,
		   grid=false,
		   yticks=(1:length(sely), map(v->tex(get_brain_area_hier(ann_ids[v])), sely)),
		   foreground_color_border=:white,
		   showaxis=:y,
		   tick_direction=:none,
		   colorbar_title=tex("Length (\\si{mm})"),
		   colorbar_ticks=0:40:120,
		   legend=:right,
		   colorbar_tickfontsize=5,
		   #tickfontsize=6,
		   tickfontsize=5,
		   colorbar=:top,
		   xlim=(0, length(h.order)+1),
		   xguide="Neurons",
		   guidefontsize=7,
		   xguide_position=:left,
		   #xticks=(1:length(h.order), map(v->fns[v], h.order)),
		   )
	p=plot(p1, p2, layout=grid(2, 1, heights=[0.3,0.6]),
	       link=:x,
	       size=(600,360),
	       )
	savefig(p, "extra/clustering.tex")
	df

end

Make.@target run_cluster5 (run_cluster4) -> begin
	df=deepcopy(run_cluster4)
	sd=map(v->(v, length(findall(df[:, :bas].==v))), sort(unique(df[:, :bas])))
	q=bar(map(v->v[1], sd), map(v->v[2], sd),
	      permute=(:x, :y),
	      #%xflip=true,
	      xmirror=true,
	      legend=false,
	      title="Soma distrubution",
	      )
	savefig(q, "extra/soma-dist.tex")
	q
end

#####################good til #
Make.@target compare_fnt (snapshots2) -> begin
	df=deepcopy(snapshots2)
	dss=Dict("17391"=>.32/.3, "18133"=>.32/.35,
		 "192095"=>1, "192096"=>1)
	f1s=map(collect(keys(dss))) do ds
		stg=maximum(df[df[:, :ds].==ds, :stg])
		table=map(readdir(cache_file_swcdir(ds, stg), join=true)) do fn
			pos=get_soma_pos(fn)
			(pos, fn)
		end

		swcs=readdir("/data/dat/tracing_FNT/$ds/swc_raw", join=true)
		outdir="/tmp/swcs-fnt-fixed"
		rm("$outdir", force=true, recursive=true)
		mkdir("$outdir")
		xyscale=dss[ds]
		cmd=gapr_cmd(`map $swcs -o "$outdir/%s%e" -x "awk \"{print \$1*$xyscale,\$2*$xyscale,\$3}\""`)
		println(cmd.exec)
		cc=run(cmd)
		check_exit(cc)

		swcs=readdir("/data/dat/tracing_FNT/$ds/swc_raw")
		res=pmap(swcs) do swc
			swc=joinpath(outdir, swc)
			pos=get_soma_pos(swc)
			refbest=nothing
			refdist=Inf
			for (pos2, fn) in table
				d=sum((pos2.-pos).^2)
				if d<refdist
					refdist=d
					refbest=fn
				end
			end
			v=open(gapr_cmd(`brief --f1score $refbest $swc`), "r") do io
				parse(Float64, split(readline(io), ' ')[1])
			end
			(v, swc, refbest)
		end
		map(res) do v
			q=check_neuron_qc(ds, stg, basename(v[3]))>1
			(v..., q)
		end
	end
	vcat(f1s...)
end
Make.@target compare_fnt_plot (compare_fnt) -> begin
	x=deepcopy(compare_fnt)
	function get_ds(tup)
		match(r"/results-(\d+)-", tup[3]).captures[1]
	end
	dss=unique(map(v->get_ds(v), x))
	pp=map(dss) do ds
		histogram(map(v->v[1], filter(v->get_ds(v)==ds && v[4], x)),
			  legend=false,
			  xguide="f1score",
			  yguide="count",
			  title=ds,
			  bins=range(0, 1, 10)
			  )
	end
	p=plot(pp..., layout=grid(2, 2), size=(1280, 960),)
	p
end

Make.@target compare_fnt_soma (compare_fnt, cluster_hy_calc) -> begin
	x=deepcopy(compare_fnt)
	pag=findfirst(allen_ids_rev()) do id
		match(r"/PAG$", get_brain_area_hier(id))!==nothing
	end
	mpo=findfirst(allen_ids_rev()) do id
		match(r"/MPO$", get_brain_area_hier(id))!==nothing
	end
	println("$pag/$mpo")
	x=map(x) do (s, f1, f2, q)
		q || return nothing
		m=match(r"/results-(\d+)-(\d+).*/(\d+@.*.swc(\.gz)?)", f2)
		m!==nothing || error("wrong m")
		ds, stg, nm2=(m.captures..., )
		m=match(r"/tmp/swcs-fnt-fixed/(\d+\.swc(\.gz)?)", f1)
		m!==nothing || error("wrong m2")
		nm1=m.captures[1]
		p2=get_soma_pos(joinpath(cache_dir_swcs_xform(), "$ds-$stg-$nm2"))
		p1=get_soma_pos("/data/dat/tracing_FNT/$ds/swc_allen_space_coronal/$nm1")
		#p1=get_soma_pos("/data/dat/tracing_FNT/$ds/swc_allen_space/$nm1")
		i=findall(run_cluster[1]) do fn
			m=match(r"\d+-(\d+)-\d+-(\d+@.*.swc(\.gz)?)", fn)
			m!==nothing || error("a")
			m.captures[1]==ds && m.captures[2]==nm2
		end
		length(i)==1 || return nothing# error("asdf $i $ds $nm2")
		i=i[1]
		(ds=ds, nm=nm2, pfnt=p1, pgapr=p2, ffnt=f1,
		 pgapr_raw=get_soma_pos(f2),
		 pfnt_raw=get_soma_pos("/data/dat/tracing_FNT/$ds/swc_raw/$nm1"),
		 cls=run_cluster[4][i],
		 pag=run_cluster[5][i, pag],
		 mpo=run_cluster[5][i, mpo],
		 )
	end
	x=filter(v->v!==nothing, x)
	return DataFrame(x)
############
end

# XXX end
