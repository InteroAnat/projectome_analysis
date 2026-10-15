function nurbs_gen_basis(deg::Int, n::Int)
	(deg=deg, knots=[clamp(i-deg, 0, n-deg+1)/(n-deg+1) for i in 0:n+deg+1])
end
function nurbs_find_idx(bas, t::Real)
	a,b=1,length(bas.knots)+1
	while a+1<b
		c=floor(Int, (a+b)/2)
		if t<bas.knots[c]
			b=c
		else
			a=c
		end
	end
	min(a-1, length(bas.knots)-bas.deg-1)
end
function nurbs_cond_sum(a, b)
	if a!==nothing
		if b!==nothing
			return a.+b
		end
		return a
	end
	if b!==nothing
		return b
	end
	error("nothing")
end
function nurbs_eval_basis(bas, tvec)
	deg=bas.deg
	first_i=Zygote.@ignore [nurbs_find_idx(bas, t) for t in tvec]
	idx0=Zygote.@ignore [clamp(i0-deg, 0, length(bas.knots)-2-2*deg) for i0 in first_i]
	vals=Zygote.@ignore [[1.0 for k in 1:length(idx0)]]
	knots_cache=Zygote.@ignore [bas.knots[i+idx0[k]] for i in 2:2*deg+1, k in 1:length(tvec)]
	for j in 1:deg
		ww=nothing
		next_vals=map(0:j) do jj
			left=jj>0 ? ww.*vals[jj] : nothing
			if jj<j
				ww=(tvec.-knots_cache[jj-j+deg+1, :])./(knots_cache[jj+deg+1, :].-knots_cache[jj-j+deg+1, :])
			end
			right=jj<j ? (1 .-ww).*vals[jj+1] : nothing
			nurbs_cond_sum(left, right)
		end
		vals=next_vals
	end
	vals, idx0
end
function nurbs_surf_create(m::Int, n::Int, A, B, C, xform=nothing)
	ubas=nurbs_gen_basis(3, m)
	vbas=nurbs_gen_basis(3, n)
	weights=[1.0 for i in 0:m, j in 0:n]
	P=[A.+B.*(i/m-.5).+C.*(j/n-.5) for i in 0:m, j in 0:n]
	P=reshape(reduce(hcat, reshape(P, :)), 3, 1+m, 1+n)
	if xform!==nothing
		P=xform(P)
	end
	(m=m, n=n, w=weights, P=P, ubas=ubas, vbas=vbas)
end
function nurbs_surf_points(surf, uvvec)
	Ni, Ni0=nurbs_eval_basis(surf.ubas, uvvec[1, :])
	Nj, Nj0=nurbs_eval_basis(surf.vbas, uvvec[2, :])
	res=map(axes(uvvec, 2)) do k
		wij=[Ni[di+1][k]*Nj[dj+1][k] for di in 0:3, dj in 0:3]
		irng=Zygote.@ignore (0:3).+(Ni0[k]+1)
		jrng=Zygote.@ignore (0:3).+(Nj0[k]+1)
		wij=wij.*surf.w[irng,jrng]
		Pts=reshape(wij, 1, 4, 4).*surf.P[:, irng, jrng]
		pt=reshape(sum(Pts, dims=(2,3))./sum(wij), 3)
		pt
	end
	reduce(hcat, res)
end
function nurbs_change(surf; shift=nothing, scale=nothing)
	newP=shift===nothing ? surf.P : shift#surf.P.+shift
	newW=scale===nothing ? surf.w : scale#surf.w.*scale
	(m=surf.m, n=surf.n, w=newW, P=newP, ubas=surf.ubas, vbas=surf.vbas)
end
function nurbs_surf_samp(surf, nsamp=3000)
	uvvec=Zygote.@ignore reduce(hcat, [rand(2) for i=1:nsamp])
	inside=Zygote.@ignore [c[1]<.8 && c[1]>.2 && c[2]<.8 && c[2]>.2 for c in eachcol(uvvec)]
	nurbs_surf_points(surf, uvvec), inside
end
function nurbs_to_mesh(surf)
	uvs=[]
	uv2vi=Dict()
	for u in 0:1:50, v in 0:1:50
		push!(uvs, [u/50; v/50])
		uv2vi[(u, v)]=length(uvs)
	end
	verts=nurbs_surf_points(surf, reduce(hcat, uvs))
	verts=[Point3(c...) for c in eachcol(verts)]
	function get_v(u, v)
		r=get(uv2vi, (u, v), nothing)
		#@info "$u $v"
		@assert r!==nothing
		r
	end
	faces=[]
	for u in 0:1:49, v in 0:1:49
		a=get_v(u, v)
		b=get_v(u+1, v)
		c=get_v(u+1, v+1)
		d=get_v(u, v+1)
		push!(faces, [a, b, c])
		push!(faces, [c, d, a])
	end
	faces=reduce(hcat, faces)
	faces=[NgonFace(c...) for c in eachcol(faces)]
	GeometryBasics.Mesh(verts, faces)
end
