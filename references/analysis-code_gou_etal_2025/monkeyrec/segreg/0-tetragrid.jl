module Tetra

using ProgressMeter
using LinearAlgebra
using StaticArrays

function generate(center, radius::BigFloat)
    @assert eltype(center) == BigFloat
    a = big.((0.0, 0.0, 1.0))
    x = big.(1.0) / 3
    y = sqrt(1 - x^2)
    b = big.((y, 0, -x))
    z, w = y .* sincos(big(π) * 2 / 3)
    c = big.((w, z, -x))
    d = big.((w, -z, -x))
    function check_lens(p, q, r)
        pq = norm(p .- q)
        qr = norm(q .- r)
        rp = norm(r .- p)
        @assert maximum(abs.([pq - qr, qr - rp, rp - pq])) < 1e-70
    end
    check_lens(a, b, c)
    check_lens(a, b, d)
    check_lens(a, c, d)
    check_lens(b, c, d)
    a = SVector(a .* radius .+ center)
    b = SVector(b .* radius .+ center)
    c = SVector(c .* radius .+ center)
    d = SVector(d .* radius .+ center)
    (a=a, b=b, c=c, d=d)
end
function workspace(root)
    env = (edge_adjs=Dict(), halfspace_cache=Dict{Int,Any}(), edgedist_cache=Dict{Int,Any}(), tetras=NamedTuple[], tree=Dict{Int,NTuple{2,Int}}(), active=Set{Int}())
    add_tetra!(env, (a=Float64.(root.a), b=Float64.(root.b), c=Float64.(root.c), d=Float64.(root.d), l=0))
    #env = (env..., tetras=map(t->(, l=t.l), env.tetras))
    env
end
function check_halfspace(env::NamedTuple, id::Union{Int,Nothing}, i, a, b, c, d, pt)
    calc() = begin
        cp=cross(a .- b, c .- b)
        cp/dot(cp, d .- b)
    end
    val = if id === nothing
        calc()
    else
        get!(calc, env.halfspace_cache, id*128+i)
    end
    dot(val, pt .- b)
end
function calc_ratios(env::NamedTuple, tetra::NamedTuple, pt; id=nothing)
    a, b, c, d = tetra[(:a, :b, :c, :d)]
    #@assert all(eltype.((a, b, c, d, pt)) .== BigFloat)
    x = check_halfspace(env, id, 1, a, b, c, d, pt)
    y = check_halfspace(env, id, 2, a, c, d, b, pt)
    z = check_halfspace(env, id, 3, a, d, b, c, pt)
    w = check_halfspace(env, id, 4, b, d, c, a, pt)
    (x, y, z, w)
end
function inside(env::NamedTuple, tetra::NamedTuple, pt; id=nothing)
    all(calc_ratios().>0)
end
function inside(tetra, pt)
    inside(NamedTuple(), tetra, pt)
end
function inside(env::NamedTuple, id::Int, pt)
    inside(env, env.tetras[id], pt, id=id)
end
function add_tetra!(env::NamedTuple, tetra)
    id = length(push!(env.tetras, tetra))
    a, b, c, d = tetra[(:a, :b, :c, :d)]
    for (p, q) in ((a, b), (a, c), (a, d), (b, c), (c, d), (d, b))
        ins = false
        adjs = get!(env.edge_adjs, (p, q)) do
            ins = true
            Int[]
        end
        if ins
            env.edge_adjs[(q, p)] = adjs
        end
        push!(adjs, id)
    end
    id
end
function dodevide!(env::NamedTuple, id, (p, q))
    function same_edge(a, b)
        (a, b) == (p, q) || (b, a) == (p, q)
    end
    a, b, c, d, l = env.tetras[id][(:a, :b, :c, :d, :l)]
    flip2 = false
    if same_edge(a, c)
        a, b, c, d = a, c, d, b
    elseif same_edge(a, d)
        a, b, c, d = a, d, b, c
    elseif same_edge(c, d)
        a, b, c, d = a, c, d, b
        flip2 = true
    elseif same_edge(d, b)
        a, b, c, d = a, d, b, c
        flip2 = true
    elseif same_edge(b, c)
        flip2 = true
    else
        @assert same_edge(a, b)
    end
    if flip2
        a, b, c, d = b, c, a, d
    end
    e = max.((a .+ b) ./ 2, (b .+ a) ./ 2)
    s = add_tetra!(env, (a=a, b=e, c=c, d=d, l=l + 1, from=(a, b)))
    t = add_tetra!(env, (a=e, b=b, c=c, d=d, l=l + 1, from=(a, b)))
    env.tree[id] = (s, t)
end
function dodevide_at!(env::NamedTuple, (p, q))
    tetras = env.edge_adjs[(p, q)]
    for i in tetras
        get(env.tree, i, nothing) === nothing || continue
        dodevide!(env, i, (p, q))
    end
end
function try_divide!(env::NamedTuple, i, maxlvl=15, maxdist=1000)
    get(env.tree, i, nothing) !== nothing && return -1
    tetra = env.tetras[i]
    tetra.l > maxlvl && return -2
    get(tetra, :inactive, false) && return -3
    a, b, c, d, l = tetra[(:a, :b, :c, :d, :l)]
    calc()=(norm(a .- b), norm(a .- c), norm(a .- d), norm(b .- c), norm(c .- d), norm(d .- b),)
    ab, ac, ad, bc, cd, db=get!(calc, env.edgedist_cache, i)
    if (m1 = max(ab, ac, ad)) > (m2 = max(bc, cd, db))
        m1 < maxdist && return -4
        if m1 == ac
            a, b, c, d = a, c, d, b
        elseif m1 == ad
            a, b, c, d = a, d, b, c
        else
            @assert m1 == ab
        end
    else
        m2 < maxdist && return -5
        if m2 == cd
            a, b, c, d = a, c, d, b
        elseif m2 == db
            a, b, c, d = a, d, b, c
        else
            @assert m2 == bc
        end
        a, b, c, d = b, c, a, d
    end
    dodevide_at!(env, (a, b))
    1
end
function locate_pt(env::NamedTuple, pt, hint=nothing)
    cur = 1
    if hint !== nothing && inside(env, hint, pt)
        cur = hint
    end
    while true
        #@assert inside(env, cur, pt)
        children = get(env.tree, cur, nothing)
        children === nothing && return cur
        hit = 0
        for c in children
            if inside(env, c, pt)
                cur = c
                hit += 1
            end
        end
        @assert hit > 0
    end
end
#=
function sweep_devide!(env.tetras, env.tree, maxlvl, maxdist)
    while true
        hits=0
        N=length(env.tetras)
        for i in 1:N
            hits+=try_divide!(env.tetras, env.tree, i, maxlvl, maxdist)
        end
        hits==0 && break
    end
end
function get_all_active(env.tetras, env.tree)
    active=Set()
    prev_hint=nothing
    @showprogress for ind in filter(i->ref_img2[i]!=0, rand(CartesianIndices(ref_img), 500_000))
        pt=(Tuple(ind)[1:3].-big(0.5)).*ref_res
        k=locate_pt(env.tetras, env.tree, pt, prev_hint)
        prev_hint=k
        #@info k
        push!(active, k)
    end
    active
end
function deactiveate!(env.tetras, active)
    for (i, t) in enumerate(env.tetras)
        i in active && continue
        env.tetras[i]=(t..., inactive=true)
    end
end
=#
function refine(env::NamedTuple, all_points, maxlvl)
    prev_hint = nothing
    @showprogress for (pt, maxdist) in all_points
        while true
            prev_hint = k = locate_pt(env, pt, prev_hint)
            push!(env.active, k)
            res = try_divide!(env, k, maxlvl, maxdist)
            #@info "k $k $res"
            res > 0 || break
        end
    end
    @info length(env.tetras)
    @info length(env.active)
end

end

#=
@rule :memdata :checksum "interm/tetra-grid-raw" let ref_img
########################
#1
#1
#0 1
#
###############################################################
####################################################################
###########################################################
############################################################
#######################################################
########################################################

    center=world/2
    root=let radius=big(1.0), tetra
        while true
            println("radius: $radius")
            tetra=Tetra.generate(center, radius)
            allin=true
            pts=[(ind&1, ind&2, ind&4).*world for ind in 0:7]
            all(pt->Tetra.inside(tetra, pt), pts) && break
            radius*=2
        end
        tetra
    end

    ind2point(ind)=(point=(Tuple(CartesianIndices(ref_img)[ind]) .- 0.5) .* ref_res, maxdist=(ref_img2[ind] != 0 ? 1000.0 : 5000.0))

    env=Tetra.workspace(root)
    all_points=LinearIndices(ref_img)
    all_points=rand(all_points, 100000)
    Tetra.refine(env, Iterators.map(ind2point, all_points), 36)
    env
end
=#
    ################################################################
    ###########################################
    ###########################################

@rule :memdata :checksum "interm/tetra-grid-raw" let ref_img_mask
    ref_img,ref_res=deepcopy(ref_img_mask.dat), ref_img_mask.res
    world=big.(size(ref_img).*ref_res)
    maxdist=big(800.0)
    npad=2

    nbins=ceil.(Int, world./maxdist)
    nbins=map(v->isodd(v) ? v : v+1, nbins)
    @info "nbins: $nbins"
    maxdist=world./nbins
    point2idx=Dict{Any,Int}()
    tetras=[]
    @showprogress for cidx in CartesianIndices((nbins..., ).+npad*2)
        function get_coord(seq, k)
            pred=((seq-1)>>(k-1))&1
            (cidx[k]+(pred!=0 ? 0 : -1)-npad).*maxdist[k]
        end
        verts=[(get_coord(seq, 1), get_coord(seq, 2), get_coord(seq, 3)) for seq in 1:8]
        configs=if isodd(sum(Tuple(cidx)))
            [
             (5,1,6,7),
             (2,1,4,6),
             (3,1,7,4),
             (8,4,7,6),
             (1,4,6,7),
            ]
        else
            [
             (1,2,5,3),
             (6,2,8,5),
             (4,2,3,8),
             (7,3,5,8),
             (2,3,8,5),
            ]
        end
        for inds in configs
            id=length(tetras)+1
            for i in inds
                pti=get!(()->length(point2idx)+1, point2idx, verts[i])
                push!(tetras, pti)
            end
        end
    end

    points=Vector{Any}(missing, maximum(values(point2idx)))
    for (pt, pti) in point2idx
        @assert points[pti]===missing
        points[pti]=[Float64.(pt)...]
    end
    points=reduce(hcat, points)
    tetras=reshape(tetras, 4, :)
    @info size(tetras)
    @info size(points)
    (points=points, tetras=tetras)
end

function sample_tetra_coefs()
    tetra=Tetra.generate(big.((0.5,0.5,0.5)), big(0.5))
    xyzs=rand(BigFloat, 3, 10000)
    samps=map(eachcol(xyzs)) do xyz
        Tetra.calc_ratios(NamedTuple(), tetra, xyz)
    end
    samps=filter(samps) do r
        all(r.>0)
    end
    samps=map(v->[v...], samps)
    samps=reduce(hcat, samps)
    if false
        @info size(samps)
        @info tetra[:a]
        pts=[reduce(.+, broadcast(.*, samp, [tetra[(:a,:b,:c,:d)]...])) for samp in eachcol(samps)]
        display(GLMakie.scatter(pts))
    end
    samps
end
    function batcross(a, b)
        @assert size(a, 1)==size(b,1)==3
        a1, a2, a3 = (view(a, k, :) for k in 1:3)
        b1, b2, b3 = (view(b, k, :) for k in 1:3)
        v1=a2.*b3-a3.*b2
        v2=a3.*b1-a1.*b3
        v3=a1.*b2-a2.*b1
        stack([v1,v2,v3], dims=1)
    end
    function batdot(a,b)
        sum(a.*b, dims=1)
    end
    import ChainRulesCore
    function ensure_non_negative(xs::AbstractArray)
        #println( "not using chainrule")
        e=1f20*eps(eltype(xs)(0))
        map(xs) do x
            x>e ?  -log(x) : -log(e)+(1/e)*((1+e-x)^2-1)
        end
    end
    function ChainRulesCore.rrule(::typeof(ensure_non_negative), xs::AbstractArray)
        #println( "using chainrule")
        e=1f20*eps(eltype(xs)(0))
        res=map(xs) do x
            x>e ?  -log(x) : -log(e)+(1/e)*((1+e-x)^2-1)
        end
        function diff(ys)
            ds=map(xs) do x
                x>e ? -1/x : (-1/e)*(1+e-x)
            end
            (ChainRulesCore.NoTangent(), ds.*ys)
        end
        (res, diff)
    end

    idx2pos(idx, res) = SVector{3}((Tuple(idx) .- 0.5f0) .* res)
    pos2idx(pos, res)=round.(Int, pos ./ res .+ 0.5f0)

function optimize_tetras(points, tetras, ref_img, ref_res; tgtvals=nothing)
    points=Float32.(points)
    tetras=Int32.(tetras)
    gausssigma=150f0

    fixed_verts=Int32[]
    for k in 1:3
        row=view(points, k, :)
        for e in extrema(row)
            append!(fixed_verts, findall(row.≈e))
        end
    end
    sort!(unique!(fixed_verts))
    if false
        display(GLMakie.scatter(points))
        GLMakie.scatter!(points[:, fixed_verts])
        GLMakie.DataInspector()
    end
    tetra_samps=sample_tetra_coefs()

    active_tetras=let
        blacks=Set{AbstractVector}()
        whites=Set{AbstractVector}()
        @showprogress for idx in CartesianIndices(ref_img)
            pos=idx2pos(idx, ref_res)
            posx=map(v->round(v/1000f0)*1000f0, pos)
            if ref_img[idx]>0.1f0
                push!(whites, posx)
            else
                push!(blacks, posx)
            end
        end
        blacks=reduce(hcat, collect(blacks))
        whites=reduce(hcat, collect(whites))
        @info "size $(size(blacks))"
        @info "size $(size(whites))"
        blacks=NearestNeighbors.KDTree(blacks)
        whites=NearestNeighbors.KDTree(whites)
        centers=map(eachcol(tetras)) do (a,b,c,d)
            a,b,c,d=map(i->view(points, :, i), (a,b,c,d))
            reduce(.+, (a,b,c,d))/4
        end
        centers=reduce(hcat, centers)
        _, wd=NearestNeighbors.knn(whites, centers, 1)
        _, bd=NearestNeighbors.knn(blacks, centers, 1)
        Int32.(findall(map(first, wd).<1500f0 .&& map(first, bd).<1500f0))
    end
    @info "nactive: $(length(active_tetras)/size(tetras,2))"
    active_points=Set(tetras[:, active_tetras])

    points=points|>Flux.gpu
    pointsdelta=fill!(similar(points), 0f0)
    tetras=tetras|>Flux.gpu
    ref_img=map(v->v>0.1f0 ? 1f0 : 0f0, ref_img)|>Flux.gpu
    ref_res=ref_res|>Flux.gpu
    all_edges=tetras[reshape([1,2,1,3,1,4,2,3,3,4,4,2], 2, :), :] |>Flux.gpu
    tetra_samps=Float32.(hcat(I(4).*0.92.+0.02, tetra_samps)) |>Flux.gpu
    position_shifts=randn(Float32, 3, 256).*gausssigma |>Flux.gpu
    sell_points=abs.(view(points, 3, :).-100*250).<2000 |>Flux.gpu
    tgtvals=tgtvals |>Flux.gpu

    function tetra_volume(verts)
        #println("size: $(size(verts))")
        a,b,c,d=(view(verts, :, k, :) for k in 1:4)
        res=batdot(batcross(a.-b, c.-b), d.-b)
        #println("xxx")
        res
    end
    stepsize=100

    function loss(K, Δpoints, obs_data)
        verts=Flux.ignore_derivatives(points).+Δpoints.*stepsize
        seltetra=Flux.ignore_derivatives(()->(rand(active_tetras, 16*1024) |>Flux.gpu))

        sel2=if K%8==0
            Colon()
        else
            seltetra
        end

        edges=verts[:, view(all_edges, :, :, sel2)]
        #println("size edges $(size(edges))")
        # size edges (3, 2, 6, 4096)
        edgelens=sum((view(edges, :, 1, :, :).-view(edges, :, 2, :, :)).^2, dims=1)
        # 1 6 4096
        edgelensavg=Flux.mean(edgelens, dims=2)
        # 1 1 4096
        l1=Flux.mean((sqrt.(edgelens)./500 .-1).^2)*100
        #l1=Flux.mean(edgelens)
        #println("here")

        volumes=tetra_volume(verts[:, view(tetras, :, sel2)])
        #println("volumes $(size(volumes)) $(size(edgelensavg))")
        volumes=reshape(volumes, :)./reshape(edgelensavg, :)
        Flux.ignore_derivatives() do
            nneg=count(volumes.<1)
            if nneg>0
                println("nneg $nneg")
            end
        end
        l3=Flux.mean(ensure_non_negative(volumes.-200))
        #println("here2")

        #sel=Flux.ignore_derivatives(()->(findall(reshape(any(sell_points[tetras], dims=1), :)) |>Flux.gpu))
        sel=tetras[:, seltetra]
        # 4 4096

        #sel I 4 4096
        vertssel=verts[:, sel]
        #F 3 4 4096
        samps=view(tetra_samps, :, 1:16)
        #r 4 16
        samps=sum(reshape(vertssel, size(vertssel)..., 1).*reshape(samps, 1, size(samps, 1), 1, :), dims=2)
        #println("samps: $(size(samps))")
        #(3, 1, 4096, 16)
        sampsshifts,sampsshiftsvals=Flux.ignore_derivatives() do
            shifts=samps.+reshape(view(position_shifts, :, 1:8), size(position_shifts, 1), :, 1, 1)
            #3 8  4096 16
            shiftsi=shifts./reshape(ref_res, :, 1, 1, 1).+0.5f0
            vals=map([view(shiftsi, k, :, :, :) for k in 1:3]...) do x, y, z
                idx=round.(Int, (x,y,z))
                if checkbounds(Bool, ref_img, idx...)
                    ref_img[idx...]
                else
                    0.0f0
                end
            end
            # 1 8 4096 16
            shifts, reshape(vals, 1, size(vals)...)
        end
        weights=exp.(-sum(((samps.-sampsshifts)./gausssigma).^2, dims=1))
        #1 8  4096 16
        sampsvals=sum(weights.*sampsshiftsvals, dims=2)./sum(weights, dims=2)
        #1 1  4096 16
        sampsvalstgt=Flux.ignore_derivatives() do
            tgt1=map(maximum(sampsvals, dims=4), minimum(sampsvals, dims=4)) do a,b
                (1-a<b-0) ? 1f0 : 0f0
            end
            tgt2=if tgtvals!==nothing
                reshape(tgtvals[seltetra], 1, 1, :, 1)
            else
                nothing
            end
            #@info "$(size(tgt1)) $(size(tgt2))"
            @something(tgt2, tgt1)
        end
        l2=Flux.mean(sum((sampsvals.-sampsvalstgt).^2, dims=4))*10000
    ###########################################################
    ######################################################
    ###################################################

        Flux.ignore_derivatives() do
            if K % 10 == 0
                tetra_samps[:, 5:end] .= tetra_samps[:, shuffle(axes(tetra_samps, 2)[5:end])]
                tetra_samps[:, 1:4] .= tetra_samps[:, shuffle(axes(tetra_samps, 2)[1:4])]
                position_shifts .= position_shifts[:, shuffle(axes(position_shifts, 2))]
                println("l1 $l1;  l2 $l2; l3 $l3; ")
                sel=findall(abs.(view(points, 3, :).-100*250).<512)
                sel=filter(i->in(i, active_points), sel|>Flux.cpu) |>Flux.gpu
                newpos=points[:, sel].+Δpoints[:, sel].*stepsize
                obs_data[]=(newpos./reshape(ref_res, :, 1))[1:2, :]|>Flux.cpu
            end
        end
        l1+l2+l3
    end

    obs_data=Ref{Any}()
    loss(0, pointsdelta, obs_data)
    obs_data=let
        obs=GLMakie.Observable(obs_data[])
        display(GLMakie.image(ref_img[:, :, 200]|>Flux.cpu, inspectable=false))
        GLMakie.scatter!(obs)
        GLMakie.DataInspector()
        obs
    end

    optst=Flux.setup(Flux.Adam(), pointsdelta)
    @showprogress for K in 1:40000
        _,g=Flux.withgradient(p->loss(K, p, obs_data), pointsdelta)
        Flux.update!(optst, pointsdelta, g[1])
        pointsdelta[:, fixed_verts].=0f0
        #pointsdelta[3, :].=0f0
    end

    points.+pointsdelta.*stepsize |>Flux.cpu
end

@rule :data "interm/tetra-grid" let ref_img_mask, env="interm/tetra-grid-raw"
    ref_img,ref_res=deepcopy(ref_img_mask.dat), ref_img_mask.res
    points, tetras=env[(:points, :tetras)]
    @info "optimize ......"
    points=optimize_tetras(points, tetras, ref_img, ref_res)
    (points=points, tetras=tetras)
end

@rule :data "interm/tetra-grid2" let ref_img_mask, prev="test-tetragrid-warp"
    ref_img, ref_res=deepcopy(ref_img_mask.dat), ref_img_mask.res
    points, tetras=deepcopy(prev[(:points, :tetras)])
    @info "optimize ......"
    points=optimize_tetras(points, tetras, ref_img, ref_res; tgtvals=map(v-> v>0.85 ? 1.0 : 0.0, prev.hitratio2))
    (points=points, tetras=tetras)
end

using StaticArrays
function warp_tetragird(tetragrid, N, ref_img, ref_res)
    plane_abcd = Array{Float64 ,3}(undef, 4, 4, size(tetragrid.tetras, 2))
    signs = Vector{Float64}(undef, size(tetragrid.tetras, 2))
    hitratio = Vector{Float16}(undef, size(tetragrid.tetras, 2))
    hitratio2 = Vector{Float16}(undef, size(tetragrid.tetras, 2))
    idx2id=Array{Int32, 3}(undef, size(ref_img))
    idx2id.=0
    idxscores=similar(idx2id, Float32)

    function coef_for_pt(pt, id)
        verts = reshape(pt, 3, 1)
        plane = view(plane_abcd, :, :, id)
        pabc=view(plane, 1:3, :)
        pd=view(plane, 4:4, :)
        sum(verts .* pabc, dims=1) .- pd
    end
    function coef_for_idx(idx, id)
        coef_for_pt(idx2pos(idx, ref_res), id)
    end
    function check_for_score(idx, id)
        coefs =coef_for_idx(idx, id)
        min(1-maximum(coefs), minimum(coefs))
    end
    function check_for(idx, id)
        check_for_score(idx, id) >0
    end
    function calc(a, b, c, d)
        @assert all(eltype.((a, b, c, d)) .== BigFloat)
        cp = cross(a .- b, c .- b)
        l = dot(cp, d .- b)
        res = (cp..., dot(cp, b)) ./ l
        @assert eltype(res) == BigFloat
        res
    end
    tetra_samps=sample_tetra_coefs()
    @showprogress for (id, (a, b, c, d)) in enumerate(eachcol(tetragrid.tetras))
        a, b, c, d = map(i -> big.(view(tetragrid.points, :, i)), (a, b, c, d))
        plane_abcd[:, 1, id] .= calc(b, d, c, a)
        plane_abcd[:, 2, id] .= calc(a, c, d, b)
        plane_abcd[:, 3, id] .= calc(a, d, b, c)
        plane_abcd[:, 4, id] .= calc(a, b, c, d)
        signs[id] = dot(cross(a .- b, c .- b), d .- b)
        nsamp, nhit, nhit2=0, 0, 0
        for K in 33:64
            pt = reduce(.+, broadcast(.*, view(tetra_samps, :, K), (a, b, c, d)))
            idx = pos2idx(pt, ref_res)
            if checkbounds(Bool, idx2id, idx...)
                isfg=ref_img[idx...] > 0.1
                if isfg
                    nhit2+=1
                end
                if check_for(CartesianIndex(idx...), id)
                    idx2id[idx...] = -id
                    idxscores[idx...] = 0
                    if isfg
                        nhit += 1
                    end
                end
            end
            nsamp += 1
        end
        hitratio[id] = nhit/nsamp
        hitratio2[id] = nhit2/nsamp
    end

    while true
        nmissing = 0
        nadd = 0
        @showprogress for idx in CartesianIndices(idx2id)
            id = idx2id[idx]
            if id>0
            elseif id<0
                id=-id
                idx2id[idx]=id
                for seq in 1:6
                    kk = ((seq - 1) >> 1) + 1
                    idx2 = Base.setindex(idx, idx[kk] + ((seq - 1) & 1 > 0 ? 1 : -1), kk)
                    if checkbounds(Bool, idx2id, idx2) && idx2id[idx2] ==0
                        if check_for(idx2, id)
                            idx2id[idx2] = -id
                            idxscores[idx2]=0
                            nadd+=1
                        end
                    end
                end
            else
                nmissing += 1
            end
        end
        @info "incr $nmissing $nadd"
        nadd==0 && break
    end
    todo=findall(idx2id.==0)
    pointtree=NearestNeighbors.KDTree(tetragrid.points)
    nearidxes, _=NearestNeighbors.knn(pointtree, reduce(hcat, map(idx->idx2pos(idx, ref_res), todo)), 4)
    tmptetras=Set{Int}()
    point2tetras=Dict{Int,Vector}()
    for (id, col) in enumerate(eachcol(tetragrid.tetras))
        for p in col
            tt=get!(()->Int[], point2tetras, p)
            push!(tt, id)
        end
    end
    @showprogress map(todo, nearidxes) do idx, idxes
        empty!(tmptetras)
            for i in idxes
                for t in point2tetras[i]
                    push!(tmptetras, t)
                end
            end
            res=maximum(Iterators.map(i->(check_for_score(idx,i), i), tmptetras))
            @info "$idx: $res"
            if res[1]<=0
                res=maximum(Iterators.map(i->(check_for_score(idx,i), i), axes(tetragrid.tetras, 2)))
            end
        idx2id[idx]=res[2]
        idxscores[idx]=res[1]
    end

    delta_init=rand(Float32, 3, N, size(tetragrid.points)[2:end]...).- .5f0
    (deepcopy(tetragrid)...,
        signs=Float32.(signs),
        plane_abcd=Float32.(plane_abcd),
        hitratio=hitratio,
        hitratio2=hitratio2,
        deltas=delta_init,
        res=reshape(Float32[ref_res...,], :, 1),
        idx2id=idx2id,
        scores=idxscores,
    )
end

@rule :memdata test_tetragrid_warp let tetragrid="interm/tetra-grid", ref_img_mask
    @info "xxxx"
    res=warp_tetragird(tetragrid, 2, deepcopy(ref_img_mask.dat), ref_img_mask.res)
    (res..., tetras=Int32.(res.tetras))
end
@rule :memdata test_tetragrid_warp2 let tetragrid="interm/tetra-grid2", ref_img_mask
    @info "xxxx"
    res=warp_tetragird(tetragrid, 2, deepcopy(ref_img_mask.dat), ref_img_mask.res)
    (res..., tetras=Int32.(res.tetras))
end

@rule :memdata show_tetra_grid let test_tetragrid_warp2
    x=deepcopy(test_tetragrid_warp2)
    faces=sizehint!(Dict(), size(x.tetras, 2)*3)
    function add_face(face, id)
        key=sort(SVector(face))
        faces[key]=(get(faces, key, ())..., (face,id))
    end
    @showprogress for (id, (a,b,c,d)) in enumerate(eachcol(x.tetras))
        add_face((b,d,c), id)
        add_face((a,c,d), id)
        add_face((a,d,b), id)
        add_face((a,b,c), id)
    end
    hitratio=x.hitratio2
    actfaces=sizehint!(Set(), length(faces))
    tetravals=sizehint!([], size(x.tetras, 2))
    meshpts=sizehint!([], length(faces))
    tetrapts=sizehint!([], size(x.tetras, 2))
    norminfo=sizehint!(Dict(), size(x.points, 2))
    edg_cnts=sizehint!(Dict(), length(faces))
    function add_norm(v, n)
        ns, nc=get(norminfo, v, (SVector((0.0, 0.0, 0.0)), 0))
        norminfo[v]=(ns.+n, nc+1)
    end
    function calc_norm(a,b,c)
        n=cross(c.-b, a.-b)
        n./norm(n)
    end
    function add_edg(a,b)
        key=sort(SVector((a,b)))
        edg_cnts[key]=get(edg_cnts, key, 0)+1
    end
    function add_tri(verts)
        a,b,c=map(v->SVector((x.points[:, v]...)), verts)
        n=calc_norm(a,b,c)
        push!(meshpts, a, b, c)
        r,s,t=verts
        add_norm(r, n)
        add_norm(s, n)
        add_norm(t, n)
        add_edg(r,s)
        add_edg(s,t)
        add_edg(t,r)
        return
        a,b,c=verts
        for (p,q) in ((a,b), (b,c), (c,a))
            for v in (p, q)
                push!(meshpts, SVector((x.points[:, v]...)))
            end
        end
    end
    function add_tet(id, lvl=0)
        pts=Flux.mean(x.points[:, x.tetras[:, id]], dims=2)
        push!(tetrapts, SVector((pts...)))
        push!(tetravals, (id, x.hitratio[id], x.hitratio2[id], lvl))
    end
    thr1,thr2=0.1,0.9
    @showprogress for (face, ids) in pairs(faces)
        nids=length(ids)
        if nids==1
            @assert hitratio[ids[1][2]]<0.1
        elseif nids==2
            a,b=map(ids) do (_, id)
                hitratio[id]
            end
            if a>thr2 && b<thr1
                add_tri(ids[1][1])
                add_tet(ids[2][2])
                add_tet(ids[1][2])
            elseif a<thr1 && b>thr2
                add_tri(ids[2][1])
                add_tet(ids[1][2])
                add_tet(ids[2][2])
            elseif a<thr1 && b<thr1
            elseif a>thr2 && b>thr2
            else
                push!(actfaces, face)
            end
        else
            @assert false
        end
    end

    thr_dp=0.95
    function test_edg(a,b)
        key=sort(SVector((a,b)))
        get(edg_cnts, key, 0)<2
    end
    for KKK in 1:80
        added = []
        @showprogress for face in actfaces
            ids = faces[face]
            nids = length(ids)
            @assert nids == 2
            a, b, c = map(v -> SVector((x.points[:, v]...)), ids[1][1])
            r,s,t=ids[1][1]
            if !test_edg(r,s) || !test_edg(s,t) || !test_edg(t,r)
                push!(added, face)
                continue
            end
            n = calc_norm(a, b, c)
            hit1, hit2 = 0, 0
            for v in ids[1][1]
                nn, nc = get(norminfo, v, (nothing, 0))
                #@info "$nn $n"
                if nc > 4
                    dp = dot(nn ./ norm(nn), n)
                    hit1 = min(hit1, dp)
                    hit2 = max(hit2, dp)
                end
            end
            if hit2 > thr_dp
                add_tri(ids[1][1])
                add_tet(ids[2][2], 2)
                add_tet(ids[1][2], 2)
                push!(added, face)
            elseif hit1 < -thr_dp
                add_tri(ids[2][1])
                add_tet(ids[1][2], 3)
                add_tet(ids[2][2], 3)
                push!(added, face)
            end
        end
        @info "nadd: $(length(added)) $thr_dp"
        for face in added
            delete!(actfaces, face)
        end
        if length(added)<100
            thr_dp-=0.05
        end
    end
    ####################
    #abc t1 t2
    #abc t1 t2
    #
#####################################################
    #filter!(v->v[2]>30000 && v[2]<31000, meshpts)
    #filter!(v->v[2]>30000 && v[2]<31000, tetrapts)
    label(_, i, _)="$((i+0, tetravals[i], tetravals[i-1]))"
    display(GLMakie.mesh([map(v->v[i], meshpts) for i in 1:3]..., color=:orange, inspectable=false))
    GLMakie.linesegments!([map(v->v[i], tetrapts) for i in 1:3]..., color=:red, inspector_label=label)
    GLMakie.DataInspector()
    return tetrapts
end


function xform_tetragrid(tetragrid, B::Union{AbstractRange,Colon}, x::AbstractArray, factor::Float32; gpudev)
    @assert eltype(x)==Float32
    siz1, siz2p... = size(x)
    @assert siz1 == 3
    x=reshape(x, siz1, :)
    # 3 N

    idx2id=tetragrid.idx2id
    idsx,plane_abc, plane_d=Flux.ignore_derivatives() do
        inds=round.(Int, x./tetragrid.res .+.5f0)
        ids=map([view(inds, k, :) for k in 1:3]...) do xi, yi, zi
            @assert checkbounds(Bool, idx2id, xi, yi, zi)
            idx2id[xi,yi,zi]
        end
        #@info "$(typeof(ids)) $(size(ids)) $(typeof(tetragrid.plane_abcd))"
        ids,tetragrid.plane_abcd[1:end-1, :, ids], tetragrid.plane_abcd[end:end, :, ids]
    end
    # N

    coefs=sum(reshape(x, siz1, 1, :).*plane_abc, dims=1) .-plane_d
    # 1 4 N
    #@info "$(typeof(idsx)) $(size(idsx))"
        #@info "$(typeof(idsx)) $(size(idsx)) $(typeof(tetragrid))"
    vertids=tetragrid.tetras[:, idsx]
    # 4 N
    
    # 3 B 4 N
    deltas=reshape(coefs, 1, size(coefs)...).*tetragrid.deltas[:, B, vertids]
    # 3 B 4 N
    deltas=sum(deltas, dims=3)
    # 3 B 1 N
    x=reshape(deltas, siz1, :, siz2p...).*factor
    @assert eltype(x)==Float32
    x
end



##########################################################################################
#optimize->points part
#
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#=
    
    (points=points, tetras=tetras, img=ref_img)
end
=#

#=
###############################################################################################
#
#faces
#
##############################################################################
###################################################################
################################################################


    tree=NearestNeighbors.KDTree(Float32.(points))
    inds=NearestNeighbors.inrange(tree, Float32.(points), 0.1)
    dists=[]
    for (i, ind) in enumerate(inds)
        for j in ind
            d=norm(points[:, i].-points[:, j])
            push!(dists, d)
        end
    end
    return dists




###################################################################################
#############################################################################
##########################################################################
#=
    radius=
    =#
    ########################
    ########################
    #####################
    points
end
=#

