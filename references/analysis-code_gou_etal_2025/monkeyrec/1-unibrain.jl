(out::Make.@x("interm/depth-mapping-mac-model.dat"))() = begin
    objs = (
        outer=[
            "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.gray_surface.rsl.gii.obj"
            ##"/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/rh.gray_surface.rsl.gii.obj"
        ],
        middle=[
            "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.mid_surface.rsl.gii.obj"
            ##"/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/rh.mid_surface.rsl.gii.obj"
        ],
        inner=[
            "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.white_surface.rsl.gii.obj"
            ##"/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/rh.white_surface.rsl.gii.obj"
        ],
    )
    objs = map(objs) do files
        map(files) do f
            load(f)
        end
    end
    function gen_tree(objs)
        pts = SVector{3,Float32}[]
        NNHelpers.sample_objs(objs, 128) do p
            push!(pts, p)
        end
        pts = reduce(hcat, pts)
        (tree=NearestNeighbors.KDTree(pts), pts=pts)
    end

    innertree, outertree = gen_tree(objs.inner), gen_tree(objs.outer)


    mid = hcat([reduce(hcat, obj.position) for obj in objs.middle]...)
    adjs = SVector{2,Int}[]
    let offset = 0
        for obj in objs.middle
            for f in GeometryBasics.faces(obj)
                a, b, c = f .+ offset
                push!(adjs, sort(SVector(a, b)))
                push!(adjs, sort(SVector(b, c)))
                push!(adjs, sort(SVector(c, a)))
            end
            offset += length(obj.position)
        end
        adjs = reduce(hcat, unique(adjs))
    end
    adjs = adjs |> Flux.gpu

    mid = mid |> Flux.gpu
    get_ends(dir, len, sel) = begin
        mid[:, sel] .+ dir[:, sel] .* len[:, sel]
    end
    calc_tite(tite, sgn) = begin
        (sgn * 100.0f0) .* Flux.softplus(tite)
    end
    function loss_dist(pts, tree, K)
        KK = 8
        pts2 = Flux.ignore_derivatives() do
            inds, dists = NearestNeighbors.knn(tree.tree, pts |> Flux.cpu, KK)
            if K % 10 == 0
                #println("dists: ", dists[1:3])
            end
            tree.pts[:, reduce(vcat, inds)] |> Flux.gpu
        end
        @check_size pts2 (3, :)
        @check_size pts (3, :)
        pts = reshape(pts, 3, 1, :)
        pts2 = reshape(pts2, 3, KK, :)
        dpts = sum((pts .- pts2) .^ 2, dims=1)
        Flux.ignore_derivatives() do
            if K % 10 == 0
                #println("dpts: ", sqrt.(dpts)[:, :, 1:3] |>Flux.cpu)
                #println(pts[:, 1, 1:3] |>Flux.cpu)
                #println(pts2[:, 1, 1:3] |>Flux.cpu)
            end
        end
        Flux.mean(dpts) / 1.0f4
    end
    loss(m, K, obs) = begin
        sel = Flux.ignore_derivatives() do
            rand(axes(mid, 2), 1024) |> Flux.gpu
        end
        len2 = sum(m.dir .^ 2, dims=1)
        mti = calc_tite(m.ti, -1)
        mte = calc_tite(m.te, 1)
        l0 = Flux.mean(m.ti .^ 2 .+ m.te .^ 2)
        inners = get_ends(m.dir, mti, :)
        outers = get_ends(m.dir, mte, :)
        l1 = loss_dist(inners[:, sel], innertree, K)
        l2 = loss_dist(outers[:, sel], outertree, K)
        l3 = 100.0f0 * (1.0f0 - Flux.mean(sum(m.dir[:, adjs[1, :]] .* m.dir[:, adjs[2, :]], dims=1) ./ sqrt.(len2[:, adjs[1, :]] .* len2[:, adjs[2, :]])))
        l4 = Flux.mean(sum((inners[:, adjs[1, :]] .- inners[:, adjs[2, :]]) .^ 2, dims=1)) / (K < 3000 ? 1.0f4 : 1.0f5)
        l5 = Flux.mean(sum((outers[:, adjs[1, :]] .- outers[:, adjs[2, :]]) .^ 2, dims=1)) / (K < 3000 ? 1.0f4 : 1.0f5)

        l = l0 + l1 + l2 + l3 + l4 + l5
        Flux.ignore_derivatives() do
            if K % 10 == 0
                e1 = get_ends(m.dir, mti, :)
                e2 = get_ends(m.dir, mte, :)
                obs.vecs[] = reshape(vcat(e1, e2), 3, :) |> Flux.cpu
                obs.from[] = e1 |> Flux.cpu
                obs.to[] = e2 |> Flux.cpu
                @info "$K, loss $l = $l0 [$l1 $l2] $l3 [$l4 $l5]"
            end
        end
        l
    end

    obs = (; vecs=Ref{Any}(), from=Ref{Any}(), to=Ref{Any}())

    model = (;
        dir=rand(Float32, 3, size(mid, 2)) .- 0.5f0,
        te=rand(Float32, 1, size(mid, 2)),
        ti=-rand(Float32, 1, size(mid, 2))) |> Flux.gpu
    loss(model, 0, obs)

    obs = let
        fig = GLMakie.Figure()
        display(fig)
        ax = GLMakie.LScene(fig[1, 1])
        foreach(o -> GLMakie.mesh!(ax, o, color=:orange), objs.inner)
        GLMakie.linesegments!(ax, reshape(mid[:, adjs], 3, :) |> Flux.cpu, color=:red)
        vecs = GLMakie.Observable(obs.vecs[])
        from = GLMakie.Observable(obs.from[])
        to = GLMakie.Observable(obs.to[])
        GLMakie.linesegments!(ax, vecs, color=:gray)
        GLMakie.scatter!(ax, from, color=:red)
        GLMakie.scatter!(ax, to, color=:green)
        #return gen_tree(objs.outer)
        (; vecs=vecs, from=from, to=to)
    end

    optst = Flux.setup(Flux.Adam(0.015), model)
    @showprogress for K in 1:100*60
        l, g = Flux.withgradient(m -> loss(m, K, obs), model)
        Flux.update!(optst, model, g[1])
    end
    model.ti .= calc_tite(model.ti, -1)
    model.te .= calc_tite(model.te, 1)
    model = model |> Flux.cpu
    mid = mid |> Flux.cpu
    df = map(eachcol(model.dir), eachcol(model.ti), eachcol(model.te), eachcol(mid)) do dir, ti, te, pos
        dir = SVector(dir...)
        ldir = norm(dir)
        dir = dir / ldir
        ti = ti[1] * ldir
        te = te[1] * ldir
        pos = SVector(pos...)
        (; dir, ti, te, pos)
    end |> DataFrame
    JLD2.jldsave(out.path, true; df)
end

(out::Make.@x("interm/depth-mapping-mac.dat"))(mdl::Make.@x("interm/depth-mapping-mac-model.dat")) = begin
    model = JLD2.load(mdl.path, "df")
    obj = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.mid_surface.rsl.gii.obj")
    @assert reduce(hcat, obj.position) == reduce(hcat, model.pos)
    nmt = NmtDat.load_images()
    charm = nmt.charm[1:end÷2, :, :, 1]
    img = charm
    res = nmt.res
    ratio = 2
    Main.xxximg = img = Images.imresize(img, ratio=ratio, method=Interpolations.Constant())
    res = res ./ ratio
    buf = map(v -> (true, Inf32, NaN32, NaN32, NaN32, NaN32), img)
    hits = Tuple[]
    sample_mesh(obj, obj.position, model.dir, model.ti, model.te; perface=48) do vec
        @assert length(vec) == 8
        cent = SVector(vec[1:3]...)
        dir = SVector(vec[4:6]...)
        ti, te = vec[7:8]
        empty!(hits)
        @assert te >= 0 "$te >=0"
        @assert ti <= 0 "$ti <=0"
        te2, ti2 = map((0:60:te*1.5f0, 0:-60:ti*1.5f0)) do rng
            outside = false
            tt = last(rng)
            for t in rng
                p = cent + dir * t
                i = NmtDat.pos2idx(p, res)
                ii = NmtDat.pos2idx(p, nmt.res)
                if !(checkbounds(Bool, charm, ii...) && charm[ii...] != 0) && !outside
                    tt = t
                    outside = true
                end
                pp = NmtDat.idx2pos(i, res)
                dd = norm(pp - cent)
                push!(hits, (outside, i, dd, t))
            end
            tt
        end
        if te2 > ti2
            for (outside, i, dd, t) in hits
                if checkbounds(Bool, buf, i...)
                    buf[i...] = min(buf[i...], (outside, dd, clamp((t - te2) / (ti2 - te2), 0, 1), cent...))
                end
            end
        end
    end
    img = stack((map(v -> v[k], buf) for k in 3:6), dims=4)
    img = cat(img, img[end:-1:1, :, :, :], dims=1)
    # for k in axes(img, 4)
    #     save("/tmp/test$k.nrrd", img[:, :, :, k])
    # end
    JLD2.jldsave(out.path, true; img, res)
end

function xyz2dxyz(xyzs, depthimg, depthres; nneighbor=1)
    allci = CartesianIndices(@view depthimg[:, :, :, 1])
    idxs = map(xyzs) do xyz
        NmtDat.pos2idx(xyz, depthres)
    end
    idx2dxyz = @showprogress "mapxyz" map(unique(idxs)) do idx
        rng = intersect(CartesianIndices(((idx[d] .+ (-nneighbor:nneighbor) for d in 1:3)...,)), allci)
        dxyzm = SVector(Inf32, NaN32)
        for ii in rng
            dxyz = @view(depthimg[ii, :])
            dd = norm(idx .- Float32.(Tuple(ii)))
            dd = isnan(dxyz[1]) ? dd * 100 + 100 : dd
            dxyz = SVector(dd, dxyz...)
            dxyzm = min(dxyzm, dxyz)
        end
        idx => (dxyzm[2], SVector(dxyzm[3:end]...))
    end |> Dict
    dxyzs = map(xyzs) do xyz
        idx2dxyz[NmtDat.pos2idx(xyz, depthres)]
    end
    @info "NaN $(count(isnan, first.(dxyzs)))"
    dxyzs
end

function dxyz2uvw(dxyzs, objflat, objphy)
    samples = Pair[]
    sample_mesh(objflat, objflat.position, objphy.position; perface=64) do vec
        @assert length(vec) == 5
        push!(samples, SVector(vec[1:2]...) => SVector(vec[3:end]...))
    end
    tree = NearestNeighbors.KDTree(last.(samples))
    nnn = first.(NearestNeighbors.knn(tree, last.(dxyzs), 1)[1])
    map(dxyzs, nnn) do dxyz, nn
        uv = isnan(dxyz[1]) ? (NaN32, NaN32) : first(samples[nn])
        SVector(uv..., dxyz[1])
    end
end

function xyz2uvw(xyzs, depthimg, depthres, objflat, objphy; nneighbor=1)
    dxyzs = xyz2dxyz(xyzs, depthimg, depthres; nneighbor)
    dxyz2uvw(dxyzs, objflat, objphy)
end

function segs2anns(segs, Mod, nmt=nothing)
    if nmt===nothing
    nmt = Mod.load_images()
end
    anns = @showprogress "segs2anns" map(segs) do seg
        idx = Mod.pos2idx(seg.pos, nmt.res)
        ####@assert checkbounds(Bool, nmt.sarm, idx..., 1)
        Mod.idx2ann(nmt, idx)
    end
end

function xyz2proj(segs,)
    nmt = NmtDat.load_images()
    hiers = @showprogress "get hier" map(segs) do seg
        idx = NmtDat.pos2idx(seg.pos, nmt.res)
        @assert checkbounds(Bool, nmt.sarm, idx..., 1)
        hier = nothing
        for tag in (:charm, :sarm)
            img = nmt[tag]
            ba = view(img, idx..., :)
            ba[1] == 0 && continue
            hier = (tag, ba...)
            break
        end
        hier
    end
    tag2tbl = map((:sarm, :charm)) do tag
        tbl = nmt[Symbol(tag, :_key)]
        tbl = map(eachrow(tbl)) do r
            r[1] => r[2]
        end |> Dict
        tag => tbl
    end |> NamedTuple
    t2name(t) = begin
        t === nothing && return ""
        tbl = tag2tbl[t[1]]
        string("nmt|", t[1], join(("|$(tbl[v])" for v in unique(t[2:end]))))
    end
    h2h = map(unique(hiers)) do hier
        hier => t2name(hier)
    end |> Dict
    map(h -> h2h[h], hiers)
end

