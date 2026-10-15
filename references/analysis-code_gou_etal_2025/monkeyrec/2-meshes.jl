function smooth_binary_mask(seg::AbstractArray{T,3}) where {T<:AbstractFloat}
    zslice = findmax(sum(seg, dims=(1, 2)))[2][3]
    GLMakie.activate!()
    fig = GLMakie.Figure()
    display(fig)
    GLMakie.image(fig[1, 1], seg[:, :, zslice])
    allindices = CartesianIndices(seg)
    coefs = @showprogress "preparing" map(allindices) do idx
        inds = intersect(allindices, CartesianIndices(((idx[d]-1:idx[d]+1 for d in 1:3)...,)))
        @assert !isempty(inds)
        n1 = count(>(0.5f0), view(seg, inds))
        n0 = length(inds) - n1
        if n0 == 0
            (0.91f0, 1.0f0)
        elseif n1 == 0
            (0.0f0, 0.09f0)
        else
            if seg[idx] > 0.5
                (0.501f0, 0.9999f0)
            else
                (0.0001f0, 0.499f0)
            end
        end
    end
    coefs = (map(first, coefs), map(last, coefs)) |> Flux.gpu
    calc_img(p) = begin
        Flux.sigmoid.(p) .* (coefs[2] .- coefs[1]) .+ coefs[1]
    end
    diffimg1(a, b) = Flux.binarycrossentropy(a, b)
    diffimg2(a, b) = Flux.mean((a .- b) .^ 2) * 10.0f0
    #diffimg2(a, b) = Flux.huber_loss(a, b; delta=0.05) * 1000.0f0
    loss(p; captimg) = begin
        captimg[] = img = calc_img(p)
        l1 = diffimg2(@view(img[1:end-1, :, :]), @view(img[2:end, :, :]))
        l2 = diffimg2(@view(img[:, 1:end-1, :]), @view(img[:, 2:end, :]))
        l3 = diffimg2(@view(img[:, :, 1:end-1]), @view(img[:, :, 2:end]))
        l1 + l2 + l3
    end
    par = zeros(Float32, size(seg))
    seg, par = (seg, par) |> Flux.gpu
    img2 = GLMakie.Observable(calc_img(par)[:, :, zslice] |> Flux.cpu)
    GLMakie.image(fig[1, 2], img2)
    st = Flux.setup(Flux.RMSProp(), par)
    @showprogress "optimizing" for iter in 1:10000
        captimg = Ref{Any}()
        l, grad = Flux.withgradient((p) -> loss(p; captimg), par)
        Flux.update!(st, par, grad[1])
        if iter % 100 == 0
            img2[] = captimg[][:, :, zslice] |> Flux.cpu
        end
    end
    Float32.(calc_img(par)) |> Flux.cpu
end
function generate_mesh_from_mask_mc(vol)
    mc = MarchingCubes.MC(1.0f0 .- vol)
    MarchingCubes.march(mc, 0.5f0)
    vertices = map(GeometryBasics.Point3f, mc.vertices)
    normals = map(GeometryBasics.Vec3f, mc.normals)
    triangles = map(t -> GeometryBasics.TriangleFace(t...), mc.triangles)
    GeometryBasics.Mesh(GeometryBasics.meta(vertices; normals=normals), triangles)
end
function generate_mesh_from_mask_surfnet(vol)
    algo = Meshing.MarchingCubes(iso=0.5f0, insidepositive=true)
    algo = Meshing.MarchingTetrahedra(iso=0.5f0, insidepositive=true)
    algo = Meshing.NaiveSurfaceNets(iso=0.5f0, insidepositive=true)
    GeometryBasics.Mesh(vol, algo)
end
function generate_mesh_from_mask_mt(vol)
    algo = Meshing.MarchingCubes(iso=0.5f0, insidepositive=true)
    algo = Meshing.MarchingTetrahedra(iso=0.5f0, insidepositive=true)
    GeometryBasics.Mesh(vol, algo)
end
function decompose_mesh(obj)
    pos2faces = Dict{Int,Set}()
    pos2vert = Dict{Int,Int32}()
    for (fi, f) in enumerate(faces(obj))
        for v in f
            p = obj.position[v]
            adjs = get!(Set, pos2faces, v + 0)
            push!(adjs, fi)
            push!(pos2vert, v => v)
        end
    end
    npos = length(pos2vert)
    npos2 = length(obj.position)
    @info "npos $npos/$npos2"
    face2comp = Dict([f => 0 for f in axes(faces(obj), 1)])
    todo = collect(axes(faces(obj), 1))
    while !isempty(todo)
        fi = pop!(todo)
        tt = face2comp[fi]
        tt > 0 && continue
        tt = tt < 0 ? -tt : (@warn "new comp"; fi)
        face2comp[fi] = tt
        for v in faces(obj)[fi]
            p = obj.position[v]
            for ffi in pos2faces[v+0]
                if face2comp[ffi] == 0
                    face2comp[ffi] = -tt
                    push!(todo, ffi)
                end
            end
        end
    end
    submeshes = GeometryBasics.Mesh[]
    for comp in unique(values(face2comp))
        subfaces = GeometryBasics.TriangleFace[]
        parfaces = GeometryBasics.faces(obj)
        for (f, c) in face2comp
            if c == comp
                a, b, c = map(v -> pos2vert[v+0], parfaces[f])
                if length(unique((a, b, c))) < 3
                    @info "trim face $f"
                    continue
                end
                ###push!(subfaces, GeometryBasics.TriangleFace(a, b, c))
                a, b, c = parfaces[f]
                push!(subfaces, GeometryBasics.TriangleFace(a, b, c))
            end
        end
        push!(submeshes, normal_mesh_newvid(obj.position, subfaces))
    end
    sort!(submeshes, by=length, rev=true)
end

function normal_mesh_newvid(ppp, fff)
    i2i = Dict{Int,Int}()
    for f in fff
        for v in f
            v = v + 0
            get!(i2i, v, length(i2i) + 1)
        end
    end
    ppp = ppp[first.(sort(collect(i2i), by=last))]
    fff = map(fff) do f
        map(v -> i2i[v+0], f)
    end
    normal_mesh(ppp, fff)
end


function gmsh_get_nodes()
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    @assert length(ntags) * 3 == length(coords)
    points = map(c -> Point3f(c), eachcol(reshape(coords, 3, :)))
    tag2idx = Dict(reverse.(enumerate(ntags)))
    (; points, tag2idx)
end
function gmsh_get_mesh()
    nodes = gmsh_get_nodes()
    faces = TriangleFace{Int}[]
    map(gmsh.model.mesh.getElements()...) do etype, etags, ntags
        if etype == 2
            @assert length(ntags) == length(etags) * 3
            for c in eachcol(reshape(ntags, 3, :))
                c = map(n -> nodes.tag2idx[n], (c...,))
                push!(faces, c)
            end
        end
    end
    Mesh(nodes.points, faces)
end
function gmsh_get_tetras()
    nodes = gmsh_get_nodes()
    tetras = NTuple{4,Int}[]
    map(gmsh.model.mesh.getElements()...) do etype, etags, ntags
        if etype == 4
            @assert length(ntags) == length(etags) * 4
            for c in eachcol(reshape(ntags, 4, :))
                c = map(n -> nodes.tag2idx[n], (c...,))
                push!(tetras, c)
            end
        end
    end
    (; nodes.points, tetras)
end

function gmsh_add_mesh(obj)
    objsurf = gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2, objsurf, collect(axes(obj.position, 1)), reduce(vcat, SVector.(obj.position)))
    objfaces = map(faces(obj)) do f
        f = SVector(f .+ 0)
        @assert length(f) == 3
        f
    end
    gmsh.model.mesh.addElementsByType(objsurf, 2, [], reduce(vcat, objfaces))
    objsurf
end

function gmsh_tetras_with_embed(obj)
    gmsh.initialize()
    try
        gmsh.clear()
        gmsh.model.add("hello")
        gmsh.option.setNumber("Mesh.MeshSizeMin", 350.0 * 3)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 350.0 * 9)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 0.0000001)

        box = gmsh.model.occ.addBox(-200, -200, -200, 64200, 78200, 50200)
        gmsh.model.occ.synchronize()
        @info "len $(length(obj))"
        objsurf = gmsh_add_mesh(obj)
        for opt in ["UntangleMeshGeometry" "" "HighOrder" "HighOrderFastCurving" "Laplace2D" "Netgen" "QuadQuasiStructured" "Relocate2D" "Relocate3D"]
            @info "opt $opt"
            ####gmsh.model.mesh.optimize(opt)
        end
        gmsh.model.mesh.createTopology(true)

        gmsh.model.mesh.embed(2, [objsurf], 3, box)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        Main.objxxx = gmsh_get_mesh()
        gmsh.fltk.run()
        gmsh_get_tetras()
    finally
        gmsh.finalize()
    end
end


function optimize_mesh(obj)
    gmsh.initialize()
    obj = try
        #239535 *1
        #124537*2
        #70960*3
        #70960
        gmsh.option.setNumber("Mesh.MeshSizeMin", 350.0 * 3)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 350.0 * 9)
        #gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        #368K
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 8)
        gmsh.clear()
        gmsh.model.add("hello")
        gmsh_add_mesh(obj)
        # may discard
        #gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.createTopology(true)
        for opt in ["UntangleMeshGeometry" "" "HighOrder" "HighOrderElastic" "HighOrderFastCurving" "Laplace2D" "Netgen" "QuadQuasiStructured" "Relocate2D" "Relocate3D"]
            @info "opt $opt"
            ###gmsh.model.mesh.optimize(opt)
        end
        #gmsh.fltk.run()
        #@assert false
        
        gmsh.model.mesh.classifySurfaces(2.0pi, true, true, 2.0pi, true)
        gmsh.model.mesh.createGeometry()
        ents = gmsh.model.getEntities(2)
        l = gmsh.model.geo.addSurfaceLoop([e[2] for e in ents], -1)
        v = gmsh.model.geo.addVolume([l], -1)
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        ##gmsh.fltk.run()
        gmsh_get_mesh()
    finally
        gmsh.finalize()
    end
    save("/tmp/meshxxx-after-xxx-write.obj", obj)
    decompose_mesh(obj)[1]
end

function visualize_with_norm(obj::AbstractString; normd=500)
    obj = FileIO.load(obj)
    visualize_with_norm(obj; normd)
end

function visualize_with_norm(obj::GeometryBasics.AbstractMesh; normd=500)
    obj = dedup_verts(obj) |> normal_mesh

    #count_mesh_edges(obj)
    #obj = fix_normals(obj)
    
    #obj = GeometryBasics.normal_mesh(obj.position, GeometryBasics.faces(obj))
    #count_mesh_edges(obj)
    Makie.mesh(obj, color=:red) |> display
    #return
    GLMakie.linesegments!(reshape(permutedims(hcat(obj.position, map((a, b) -> a + b * 700, obj.position, obj.normals)), (2, 1)), :) |> collect)
    #MakieCore.LineSegments{Tuple{Vector{GeometryBasics.Point{3, Float32}}}}
end

flatmap_data(Tag) = Make.@x("interm/flat-map-$Tag.dat")
(out::flatmap_data(Tag))() where {Tag} = begin
    cfg = (;
        test=(;
            clipfn=(obj) -> begin
                vmax = findmin(p -> -p[2], obj.position)[2]
                dist1 = dijkstra_vertex(obj, vmax)
                maxd = maximum(first, dist1)
                (v) -> dist1[v][1] < 40000
            end,
            rightfn=p -> p[1],
        ),
        lefths=(;
            clipfn=(obj1) -> begin
                nmt = NmtDat.load_images()
                charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res), CartesianIndices(nmt.seg)[nmt.charm[:, :, :, 1].>0])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj, findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 10000
            end,
            rightfn=p -> p[2],
        ),
        leftfrontal=(;
            clipfn=(obj1) -> begin
                nmt = NmtDat.load_images()
                charmpos = map(i -> NmtDat.idx2pos(SVector(Tuple(i)), nmt.res), CartesianIndices(nmt.seg)[nmt.charm[:, :, :, 1].==1])
                tree = NearestNeighbors.KDTree(charmpos)
                dists = reduce(vcat, NearestNeighbors.knn(tree, obj1.position, 1)[2])
                dist1 = dijkstra_vertex(obj, findmax(dists)[2])
                (v) -> dists[v] < 500 || dist1[v][1] > 60000
            end,
            rightfn=p -> p[1],
        ),
    )[Tag]
    #obj00 = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.gray_surface.rsl.gii.obj")
    obj05 = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.mid_surface.rsl.gii.obj")
    #obj10 = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.white_surface.rsl.gii.obj")
    obj = refine_mesh(obj05)
    obj2 = clip_mesh(cfg.clipfn(obj), obj)
    vmax = findmin(cfg.rightfn, obj2.position)[2]
    obj3 = flatten_mesh(obj2, vmax)
    JLD2.@save out.path obj = obj3 objphy = obj2
end

function to3d(obj::GeometryBasics.AbstractMesh)
    GeometryBasics.Mesh(map(p -> GeometryBasics.Point3(p..., 0.0), obj.position), GeometryBasics.faces(obj))
end

function distmap(img::AbstractArray{<:Real,3})
    allindices = CartesianIndices(img)
    out = similar(img, Float32)
    tags = map(allindices) do idx
        near = intersect(allindices, CartesianIndices(((idx[d]-1:idx[d]+1 for d in 1:3)...,)))
        n1 = count(>(0.5), view(img, near))
        if img[idx] < 0.5
            n1 > 0 ? 0 : -1
        else
            1
        end |> Int8
    end
    idx2pos(idx) = SVector((idx[d] for d in 1:3)...,)
    zindices = allindices[tags.==0]
    tree = NearestNeighbors.KDTree(idx2pos.(zindices))
    out[zindices] .= 0
    negindices = allindices[tags.<0]
    out[negindices] .= 0
    pindices = allindices[tags.>0]
    dists = reduce(vcat, NearestNeighbors.knn(tree, idx2pos.(pindices), 1)[2])
    map(pindices, dists) do i, d
        out[i] = d
    end
    out
end

function mesh_disttosurf(objphy::GeometryBasics.AbstractMesh, nmt)
    segcoarse = map(v -> (v > 0 ? 1.0f0 : 0.0f0), nmt.seg)
    segcoarse[end÷2+1:end, :, :] .= 0
    disttosurf = segcoarse |> distmap
    save("/tmp/test.nrrd", disttosurf)
    dists = map(p -> disttosurf[NmtDat.pos2idx(p, nmt.res)...], objphy.position)
end

function mesh_annotate_vert(objphy::GeometryBasics.AbstractMesh, nmt, highlights)
    getter = NmtDat.getter(nmt)
    prevlen = length(highlights)
    if !(highlights isa Dict)
        highlights = map(reverse, enumerate(highlights)) |> Dict
    end
    @assert prevlen == length(highlights)

    charmpos = let
        charmindices = CartesianIndices(nmt.seg)[nmt.charm[:, :, :, 1].>0]
        charmpos = map(charmindices) do i
            NmtDat.idx2pos(SVector(Tuple(i)), nmt.res)
        end
        tree = NearestNeighbors.KDTree(charmpos)
        indices = NearestNeighbors.knn(tree, objphy.position, 16, true)[1]
        map(indices, objphy.position, objphy.normals) do inds, p0, n0
            for i in inds
                if dot(charmpos[i] .- p0, n0) < 0
                    return charmpos[i]
                end
            end
            SVector(250.0f0, 250.0f0, 250.0f0)
        end
    end
    hiers = getter(reduce(hcat, charmpos))[2]

    hier2dist = map(unique(hiers)) do hh
        v = MVector{length(highlights),Float32}(undef)
        v .= 0
        h = hh
        while count('|', h) > 0
            if (i = get(highlights, h, 0)) != 0
                v[i] = 1
            end
            h = replace(h, r"\|[^|]+$" => "")
        end
        hh => SVector(v)
    end |> Dict
    map(hiers) do hh
        hier2dist[hh]
    end
end

generate_mesh_raw(atlas, id, lr) = Make.@x("interm/raw-mesh-$atlas-$id-$lr.obj")
generate_mesh(atlas, id, lr) = Make.@x("interm/mesh-$atlas-$id-$lr.obj")
## :area_10 :area_13m :area_44 :area_45 :area_46d :area_8B :area_8Bm :area_9d :area_9m :MT :TEa :TEm :V1 :V2 :V4 :V6

function expand_boundary(img)
    siz1, siz2, siz3 = size(img)
    img = cat(zeros(Float16, siz1, siz2, 1), img, zeros(Float16, siz1, siz2, 1), dims=3)
    siz1, siz2, siz3 = size(img)
    img = cat(zeros(Float16, siz1, 1, siz3), img, zeros(Float16, siz1, 1, siz3), dims=2)
    siz1, siz2, siz3 = size(img)
    cat(zeros(Float16, 1, siz2, siz3), img, zeros(Float16, 1, siz2, siz3), dims=1)
end
function fix_mesh_position(mesh, voxsize, fixoff=nothing)
    fixoff = @something(fixoff, (0.0f0, 0.0f0, 0.0f0))
    mesh.position .= map(mesh.position) do pt
        (pt .- (0.5f0, 0.5f0, 0.5f0) .- fixoff) .* voxsize
    end
    mesh
end

(out::generate_mesh_raw(:sample, Samp, :lr))(
) where {Samp} = begin
    seg, dsfactor, fixoff = if Samp === Symbol("fmost234197.ch1")
        img = @time "load" load("monkeytemp/maskG97-diff.nrrd") .> 180
        dsfactor = 0.3
        Images.imresize(img, ratio=dsfactor, method=Interpolations.Linear()) .- 0.31f0, dsfactor, dsfactor .* (23, 77, 197)
    elseif startswith("fmost234197.ch1")("$Samp")
        img = @time "load" load("monkeytemp/maskG97-diff.nrrd").data
        plane(v) = dot((0.888, 0.011, 0.459), v) - 48677 > 0
        planei(i) = plane((Tuple(i) .- 0.5) .* (65, 65, 60))
        ff = if endswith("$Samp", ".fh")
            (i, v) -> v > 0
        elseif endswith("$Samp", ".ss")
            (i, v) -> planei(i) > 0 ? v > 0 : v > 180
        elseif endswith("$Samp", ".hs")
            (i, v) -> planei(i) > 0 ? (0 < v < 180) : false
        else
            @assert false
        end
        img = Float32.(map(ff, CartesianIndices(img), img))
        img2 = similar(img)
        GC.gc()
        dsfactor = 0.3
        Images.imresize(img, ratio=dsfactor, method=Interpolations.Linear()) .- 0.31f0, dsfactor, dsfactor .* (23, 77, 197)
    else
        mip = "/data/dat/dsimgs/$Samp@ch00-mip.nrrd"
        avg = "/data/dat/dsimgs/$Samp@ch00-avg.nrrd"
        x = load(avg.path)
        x[:, :, 1:4] .= 0
        x2 = Images.imfilter(Float32, x, Images.Kernel.gaussian((2, 2, 2)))
        y = load(mip.path)
        y2 = Images.imfilter(Float32, y, Images.Kernel.gaussian((2, 2, 2)))
        dsfactor = 0.3
        Images.imresize(map((i, u, v) -> i[3] < 0 ? Flux.sigmoid((v - 300) / 10) : Flux.sigmoid((u - 430) / 10), CartesianIndices(x2), x2, y2), ratio=dsfactor, method=Interpolations.Linear()), dsfactor, nothing
    end
    GC.gc()
    global xxximg = deepcopy(seg)
    seg = expand_boundary(seg)
    mesh = generate_mesh_from_mask_mc(seg)
    mesh = fix_mesh_position(mesh, (65, 65, 60) ./ dsfactor, fixoff)
    save(out.path, mesh)
end

(out::generate_mesh_raw(:nmt, :brain, :lr))() = begin
    nmt = NmtDat.load_images()
    seg = NmtDat.normalize_tissue(nmt)
    seg = expand_boundary(seg)
    vol = smooth_binary_mask(seg)
    mesh = generate_mesh_from_mask_mc(vol)
    mesh = fix_mesh_position(mesh, nmt.res)
    save(out.path, mesh)
end
(out::generate_mesh_raw(:nmt, Area, LR))() where {Area,LR} = begin
    nmt = NmtDat.load_images()
    areaname = "$Area"
    hits = map((:charm, :sarm)) do k
        tbl = nmt[Symbol(k, :_key)]
        hits2 = findall(v -> v[2] == areaname, eachrow(tbl))
        k => tbl[hits2, 1]
    end
    @info "Hits: $hits"
    @assert length(mapreduce(last, vcat, hits)) == 1
    seg = nothing
    for (k, h) in hits
        if length(h) > 0
            h = h[1]
            img = nmt[k]
            seg = map(CartesianIndices(nmt.seg)) do idx
                Float32(count(==(h), view(img, idx, :)) > 0 ? 1.0 : 0.0)
            end
        end
    end
    if LR == :lr
    elseif LR == :l
        seg[(size(nmt.seg, 1)÷2+1):end, :, :] .= 0
        seg = seg[1:(size(nmt.seg, 1)÷2+1), :, :]
    elseif LR == :r
        seg[1:(size(nmt.seg, 1)÷2), :, :] .= 0
    else
        @assert false
    end
    seg = expand_boundary(seg)
    vol = smooth_binary_mask(seg)
    mesh = generate_mesh_from_mask_mc(vol)
    mesh = fix_mesh_position(mesh, nmt.res)
    save(out.path, mesh)
end

function mirror_mesh(mirrorby)
    (mesh) -> begin
        vv = map(p -> Point3d(mirrorby(p)), mesh.position)
        ff = map(f -> reverse(f .+ 0), faces(mesh))
        normal_mesh(vv, ff)
    end
end

function add_mirror_mesh(mirrorby)
    (mesh) -> begin
        vv = collect(mesh.position)
        ff = map(f -> f .+ 0, faces(mesh))
        lenv = length(vv)
        for v in mesh.position
            push!(vv, mirrorby(v))
        end
        for f in faces(mesh)
            push!(ff, reverse(f .+ lenv))
        end
        @warn "fix lost"
        normal_mesh(vv, ff)
    end
end
function fixlost(mesh, Atlas, Area, LR)
    LR !== :lr && return mesh
    mod = if Atlas === :nmt
        NmtDat
    elseif Atlas === :allen
        Allen
    else
        return mesh
    end
    activepos = reduce(vcat, [mesh.position[f] for f in faces(mesh)])
    lrs = unique(map(v -> mod.isleft(v), activepos))
    @info "fix $Atlas $Area $LR $lrs"
    length(lrs) != 1 && return mesh
    add_mirror_mesh(mod.mirror)(mesh)
end

(out::generate_mesh(Atlas, Area, LR))(in::generate_mesh_raw(Atlas, Area, LR)) where {Atlas,Area,LR} = begin
    mesh = load(in.path)
    save("/tmp/meshxxx-after-load.obj", mesh)
    # very much needed
    mesh = inflate_mesh(mesh; niter=250, rng=9.0f0)
    save("/tmp/meshxxx-after-inflate.obj", mesh)
    #mesh = fix_normals(mesh)
    dirs_vf = []
    dirs_vv = []
    for f in faces(mesh)
        a, b, c = f
        na, nb, nc = mesh.normals[[a, b, c]]
        a, b, c = mesh.position[[a, b, c]]
        n = cross(b .- a, c .- a)
        n = n ./ norm(n)
        push!(dirs_vf, dot(n, na))
        push!(dirs_vf, dot(n, nb))
        push!(dirs_vf, dot(n, nc))
        push!(dirs_vv, dot(nb, na))
        push!(dirs_vv, dot(nc, nb))
        push!(dirs_vv, dot(na, nc))
    end
    ##display(GLMakie.hist(dirs_vf))
    extr = extrema(dirs_vf)
    extr2 = extrema(dirs_vv)
    @info "eeee $extr"
    @info "eeee $extr2"
    mesh = decompose_mesh(mesh)[1]
    save("/tmp/meshxxx-after-decomp.obj", mesh)
    mesh = optimize_mesh(mesh)
    mesh = fixlost(mesh, Atlas, Area, LR)
    save("/tmp/meshxxx-after-opt.obj", mesh)
    mesh = try
        fix_normals(mesh)
    catch
        @error "error fix normals"
        mesh
    end
    save("/tmp/meshxxx-after-fixnorm.obj", mesh)
    save(out.path, mesh)
end

function inflate_mesh(obj::GeometryBasics.AbstractMesh; niter=10000, rng=1)
    edg(e) = sort(Vec2i(Int.((e .+ 0))))
    pts = reduce(hcat, obj.position)
    edgs = Vec2i[]
    for f in GeometryBasics.faces(obj)
        a, b, c = f
        for e in ((a, b), (b, c), (c, a))
            push!(edgs, edg(e))
        end
    end
    sort!(unique(edgs))
    for (e1, e2) in edgs
        n = norm(obj.position[e1] .- obj.position[e2])
        if n > 700
            ###@info "long::: $e1 $e2 $n"
        end
    end
    adj = reduce(hcat, edgs)
    # pts0 = minimum(pts, dims=2)
    # pts1 = maximum(pts, dims=2)
    # rng = median(pts1 .- pts0)
    # pts = pts ./ rng
    # pts = pts .- median(pts, dims=2)

    fig = Makie.Figure()
    display(fig)
    obs1 = Makie.Observable(reduce(vcat, [SVector((1.0f0, 1.0f0)) for i in axes(adj, 2)]))
    Makie.linesegments(fig[1, 1], reshape(pts[:, adj], 3, :), color=obs1)
    Makie.wireframe(fig[2, 1], obj)
    obs = Makie.Observable(reshape(pts[:, adj], 3, :))
    Makie.linesegments(fig[1:2, 2], obs)
    Makie.DataInspector()

    device = Flux.cpu
    pts, adj = (pts, adj) |> device

    newpts(d) = pts .+ Flux.tanh_fast(d) .* rng
    dump(edgs, d) = begin
        obs[] = reshape(edgs, 3, :) |> Flux.cpu
        dd = sqrt.(sum((Flux.tanh_fast(d) .* rng) .^ 2, dims=1))[:, adj] |> Flux.cpu
        lens = sqrt.(sum((view(edgs, :, 1, :) .- view(edgs, :, 2, :)) .^ 2, dims=1)) |> Flux.cpu
        #obs1[] = reduce(vcat, [SVector((lens[1, i], lens[1, i])) for i in axes(adj, 2)])
        obs1[] = reshape(dd, :)
    end
    loss(d, K) = begin
        pts2 = newpts(d)
        edgs = @view pts2[:, adj]
        l1 = Flux.mean(Flux.relu.(sum((view(edgs, :, 1, :) .- view(edgs, :, 2, :)) .^ 2, dims=1) .- 1.0f2))
        l2 = 0#Flux.mean(Flux.softplus.((0.25f0 .- sum(pts2 .^ 2, dims=1)) * 10)) / 10.0f0
        l2 = 0.01 * Flux.mean(sum(d .^ 2, dims=1))
        Flux.ignore_derivatives() do
            if K % 10 == 0
                dump(edgs, d)
                @info "$K $l1 $l2"
            end
        end
        l1 + l2
    end

    delta = similar(pts)
    delta .= 0

    Flux.train!(loss, delta, 1:niter, Flux.setup(Flux.Adam(0.03), delta))
    pos = map(GeometryBasics.Point3f, newpts(delta) |> Flux.cpu |> eachcol)
    GeometryBasics.normal_mesh(pos, GeometryBasics.faces(obj))
end

(out::Make.@x("interm/test-gmsh.dat"))() = begin
    nmt = NmtDat.load_images()
    id = maximum(view(nmt.charm, :, :, :, 5))
    vol = view(nmt.charm, :, :, :, 5) .== id
    vol[:, 1:90, :] .= 0
    Makie.volume(vol) |> display

    gmsh.initialize()
    try
        spheres = map(findall(>(0), vol)) do ci
            pos = NmtDat.idx2pos(SVector(Tuple(ci)), nmt.res)
            gmsh.model.occ.addSphere(pos..., 250.0f0 * sqrt(3.01f0))
        end
        spheres1 = map(v -> (UInt32(3), v), spheres[1:100])
        spheres2 = map(v -> (UInt32(3), v), spheres[101:end])
        odt, odtm = gmsh.model.occ.fuse(spheres1, spheres2)
        @info "$odt"
        @info "$odtm"
    finally
        gmsh.finalize()
    end
    #
end
