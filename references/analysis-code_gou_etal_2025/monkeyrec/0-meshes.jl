function fix_mesh_object(obj)
    function checknormals(msh)
        dps = map(GeometryBasics.faces(msh)) do f
            a, b, c = msh.position[f]
            n = cross(b - a, c - a)
            n = n ./ norm(n)
            map(f) do v
                dot(n, msh.normals[v])
            end
        end
        reduce(vcat, dps)
    end
    function renorm(msh)
        msh = GeometryBasics.triangle_mesh(msh)
        y, _ = GeometryBasics.pop_pointmeta(msh, :normals)
        y = GeometryBasics.Mesh(collect(y.position), GeometryBasics.faces(y))
        #@info "$(typeof(y))"
        GeometryBasics.normal_mesh(y)
    end
    function repos(msh; dp=GeometryBasics.Point(32000.0, 27750.0, 8000.0))
        for (i, p) in pairs(msh.position)
            msh.position[i] = (p .* 100000) + dp
        end
        msh
    end
    obj = obj |> repos |> renorm
    quantile(filter(!isnan, checknormals(obj)), [0.1, 0.5, 0.9])
    obj
end

function count_mesh_edges(obj)
    edgs = Dict{Any,Int}()
    for f in GeometryBasics.faces(obj)
        a, b, c = obj.position[f.+0]
        for e in ((a, b), (b, c), (c, a))
            e = (sort([e...,])...,)
            edgs[e] = get(edgs, e, 0) + 1
        end
    end
    println(extrema(map(e -> norm(e[1] .- e[2]), collect(keys(edgs)))))
    for (e, c) in edgs
        # if e[1]==e[2]
        #     println(e, "====>", c)
        # end
        # if c != 1
        #     println(e, "=>", c)
        # end
        if c != 2
            #Makie.linesegments!([Vec3f(e[1]...), Vec3f(e[2]...)], color=:blue)
            println(e, "=>", c)
            if c == 4
                ff = TriangleFace{Int}[]
                for f in GeometryBasics.faces(obj)
                    a, b, c = obj.position[f.+0]
                    for e2 in ((a, b), (b, c), (c, a))
                        e2 = (sort([e2...,])...,)
                        if e2 == e
                            println(f)
                            push!(ff, f .+ 0)
                        end
                    end
                end
                Main.xxobj = Mesh(obj.position, unique!(ff))
            end
        end
    end
    @assert false
end

function dedup_faces(obj::AbstractMesh)
    newfaces = map(faces(obj)) do f
        t1 = Tuple(f .+ 0)
        t2 = (t1[2:end]..., t1[1])
        t3 = (t2[2:end]..., t2[1])
        min(t1, t2, t3)
    end
    @info "prev $(length(newfaces))"
    newfaces = newfaces |> unique
    @info "after $(length(newfaces))"
    newfaces = map(f -> TriangleFace{Int}(f), newfaces)
    #eee=popat!(newfaces, 8629)
    #@info "length $(length(newfaces)) $eee"
    #newfaces = newfaces[8630:end]
    Mesh(coordinates(obj), newfaces)
end

function dedup_verts(obj::AbstractMesh; keep_degenerate=false)
    verts = metafree(coordinates(obj))
    newverts = unique(verts)
    v2i = Dict(reverse.(enumerate(newverts)))
    newfaces = map(faces(obj)) do f
        F = typeof(f)
        f2 = map(i -> v2i[verts[i]], f)
        if length(unique(f2)) < 3 && !keep_degenerate
            @info "missing $f2"
            return missing
        end
        F(f2)
    end
    Mesh(newverts, newfaces |> skipmissing |> collect)
end

function fix_normals(obj)
    edge2adj = Dict{Any,Set}()
    toedge(e) = sort([e...])
    tovert(v) = v + 0
    for (fi, f) in enumerate(GeometryBasics.faces(obj))
        a, b, c = map(v -> tovert(v), f)
        for e in ((a, b), (b, c), (c, a))
            adj = get!(Set, edge2adj, toedge(e))
            push!(adj, fi)
        end
    end
    face2ori = Dict{Int,Int}()
    todo = collect(axes(GeometryBasics.faces(obj), 1))
    while !isempty(todo)
        fi = pop!(todo)
        ori = get(face2ori, fi, 0)
        a, b, c = map(v -> tovert(v), GeometryBasics.faces(obj)[fi])
        if length(unique((a, b, c))) < 3
            face2ori[fi] = -5
        end
        if ori == 0
            ori = 1
            @warn "I$ori $a, $b, $c"
            face2ori[fi] = ori
        end
        edges = ((a, b), (b, c), (c, a))
        for e in ((a, b), (b, c), (c, a))
            for ffi in edge2adj[toedge(e)]
                ffi == fi && continue
                aa, bb, cc = map(v -> tovert(v), GeometryBasics.faces(obj)[ffi])
                ori2 = if (aa, bb) in edges || (bb, cc) in edges || (cc, aa) in edges
                    -ori
                else
                    ori
                end
                ori22 = get(face2ori, ffi, 0)
                if ori22 == 0
                    face2ori[ffi] = ori2
                    @info "$ori2 $aa, $bb, $cc"
                    push!(todo, ffi)
                else
                    #@assert ori22 == ori2 "$ori2 $ori22 $aa, $bb, $cc"
                end
            end
        end
    end
    vert2id = Dict{Any,Int}()
    vertset = Set(collect(obj.position))
    verts = map(enumerate(vertset)) do (i, p)
        vert2id[p] = i
        p
    end
    _, maxori = maximum([(count(p -> p[2] == ori, face2ori), ori) for ori in unique(values(face2ori))])
    faces = map(enumerate(GeometryBasics.faces(obj))) do (fi, f)
        a, b, c = map(v -> vert2id[obj.position[v]], f)
        if face2ori[fi] == maxori
            GeometryBasics.TriangleFace(a, b, c)
        else
            GeometryBasics.TriangleFace(a, c, b)
        end
    end
    normal_mesh(verts, faces)
end

# taggg
# tag


function remote_mktempdir(f::Function, login, prefix)
    dir = open(rstrip ∘ readline, `ssh $login mktemp -d $prefix`)
    try
        return f(dir)
    finally
        run(`ssh $login rm -rvf $dir`)
    end
end

function with_addprocs(f::Function, args...; kwargs...)
    pids = addprocs(args...; kwargs...)
    try
        return f(pids)
    finally
        rmprocs(pids)
    end
end

function collect_spat_trans_code(pids)
    fn = "0-spat-trans-prep.jl"
    code = read(fn, String)
    @everywhere pids include_string(Main, code, fn)
end

# XXX: becareful when collapse verts
function vertex_face_adj(obj::AbstractMesh)
    v2faces = Dict{Int,Set{Int}}()
    for (fi, f) in enumerate(GeometryBasics.faces(obj)), v in f
        adj = get!(Set{Int}, v2faces, v)
        push!(adj, fi)
    end
    v2faces
end

function dijkstra_vertex(obj::AbstractMesh, seed1, seedtags::Pair...)
    vert2tag = map(_ -> (Inf, 0), axes(obj.position, 1))
    todo = DataStructures.PriorityQueue{Tuple{Int,Int},Float64}()
    push!(todo, (seed1, 1) => 0.0)
    for (s, t) in seedtags
        push!(todo, (s, t) => 0.0)
    end
    v2adj = vertex_face_adj(obj)
    allfaces = GeometryBasics.faces(obj)
    while !isempty(todo)
        (v, t), d = DataStructures.dequeue_pair!(todo)
        vert2tag[v] = (d, t)
        p = obj.position[v]
        for fi in v2adj[v], vv in allfaces[fi]
            dd = d + norm(obj.position[vv] .- p)
            if dd < vert2tag[vv][1] && dd < get(todo, (vv + 0, t), Inf)
                todo[(vv + 0, t)] = dd
            end
        end
    end
    vert2tag
end

function clip_mesh(f::Function, obj::AbstractMesh)
    selfaces = filter(GeometryBasics.faces(obj)) do fac
        all((f(v) for v in fac))
    end
    v2i = Dict{Int,Int}()
    for f in selfaces
        for v in f
            get!(v2i, v + 0, length(v2i) + 1)
        end
    end
    pts = similar(obj.position, length(v2i))
    for (v, i) in v2i
        pts[i] = obj.position[v]
    end
    selfaces = map(selfaces) do f
        a, b, c = f
        a, b, c = map((a, b, c)) do v
            v2i[v+0]
        end
        GeometryBasics.TriangleFace(a, b, c)
    end
    GeometryBasics.Mesh(pts, selfaces)
end

_edgid(e) = sort(Vec2i(e))
function find_boundary(obj::AbstractMesh)
    edg2faces = Dict{SVector,Int}()
    for f in GeometryBasics.faces(obj)
        a, b, c = f .+ 0
        for e in ((a, b), (b, c), (c, a))
            e2 = _edgid(e)
            edg2faces[e2] = get(edg2faces, e2, 0) + 1
        end
    end
    v2v = Dict{Int,Int}()
    for f in GeometryBasics.faces(obj)
        a, b, c = f .+ 0
        for e in ((a, b), (b, c), (c, a))
            e2 = _edgid(e)
            nfac = edg2faces[e2]
            @assert 1 <= nfac <= 2
            if nfac == 1
                v2v[e[1]] = e[2]
            end
        end
    end
    verts = sizehint!(Int[], length(v2v))
    s = first(v2v)[1]
    push!(verts, s)
    v = v2v[s]
    while v != s
        push!(verts, v)
        v = v2v[v]
    end
    verts
end

function flatten_mesh(objphy::AbstractMesh, anchor_right::Integer; niter=30000)
    fig = Makie.Figure()
    display(fig)
    GLMakie.mesh(fig[1, 1], objphy, color=:orange)
    GLMakie.wireframe!(fig[1, 1], objphy, color=:black)
    boundary = let
        boundary = find_boundary(objphy)
        dists = dijkstra_vertex(objphy, anchor_right)
        i = findmin(v -> dists[v][1], boundary)[2]
        vcat(boundary[i:end], boundary[1:i-1])
    end
    nsep = 7
    dists = map([boundary[end÷nsep*i+1] for i in (1:nsep) .- 1]) do s
        map(first, dijkstra_vertex(objphy, s))
    end
    for dd in dists
        dd .= maximum(filter(!isinf, dd)) .- dd
    end
    anchors = [Vec2f(cos(r), sin(r)) for r in range(0, length=nsep, step=2pi / nsep)]
    pos2 = map(dists...) do abc...
        GeometryBasics.Point2f(sum(map(.*, abc, anchors)) ./ sum(abc))
    end
    obj = GeometryBasics.Mesh(pos2, GeometryBasics.faces(objphy))

    v2i = Dict{Int,Int}()
    for v in boundary
        get!(v2i, v, length(v2i) + 1)
    end
    nboundary = length(v2i)
    for f in GeometryBasics.faces(obj)
        for v in f
            get!(v2i, v, length(v2i) + 1)
        end
    end
    pts = Vector{Vec2f}(undef, length(v2i))
    pts2 = similar(pts, Vec3f)
    for (v, i) in v2i
        pts[i] = obj.position[v]
        pts2[i] = objphy.position[v]
    end
    adj = Vec2i[]
    for f in GeometryBasics.faces(obj)
        a, b, c = f .+ 0
        for e in ((a, b), (b, c), (c, a))
            e2 = map(v -> v2i[v], e)
            v1, v2 = map(v -> obj.position[v], e)
            len1 = norm(v1 .- v2)
            vv1, vv2 = map(v -> pts[v], e2)
            len2 = norm(vv1 .- vv2)
            @assert len1 == len2
            if len2 > 0.04
                @info "length: $len1 $len2"
            end
            push!(adj, _edgid(e2))
        end
    end
    for e2 in adj
        vv1, vv2 = map(v -> pts[v], e2)
        len2 = norm(vv1 .- vv2)
        if len2 > 0.03
            @info "length: $len2"
        end
    end
    pts = reduce(hcat, pts)
    boundaryscore = map(v -> minimum(map(d -> d[v], dists)), axes(pts, 2))
    adj = sort(unique(adj), by=(a) -> min(boundaryscore[a[1]], boundaryscore[a[2]]))
    adj = reduce(hcat, adj)
    println("size adj ", size(adj))
    pts_obs = Makie.Observable(pts)
    GLMakie.scatter(fig[1, 2], pts_obs)
    adj_backup = deepcopy(adj)
    lines_obs = map(pts_obs) do p
        p[:, reshape(adj_backup, :)]
    end
    GLMakie.linesegments!(lines_obs, color=:black)
    Makie.DataInspector()

    disttob = dijkstra_vertex(objphy, boundary[1], [v => 1 for v in boundary[2:end]]...)
    disttob = map(first, disttob)
    mdisttob = maximum(disttob)
    www = map(d -> 1 / (1 + ((d / mdisttob))^2), disttob) |> Flux.gpu

    pts = pts |> Flux.gpu
    pts2 = reduce(hcat, pts2) |> Flux.gpu
    adj = adj |> Flux.gpu
    adjb = [Vec2i(v2i[boundary[i]], v2i[boundary[i%length(boundary)+1]]) for i in axes(boundary, 1)]
    adjb = reduce(hcat, adjb) |> Flux.gpu
    edgs2 = pts2[:, adj]
    len2 = sqrt.(sum((edgs2[:, 1, :] .- edgs2[:, 2, :]) .^ 2, dims=1)) .+ 1.0f1
    edgsb2 = pts2[:, adjb]
    lenb2 = sqrt.(sum((edgsb2[:, 1, :] .- edgsb2[:, 2, :]) .^ 2, dims=1)) .+ 1.0f1
    medlen2 = median(len2 |> Flux.cpu)
    println("size adj ", size(adj))
    println("size pts ", size(pts))
    newpts(d) = pts .+ d
    loss(d, iter, adjsel) = begin
        p = newpts(d)
        adjx = adj[:, adjsel]
        edgs = p[:, adjx]
        edgsb = p[:, adjb]
        println(size(edgs))
        @assert size(edgs)[1:2] == (2, 2)
        l1 = Flux.mean(sum((edgs[:, 1, :] .- edgs[:, 2, :]) .^ 2, dims=1) ./ len2[:, adjsel] .* sum(www[adjx], dims=1)) * medlen2
        l3 = Flux.mean(sum((edgsb[:, 1, :] .- edgsb[:, 2, :]) .^ 2, dims=1) ./ lenb2) * medlen2 * 0.1
        l2 = Flux.mean(Flux.softplus((1 .- sum(p[:, 1:nboundary] .^ 2, dims=1)) .* 1.0f2) ./ 1.0f3)
        Flux.ignore_derivatives() do
            if iter % 100 == 0
                pts_obs[] = p |> Flux.cpu
            end
            @info "$iter $(l1+l2+l3) $l1 $l2 $l3 $medlen2"
        end
        l1 + l2 + l3
    end
    delta = zeros(Float32, size(pts)) |> Flux.gpu
    st = Flux.setup(Flux.Adam(), delta)
    @showprogress for K in 1:niter
        adjx = if niter < niter ÷ 3
            1:size(adj, 2)÷8
        elseif niter < niter ÷ 3 * 2
            1:size(adj, 2)÷4
        else
            1:size(adj, 2)
        end
        #adjx = rand(axes(adj, 2), 4096) |> Flux.gpu
        l, g = Flux.withgradient(d -> loss(d, K, adjx), delta)
        Flux.update!(st, delta, g[1])
    end
    pts = newpts(delta) |> Flux.cpu
    verts = [GeometryBasics.Point2f(pts[1:2, v2i[v]]...) for v in axes(obj.position, 1)]
    GeometryBasics.Mesh(verts, GeometryBasics.faces(objphy))
end

function color_mesh_verts(obj, bg, sites...; nseed=7, sigma=2500)
    tree = NearestNeighbors.KDTree(obj.position)
    seeds = collect(map(SVector ∘ first, sites))
    nearest = NearestNeighbors.knn(tree, reduce(hcat, seeds), nseed)[1]
    vert2dists = map(enumerate(seeds)) do (k, seedpos)
        map(first, dijkstra_vertex(obj, nearest[k][1], map(v -> v => 1, nearest[k][2:end])...))
    end
    g = Distributions.Gaussian(0, sigma)
    gm = Distributions.pdf.(g, 0)
    map(vert2dists...) do dists...
        w = Distributions.pdf.(g, dists) ./ gm
        wbg = 1 - sum(w)
        sum((bg, map(last, sites)...) .* (wbg, w...))
    end
end

function _to_meshes(obj)
    Meshes.SimpleMesh(map(p -> Meshes.Point(p...), obj.position), map(f -> Meshes.connect((f...,)), GeometryBasics.faces(obj)))
end
function _from_meshes(obj)
    ff = map(t -> GeometryBasics.TriangleFace(t.indices...), Meshes.topology(obj).connec)
    GeometryBasics.Mesh(map(v -> GeometryBasics.Point(map(c -> c.val, getfield(v.coords, :coords))...), obj.vertices), ff)
end
function refine_mesh(obj)
    obj2 = _to_meshes(obj)
    obj3 = Meshes.refine(obj2, Meshes.TriSubdivision()) |> _from_meshes
    #obj3 = Meshes.refine(obj2, Meshes.CatmullClark()) |> _from_meshes
    len1 = length(obj)
    len3 = length(obj3)
    @info "len1 $len1 len3 $len3"
    obj3
end

function sample_mesh(fn::Function, obj::GeometryBasics.AbstractMesh, targets...; perface=64)
    samps = map(eachcol(rand(Float32, 2, perface))) do (l1, l2)
        r1 = 1 - sqrt(l1)
        r2 = (1 - r1) * l2
        r3 = 1 - r1 - r2
        [r1, r2, r3]
    end
    allverts = map(targets...) do vals...
        t = ()
        for v in vals
            t = (t..., v...)
        end
        t
    end
    @showprogress "sample mesh" for f in GeometryBasics.faces(obj)
        verts = ((allverts[v] for v in f .+ 0)...,)
        for r123 in samps
            p = mapreduce(.*, .+, r123, verts)
            fn(p)
        end
    end
end

