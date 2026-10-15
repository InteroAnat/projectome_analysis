function load_nmr_data()
    HDF5.h5open("contrib/emfmri.ctx.ipsi.nmt20sym.mat") do f
        to_string(v) = (@assert size(f[v], 1) == 1; join(map(Char, f[v][:, :])[:]))
        subjs = map(to_string, f["subj_list"][:, :][:])
        connmats = map(v -> f[v][:, :], f["conn_mat_ctx"][:, :][:])
        subj_sites = map(v -> Int.(f[v][:, :]), f["channel_no_sum"][:, :][:])
        stim_verts = map(v -> Int.(f[v][:, :]), f["channel_vtxNo_sum"][:, :][:])

        area_conn = f["conn_mat_ctx_area"][:, :]
        area_names = map(to_string, f["label_name_ctx"][:, :][:])
        vert_area = map(Int, f["label_mat_ctx"][:, :][:])

        surf = load_vtk_surf("contrib/serfaces/lh.mid..vtk")
        surfinf = load_vtk_surf("contrib/serfaces/lh.mid_inflated..vtk")
        vert_pos = deepcopy(surf.position)
        dats = map(subjs, connmats, subj_sites, stim_verts) do subj, mat, sites, sitevert
            sn = [Symbol("$subj$s") for s in sites[:]]
            sites = map(sn, sitevert) do site, sitevert
                sitepos = vert_pos[sitevert]
                (; site, sitevert, sitepos)
            end |> DataFrame
            DataFrame(mat, sn), sites
        end
        vert_conn = reduce(hcat, first.(dats))
        sites = reduce(vcat, last.(dats))
        verts = map(vert_area, vert_pos) do area, pos
            areaname = area == 0 ? "" : area_names[area]
            (; area, pos, areaname)
        end |> DataFrame
        areas = map(area_names) do name
            (; name)
        end |> DataFrame
        (; area_conn, areas,
            vert_conn, sites, verts,
            surf, surfinf,
        )
    end
end

function load_vtk_surf(io::IO)
    vtkout, vtkascii, vtkpoly = 0, 0, 0
    nverts, npolys, npolynums = fill(nothing, 3)
    verts = GeometryBasics.Point3f[]
    polys = GeometryBasics.AbstractNgonFace[]
    fix_point(p) = begin
        p .* 500 .+ (32000.0f0, 39000.0f0, 25000.0f0)
        #(p .* 1000) .+ (32000.0, 27750.0, 8000.0)
    end
    for l in eachline(io)
        l = lstrip(l)
        isempty(l) && continue
        l[1] == '#' && continue
        l == "vtk output" && (vtkout += 1; continue)
        l == "ASCII" && (vtkascii += 1; continue)
        l == "DATASET POLYDATA" && (vtkpoly += 1; continue)
        if vtkout == 1 && vtkascii == 1 && vtkpoly == 1
            if nverts !== nothing && length(verts) < nverts
                push!(verts, parse.(Float32, split(rstrip(l), r"\s+")))
            elseif npolys !== nothing && length(polys) < npolys
                pp = parse.(Int, split(rstrip(l), r"\s+"))
                @assert length(pp) == pp[1] + 1
                npolynums -= length(pp)
                push!(polys, NgonFace(pp[2:end]...) .+ 1)
            elseif startswith("POINTS")(l)
                m = match(r"^POINTS\s+(\d+)\s+float$", l)
                @assert m !== nothing "wrong line $l"
                nverts = parse(Int, m.captures[1])
            elseif startswith("POLYGONS")(l)
                m = match(r"^POLYGONS\s+(\d+)\s+(\d+)$", l)
                @assert m !== nothing "wrong line $l"
                npolys = parse(Int, m.captures[1])
                npolynums = parse(Int, m.captures[2])
            else
                @assert false
            end
        else
            @assert false
        end
    end
    @assert length(polys) == npolys
    @assert length(verts) == nverts
    @assert npolynums == 0
    GeometryBasics.normal_mesh(fix_point.(verts), polys)
end
function load_vtk_surf(fn::AbstractString)
    open(load_vtk_surf, fn)
end
# contrib/serfaces/lh.inflated..vtk
# *contrib/serfaces/lh.mid..vtk
# contrib/serfaces/lh.mid_inflated..vtk
# contrib/serfaces/lh.orig..vtk
# *contrib/serfaces/lh.pial..vtk
# contrib/serfaces/lh.smoothwm..vtk
# *contrib/serfaces/lh.sphere..vtk
# contrib/serfaces/lh.white..vtk

function get_injsite_population_data(df, arbors, nmrdat, depthimg, depthres)
    injs = sort(unique(df.inj))
    injs = map(injs) do inj
        dfx = filter(r -> r.inj == inj && !r.qroot, df)
        injxyz = SVector(median(reduce(hcat, dfx.soma_pos), dims=2)...)
        if !NmtDat.isleft(injxyz)
            injxyz = NmtDat.mirror(injxyz)
        end
        (; inj, injxyz)
    end |> DataFrame
    injxyzm = last.(xyz2dxyz(injs.injxyz, depthimg, depthres))
    insertcols!(injs, :injxyzm => injxyzm)

    arbors = filter(r -> r.ipsi, arbors)
    leftjoin!(arbors, df, on=:path) |> dropmissing!
    segs = Data.load_swcs_as_segs(arbors.arborpath, arbors.soma_pos)
    segxyzm = last.(xyz2dxyz([s.pos for s in segs], depthimg, depthres))
    vertpos = nmrdat.verts.pos
    tree = NearestNeighbors.KDTree(vertpos)
    segnn = NearestNeighbors.knn(tree, segxyzm, 3)[1]
    vals = zeros(size(nmrdat.verts, 1), size(injs, 1))
    inj2row = Dict(reverse.(enumerate(injs.inj)))
    @showprogress "sum" foreach(segs, segxyzm, segnn) do seg, xyzm, nn
        seg.ic != :I && return
        any(isnan.(xyzm)) && return
        inj = inj2row[arbors[seg.nrn, :inj]]
        v2w = map(nn) do n
            d = norm(vertpos[n] .- xyzm) / 1000
            n => exp(-d * d)
        end
        ws = sum(last.(v2w))
        foreach(v2w) do (v, w)
            vals[v, inj] += w / ws * seg.len
        end
    end
    vert_conn = DataFrame(vals, injs.inj)
    (; vert_conn, injs,)
end

function match_inj_site(df, arbors, nmrdat)

    dfcmp = map(eachrow(dfinj)) do r1
        map(collect(nmrdat.site_vert)) do (site, vert)
            sitedist = norm(r1.injxyzm .- vertpos[vert])
            (; r1.inj, site, sitedist)
        end
    end
    dfcmp = reduce(vcat, dfcmp) |> DataFrame
    dfcmp

    # injdist
    # stim stimcent stimdist
end

