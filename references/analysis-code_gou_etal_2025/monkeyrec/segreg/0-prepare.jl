## common

@rule out = "input/fmost$dataset.catalog" let
    @info "download catalog $out"
    #url="https://yanlab.org.cn:30003/api/catalog/fmost$dataset/aaaaaa"
    url = "https://10.10.41.224:8002/api/catalog/fmost$dataset/aaaaaa"
    run(`curl -ko $out $url`)
end

@rule :phony "rm-swc-$ds" let
    run(`bash -c "rm input/*$ds*.swc* -v"`)
end

# XXX load high-res, adj contrast JiT
@rule :checksum :memdata "moving-imgs-$dataid" let info = "moving-imgs-info-$dataid"
    info = deepcopy(info)
    function get_merged(a, others...; adj, cats)
        res, off = parse_ds_res(cats)
        imgs = map([a, others...]) do fn
            @info "load $fn ..."
            FileIO.load(fn).data
        end
        siz = reduce(max, map(size, imgs))
        imgs = map(imgs) do x
            ds = siz .- size(x)
            @info "ds ... $ds"
            @assert all(ds .< 2)
            xx = similar(x, siz)
            xx .= 0
            xx[axes(x)...] .= x
            xx
        end
        r = stack(imgs, dims=4)
        (data=r, res=res, adj=adj, off=off)
    end
    parts = map(enumerate(info)) do (i, part)
        @info "Loading part $i ..."
        get_merged(part.files...; adj=part.adj, cats=part.cats)
    end
    (parts...,)
end

function parse_ds_res(fns::AbstractVector{<:AbstractString})
    res = nothing
    off = nothing
    for fn in fns
        ini = read(IniFile.Inifile(), fn)
        for (sec, _) in IniFile.sections(ini)
            if match(r"CH\d+_WEBM", sec) === nothing
                val = split(get(ini, sec, "direction"), r"\s+")[1:4:9]
                val = map(v -> parse(Float32, v), val)
                if res === nothing
                    res = val
                else
                    @assert res == val "$fn"
                end
                val = split(get(ini, sec, "origin"), r"\s+")
                val = map(v -> parse(Float32, v), val)
                if off === nothing
                    off = val
                else
                    @assert off == val
                end
            end
        end
    end
    res, off
end

@rule out = "input/raw-fmost$idpartch.swc.gz" let
    exe = expanduser("~/Desktop/gapr/build/source/export/gapr-export")
    conf = expanduser("~/Desktop/gapr/confdir/client.conf")
    conf2 = expanduser("~/Desktop/gapr/confdir/client2.conf")
    for cmd in [
        `$exe -z -c $conf root@yanlab.org.cn:30003/fmost$idpartch $out`
        `$exe -z -c $conf root@goulf-ts:33003/fmost$idpartch $out`
        `$exe -z -c $conf2 root@10.10.42.8:8002/fmost$idpartch $out`
        ``
    ]
        try
            @info "$cmd"
            run(cmd)
            !isfile(out) && error("no out")
            @info " xx $cmd"
            break
        catch
            isempty(cmd) && rethrow()
        end
    end
end

@rule out = "input/fmost$idpartch.swc.gz" let raw = "input/raw-fmost$idpartch.swc.gz"
    @info "...."
    exe = expanduser("~/Desktop/gapr/build/source/map/gapr-map")
    run(`$exe --decimate=27:3 -o $out $raw`)
end

@rule webm = "/data/dat/dsimgs/$ds@$avg.webm" let
    url = "https://goulf-ts:8002/api/data/$ds/$avg.webm"
    @info "download: $url"
    run(`curl -ko $webm $url`)
end

@rule nrrd = "/data/dat/dsimgs/$dsavg.nrrd" let webm = "/data/dat/dsimgs/$dsavg.webm"
    @info "Convert $nrrd..."
    nrrdtemp = nrrd * ".temp"
    exe = expanduser("~/Desktop/gapr/build/source/convert/gapr-convert")
    run(`$exe --to-nrrd $webm $nrrdtemp`)
    mv(nrrdtemp, nrrd, force=true)
end


@rule :checksum :memdata ref_img let
    @deps path = "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz"
    @deps path_seg = "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_segmentation.nii.gz"
    ratio = 2.0
    @info "Load reference $path..."
    img = ImageTransformations.imresize(NIfTI.niread(path).raw, ratio=ratio, method=Interpolations.Linear())
    @info "Load reference $path_seg..."
    tissuetype(vv) = begin
        # 1. !CSF
        # 2. *WM
        # 3. *scGM
        # 4. *GM
        # 5. !BV
        if vv < 0.5
            0.0f0
        elseif vv < 1.5
            0.05f0
        elseif vv < 2.5
            1.0f0
        elseif vv < 3.5
            0.5f0
        elseif vv < 4.5
            0.3f0
        elseif vv < 5.5
            0.05f0
        else
            @assert false
        end
    end
    seg = Float32.(ImageTransformations.imresize(tissuetype.(NIfTI.niread(path_seg).raw), ratio=ratio, method=Interpolations.Constant()))
    @assert size(img) == size(seg)
    res = Array{Float32,4}(undef, size(img)..., 3)
    res[:, :, :, 1] .= img[:, :, :] ./ 1024.0
    fgbg(vv) = (vv != 0 ? 1.0 : 0.0)
    res[:, :, :, 2] .= fgbg.(seg[:, :, :])
    #=
    segs=unique(seg)
    @info "$segs"
    xlate=[
           # 2. *WM
           2.0 => 1.0,
           # 3. *scGM
           3.0 => 0.5,
           # 4. *GM
           4.0 => 0.3,
           # 5. !BV
           5.0 => 0.07,
           # 1. !CSF
           1.0=>0.05,
    ]
    res3=similar(seg)
    res3.=0.0
    prevmask=deepcopy(res3)
    @showprogress for (f, t) in xlate
        curmask=fgbg.(seg.-f)
        prevmask=min.(prevmask.+1.0.-curmask, 1.0)
        filt=ImageFiltering.imfilter(prevmask, ImageFiltering.Kernel.gaussian((2f0,2f0,2f0), (9,9,9)), ImageFiltering.Algorithm.FFT())
        res3[filt.>0.7 .&& res3.==0].=t
    end
    =#
    res[:, :, :, 3] .= seg
    #res[:, :, :, 3] .= tissuetype.(seg[:, :, :])
    (dat=res, res=Float32.([250.0, 250.0, 250.0] ./ ratio))
end

@rule :checksum :memdata ref_img2 let
    @deps path = "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz"
    @deps path_seg = "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym/NMT_v2.0_sym_segmentation.nii.gz"
    @info "Load reference $path..."
    img = NIfTI.niread(path).raw
    @info "Load reference $path_seg..."
    seg = NIfTI.niread(path_seg).raw
    res = map(img, seg) do v1, vv
        v2, v3, v4 = if vv < 0.5
            0.0f0, 0.0f0, 0.0f0
        elseif vv < 1.5
            0.0f0, 0.0f0, 0.0f0
        elseif vv < 2.5
            1.0f0, 1.0f0, 1.0f0
        elseif vv < 3.5
            1.0f0, 1.0f0, 0.0f0
        elseif vv < 4.5
            1.0f0, 0.0f0, 0.0f0
        elseif vv < 5.5
            0.0f0, 0.0f0, 0.0f0
        else
            @assert false
        end
        SVector(v1 / 1024.0f0, v2, v3, v4)
    end
    res = stack((map(v -> v[i], res) for i in axes(res[1], 1)), dims=4)
    (dat=Float16.(res), res=Float32.(SVector(250.0, 250.0, 250.0)))
end

@rule :checksum :memdata ref_img_mask let ref_img
    img = deepcopy(ref_img.dat[:, :, :, 3])
    img0 = map(v -> v > 0.1 ? 1.0 : 0.0, img)

    img = ImageTransformations.imresize(img0, ratio=0.5, method=Interpolations.Constant())
    img = ImageTransformations.imresize(img, ratio=2.0, method=Interpolations.Linear())
    img = map(v -> v > 0.5 ? 1.0 : 0.0, img)

    img2 = ImageFiltering.imfilter(img, ImageFiltering.Kernel.gaussian((1.3f0, 1.3f0, 1.3f0), (9, 9, 9)), ImageFiltering.Algorithm.FFT())
    img = map(v -> v > 0.6 ? 1.0 : 0.0, img2)
    (dat=img, res=ref_img.res)
end

@rule :checksum :memdata "ref-pts-$dataid" let fn = "data/stitch/bootstrap-$dataid.txt", stitch = "stitch-info-$dataid"
    res = []
    open(fn) do io
        for l in eachline(io)
            l = lstrip(l)
            isempty(l) && continue
            l[1] == '#' && continue
            m = match(r"^\s*\(([-0-9\.]+),\s*([-0-9\.]+),\s*([-0-9\.]+)\)/(\d+)/(\d+)\s*(#.*)?", l)
            @assert m !== nothing
            xyz = map((x) -> parse(Float64, x), m.captures[1:3])
            spc = parse(Int, m.captures[4])
            typ = parse(Int, m.captures[5])
            push!(res, (pos=[xyz...,], block=spc, type=typ))
        end
    end
    function fix_pos(B, pos)
        if B == 1
            pos[3] += 100
        end
        pos
    end
    chkdup = Set()
    for (i, file) in enumerate(stitch)
        open(file) do io
            for l in eachline(io)
                l = lstrip(l)
                isempty(l) && continue
                f = split(l, ',')
                @assert length(f) >= 6
                pos = map(v -> parse(Float64, v), f[3:5])
                t = parse(Int, f[6])
                n = parse(Int, f[2])
                b = parse(Int, f[1])
                if n == 0
                    t = t + i * 100_000
                else
                    t = t + i * 1000_000
                end
                push!(res, (node=n, pos=fix_pos(b, pos), block=b, type=t, ch=i))
                if n != 0
                    kkk = (n, b, i)
                    if in(chkdup, kkk)
                        error("duplicate id: $n")
                    end
                    push!(chkdup, kkk)
                end
            end
        end
    end
    res
end

@rule :+ out = "input/fmost233001xx$part.ch$ch.catalog" let in = "input/fmost233001.ch$ch.catalog"
    ini = open(in) do io
        read(IniFile.Inifile(), io)
    end
    zoff, zsiz = if part == "1"
        1, 158
    elseif part == "2"
        188, 708
    else
        error("no such part")
    end
    zres = 60.0f0
    zoff = zres * (zoff - 1)
    for (n, sect) in IniFile.sections(ini)
        sect["origin"] = "0.0 0.0 $zoff"
        if endswith("_MIP")(n) || endswith("_AVG")(n)
            sect["size"] = replace(sect["size"], r"(\s+)\d+$" => s"\1") * "$zsiz"
        end
    end
    open(out, "w") do io
        write(io, ini)
    end
end

@rule :+ nrrd = "/data/dat/dsimgs/fmost233001xx$part.ch$ch@ch00-$tag.nrrd" let in = "/data/dat/dsimgs/fmost233001.ch$ch@ch00-$tag.nrrd"
    img = FileIO.load(in)
    if part == "1"
        img = img[:, :, 1:158]
    elseif part == "2"
        img = img[:, :, 188:end]
    else
        error("no such part")
    end
    @info "Convert xxxx $nrrd..."
    nrrdtemp = nrrd * ".temp.nrrd"
    FileIO.save(nrrdtemp, img)
    mv(nrrdtemp, nrrd, force=true)
end

import Gapr
@rule :+ out = "input/fmost233001xx$part.ch$ch.swc.gz" let in = "input/fmost233001.ch$ch.swc.gz"
    swc = Gapr.load_swc(in)
    filters = Gapr.split_swc(swc) do i, n
        n.pos[3] < 11020.0 ? "1" : "2"
    end
    open(pipeline(`gzip -c6 -`, out), "w") do io
        Gapr.save_swc(io, swc; filter=get(filters, part, ()))
    end
end

