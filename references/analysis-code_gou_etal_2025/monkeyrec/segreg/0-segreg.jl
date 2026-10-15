###################################################################
###################################################################
macro register_dataset(name::AbstractString, info::Expr)
    info = eval(info)
    for part in info.parts
        eval(quote
            @rule :memdata $("moving-imgs-partinfo-$part") let
                @deps cats = $(["input/$part$chan.catalog" for chan in info.chans])
                @deps files = $(["/data/dat/dsimgs/$part$chan@$img.nrrd" for chan in info.chans for img in info.dsimgs])
                (files=files, cats=cats, adj=$(info.imgadj))
            end
        end)
    end
    eval(quote
        @rule :memdata $("moving-imgs-info-$name") let
            @deps info_parts = $(["moving-imgs-partinfo-$part" for part in info.parts])
            (info_parts...,)
        end
    end)

    for (chidx, chan) in enumerate(info.chans)
        eval(quote
            @rule :memdata $("swc-chaninfo-$name-ch$chidx") let
                @deps swcs = $(["input/$part$chan.swc.gz" for part in info.parts])
                swcs
            end
        end)
    end

    eval(quote
        @rule :memdata $("stitch-info-$name") let
            @deps stitch = $(["data/stitch/$name.ch$chan.txt" for chan in 1:length(info.chans)])
            stitch
        end
    end)

    eval(quote
        @rule :phony $("input-$name") let
            @deps $(["swc-chaninfo-$name-ch$chan" for chan in 1:length(info.chans)])
        end
    end)

    eval(quote
        @rule :phony $("temp-$name") let
            @deps $(["data/temp/xformed-$name.ch$chan.swc.gz" for chan in 1:length(info.chans)])
            @deps $(["data/temp/xformed-$name-part$pt.ch$chan.catalog" for pt in 1:length(info.parts) for chan in 1:length(info.chans)])
        end
    end)
    eval(quote
        @rule :phony $("final-$name") let
            @deps $(["data/final/warped-$name.ch$chan.swc.gz" for chan in 1:length(info.chans)])
            @deps $("data/final/warpimg-$name.nrrd")
        end
    end)
    quote end
end

@rule out="data/temp/xformed-$name.ch$ch.swc.gz" let xform="interm/xform2-$name", ref_pts="ref-pts-$name"
    @deps knn="knn-tree2-$name"
    @deps swcs="swc-chaninfo-$name-ch$ch"
    xform,_=deepcopy(xform)
    ch=parse(Int, ch)
    ref_pts=filter(p->get(p, :ch, nothing)==ch, ref_pts)
    @info "Join $swcs........................-- $(length(ref_pts))"
    xform_swc(out, knn, xform, ref_pts, swcs...; prec=0.0002)
end

