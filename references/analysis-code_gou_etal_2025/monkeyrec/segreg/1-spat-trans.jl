@rule :memdata spat_trans_img let img = "../monkeyrec/interm/spat-trans.nrrd"
    img = FileIO.load(img)
    (data=reshape(img, size(img)..., 1), res=(125.0f0, 125.0f0, 125.0f0))
end

@rule :memdata spat_trans_meta let dat = "../monkeyrec/interm/spat-trans-img.dat"
    return Jmake.priv.load_data(splitext(dat)[1]).data
end

@rule :memdata spat_trans_ref_img let ref_img
    img = ref_img.dat[1:round(Int, size(ref_img.dat, 1) / 2), :, :, :]
    img1 = img[:, :, :, 1]
    img2 = img[:, :, :, end]
    msk = abs.(img2 .- 1) .> 0.01
    img2[msk] .= 0
    img1[msk] .= 0
    (dat=Flux.stack((img1, img2), dims=4), res=ref_img.res)
end

@rule :data "interm/spat-trans-init" let ref_img = "spat-trans-ref-img", spat_trans_img, spat_trans_meta
    ref_pts = [
        (pos=(32218.25, 16243.43, 70496.08), blk=1, grp=1), #tele
        (pos=(28209.94, 72480.79, 31894.61), blk=0, grp=1),
        (pos=(18153.89, 55233.68, 39473.20), blk=1, grp=2), #tp down
        (pos=(17869.98, 52346.27, 6157.76), blk=0, grp=2),
        (pos=(29645.77, 21154.23, 423.76), blk=1, grp=3), # post
        (pos=(23948.58, 2681.37, 18989.54), blk=0, grp=3),
        (pos=(9314.47, 25272.76, 31197.63), blk=1, grp=4), # tpleft
        (pos=(2903.00, 36459.53, 21453.76), blk=0, grp=4),
    ]
    minz, maxz = extrema(map(v -> v.z, spat_trans_meta))
    minz1, maxz1 = extrema(map(v -> v.pos[3], filter(v -> v.blk == 1, ref_pts)))
    ref_pts = map(ref_pts) do p
        if p.blk == 1
            pos3 = (p.pos[3] - minz1) / (maxz1 - minz1) * (maxz - minz) + minz
            return (; p..., pos=(p.pos[1:2]..., pos3))
        end
        return p
    end

    @info "initial xform ...... ..........."
    prob = SegReg.Problem(ref_img, (spat_trans_img,), ref_pts)
    show(prob)
    xform = SegReg.Network(prob)
    show(xform)
    open("/tmp/segreg1.log", "w") do log_io
        xform = prob(xform;
            landmark_weights=(10.0, 1000.0, 100000.0),
            n_iter=20000,
            log_io
        )
        xform = prob(xform;
            landmark_weights=(1000.0, 100.0, 10.0),
            n_iter=20000,
            log_io
        )
        xform
    end
end

@rule :data "interm/spat-trans-affine" let ref_img = "spat-trans-ref-img", spat_trans_img, spat_trans_meta
    @deps xform = "interm/spat-trans-init"
    @info "affine xform ............ "
    xform = deepcopy(xform)
    prob = SegReg.Problem(ref_img, (spat_trans_img,), [])
    show(prob)
    xform = SegReg.Network(xform, unet=:new, slicers=(spat_trans_meta,))
    show(xform)
    open("/tmp/segreg2.log", "w") do log_io
        xform = prob(xform;
            landmark_weights=(10000.0, 1000.0, 100.0),
            n_iter=10000,
            dpts_scale=3000,
            log_io
        )
        SegReg.Network(xform, unet=:delete), prob.group_pos
    end
end

@rule outfile = "data/final/warpimg-spat-trans.nrrd" let ref_img = "spat-trans-ref-img", spat_trans_img
    @deps xform = "interm/spat-trans-affine"
    xform_pars, point_gpos = deepcopy(xform)
    prob = SegReg.Problem(ref_img, (spat_trans_img,), [])
    img, warp = SegReg.get_warp_img(prob, xform_pars)
    FileIO.save(outfile, img[:, :, :, 1])
end

#########################################

########################
###############################
###################
##########################
###################
################################################################
#####################################
########################

##############################################

function xxx()
    #################################
    # XXX
    # * special sampler in SegReg
    # * mis-alignment sections: affine matrix/(even warp) for each section 
    #######################
    ##################
    ###############################
    #################################

    @assert length(dat) == 119
    pts = reduce(hcat, reduce(vcat, dat))
    #return pts
    exts = extrema(pts, dims=2)
    #return exts
    siz = map(exts) do (a, b)
        ceil(Int, (b - a) / 125) + 5
    end
    off = map(first, exts)
    img = zeros(Float32, siz...)
    for col in eachcol(pts)
        idx = ceil.(Int, (col .- off) ./ 125) .+ 2
        img[idx...] = 128
    end
    FileIO.save("/tmp/test.nrrd", img)
    pts


    #       "/regions-macaque2.csv"
    #        "/regions-macaque3.csv"
    #
    #
    #            
    #             
    #
    #y=map(x) do (a,b);
    #m.captures[2:end]; end
    #3-element Vector{Vector{Union{Nothing, SubString{String}}}}:
    #["20230503", "macaque1", "contour2"]
    #["20230619", "macaque2", "contour2"]
    #["20230619", "macaque3", "MQ121"]
    #
end
