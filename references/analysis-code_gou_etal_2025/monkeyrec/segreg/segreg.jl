import CUDA
import Flux
import IniFile
using LinearAlgebra
import Statistics
import FileIO
import NIfTI
import ImageTransformations
import ImageFiltering
import Interpolations
import GLMakie
import Metalhead
using ProgressMeter
using ColorTypes
using Random
import NearestNeighbors
import IniFile
import GeometryBasics
import JLD2

import GPUArrays
GPUArrays.allowscalar(false)

import PartialFunctions
function PartialFunctions.PartialFunction(expr_string::String, f, args::T, kwargs::N) where {T<:Tuple,N<:NamedTuple}
    return PartialFunctions.PartialFunction{nothing,nothing,typeof(f),T,N}(expr_string, f, args, kwargs)
end

import Revise
using Jmake

import SegReg
import Gapr

Revise.includet("0-prepare.jl")
Revise.includet("0-segreg.jl")
Revise.includet("0-tetragrid.jl")
Revise.includet("1-datasets.jl")
Revise.includet("1-spat-trans.jl")
Revise.includet("1-segment.jl")








function warp_net(N::Int)
    nx, ny = 32, 64
    block() = Flux.Chain(
        Flux.Parallel(
            +,
            identity,
            Flux.Chain(
                Flux.Conv((1,), nx => ny, bias=false, groups=4),
                Flux.BatchNorm(ny),
                Flux.tanh_fast,
                #Flux.leakyrelu,
                Flux.Conv((1,), ny => nx, bias=false, groups=4),
                Flux.BatchNorm(nx),
            ),
        ),
        Flux.tanh_fast,
        #Flux.leakyrelu,
    )
    model = Flux.Chain(
        Flux.Conv((1,), 3 => nx),
        block(),
        block(),
        block(),
        block(),
        block(),
        #block(),
        #block(),
        #block(),
        #block(),
        #block(),
        #block(),
        Flux.Conv((1,), nx => 3 * N),
    )
    model = model |> gpudev
    optst = Flux.setup(Flux.Adam(), model)
    loss(m, d) = begin
        sum((xfunc_warp3(m, 1:N, d, 3000.0f0) ./ 3000 .- 0) .^ 2; gpudev)
    end
    gen_data() = rand(3, 16, 4) .* (100000) |> gpudev
    loss(model, gen_data())
    for iter in 1:1000
        dat = gen_data()
        l, g = Flux.withgradient(m -> loss(m, dat), model)
        Flux.update!(optst, model, g[1])
        if iter % 100 == 0
            @info ". $iter $l"
        end
    end
    model
end

@rule :memdata eval_warp let ref_img, bones
    fixed_pts = [160 128 68
        353 148 67
        159 366 67
        361 389 67
        130 68 191
        405 108 191
        97 415 191
        388 425 191
        255 303 334]

    warps = [255 322 265 1000 -300
        259 310 237 -1000 300
        352 366 210 200 500]

    egs = []
    gpudev = Flux.FluxCUDADevice(nothing)
    for r in eachrow(fixed_pts)
        push!(egs, [r .* ref_img.res..., 0, 0, 0])
    end
    for r in eachrow(warps)
        push!(egs, [r[1:3] .* ref_img.res..., r[4:end]..., 0])
    end
    egs = Float32.(reduce(hcat, egs)) |> gpudev

    pars = (warp=warp_spline(2), warpfactor=Ref(1000.0f0)) |> gpudev

    bones = bones |> gpudev

    function loss(m, K)
        l1 = loss_bones(m, bones; gpudev)
        vals = xfunc_warp3(m.warp, :, view(egs, 1:3, :), m.warpfactor[]; gpudev)
        l2 = sum(sum(((vals[:, 1, :] .- egs[4:end, :]) ./ 100.0f0) .^ 2, dims=1))
        if K % 100 == 0
            println("$K $l1 $l2")
        end
        l1 + l2
    end
    loss(pars, 0)


    optst = Flux.setup(Flux.Adam(), pars)
    for K in 1:1000
        l, g = Flux.withgradient(pars) do m
            loss(m, K)
        end
        Flux.update!(optst, pars, g[1])
    end

    function gen_img()
        ref_img = ref_img |> gpudev
        out = similar(ref_img.dat, size(ref_img.dat)[1:3]..., 3)
        @showprogress for z in 1:size(out, 3)
            inds = CartesianIndices(out)[:, :, z, 1] |> gpudev
            inds = reshape(inds, size(inds)..., 1)
            pos1i = stack([map(i -> i[k], inds) for k in 1:3], dims=1)
            pos1 = pix2position(pos1i, ref_img.res)
            pos2d = xfunc_warp3(pars.warp, 1, pos1, pars.warpfactor[]; gpudev)
            for c in 1:3
                out[map(i -> i + CartesianIndex(0, 0, 0, c - 1), inds)] .= pos2d[c, :, :]
            end
            ############################################################################
            ############################################################################
            ############################################################################
            ############################################################################
            #dx=xfunc_warp3(xform_pars.warp, :, pos1, xform_pars.warpfactor[])[:, 2, :, :]
        end
        out |> Flux.cpu
    end
    return gen_img()

end
################################################

###################################
##################################
#
#
#
#
#
#
#
#####################################
#
#ref: (512, 624, 400, 3) [125.0, 125.0, 125.0]
#moving 1: (401, 287, 69, 4) [130.0, 130.0, 120.0]
#moving 2: (486, 359, 477, 4) [130.0, 130.0, 120.0]
######################################
#affine_xform((mat.*matd, shift.*(1.0f0 .+shiftd)), B, pos)
#return sum(vals2)

##############################################################
#=
    function train_segment(unet, unetdata, moving_imgs; niter=1000)
        function get_seg_vis()
            (ref, dat, msk)=take!(chan)
            @assert size(ref)==(256,256, 3)
            @assert size(dat)==(256,256, 4)
            @assert size(msk)==(256,256)
            dat=reshape(dat, size(dat)..., 1)
            out=unet(dat)
            img=map(dat[:, :, 1, 1], out[:, :, 3, 1]) do r,g
                RGB{Float32}(r, g, 0.0f0)
            end
            ImageTransformations.imresize(img|>Flux.cpu, ratio=3)
            #reshape(img, size(img)[1:2]...)
        end

        obs=let
            scene=GLMakie.Scene(camera=GLMakie.campixel!)
            obs=GLMakie.Observable(get_seg_vis()|>Flux.cpu)
            GLMakie.image!(scene,obs)
            display(scene)
            GLMakie.DataInspector(scene)
            obs
        end
        function get_train_data()
            samps=map(1:4) do k
                take!(chan)
            end
            ref=stack([samps[k][1] for k in 1:4], dims=4)
            dat=stack([samps[k][2] for k in 1:4], dims=4)
            msk=stack([samps[k][3] for k in 1:4], dims=3)
            (ref=ref,dat=dat,msk=Float32.(msk))
        end
        function unet_loss(m, dat)
            #sum(((out.-ref).^2).*reshape(msk, size(msk)[1:2]..., 1, size(msk)[end]))/sum(msk)
        end

        unet_loss(unet, get_train_data())

        optst=Flux.setup(Flux.Adam(), unet)

        @info "start..."
        @showprogress for K in 1:niter
            dat=get_train_data()
            l,g=Flux.withgradient(m->unet_loss(m, dat), unet)
            Flux.update!(optst, unet, g[1])
            if K%300==0
                @info "$K dist $l"
                obs[]=get_seg_vis() |>Flux.cpu
            end
        end
        unet |> Flux.cpu
    end
    =#



function reverse_map_impl(kkkkk, x0, x1, opt, B, pars; prec)
    l = 10000
    traj = Float64[]
    s = 2048.0f0
    while l >= prec && s > 128
        dx = similar(x0)
        dx .= 0.0f0
        function lossfunc(xx)
            dd = SegReg.execute_xform(pars, B, x0 .+ xx .* s) .- x1
            d = sum((dd ./ 10.0f0) .^ 2, dims=1)
            Flux.mean(d) + maximum(d)
        end
        optst = Flux.setup(opt, dx)
        for iter in 1:128
            l, g = Flux.withgradient(lossfunc, dx)
            Flux.update!(optst, dx, g[1])
            push!(traj, l)
            l < prec && break
            ##@info "losss $iter/$s $l"
        end
        x0 = x0 .+ dx .* s
        s = s / 2
    end
    dd = SegReg.execute_xform(pars, B, x0) .- x1
    dd = sqrt.(sum(dd .^ 2, dims=1))
    x0, traj, reshape(dd, :)
end

function reverse_map(pars, B, pts, init; prec)
    pars = pars |> Flux.gpu
    pts = pts |> Flux.gpu
    init = init |> Flux.gpu
    siz1, siz2p... = size(init)
    @assert siz1 == 3
    @assert length(siz2p) == 1
    @assert size(pts) == size(init)
    K = siz2p[1]
    ddd = similar(init, (K,))
    ddd .= 0.0f0
    allopts = [Flux.AdaGrad, Flux.Nesterov, Flux.RMSProp]
    batsiz = 256 * 1024
    @showprogress for k in 1:batsiz:K
        rng = k:min(k + batsiz - 1, K)
        #ff=GLMakie.Figure()
        #display(ff)
        #GLMakie.DataInspector(ff)
        #for (iii,opt) in enumerate(allopts)
        #ax = GLMakie.Axis(ff[(iii-1)%4+1, floor(Int, (iii-1)/4)+1], title=string(opt), yscale=log10)
        #try
        vals, traj, dd = reverse_map_impl(k, init[:, rng], view(pts, :, rng), Flux.AdaGrad(), B, pars; prec)
        init[:, rng] .= vals
        ddd[rng] .= dd
        #GLMakie.lines!(ax, axes(traj, 1), traj)
        #GLMakie.autolimits!(ax)

        ####################################
        #############################
        ###############################
        #catch e
        #e isa InterruptException && break
        #end
        #end
    end
    init, ddd
end

function xform_swc(outfn, knn, pars, points, infns...; prec=0.002)
    Random.seed!(1)
    knn = map(knn) do pts
        @assert size(pts, 1) == 6
        tree = NearestNeighbors.KDTree(pts[4:end, :])
        (tree=tree, from=pts[1:3, :])
    end
    function invmap(B, pts)
        siz1, siz2p... = size(pts)
        @assert siz1 == 3
        pts = reshape(pts, siz1, :) |> Flux.cpu
        inds, dists = NearestNeighbors.knn(knn[B].tree, pts, 1)
        reshape(knn[B].from[:, map(first, inds)], siz1, siz2p...) |> Flux.gpu
    end
    pars = pars |> Flux.gpu
    SegReg.prepare_xform!(pars)
    @info "xform.: $(length(points))"
    id_inc = 1_000_000_000
    paired = Dict{Int,Int}()
    t2n = Dict{Int,Int}()
    for p in points
        n = get(p, :node, nothing)
        if n !== nothing
            n += id_inc * (p.block - 1)
            paired[n] = p.type
        end
    end
    marks = []
    swcs = []
    for (B, infn) in enumerate(infns)
        swc = Gapr.load_swc(infn; idoffset=id_inc * (B - 1))
        isempty(swc.nodes) && continue
        filter!(swc.annots) do a
            if a.key == :root && startswith("alien", a.val)
                push!(marks, a.id)
                return false
            end
            true
        end
        pts = reduce(hcat, map(n -> n.pos, swc.nodes)) |> Flux.gpu
        pts2, ddd = reverse_map(pars, B, pts, invmap(B, pts); prec) |> Flux.cpu
        degrees = collect(Gapr.nodedegree(swc).degree)
        for (idx, v) in enumerate(degrees)
            if v <= 1
                push!(marks, swc[idx].id)
            end
        end
        for (idx, n) in enumerate(swc.nodes)
            Gapr.addannot!(swc, (id=n.id, key=:previd, val="$(n.id)"))
            at = get(paired, n.id, nothing)
            if at !== nothing
                n.typ = 3
                n2 = get(t2n, at, nothing)
                if n2 !== nothing
                    push!(marks, n.id)
                    push!(marks, n2)
                    Gapr.addloop!(swc, (n.id, n2))
                else
                    t2n[at] = n.id
                end
            end
            if n.paridx != 0 && get(paired, swc[n.paridx].id, nothing) !== nothing
                n.typ = 3
            end
            n.rad = ddd[idx]
            n.pos = SVector(view(pts2, :, idx)...)
        end
        push!(swcs, swc)
    end
    swcs = Gapr.join_swc!([s.data_ for s in swcs]...)
    swcs = Gapr.sort_swc!(swcs)
    swcs = take!(swcs)
    has_err = Set{Int64}()
    for ann in swcs.annots
        if ann.key === :error
            push!(has_err, ann.id)
        end
    end
    for a in unique(marks)
        a in has_err && continue
        Gapr.addannot!(swcs, (id=a, key=:error, val="HIGHLIGHTTERMS"))
    end
    for n in swcs.nodes
        n.pos = round.(n.pos, digits=3)
        n.rad = round(n.rad, digits=3)
    end
    Gapr.save_swc(outfn, swcs)
end



function sample_img_impl(img, inds, chrng)
    @assert length(size(img)) == 4
    sz = axes(img)[1:3]
    inds = map(inds) do i
        all(in.(Tuple(i), sz)) ? i : CartesianIndex(0, 0, 0)
    end
    sel = map(i -> i != CartesianIndex(0, 0, 0), inds)
    selinds = inds[sel]
    res = map(enumerate(chrng)) do (i, ch)
        res = similar(img, size(inds)...)
        res .= 0.0f0
        indsx = map(i -> CartesianIndex(i, ch), selinds)
        tmp = img[indsx]
        res[sel] .= tmp
        res
    end
    res
    #stack(res, dims=4)
end
function sample_img(img, inds, ch::Int)
    sample_img_impl(img, inds, ch:ch)[1]
end
function indices_to_pos(indices, res)
    res = res |> Flux.cpu
    stack([(map(i -> i[k], indices) .- 0.5f0) .* res[k] for k in 1:3], dims=1)
end
function indices_from_pos(pos, res)
    siz1, siz2p... = size(pos)
    pos = reshape(pos, siz1, :)
    res = res |> Flux.cpu
    ret = map((args...) -> CartesianIndex(args...), [round.(Int, pos[k, :] ./ res[k] .+ 0.5) for k in 1:3]...)
    reshape(ret, siz2p...)
end

@rule :data "interm/xform1-$name" let ref_img, moving_imgs = "moving-imgs-$name", ref_pts = "ref-pts-$name"
    @info "initial xform $name...... ..........."
    prob = SegReg.Problem(ref_img, moving_imgs, ref_pts)
    show(prob)
    xform = SegReg.Network(prob)
    show(xform)
    open("/tmp/segreg1-$name.log", "w") do log_io
        xform = prob(xform;
            landmark_weights=(10.0, 1000.0, 100000.0),
            n_iter=30000,
            log_io
        )
        xform = SegReg.Network(xform, warp=:newbspline0)
        xform = prob(xform;
            landmark_weights=(1000.0, 100.0, 100.0),
            n_iter=20000,
            log_io
        )
        xform, prob.group_pos
    end
end

@rule :data "interm/xform2-$name" let ref_img, moving_imgs = "moving-imgs-$name", ref_pts = "ref-pts-$name"
    @deps xform = "interm/xform1-$name"
    @info "warp xform $name............ "
    xform, group_pos = deepcopy(xform)
    prob = SegReg.Problem(ref_img, moving_imgs, ref_pts, group_pos)
    show(prob)
    xform = SegReg.Network(xform, unet=:new, warp=:newbspline)
    SegReg.load_pretrain!(xform) do tag
        JLD2.load("pretrain/$tag.jld2", "model_state")
    end
    show(xform)
    open("/tmp/segreg2-$name.log", "w") do log_io
        xform = prob(xform;
            landmark_weights=(10000.0, 1000.0, 100.0),
            n_iter=10000,
            dpts_scale=3000,
            log_io
        )
        SegReg.save_pretrain(xform) do state, tag
            JLD2.jldsave("pretrain/$name-$tag.jld2", model_state=state)
        end
        SegReg.Network(xform, unet=:delete), prob.group_pos
    end
end
##########################################
##################################################################################
################################################################################
################################################################################
#################################################################################
######################################################################
##############
########################################################################################################
########################################################################################################
########################################################################################################

@rule :data "interm/xform3-$name" let ref_img2, moving_imgs = "moving-imgs-$name", ref_pts = "ref-pts-$name"
    @deps xform = "interm/xform2-$name"
    @info "warp xform3 $name......... "
    xform, group_pos = deepcopy(xform)
    prob = SegReg.Problem(ref_img2, moving_imgs, ref_pts, group_pos)
    show(prob)
    xform = SegReg.Network(xform, unet=:new, warp=:newbspline)
    xform = SegReg.Network2(xform)
    show(xform)
    open("/tmp/segreg3-$name.log", "w") do log_io
        xform = SegReg.test3(prob, 0, xform;
            landmark_weights=(100000.0, 10000.0, 100.0),
            n_iter=30000,
            dpts_scale=3000,
            log_io
        )
        ########################################################################################################
        # SegReg.save_pretrain(xform) do state, tag
        #     JLD2.jldsave("pretrain/final-$name-$tag.jld2", model_state=state)
        # end
        SegReg.Network(xform, unet=:delete), prob.group_pos
    end
end


function get_pixcls(inch)
    # . !CSF # . !BV
    # . *WM # . *GM
    nx, ny = 16, 32
    block() = Flux.Chain(
        Flux.Parallel(
            +,
            identity,
            Flux.Chain(
                Flux.Conv((3, 3), nx => ny, pad=1, bias=false),
                Flux.BatchNorm(ny),
                Flux.leakyrelu,
                Flux.Conv((1, 1), ny => nx, bias=false),
                Flux.BatchNorm(nx),
            ),
        ),
        Flux.leakyrelu,
    )
    Flux.Chain(
        Flux.Conv((1, 1), inch => nx),
        block(),
        block(),
        Flux.Conv((1, 1), nx => 1),
    )
end

function calc_jacobian(img, res)
    out = similar(img, size(img)[1:end-1])
    function get_offset(img, ci, col, o)
        ci1 = MVector(Tuple(ci))
        ci1[col] += o
        if checkbounds(Bool, out, ci1...)
            ci1, 1
        else
            MVector(Tuple(ci)), 0
        end
    end
    function deriv(img, ci, row, col)
        ci1, d1 = get_offset(img, ci, col, -1)
        ci2, d2 = get_offset(img, ci, col, 1)
        dd = d1 + d2
        (img[ci2..., row] - img[ci1..., row] + (ci2[row] - ci1[row]) * res[row] * 0) / (dd * res[col])
    end
    nrow = size(img)[end]
    ncol = length(size(out))
    @showprogress for ci in CartesianIndices(out)
        mat = SMatrix{nrow,ncol}((deriv(img, ci, row, col) for row in 1:nrow, col in 1:ncol)...)
        out[ci] = det(mat)
    end
    out
end

@rule :memdata "jacob-$dataid" let xform = "interm/xform3-$dataid", ref_img, moving_imgs = "moving-imgs-$dataid"
    xform_pars, point_gpos = deepcopy(xform)
    _, warp = get_warp_img(ref_img, moving_imgs, xform_pars; gpudev=Flux.FluxCUDADevice(nothing))
    warp
    #calc_jacobian(img)
end
#############################
############################################################################
############################################################################
############################################################################
############################################################################

@rule outfile = "data/final/warpimg-$dataid.nrrd" let xform = "interm/xform3-$dataid", ref_img, moving_imgs = "moving-imgs-$dataid"
    xform_pars, point_gpos = deepcopy(xform)
    prob = SegReg.Problem(ref_img, moving_imgs, [])
    img, warp = SegReg.get_warp_img(prob, xform_pars)
    FileIO.save(outfile, img[:, :, :, 1])
end

@rule :memdata "warpimg-$dataid" let xform = "interm/xform2-$dataid", ref_img, moving_imgs = "moving-imgs-$dataid"
    xform_pars, point_gpos = deepcopy(xform)
    prob = SegReg.Problem(ref_img, moving_imgs, [])
    img, warp = SegReg.get_warp_img(prob, xform_pars)
    img
end

@rule :memdata "xxx-fw-warpimg-$dataid" let xform = "interm/xform2-$dataid", ref_img, moving_imgs = "moving-imgs-$dataid"
    xform_pars, point_gpos = deepcopy(xform)
    prob = SegReg.Problem(ref_img, moving_imgs, [])
    src = FileIO.load("/home/goulf/MASK_Result of MASK_NMT_v2.0_sym_fh.nii.nrrd").data
    src_shift = (200.5 - 128.5, 181.25 - 137.46, 152.65 - 103.06,)
    img = SegReg.get_fw_warp_img(prob, xform_pars, src, src_shift)
    img
end


@rule out = "output/stitch-bootstrap-$dataid.txt" let xform2 = "xform2-$dataid", ref_pts = "ref-pts-$dataid"
    xform, grp2pts = deepcopy(xform2)
    gpudev = Flux.FluxCUDADevice(nothing)
    xform = xform |> gpudev
    xform = SegReg.prepare_xform!(xform; gpudev)
    open(out, "w") do io
        for pts in ref_pts
            blk = pts.block
            blk == 0 && continue
            t = pts.type
            t >= 100 && continue
            pos0 = grp2pts[t]
            posb = execute_xform(xform, blk, reshape(pos0, 3, 1, 1) |> gpudev; gpudev) |> Flux.cpu
            println(io, "(", join(map(string, pos0), ","), ")/0/$t")
            println(io, "(", join(map(string, posb), ","), ")/$blk/$t")
        end
    end
end

@rule :memdata "knn-tree$stage-$name" let xform = "interm/xform$stage-$name", ref_img, moving_imgs = "moving-imgs-$name"
    Random.seed!(1)
    xform, _ = deepcopy(xform) |> Flux.gpu
    # XXX
    SegReg.prepare_xform!(xform)
    ref_img = ref_img |> Flux.gpu
    moving_imgs = moving_imgs |> Flux.gpu
    nblk = length(moving_imgs)
    from_to = [[] for _ in 1:nblk]
    @showprogress for _ in 1:300
        idx = rand(keys(ref_img.dat), 10_000) |> Flux.gpu
        #idx=idx[ref_img.dat[idx].>.1f0]
        pos0i = stack([map(i -> i[k], idx) for k in 1:3], dims=1) |> Flux.gpu
        pos0 = SegReg.pix2position(pos0i, ref_img.res)
        pos1 = SegReg.execute_xform(xform, :, pos0)
        pos0 = pos0 |> Flux.cpu
        @assert nblk == size(pos1, 2)
        map(1:nblk) do B
            moving_img = moving_imgs[B]
            posb = pos1[:, B, :]
            posbi = SegReg.position2pix(posb, moving_img.res, moving_img.off) |> Flux.cpu
            @assert size(pos0, 2) == size(posb, 2)
            posb = posb |> Flux.cpu
            for j in 1:size(posb, 2)
                if checkbounds(Bool, moving_img.data, posbi[:, j]..., 1)
                    push!(from_to[B], (view(pos0, :, j)..., view(posb, :, j)...))
                end
            end
        end
    end
    [stack(unique(from_to[B]), dims=2) |> Flux.cpu for B in 1:nblk]
end



@rule out = "data/final/warped-$name.ch$ch.swc.gz" let xform = "interm/xform3-$name", ref_pts = "ref-pts-$name"
    @deps knn = "knn-tree3-$name"
    @info "....."
    @deps swcs = "swc-chaninfo-$name-ch$ch"
    xform, _ = deepcopy(xform)
    ch = parse(Int, ch)
    ref_pts = filter(p -> get(p, :ch, nothing) == ch, ref_pts)
    @info "Join $swcs.....................-- $(length(ref_pts))"
    xform_swc(out, knn, xform, ref_pts, swcs...; prec=0.002)
end

@rule out = "data/temp/xformed-$name-part$blk.ch$ch.catalog" let xform = "interm/xform2-$name", infos = "moving-imgs-info-$name"
    function printarr(vec)
        join(map(v -> round(v, digits=5), vec), " ")
    end
    xform, _ = deepcopy(xform) |> Flux.gpu
    SegReg.prepare_xform!(xform)
    xform = xform |> Flux.cpu
    for (b, info) in enumerate(infos)
        for (c, cat) in enumerate(info.cats)
            ("$b" == blk && "$c" == ch) || continue
            @assert isfile(cat)
            a = inv(xform.affine.cache_mats[][:, :, b])
            c = -inv(xform.affine.cache_mats[][:, :, b]) * xform.affine.cache_vecs[][:, b]
            ini = read(IniFile.Inifile(), cat)
            for (sect, _) in IniFile.sections(ini)
                IniFile.set(ini, sect, "origin", printarr(c))
                vv = map(v -> parse(Float64, v), split(IniFile.get(ini, sect, "direction", nothing), r"\s+"))
                vv = reshape(vv, 3, 3)
                if match(r"CH\d+_WEBM", sect) !== nothing
                    IniFile.set(ini, sect, "direction", printarr(a * vv))
                else
                    IniFile.set(ini, sect, "direction", printarr(a * vv))
                end
            end
            @info "$out"
            open("$out", "w+") do io
                IniFile.write(io, ini)
            end
        end
    end
end
###################################################
####################################################

function sample_objs(fn::Function, objs, perface=64)
    for obj in objs
        samps = map(eachcol(rand(Float32, 2, perface))) do (l1, l2)
            r1 = 1 - sqrt(l1)
            r2 = (1 - r1) * l2
            r3 = 1 - r1 - r2
            [r1, r2, r3]
        end
        @showprogress for f in obj
            verts = map(v -> [v.position.data...], f)
            for r123 in samps
                p = mapreduce(.*, .+, r123, verts)
                fn(p)
            end
        end
    end
end
function generate_bones(ref_img, objs, L)
    ref_img, ref_res = ref_img.dat[:, :, :, 3], ref_img.res
    N = 900_000
    xyz = rand(Float32, 3, N)
    dxyz = randn(Float32, 3, N)
    rng = size(ref_img) .* ref_res
    msk1 = similar(ref_img)
    msk2 = similar(ref_img)
    msk1 .= 0
    msk2 .= 0
    sample_objs(objs) do p
        p = round.(Int, p ./ ref_res)
        if checkbounds(Bool, msk1, p...)
            msk1[p...] = 1.0
        end
    end
    bones = @showprogress map(eachcol(vcat(xyz, dxyz))) do seed
        @assert length(seed) == 6
        xyz = seed[1:3] .* rng
        dxyz = seed[4:6]
        dxyz = dxyz .* (L / norm(dxyz))
        xyz2 = xyz .+ dxyz
        hits = 0
        inds = []
        for i in 0:0.03:1.000001
            p = xyz .* i .+ xyz2 .* (1 - i)
            p = round.(Int, p ./ ref_res)
            push!(inds, p)
            v = get(ref_img, (p...,), -1.0f0)
            if v >= 0.1
                hits += 1
            end
            get(msk1, (p...,), 1.0f0) > 0 && return missing
        end
        hits < 30 && return missing
        for p in inds
            msk2[p...] = 1.0f0
        end
        [xyz..., xyz2...]
    end
    reshape(reduce(hcat, skipmissing(bones)), 3, 2, :), msk1, msk2
end

@rule :memdata "objs-$part" let files = ["/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/$lr.$(part)_surface.rsl.gii.obj" for lr in ["lh", "rh"]]
    map(FileIO.load, files)
end

function gen_cortex(ref_img, objs_list...)
    ref_img, ref_res = ref_img.dat[:, :, :, 3], ref_img.res
    res = similar(ref_img)
    res .= ref_img * 0.4
    for (objs, val) in objs_list
        sample_objs(res, ref_res, objs, val)
    end
    res
end

#gen_cortex(ref_img, objs_mid=>0.8, objs_white=>1.0, objs_gray=>0.6), ref_img.res

@rule :memdata "sample-objs-$part" let objs = "objs-$part"
    function get_points(objs)
        pts = []
        sample_objs(objs, 96) do pt
            push!(pts, pt)
        end
        println("len ", length(pts))
        shuffle!(pts)
        reduce(hcat, pts[1:12_000_000])
    end
    get_points(objs)
end


@rule :memdata bones let ref_img
    @info "..."
    objs = [
        "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.gray_surface.rsl.gii.obj"
        "/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/rh.gray_surface.rsl.gii.obj"
    ]
    objs = map(FileIO.load, objs)
    function display_bones(bones, objs)
        p = GLMakie.linesegments(reshape(bones, 3, :), color=:red)
        for obj in objs
            GLMakie.mesh!(obj, color=:orange, alpha=0.3, transparency=true)
        end
        GLMakie.display(p)
    end
    bones, msk1, msk2 = generate_bones(ref_img, objs, 3000)
    display_bones(bones, objs)
    bones
end
#=


############################################################
#############################################################
############################################################
#import AxisArrays
#import CoordinateTransformations
#import Rotations

###############################
#cc
############################
####################



function xform_affine_inv(pars, x::AbstractArray)
    siz1, siz2p... = size(x)
    @assert siz1 == 3
    mat = inv(pars.affine_m |> Flux.cpu) |> gpudev
    reshape(mat * (reshape(x, siz1, :) .- reshape(pars.affine_v, :, 1)), siz1, siz2p...)
end

######################################
######################################
function pix2ref(i, x, scale=1.0)
    ((i[1]-.5)*x,
            (i[2]-.5)*x,
            (i[3]-.5)*x,)./scale
end
#=
function ref2pix(xyz, scale=1.0)
    (round(Int, xyz[1]/xres+.5),
     round(Int, xyz[2]/xres+.5),
     round(Int, xyz[3]/xres+.5),)
end
=#

function pix2block(i, scale=1.0)
    [(i[1]-.5)*65,
            (i[2]-.5)*65,
            (i[3]-.5)*60,]./(scale*scale_moving())
end
function block2pixf(xyz, scale=1.0)
    #65 0 0 0 65 0 0 0 60
    #x971 y717 z953
    #x802 y574 z138
    xyz=xyz.*(scale_moving()*scale)
    (xyz[1]/65.0+.5,
     xyz[2]/65.0+.5,
     xyz[3]/60.0+.5,)
end


function gen_output(xform, xform_pars, ref, parts...)
    @info "Gen output..."
    ref_s=2
    ref1=scale_image(ref[:, :, :, 1], ref_s)
    ref1=(ref1|>gpudev)[:, :, :, 1]
    println("ref1 $(size(ref1))")
    # WHD
    parts=map(x->x[:, :, :, 2], parts|>gpudev)
    # WHD
    posi=collect(keys(ref1)) |> gpudev
    posi=map(1:3) do i
        map(x->x[i], posi)
    end
    posi=stack(posi, dims=1)
    # 3WHD
    pos=pix2refv(posi, ref_s)
    # 3WHD
    xform_pars=xform_pars|>gpudev
    parts=map(enumerate(parts)) do (B, part)
        sz=axes(part)
        posj=block2pixfv(xform(xform_pars, B, pos))
        # 3WHD
        posjj=expandvox(posj, n=2, σ=.1)
        # 3dWHD
        println("$(typeof(posjj))")
        println("$(size(posjj))")
        li=LinearIndices(part)
        posjj=map(posjj[1, :, :, :, :], posjj[2, :, :, :, :], posjj[3, :, :, :, :]) do i,j,k
            ijk=round.(Int, (i, j, k))
            all(in.(ijk, sz)) ? li[ijk...] : 0
        end
        println("$(typeof(posjj))")
        println("$(size(posjj))")
        vvv=map(posjj) do ijk
            Float32(ijk>0 ?  part[ijk] : 0.0)
        end
        #vvv=similar(posjj, Float32)
        #vvv.=0.0
        #vvv[posjj.!=0].=part[posjj[posjj.!=0]]
        reshape(sum(vvv, dims=1), size(ref1)...)./size(posjj, 1)
    end
    println("$(typeof(parts[1]))")
    println("$(size(parts[1]))")
    out=map(ref1, parts[1], parts[2]) do b,r,g
        UInt16(clamp(round(Int, (r+g)*65535.0), 0, 65500))
        #RGB{Float32}(r,g,b)
    end
    out|>Flux.cpu
end

############################################################
##############################################
##############################################
#################################################################


#################################


        #=
        ################################################
        ##############################################
        ############################################
        #
function gen_output(xform, xform_pars, ref, parts...)
        # 3WHD
        posjj=expandvox(posj, n=2, σ=.1)
        # 3dWHD
        println("$(typeof(posjj))")
        println("$(size(posjj))")
        li=LinearIndices(part)
        posjj=map(posjj[1, :, :, :, :], posjj[2, :, :, :, :], posjj[3, :, :, :, :]) do i,j,k
            ijk=round.(Int, (i, j, k))
            all(in.(ijk, sz)) ? li[ijk...] : 0
        end
        println("$(typeof(posjj))")
        println("$(size(posjj))")
        vvv=map(posjj) do ijk
            Float32(ijk>0 ?  part[ijk] : 0.0)
        end
        #vvv=similar(posjj, Float32)
        #vvv.=0.0
        #vvv[posjj.!=0].=part[posjj[posjj.!=0]]
        reshape(sum(vvv, dims=1), size(ref1)...)./size(posjj, 1)
    end
    println("$(typeof(parts[1]))")
    println("$(size(parts[1]))")
    out=map(ref1, parts[1], parts[2]) do b,r,g
        UInt16(clamp(round(Int, (r+g)*65535.0), 0, 65500))
        #RGB{Float32}(r,g,b)
    end
    out|>Flux.cpu
end
=#


    ######################################
    ######################################
    #######################################

    #############################
    # 
    #
    ####################################
    #####################################
##########################################


##########################################
##########################################
########


#affine
#  affine mat+shift
#  latern-pts
#  image
#
#  +img
#  +img
#=
            w*sum(map(x->(abs(x)>1 ? abs(x) : x^2), (t2c[pt.type] .- pts[:, j]) ./ 50))
                sum((aaa .- bbbb).^2 )




    pars=xform_pars=[matd, shiftd]
    (mat.*matd, shift.*(1.0f0 .+shiftd))
            =#
#
# 
#


function optimize_affine(ts=nothing)
    matd=similar(mat)
    matd.=1.0
    shiftd=similar(shift)
    shiftd.=0.0
    points=points|>gpudev
    function xform(pt, B)
        B==0 && return pt
        xform_pars[1][:, :, B]*pt.+xform_pars[2][:, B]
    end
    ref_s, blk_s=1.0, 0.5

    ref1=ref[:, :, :, 1] |> gpudev
    bbb=similar(ref1) |> gpudev
    bbb.=0.0

    pttyp=map(x->x.type, points)
    types=pttyp |> unique
    ptblk=map(v->v.block, points)
    blocks=unique(ptblk)
    pttgt=reduce(hcat,map(x->x.pos, points)) |> gpudev

    ##############################################################################
    #########################################################################
    #################################################################################
end
    ##############################################
    ##########################################
    #######################
    #img R M...
    #PS R M...
    #XF M...
    #####################
    #####################


    function get_seg_for_train(xform, xform_pars, ref, part)
        @info "Gen block seg..."
        ref=ref|>gpudev
        part=part|>gpudev
        xform_pars=xform_pars|>gpudev
        rng=axes(part.dat)[1:3]
        msk=similar(ref.dat, Bool, rng)
        msk.=false
        res=map(axes(ref.dat)[4]) do ch
            x=similar(ref.dat, Float32, rng)
            x.=0
            x
        end
        for K in 1:16
            idx0=collect(CartesianIndices(axes(ref.dat)[1:3])) |> gpudev
            idx0=repeat(idx0, inner=(4, 1, 1))
            idx=stack([map(i->Tuple(i)[j], idx0) for j in 1:3], dims=1)
            idx=idx+(rand(Float64, size(idx)) |>gpudev) .-.5
            pos=pix2refv(idx, ref.res)
            pos=block2pixfv(xform(xform_pars, pos), part.res)
            dumbidx=CartesianIndex(0,0,0)
            idx1=map([pos[j, :, :, :] for j in 1:3]...) do i,j,k
                i,j,k=round.(Int, (i,j,k))
                all(in.((i,j,k), rng)) ? CartesianIndex(i,j,k) : dumbidx
            end
            sel=map(i->i!=dumbidx, idx1)
            idx0=idx0[sel]
            idx1=idx1[sel]

            msk[idx1].=true
            for ch in axes(ref.dat)[4]
                res[ch][idx1].=ref.dat[:, :, :, ch][idx0]
            end
        end
        return (stack(res, dims=4), msk)|>Flux.cpu
    end


    function apply_segment(unet, img, msk)
        unet=unet|>gpudev
        img=img|>gpudev
        out=similar(img, size(img)[1:end-1]..., 3)
        out.=0
        samp_pars=collect(unet_samp_pars(img; stepz=1, stepxy=96))
        siz=size(img)[1:3]

        function evaluate(acc)
            samp2=stack(map(x->x[2], acc), dims=4)
            x=unet(samp2)
            for (k, (idx, samp2, xyrng)) in enumerate(acc)
                sum(x[:, :, 2, k])<(256*256/20) && continue
                out[idx..., :].=x[xyrng..., :, k]
            end
        end

        acc=[]
        @showprogress for (ax, i, j, k) in samp_pars
            ax!=3 && continue
            idx=samp_pars_to_idx(siz, (ax, i, j, k))
            samp=img[idx..., :]
            samp2=clamp_to_size(samp, (256,256))
            #@info "$ax $i $j $k"
            sum(msk[idx...])<(256*256/20) && continue
            push!(acc, (idx, samp2, axes(samp)[1:2]))
            if length(acc)>=8
                evaluate(acc)
                empty!(acc)
            end
        end
    if !isempty(acc)
        evaluate(acc)
        empty!(acc)
    end
    out |> Flux.cpu
    end

function get_unet(chin, chout, dims=3)
    try
        return @something(Main.__unet)
    catch e
        !(e isa Union{UndefVarError,ArgumentError}) && rethrow(e)
    end
    tuple(i)=(fill(i, dims)..., )
    conv7(ci, co) = Flux.Conv(tuple(7), ci => co, pad=3, stride=2, bias=false)
    conv1(ci, co) = Flux.Conv(tuple(1), ci => co, bias=false)
    conv1relu(ci, co) = Flux.Conv(tuple(1), ci => co, Flux.relu, bias=false)
    function conv3(ci, co)
        conv=Flux.Conv(tuple(3), ci => co, pad=1, bias=false)
        return conv
        function(x...)
            println("conv3 $ci $co");
            conv(x...)
        end
    end
    conv3relu(ci, co) = Flux.Conv(tuple(3), ci => co, Flux.relu, pad=1, bias=false)
    bn(c) = Flux.BatchNorm(c)
    bnrelu(c) = Flux.BatchNorm(c, Flux.relu)
    blk3(ci, co) = Flux.Chain(conv3(ci, co), bnrelu(co))
    unet_combine(ci, co) = Flux.Chain(blk3(ci, co), blk3(co, co))
    function cat_channels(xs...)
        #=
        for x in xs
            println(" cat $(size(x))")
        end
        =#
        cat(xs...; dims = Val(dims+1))
    end

    function resblk(ci, co, cb)
        Flux.SkipConnection(
            Flux.Chain(
                bn(ci),
                conv1relu(ci, co),
                bn(co),
                conv3relu(co, cb),
            ),
        cat_channels,
        )
    end

    function pixelshuffle(ci, co, r=2)
        cx = co * (r^dims)
        Flux.Chain(
            Flux.Chain([
                conv1(ci, cx),
                bnrelu(cx),
            ]),
            Flux.PixelShuffle(r),
        )
    end

    function final()
        Flux.Chain(
            Flux.Chain(
                conv3(chin, chin),
                bn(chin),
                Flux.relu,
                conv3(chin, chin),
                bn(chin),
            ),
            Flux.Chain(
                conv1(chin, chout),
                bnrelu(chout),
            ),
        )
    end

    function upscale(ci, ca, cb, co, blks, inner)
        Flux.Chain(
            blks...,
            Flux.SkipConnection(
                Flux.Chain(
                    inner(ci, ca),
                    pixelshuffle(ca, cb),
                ),
                Flux.Parallel(
                              cat_channels,
                    identity,
                    bn(ci),
                ),
            ),
            Flux.relu,
            unet_combine(ci + cb, co),
        )
    end

    configs = [
        (pool=Flux.MaxPool(tuple(3), pad=1, stride=2),),
        (pool=Flux.MeanPool(tuple(2)),),
        (pool=Flux.MeanPool(tuple(2)),),
        (pool=Flux.MeanPool(tuple(2)),),
    ]

    function inner2f(i, c128, c32)
        if i >= length(configs)
            return function (ci, co)
                blks = [resblk(cc, c128, c32) for cc in ci:c32:(co-1)]
                Flux.Chain(
                    configs[i].pool,
                    Flux.Chain(
                        blks...,
                        bnrelu(co),
                        identity,
                    ),
                )
            end
        end
        function (ci, co)
            @info "$i $ci $co"
            blks = Any[resblk(cc, c128, c32) for cc in ci:c32:(co*2-1)]
            push!(blks, bn(co * 2), conv1relu(co * 2, co))
            inner = inner2f(i + 1, c128, c32)
            Flux.Chain(
                configs[i].pool,
                upscale(co, co * 2, co * 2, co, blks, inner),
            )
        end
    end

    function prepare(ci, co)
        Flux.Chain(
            conv7(ci, co),
            upscale(co, co * 2, co * 2, co, [bnrelu(co)],
                inner2f(1, 128, 32),
            ),
        )
    end

    net=Flux.Chain(upscale(chin, 64, 64, chin, [], prepare),
        final(),
    )
    #=
    open(pipeline(`sed 's,(,{\n,g;s,),;\n},g;s/, /\;\n/g;s,\[,{\n,g;s,\],},g;'`, `indent`, "/tmp/unet.txt"), "w") do io
        println(io, x)
    end
    =#
    Main.__unet=net|>gpudev
end

function main()



    #@time out1=gen_output(affine_xform, xform1, ref, parts...)
    #FileIO.save("/tmp/test-1.nrrd", out1 |> Flux.cpu)


###################################################################
###################################################################
################################################################
    x=get_unet(4, 3) |> gpudev
    y=randn(Float32, 64, 64, 64, 4, 1) |> gpudev
    y=x(y)
    return y




    ############################################################
    ################################################
    return get_seg_for_train(affine_xform, xform2, ref, 1, parts[1])
    #return xform2
####################################
################################
#####################

    #ImageTransformations.warp(
end

#=

    #types=map(x->x.type, pts)
    #types=unique(filter(t->length(filter(x->x.type==t, pts))>1, types))
    #x=[filter(x->x.type==t && x.block==b, pts)[1].pos for t in types, b in [0, 2]]
    #return hcat(x[:, 1]...), hcat(x[:, 2]...)
    #
    #
    #
    #
    #


=#
    #=
    x=fill(([1.0,1,1], [3.0,3,3]),10000) |> gpudev
    xx = Flux.Params(x)
    z=Float32[3.0,3.0,3.0]|>gpudev
    function www()
        x2=map(x) do y
            t=CoordinateTransformations.Translation(y[1])(z)
            sum(t)
        end
        sum(x2)
    end
    www()
    for i=1:100
    Flux.withgradient(xx) do
        www()
    end
end
=#


################################################################################





@rule :memdata "unetdata-$dataid" let xform1="xform1-$dataid", ref_img, moving_imgs="moving-imgs-$dataid"
    xform_pars, _= deepcopy(xform1) |>gpudev
    @info "get data ..."
    map(1:length(moving_imgs)) do B
        get_seg_for_train(xform_affine, xform_pars[B], ref_img, moving_imgs[B])
    end
end

@rule :memdata "unet-$dataid" let unetdata="unetdata-$dataid", moving_imgs="moving-imgs-$dataid"
    #return unetdata|>Flux.cpu
    ch1=size(moving_imgs[1].dat, 4)
    ch2=size(unetdata[1][1], 4)
    @info "$ch1 => $ch2 .."

    unet=train_segment(unet, unetdata, moving_imgs; niter=10000)
end

@rule :memdata "seg-$dataid" let unet="unet-$dataid", unetdata="unetdata-$dataid", moving_imgs="moving-imgs-$dataid"
    @info "oooooo"
    map(enumerate(moving_imgs)) do (B, img)
        (dat=apply_segment(unet, img.dat, unetdata[B][2]), res=img.res)
    end
end





################################################################################
###############################################################################
###############################################################################
###############################################################################
############################################################
#######################################################
##############################################
############################################
#########################################


#=
@rule "sed.catalog.sh" let xform2
    gen_cat(xform2)
end

@rule "/tmp/test-2.nrrd" let xform2, ref_img, moving_imgs
    @time out1=gen_output(affine_xform, xform2, ref_img, moving_imgs...)
    FileIO.save("/tmp/test-2.nrrd", out1 |> Flux.cpu)
end

=#


###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################
#############################################################################################################
#############################################################################################################

Jmake.@makefile "/tmp/testjmake/makefile"
#@make #all_swcs ref_pts
=#

############################
function xform_points_ssh(nm, pts)
    newxyz = []
    open(`ssh -C goulf-ts "LD_LIBRARY_PATH=/tmp /tmp/ld-linux-x86-64.so.2 /tmp/join --swc" "$nm"`, write=true, read=true) do io
        function write_dat()
            write(io, "x,y,z,label\n")
            for (x, y, z) in pts
                write(io, "$x,$y,$z,0\n")
            end
            close(io.in)
        end
        tsk = schedule(Task(write_dat))
        readline(io)
        for l in eachline(io)
            f = split(l, r",")
            t = (map(v -> parse(Float64, v), f[1:3])...,)
            push!(newxyz, t)
        end
        wait(tsk)
    end
    @assert length(pts) == length(newxyz)
    newxyz
end

function parse_swc_line(l)
    f = split(l)
    id = parse(Int64, f[1])
    t = parse(Int64, f[2])
    pos = (map(v -> parse(Float64, v), f[3:5])...,)
    r = parse(Float64, f[6])
    par = parse(Int64, f[7])
    (id=id, typ=t, pos=pos, r=r, par=par)
end
function parse_gapr_annot(l)
    m = match(r"^#Gapr!(\d+)@(\w+)=(.*)$", l)
    m !== nothing || return nothing
    (id=parse(Int64, m.captures[1]), key=m.captures[2], val=m.captures[3])
end
function format_swc_line(id, typ, pos, r, par)
    (x, y, z) = pos
    "$id $typ $x $y $z $r $par\n"
end
function format_gapr_annot(id, key, val)
    "#Gapr!$id@$key=$val\n"
end


@rule out = "data/final/alt-warped-fmost221059.ch1.swc.gz" let ref_pts = "ref-pts-fmost221059"
    xform_par = ["-/home/goulf/test-out-2-left-xform",
        "-/home/goulf/test-out-2-right-xform"]
    fn_pairs = "/home/goulf/Desktop/gapr/nodist/stitch/point-set-manual.txt"
    id_inc = 1_000_000_000
    ref_pts = filter(v -> get(v, :node, 0) != 0 && v.ch == 1, deepcopy(ref_pts))

    @deps swcs = "swc-chaninfo-fmost221059-ch1"
    @info "warp swc......"

    function get_manual_pairs(pts)
        lbl2id = Dict()
        for pt in pts
            id, lbl, blk = pt[[:node, :type, :block]]
            @assert id != 0
            @assert blk > 0
            id += id_inc * (blk - 1)
            lbl2id[lbl] = (get(lbl2id, lbl, ())..., id)
        end
        map(values(lbl2id)) do p
            @assert length(p) == 2
            (p...,)
        end
    end
    nn_pairs = get_manual_pairs(ref_pts)

    mm_store = Vector{Tuple{Int64,Int64}}()
    mm_dict = Dict{Int64,Int64}()
    function add_adj(a, b)
        prev = get(mm_dict, a, 0)
        push!(mm_store, (prev, b))
        mm_dict[a] = length(mm_store)
    end
    function rm_adj(a, b)
        ii = mm_dict[a]
        while ii != 0
            (pp, nn) = mm_store[ii]
            if nn == b
                mm_store[ii] = (pp, -1)
            end
            ii = pp
        end
    end

    id2name = []
    id2node = Dict{Int64,Any}()
    vpos = []
    idadd = 0
    function handle_line(l)
        if l[1] == '#'
            ann = parse_gapr_annot(l)
            if ann !== nothing && ann.key == "root"
                push!(id2name, (ann.id + idadd, ann.val))
            end
            return
        end
        (id, t, pos, r, par) = parse_swc_line(l)
        id += idadd
        if par != -1
            par += idadd
        end
        push!(vpos, pos)
        prev = get!(id2node, id, (length(vpos), t, r))
        @assert prev[1] == length(vpos)
        if par != -1
            add_adj(id, par)
            add_adj(par, id)
        end
    end
    for (blk, fn) in enumerate(swcs)
        prev_len = length(vpos)
        open(`gzip -cd $fn`, "r") do io
            for l in eachline(io)
                length(l) > 0 || continue
                handle_line(l)
            end
        end
        idadd += id_inc
        new_pos = xform_points_ssh(xform_par[blk], vpos[(prev_len+1):end])
        vpos[(prev_len+1):end] .= new_pos
    end
    for (id, par) in nn_pairs
        add_adj(id, par)
        add_adj(par, id)
    end

    open(pipeline(`gzip -c -`, out), "w") do ooo
        for (idx, ppp) in pairs(id2name)
            curset = Set()
            todo = []
            handle_node = function (io, id, newpar; forcet=0)
                (idx, t, r) = id2node[id]
                id2node[id] = (0, 0, 0)
                if idx == 0
                    if newpar != -1
                        println(io, "#Gapr!$id/$newpar")
                    end
                    return
                end
                @assert idx != 0
                (x, y, z) = vpos[idx]
                if forcet == 2
                    t = forcet
                end
                write(io, format_swc_line(id, t, (x, y, z), r, newpar))
                #if rand()<0.01
                #annot=get_nmt_hier((x, y, z))
                write(io, format_gapr_annot(id, "prev.id", "$id"))
                #end
                push!(curset, id)
                ii = get(mm_dict, id, 0)
                mm_dict[id] = 0
                while ii != 0
                    (pp, nn) = mm_store[ii]
                    if nn != -1
                        push!(todo, (id, nn, t))
                        rm_adj(nn, id)
                    end
                    ii = pp
                end
            end
            qroot = match(r"^qroot", ppp[2]) !== nothing
            handle_node(ooo, ppp[1], -1, forcet=(qroot ? 2 : 0))
            while length(todo) > 0
                (par, id, part) = pop!(todo)
                handle_node(ooo, id, par, forcet=part)
            end
            write(ooo, format_gapr_annot(ppp[1], "root", ppp[2]))
        end
    end
end

