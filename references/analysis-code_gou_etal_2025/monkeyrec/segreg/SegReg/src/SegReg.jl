########################################################################
module SegReg

using Base: mapreduce_first
using StaticArrays
using ProgressMeter
import Flux
import CUDA
import Metalhead
import GLMakie
import LinearAlgebra
import Statistics

using Random

macro check_size(s::Symbol, arr, siz)
    s!==:Bool && throw(ArgumentError("Bool?"))
    quote
        tgt = $(esc(siz))
        s = size($(esc(arr)))
        length(tgt) != length(s) && return false
        for i in 1:length(tgt)
            if tgt[i] != Colon() && tgt[i] != s[i]
                return false
            end
        end
        true
    end
end

macro check_size(arr, siz)
    expr = "size($arr) == $siz"
    quote
        tgt = $(esc(siz))
        s = size($(esc(arr)))
        msg = "failed size check: $($expr): $s"
        length(tgt) != length(s) && error(msg)
        for i in 1:length(tgt)
            if tgt[i] != Colon() && tgt[i] != s[i]
                error(msg)
            end
        end
    end
end


include("affine.jl")
include("warp-spline.jl")
include("warp-field.jl")
include("segment.jl")

#=NOTE:
#* fixed==UNet(morph(moving))
#! UNet(fixed) makes no sense.   UNet(moving) gives a segmenter.
#! fixed==morph(UNet(moving)), impossible to call UNet with trivial slices.
#! morph(fixed)==UNet(moving), impossible to call UNet across moving blocks.
#* moving_xyz = fixed_xyz |> warp |> affine.
#    In this order, could combine moving blocks with affine transformations.
#* A single UNet.
#* One affine per moving block.
*#XXX: Maybe a single warp is enough?

UNET **before sampling
=#

#=NOTE: for series of sections
#* XXX: per-section affine/warp?
=#
# XXX: (generic rrule for: bspline,tetra,persection affine)

###################################################
###################################################
const unet_siz = 256

struct Volume{T,C}
    data::AbstractArray{T,4}
    resolution::SVector{3,Float32}
    offset::SVector{3,Float32}
    chan::Val{C}
end
function Volume(tup::NamedTuple)
    data = @something(get(tup, :data, nothing), get(tup, :dat, nothing))
    res = @something(get(tup, :resolution, nothing), get(tup, :res, nothing))
    off = @something(get(tup, :offset, nothing), get(tup, :off, nothing), SVector(0.0f0, 0.0f0, 0.0f0))
    T = eltype(data)
    C = size(data)[end]
    Volume{T,C}(data, res, off, Val(C))
end
Flux.@layer Volume

struct Landmark
    ##node::Int64
    pos::SVector{3,Float32}
    block::Int32
    group::Int32
    ##channel::Int32
end
function Landmark(tup::NamedTuple)
    pos = @something(get(tup, :position, nothing), get(tup, :pos, nothing))
    blk = @something(get(tup, :block, nothing), get(tup, :blk, nothing))
    grp = @something(get(tup, :group, nothing), get(tup, :grp, nothing), get(tup, :type, nothing), get(tup, :typ, nothing))
    Landmark(pos, blk, grp)
end
function Base.convert(::Type{Landmark}, tup::NamedTuple)
    Landmark(tup)
end

struct Problem{Tf,Cf,Nm,Tm,Cm}
    fixed::Volume{Tf,Cf}
    moving::NTuple{Nm,Volume{Tm,Cm}}
    landmarks::AbstractVector{Landmark}
    # priv
    group_pos::AbstractDict{Int32,SVector{3,Float32}}
end

function Problem(fixed, moving, landmarks, group_pos=nothing)
    fixed = Volume(fixed)
    moving = ((Volume(b) for b in moving)...,)
    landmarks = collect(Landmark, landmarks)
    group_pos = @something(group_pos, Dict{Int32,SVector{3,Float32}}())
    Tf, Cf = typeof(fixed).parameters
    #Tf,k=eltype(fixed.data)
    #Cf=size(fixed.data, 4)
    Nm = length(moving)
    Tm, Cm = typeof(first(moving)).parameters
    #Tm=eltype(first(moving).data)
    #Cm=size(first(moving).data, 4)
    Problem{Tf,Cf,Nm,Tm,Cm}(fixed, moving, landmarks, group_pos)
end
function Base.show(io::IO, prob::Problem)
    Tf, Cf, Nm, Tm, Cm = typeof(prob).parameters
    show_vol(io, b) = println(io, "                $(size(b.data)[1:end-1]) $(b.resolution)")
    println(io, "Fixed : $Tf $Cf")
    show_vol(io, prob.fixed)
    println(io, "Moving: $Tm $Cm x$Nm")
    for b in prob.moving
        show_vol(io, b)
    end
    println(io, "Landmarks: x$(length(prob.landmarks))")
end
#=

function RegisterEnv(ref_img, moving_imgs, points, ref_rat, moving_rat)
    function scale_image(image, ratio)
        dat=image.dat|>Flux.cpu
        @assert length(size(dat))==4
        dat=map(1:size(dat, 4)) do k
            ImageTransformations.imresize(dat[:, :, :, k], ratio=ratio)
        end
        (dat=stack(dat, dims=4), res=image.res./ratio)
    end
    if ref_rat!=1
        ref_img=scale_image(ref_img, ref_rat)
    end
    if moving_rat!=1
        moving_imgs=map(img->scale_image(img, moving_rat), moving_imgs)
    end
    RegisterEnv(ref_img, moving_imgs, points)
end

=#

################################################
################################################
################################################
################################################
################################################
##########################
##########################
#########################################################
#########################################################
#########################################################
###############################################
struct UNet{Cf,Cm}
    unet::Flux.Chain
    chan_fixed::Val{Cf}
    chan_moving::Val{Cm}
end
function UNet{Cf,Cm}(::UndefInitializer) where {Cf,Cm}
    UNet{Cf,Cm}(Flux.Chain(), Val(Cf), Val(Cm))
end
function UNet{Cf,Cm}() where {Cf,Cm}
    backb = Metalhead.resnet(Metalhead.basicblock, [3, 5, 7]; inchannels=Cm, activation=Flux.leakyrelu)
    unet = Metalhead.UNet((unet_siz, unet_siz), Cm, 4, backb[1])
    unet = Flux.Chain(unet, Flux.Conv((1, 1), 4 => Cf, Flux.relu), Flux.tanh_fast)
    UNet{Cf,Cm}(unet, Val(Cf), Val(Cm))
end
function UNet{Cf,Cm}(::Missing) where {Cf,Cm}
    unet = Flux.Chain(
        Flux.Conv((3, 3), Cf => 4, pad=1, bias=false),
        Flux.BatchNorm(4),
        Flux.NNlib.relu,
        Flux.Conv((3, 3), 4 => Cm, pad=1, bias=true),
        Flux.sigmoid_fast,
    )
    UNet{Cf,Cm}(unet, Val(Cf), Val(Cm))
end
Flux.@layer UNet

function Base.show(io::IO, u::UNet{Cf,Cm}) where {Cf,Cm}
    print(io, "UNet(", Cm, " => ", Cf, ")")
end

function load_pretrain!(f::Function, unet::UNet{Cf,Cm}) where {Cf,Cm}
    tag = "segreg-unet-$Cm"
    state = try
        f(tag)
    catch e
        @error "failed to load $e"
        nothing
    end
    if state !== nothing
        @info "load model state $tag"
        Flux.loadmodel!(unet.unet[1], state)
    end
end

function save_pretrain(f::Function, unet::UNet{Cf,Cm}) where {Cf,Cm}
    state = Flux.state(unet.unet[1])
    tag = "segreg-unet-$Cm"
    try
        f(state, tag)
    catch
    end
end




struct Slicer
    zvals::AbstractVector{Float32}
    aff_mats::AbstractArray{Float32,3}
    aff_vecs::AbstractArray{Float32,2}
end
function Slicer(::UndefInitializer)
    Slicer(zeros(Float32, 0), zeros(Float32, 2, 2, 0), zeros(Float32, 2, 0))
end
function Slicer(infos)
    zvals = map(infos) do i::NamedTuple
        @something(get(i, :z, nothing), get(i, :zval, nothing))
    end
    m = stack([LinearAlgebra.I(2) * 1.0f0 for _ in axes(zvals, 1)], dims=3)
    v = zeros(Float32, 2, length(zvals))
    Slicer(zvals, m, v)
end
Flux.@layer Slicer trainable = (aff_mats, aff_vecs)

struct Network{Nm,Cf,Cm,Tw}
    warp::Tw
    affine::Affine{Nm}
    unet::UNet{Cf,Cm}
    slicers::NTuple{Nm,Slicer}
end

function load_pretrain!(f::Function, net::Network)
    if !isempty(net.unet.unet)
        load_pretrain!(f, net.unet)
    end
end
function save_pretrain(f::Function, net::Network)
    if !isempty(net.unet.unet)
        save_pretrain(f, net.unet)
    end
end

function Network(prob::Problem{Tf,Cf,Nm,Tm,Cm}; kwargs...) where {Tf,Cf,Nm,Tm,Cm}
    aff = Affine{Nm}(; kwargs...)
    unet = UNet{Cf,Cm}(undef)
    slicers = ((Slicer(undef) for _ in 1:Nm)...,)
    Network{Nm,Cf,Cm,Nothing}(nothing, aff, unet, (slicers...,))
end

function Network(net::Network{Nm,Cf,Cm}; unet=:keep, warp=:keep, slicers=:keep, warpsize=nothing) where {Nm,Cf,Cm}
    aff = net.affine
    unet = if unet === :keep
        net.unet
    elseif unet === :new
        UNet{Cf,Cm}()
    elseif unet === :delete
        UNet{Cf,Cm}(undef)
    else
        @assert false
    end
    slicers = if slicers === :keep
        net.slicers
    elseif slicers === :delete
        ((Slicer(undef) for _ in 1:Nm)...,)
    else
        map(Slicer, slicers)
    end
    warp = if warp === :keep
        net.warp
    elseif warp === :newbspline0
        nothing
        #WarpSpline{Nm}(grid=(8, 10, 6))
    elseif warp === :newbspline
        WarpSpline{Nm}(grid=(10, 12, 8))
        #WarpSpline{Nm}(grid=(12, 16, 10))
    elseif warp === :newfield
        WarpField{Nm}(warpsize)
    elseif unet === :delete
        nothing
    else
        @assert false "wrong warp arg: $warp"
    end
    if net.warp !== nothing && warp !== net.warp
        initialize!(warp, net.warp)
    end
    Tw = typeof(warp)
    Network{Nm,Cf,Cm,Tw}(warp, aff, unet, slicers)
end
Flux.@layer :expand Network

struct Transform{Cf,Nm,Cm}
    # priv
end
struct TransformExt
    xform::Transform
end
function Transform(prob::Problem, tup::NamedTuple)
    #_,Cf,Nm,_,Cm=typeof(prob).parameters
    #_,Cf,Nm,_,Cm=typeof(prob).parameters
    Transform{Cf,Nm,Cm}((;), tup)
end
function Transform(base::Transform{Cf,Nm,Cm}, tup::NamedTuple) where {Cf,Nm,Cm}
    Transform{Cf,Nm,Cm}(base, tup)
end
function Transform{Cf,Nm,Cm}(base::NamedTuple, tup::NamedTuple) where {Cf,Nm,Cm}
    tup = (; base..., tup...,)
    function simple_init(sym, args...)
        dat = get(tup, sym, nothing)
        @assert dat !== nothing
        if dat isa Function
            dat = dat(args...)
        end
        dat
    end
    rotate = simple_init(:rotate, Float32, 3, Nm)
    scale = simple_init(:scale, Float32, 3, Nm)
    translate = simple_init(:translate, Float32, 3, Nm)
    Transform{Cf,Nm,Cm}(rotate, scale, translate)
end

function get_slice_space(ref_img)
    @assert length(size(ref_img.data)) == 4
    siz = size(ref_img.data)
    map(1:3) do ax
        @assert siz[ax] >= unet_siz
        slices = Set()
        ax2, ax3 = setdiff(1:3, [ax])
        idx = Any[:, :, :]
        for i in 1:1:siz[ax]
            idx[ax] = i
            for j in 1:32:(siz[ax2]+1-unet_siz)
                for k in 1:32:(siz[ax3]+1-unet_siz)
                    idx[ax2] = j:(j+unet_siz-1)
                    idx[ax3] = k:(k+unet_siz-1)
                    @assert checkbounds(Bool, ref_img.data, idx..., 1)
                    push!(slices, (idx...,))
                end
            end
        end
        ax => slices
    end |> Dict
end


function prepare_xform!(net::Network)
    net.affine()
end

struct Network2{Nm,Cf,Cm,Tw,Tseg,Tcon}
    valcf::Val{Cf}
    valcm::Val{Cm}
    warp::Tw
    affine::Affine{Nm}
    unetseg::Tseg
    conv::Tcon
    slicers::NTuple{Nm,Slicer}
end
Flux.@layer :expand Network2

function Network2(prev::Network{Nm,Cf,Cm,Tw}) where {Nm,Cf,Cm,Tw}
    segunet = create_seg_unet(512, Cm => 3)
    Network2{Nm,Cf,Cm,Tw,typeof(segunet),UNet{Cf,Cm}}(Val(Cf), Val(Cm), prev.warp, prev.affine, segunet, UNet{Cf,Cm}(missing), prev.slicers)
end

function prepare_xform!(net::Network2)
    net.affine()
end


# XXX: jacobian : here or otherplace
function execute_xform(pars::Union{Network,Network2}, B::Union{AbstractRange,Colon}, x0::AbstractArray; retd=false)
    # XXX fixed to moving mapping
    #  ***Warp then Affine
    #  ***Affine 2blocks
    #  ***Warp 2blocks
    #  .. image-sample after affine
    #  super sample affine
    @assert eltype(x0) == Float32
    @assert length(size(x0)) >= 2
    siz1, siz2p... = size(x0)
    @assert siz1 == 3
    x0 = reshape(x0, siz1, :)
    dx = if pars.warp !== nothing
        pars.warp(x0, B)
    else
        Flux.ignore_derivatives(() -> reshape(fill!(similar(x0), 0), siz1, 1, :))
    end
    @assert eltype(dx) == Float32
    x = permutedims(pars.affine(permutedims(reshape(x0, siz1, 1, :) .+ dx, (1, 3, 2)), B), (1, 3, 2))
    @assert eltype(x) == Float32

    x = reshape(x, siz1, :, siz2p...)
    #@assert !any(isnan.(x))
    if retd
        x, reshape(dx, siz1, :, siz2p...)
    else
        x
    end
end
function execute_xform(pars::Union{Network,Network2}, B::Int, x::AbstractArray)::AbstractArray
    x = execute_xform(pars, B:B, x)
    siz1, siz2, siz3p... = size(x)
    @assert siz2 == 1 "$(size(x))"
    reshape(x, siz1, siz3p...)
end

function loss_valid_scales(aff::Affine)::Float32
    sum(Flux.relu(-log2.(0.5f0 .- abs.(aff.scale .- 1)) .- 3))
end

function points_spat_dist(pts1, pts2; w=nothing, sumw=nothing)
    @assert eltype(pts1) == Float32
    @assert eltype(pts2) == Float32
    #x=Flux.huber_loss(pts1llll
    if w === nothing
        x = reshape(sum(((pts1 .- pts2) ./ 1000) .^ 2, dims=1), :)
        sum(x) / length(x)
    else
        @assert eltype(w) == Float32
        x = reshape(sum(((pts1 .- pts2) ./ reshape(w, 1, :)) .^ 2, dims=1), :)
        sum(x) / length(x)
        #sum(x.*w)/(sum(w)+0.001f0)
    end
end

function loss_points_perblk(sel, ref_pts, moving_pts)
    points_spat_dist(view(ref_pts, :, sel), view(moving_pts, :, sel))
end

function loss_points(ref_pts, moving_pts, pointsblk, intragrp, intraweight, xform_pars, obsinfo)::Float32
    if isempty(ref_pts) && isempty(moving_pts)
        return 0.0f0
    end
    lgrp = points_spat_dist(view(ref_pts, :, intragrp.first), view(ref_pts, :, intragrp.second); w=intraweight)
    ref_pts2 = execute_xform(xform_pars, :, ref_pts)
    lblk = loss_points_perblk(pointsblk[0], ref_pts, moving_pts)
    for B in 1:size(ref_pts2, 2)
        lblk += loss_points_perblk(pointsblk[B], view(ref_pts2, :, B, :), moving_pts)
    end
    Flux.ignore_derivatives() do
        if obsinfo.k % 100 == 0
            obsinfo.pts[] = ref_pts |> Flux.cpu
        end
    end
    lblk * 10 + lgrp
end

function loss_bones(pars, bones)
    return 0.0f0 #XXX
    bones === nothing && return 0.0f0
    pts = Flux.ignore_derivatives() do
        permutedims(bones[:, :, rand(axes(bones, 3), 64 * 1024)], (1, 3, 2))
    end
    x = xfunc_warp3(pars.warp, :, pts, pars.warpfactor[])
    x = sum(((view(x, :, :, :, 1) .- view(x, :, :, :, 2)) ./ 700) .^ 2, dims=1)
    sum(x) / length(x)
end

function sample_valid_slices(slice_space, ref_img, N)
    samps = map(1:3) do ax
        slices = slice_space[ax]
        ret = []
        while length(ret) * 3 < N
            slice = rand(slices)
            if sum(view(ref_img.data, slice..., 2)) < (unet_siz * unet_siz / 30)
                delete!(slices, slice)
                continue
            end
            push!(ret, slice)
        end
        ret
    end
    samps = reduce(vcat, samps)
    resize!(shuffle!(samps), N)
end


function pix2position(i::AbstractArray, res)
    siz1, siz2p... = size(i)
    @assert siz1 == 3
    i = reshape(i, siz1, :)
    pos = Float32.(i .- 0.5f0) .* reshape(res, siz1, 1)
    @assert eltype(pos) == Float32
    reshape(pos, siz1, siz2p...)
end
function position2pix(x::AbstractArray, res, off=nothing)
    siz1, siz2p... = size(x)
    @assert siz1 == 3
    @assert eltype(x) == Float32
    x = reshape(x, siz1, :)
    if off !== nothing
        x = x .- reshape(off, 3, 1)
    end
    x = x ./ reshape(res, 3, 1) .+ 0.5f0
    reshape(x, siz1, siz2p...)
end

function prepare_images_batch(slices, ref_img)
    imgs1 = map(slices) do samp
        view(ref_img.data, samp..., :)
    end
    imgs1 = stack(imgs1, dims=4)

    inds = CartesianIndices(ref_img.data)
    pos1i = map(slices) do samp
        slicei = view(inds, samp..., 1)
        stack([map(i -> i[k], slicei) for k in 1:3], dims=1)
    end
    pos1i = stack(pos1i, dims=4) |> Flux.gpu
    pos1 = pix2position(pos1i, ref_img.resolution)
    @assert size(pos1) == (3, unet_siz, unet_siz, 4)
    @assert eltype(pos1) == Float32
    @check_size imgs1 (unet_siz, unet_siz, :, 4)
    @assert eltype(imgs1) == Float32
    if rand(1:2) == 1
        imgs1 = imgs1[end:-1:1, :, :, :]
        pos1 = pos1[:, end:-1:1, :, :]
    end
    if rand(1:2) == 1
        imgs1 = imgs1[:, end:-1:1, :, :]
        pos1 = pos1[:, :, end:-1:1, :]
    end
    if rand(1:2) == 1
        imgs1 = permutedims(imgs1, (2, 1, 3, 4))
        pos1 = permutedims(pos1, (1, 3, 2, 4))
    end
    imgs1, pos1
end

function to_float(x::AbstractArray{T}) where {T}
    if T === UInt16
        x ./ 65535.0f0
    elseif T === Float32
        x
    else
        @assert false
    end
end

function prepare_block_img_avg(pos2be, moving_img)
    #@assert !any(isnan.(pos2be))
    pos2i = position2pix(pos2be, moving_img.resolution, moving_img.offset)
    #@assert !any(isnan.(pos2i))
    #@assert size(pos2i)==(3, 8, unet_siz*unet_siz*4)
    linds = LinearIndices(view(moving_img.data, :, :, :, 1))
    imgdata = moving_img.data
    vals1 = map(view(pos2i, 1, :, :), view(pos2i, 2, :, :), view(pos2i, 3, :, :)) do i, j, k
        i, j, k = round.(Int, (i, j, k))
        checkbounds(Bool, imgdata, i, j, k, 1) ? linds[i, j, k] : 0
    end
    mask = map(v -> (v > 0 ? 1.0f0 : 0.0f0), vals1)
    #println("extream $(extrema(vals1))")
    vals1 = stack([view(moving_img.data, :, :, :, k)[map(v -> max(v, 1), vals1)] for k in 1:size(moving_img.data, 4)], dims=3) |> to_float
    #@assert size(vals1)==(8, unet_siz*unet_siz*4, 4)
    mask, vals1, pos2be
end

function prepare_block_img_avg_slicer(slicer::Slicer, pos2b, pos2be, moving_img)
    @check_size pos2b (3, :)
    @check_size pos2be (3, :, :)
    zvals = slicer.zvals
    nslices = length(zvals)
    zidxes = map(view(pos2b, 3, :)) do z
        i = 0
        while i < nslices
            z < zvals[i+1] && break
            i += 1
        end
        i
    end |> Flux.gpu
    #@info "$(size(zidxes))"
    # derivative on pos2b


    # XXX: redo following
    pos2i = position2pix(pos2be, moving_img.resolution, moving_img.offset)
    linds = LinearIndices(view(moving_img.data, :, :, :, 1))
    imgdata = moving_img.data
    ######################
    ###########################
    #################
    ########## pos2(z/find enclosing slices)
    #inslice expand
    #read voxels
    #calc weights
    #

    function func(i, j, k)::Int
        #vals1 = map(view(pos2i, 1, :, :), view(pos2i, 2, :, :), reshape(Float32.(zidxes), 1, :)) do i, j, k
        (k == 0 || k == nslices) && (k = 0)
        i, j, k = round.(Int, (i, j, k))
        checkbounds(Bool, imgdata, i, j, k, 1) ? linds[i, j, k] : 0
    end
    et = Base.Broadcast.eltypes((pos2i[1, :, :], pos2i[2, :, :], reshape(zidxes, 1, :)))
    @info "$(et)"
    et = Base._return_type(func, et)
    @info "$(et)"
    @info "$(et)"
    vals1 = broadcast(func, pos2i[1, :, :], pos2i[2, :, :], reshape(zidxes, 1, :))
    @info "sizzzzzzzzz $(size(vals1))"
    mask = map(v -> (v > 0 ? 1.0f0 : 0.0f0), vals1)
    #println("extream $(extrema(vals1))")
    vals1 = stack([view(moving_img.data, :, :, :, k)[map(v -> max(v, 1), vals1)] for k in 1:size(moving_img.data, 4)], dims=3) |> to_float
    #@assert size(vals1)==(8, unet_siz*unet_siz*4, 4)
    mask, vals1, pos2be
end

function sample_image_sliced(slicers, moving_imgs, pos2)
    expandrad = rand([100, 300, 600, 900])
    gausssigma = 300
    n_expand = 8
    siz1, siz2, siz3p... = size(pos2)
    pos2 = reshape(pos2, siz1, siz2, :)
    pos2e = Flux.ignore_derivatives() do
        expandpos(pos2; n=n_expand, σ=expandrad)
    end
    @check_size pos2e (siz1, n_expand, siz2, :)

    valsw3 = map(1:siz2) do B
        pos2b = view(pos2, :, B, :)
        mask, vals1, pos2be = Flux.ignore_derivatives() do
            # XXX: derivatives for slicer
            prepare_block_img_avg_slicer(slicers[B], pos2b, view(pos2e, :, :, B, :), moving_imgs[B])
        end
        @check_size pos2b (3, :)
        @check_size pos2be (3, n_expand, :)
        weight = exp.(-sum(((pos2be .- reshape(pos2b, siz1, 1, :)) ./ gausssigma) .^ 2, dims=1))
        weight = reshape(weight, size(mask)...) .* mask
        vals2 = sum(vals1 .* reshape(weight, size(weight)..., 1), dims=1)
        vals2 = reshape(vals2, size(vals2)[2:end]...)
        weight = reshape(sum(weight, dims=1), size(weight)[2:end]..., 1)
        @check_size weight (:, 1)
        @check_size vals2 (:, 1)
        vals2, weight
    end
    vals3, weight2 = reduce(.+, valsw3)
    vals3 = reshape(vals3 ./ (weight2 .+ 0.001f0), siz3p..., :)
    @info "$(size(vals3)) $siz3p"
    permutedims(vals3, (1, 2, 4, 3))
end

const expand_rand = Ref{Any}()
function expandpos(x::AbstractArray; n=8, σ=3000.0f0)
    siz1, siz2p... = size(x)
    @assert siz1 == 3
    @assert eltype(x) == Float32
    rr = if isdefined(expand_rand, :x) && size(expand_rand[]) == (siz1, n, prod(siz2p))
        expand_rand[]
    else
        rrr = rand(Float32, siz1, n, prod(siz2p)) .- 0.5f0
        rrr[:, 1, :] .= 0
        expand_rand[] = Flux.ignore_derivatives(rrr)
    end

    m = rr .* (2σ) |> Flux.gpu
    #m=randn(Float32, siz1, n, 1).*σ |> gpudev
    x = reshape(x, siz1, 1, :) .+ m
    reshape(x, siz1, :, siz2p...)
end

function pix2position(i::AbstractArray, res)
    siz1, siz2p... = size(i)
    @assert siz1 == 3
    i = reshape(i, siz1, :)
    pos = Float32.(i .- 0.5f0) .* reshape(res, siz1, 1)
    @assert eltype(pos) == Float32
    reshape(pos, siz1, siz2p...)
end
function position2pix(x::AbstractArray, res, off=nothing)
    siz1, siz2p... = size(x)
    @assert siz1 == 3
    @assert eltype(x) == Float32
    x = reshape(x, siz1, :)
    if off !== nothing
        x = x .- reshape(off, 3, 1)
    end
    x = x ./ reshape(res, 3, 1) .+ 0.5f0
    reshape(x, siz1, siz2p...)
end


function sample_image(slicers, moving_imgs, pos2)
    if mapreduce(v -> length(v.zvals), +, slicers) > 0
        return sample_image_sliced(slicers, moving_imgs, pos2)
    end
    expandrad = rand([100, 300, 600, 900])
    gausssigma = 300
    n_expand = 8

    siz1, siz2, siz3p... = size(pos2)
    pos2 = reshape(pos2, siz1, siz2, :)
    pos2e = Flux.ignore_derivatives() do
        expandpos(pos2; n=n_expand, σ=expandrad)
    end
    @check_size pos2e (siz1, n_expand, siz2, :)
    @check_size pos2 (siz1, siz2, :)

    valsw3 = map(1:siz2) do B
        pos2b = view(pos2, :, B, :)
        mask, vals1, pos2be = Flux.ignore_derivatives() do
            prepare_block_img_avg(view(pos2e, :, :, B, :), moving_imgs[B])
        end
        @check_size mask (n_expand, :)
        @check_size vals1 (n_expand, :, :)
        @check_size pos2b (3, :)
        @check_size pos2be (3, n_expand, :)
        weight = exp.(-sum(((pos2be .- reshape(pos2b, siz1, 1, :)) ./ gausssigma) .^ 2, dims=1))
        weight = reshape(weight, size(mask)...) .* mask
        vals2 = sum(vals1 .* reshape(weight, size(weight)..., 1), dims=1)
        vals2 = reshape(vals2, size(vals2)[2:end]...)
        weight = reshape(sum(weight, dims=1), size(weight)[2:end]..., 1)
        @check_size weight (:, 1)
        @check_size vals2 (:, :)
        vals2, weight
    end
    vals3, weight2 = reduce(.+, valsw3)
    summsk = Flux.ignore_derivatives() do
        permutedims(reshape(weight2 .> 0, siz3p..., :), (1, 2, 4, 3))
    end
    permutedims(reshape(vals3 ./ (weight2 .+ 0.001f0), siz3p..., :), (1, 2, 4, 3)), summsk
end


function image_adj_contrast(img::AbstractArray)::AbstractArray
    mm1 = Statistics.quantile(view(img, :) |> Flux.cpu, 0.95)
    img = clamp.(img ./ (mm1 + 0.001f0), 0, 1)
    mm = maximum(img)
    img[1:64:end, 1:64:end] .= 1.0f0
    #println("mm $mm1 $mm")
    #@assert mm>0.7f0
    img
end

# XXX: better obs passing
function loss_image(pars::Network, ref_img, moving_imgs, slice_space, obsinfo)
    unet = pars.unet.unet
    isempty(unet) && return 0.0f0, 0.0f0, 0.0f0
    n_bat = 4

    imgs1, pos1 = Flux.ignore_derivatives() do
        slices = sample_valid_slices(slice_space, ref_img, n_bat)
        prepare_images_batch(slices, ref_img)
    end

    pos2, pos2d000 = execute_xform(pars, :, pos1; retd=true)
    siz1, siz2, siz3p... = size(pos2)
    @assert (siz1, siz2, siz3p...) == (3, length(moving_imgs), unet_siz, unet_siz, n_bat)
    pos2d = sum((pos2d000 ./ 1000) .^ 2, dims=1)

    imgs2, mask2 = sample_image(pars.slicers, moving_imgs, pos2)
    #@assert size(imgs2)==(unet_siz,unet_siz,4,4)
    imgs3 = unet(imgs2)
    # imgs3 = unet.layers[1](imgs2)
    # imgs3 = Flux.Chain(unet.layers[2:end])(imgs3)
    pcls = 0.0
    ŷcls = nothing
    if false # XXX (pixcls=get(pars, :pixcls, nothing))!==nothing
        # w h C B
        ycls = view(imgs1, :, :, 3, :) .> 0.1f0
        ŷcls = pixcls(imgs2)
        pcls = Flux.logitbinarycrossentropy(reshape(ŷcls, size(ycls)...), ycls)
        # L 1 B
        # w h C B
        #12
    end
    Flux.ignore_derivatives() do
        if obsinfo.k % 100 == 0
            imgs = (imgs1[:, :, end, 2] .+ mask2[:, :, end, 2], image_adj_contrast(imgs2[:, :, 1, 2]), imgs3[:, :, end, 2])
            if ŷcls !== nothing
                imgs = (imgs..., Flux.sigmoid.(reshape(ŷcls, size(imgs[1])..., :)[:, :, 2]))
            end
            obsinfo.img[] = vcat(imgs...) |> Flux.cpu
            tmp = pos2d000[:, rand(axes(pos2d000, 2)), :, :, 2] |> Flux.cpu
            tmp0 = similar(tmp)
            for ci in CartesianIndices(tmp0)
                xyz = ci[1]
                tmp0[ci] = xyz == 1 ? 0 : ci[xyz] * 250
            end
            tmp .= tmp .+ tmp0
            obsinfo.warp[] = reshape(vcat(tmp0, tmp), 3, :)
            show(obsinfo.warp[][:, 1:4])
            @info "$(size(obsinfo.warp[]))"
        end
    end
    imgs1xx = Flux.ignore_derivatives() do
        imgs1 .* mask2
    end
    (sum(((imgs3 .- imgs1xx) .^ 2))) / (unet_siz * unet_siz), 0, pcls
end


function (prob::Problem{Tf,Cf,Nm,Tm,Cm})(xform::Network{Nm,Cf,Cm};
    landmark_weights=(1.0, 1.0, 1.0),
    n_iter=10000,
    log_io=stderr,
    #bones=nothing,
    dpts_scale=10000
) where {Tf,Cf,Nm,Tm,Cm}

    println(log_io, "init ================")
    #CUDA.memory_status(log_io)
    ref_img = prob.fixed |> Flux.gpu
    moving_imgs = prob.moving |> Flux.gpu
    xform = xform |> Flux.gpu
    #bones=bones |> gpudev

    groups = Dict{Int,Vector{Int}}()
    intragrp = Pair{Int,Int}[]
    ref_pts = map(enumerate(prob.landmarks)) do (i, pt)
        grp = get!(Vector{Int}, groups, pt.group)
        for j in grp
            push!(intragrp, j => i)
        end
        push!(grp, i)
        get(prob.group_pos, pt.group, pt.pos)
    end
    ref_pts = Float32.(reduce(hcat, ref_pts, init=zeros(Float32, 3, 0))) |> Flux.gpu
    moving_pts = map(pt -> pt.pos, prob.landmarks)
    moving_pts = Float32.(reduce(hcat, moving_pts, init=zeros(Float32, 3, 0))) |> Flux.gpu
    @assert size(ref_pts, 1) == 3 && size(moving_pts, 1) == 3
    intraweight = map(intragrp) do (a, b)
        t = prob.landmarks[a].group
        @assert t == prob.landmarks[b].group
        landmark_weights[t >= 1000_000 ? 3 : (t >= 100_000 ? 2 : 1)] |> Float32
    end |> Flux.gpu
    intragrp = (map(first, intragrp) => map(last, intragrp)) |> Flux.gpu
    pointsblk = Dict([B => findall(pt -> pt.block == B, prob.landmarks) for B in 0:length(moving_imgs)]) |> Flux.gpu
    println(log_io, "init2 ================")
    #CUDA.memory_status(log_io)
    flush(log_io)

    ###################################################
    ###########################################################################
    #################################################################
    #########################################################################################################

    ############################################
    ##################################
    ###############################################################################
    #=
        function get_slice_indices(ax, i)
            i=round(Int, size(ref_img.dat, ax)*i)
            idx=Any[:, :, :]
            idx[ax]=i
            CartesianIndices(ref_img.dat[:, :, :, 1])[idx...] |> gpudev
        end
        function get_image_tovis(ax, i)
            indices=get_slice_indices(ax, i)
            img_b=ref_img.dat[map(i->CartesianIndex(i, 3), indices)]
            pos=indices_to_pos(indices, ref_img.res)
            parts=map(enumerate(moving_imgs)) do (B, img)
                pos2=execute_xform(prepare_xform(xform_pars), B, pos)
                inds2=indices_from_pos(pos2, img.res)
                sample_img(img.dat, inds2, 1)
            end
            res1=map(img_b) do b
                r=0.0
                g=0.0
                ColorTypes.RGB{Float32}(r,g,b)
            end
            res2=map(parts...) do rg...
                r=0.0
                g=0.0
                for (i,v) in enumerate(rg)
                    if i%2==1
                        r+=v
                    else
                        g+=v
                    end
                end
                RGB{Float32}(r,g,0)
            end
            vcat(res1,res2)
        end
        =#

    get_new_pts(dpts) = ref_pts .+ dpts .* dpts_scale
    slice_space = get_slice_space(ref_img)
    #xform_pars=(;xform_pars..., warpfactor=Ref(10.0f0))

    function loss_func((pars, dpts), obsinfo)::Float32
        prepare_xform!(pars)
        lscale = loss_valid_scales(pars.affine)
        lpoint = loss_points(get_new_pts(dpts), moving_pts, pointsblk, intragrp, intraweight, pars, obsinfo)
        limage, limage2, limage3 = loss_image(pars, ref_img, moving_imgs, slice_space, obsinfo)
        if pars.warp !== nothing
            limage2 = regularize(pars.warp)
        end
        #limage2=0
        lbones = 0.0f0# XXX loss_bones(pars, bones)
        println(log_io, "lscale=$lscale lpoint=$lpoint limage=$limage lbones=$lbones limage2=$limage2 limage3=$limage3")
        lscale + lpoint + limage * 5 + 0.001 * limage2 + lbones * 0.1 + limage3 * 5
    end
    ####################################################################
    ####################################################################

    ref_pts_d = zeros(Float32, size(ref_pts)...) |> Flux.gpu
    obs_pts = Ref{Any}()
    obs_img = Ref{Any}()
    obs_warp = Ref{Any}()
    Flux.ignore_derivatives() do
        loss_func((xform, ref_pts_d), (k=0, pts=obs_pts, img=obs_img, warp=obs_warp))
    end
    fig = GLMakie.Figure()
    display(fig)
    obs_pts = let
        if isassigned(obs_pts)
            label(_, i, _) = "$(prob.landmarks[i])"
            obs = GLMakie.Observable(obs_pts[])
            p = GLMakie.scatter(fig[1, 1], obs, color=map(x -> (:black, :orange, :red, :magenta)[x.block+1], prob.landmarks), markersize=6, inspector_label=label)
            ##GLMakie.DataInspector(p)
            obs
        end
    end
    obs_deltas = GLMakie.Observable(rand(Float32, 3, 0))
    ax = GLMakie.LScene(fig[1, 3], show_axis=false)
    GLMakie.linesegments!(ax, obs_deltas,)
    GLMakie.scatter!(ax, [GLMakie.Vec3f(0, 0, 0), GLMakie.Vec3f(200, 200, 200)])

    obs_img = let
        if isdefined(obs_img, :x)
            obs = GLMakie.Observable(obs_img[])
            plt = GLMakie.image(fig[2, 1:3], obs)
            obs
        end
    end
    obs_warp = let
        if isdefined(obs_warp, :x)
            obs = GLMakie.Observable(obs_warp[])
            GLMakie.linesegments(fig[1, 2], obs)
            obs
        end
    end

    function dump(t, l=0)
        if t isa Tuple
            for i in 1:length(t)
                println(repeat(" ", l), i)
                dump(t[i], l + 1)
            end
        elseif t isa NamedTuple
            for k in propertynames(t)
                println(repeat(" ", l), k)
                dump(getproperty(t, k), l + 1)
            end
        else
            println(repeat(" ", l), typeof(t))
        end
    end
    #optst=Flux.setup(Flux.AdaGrad(), (xform_pars, ref_pts_d))
    optst = Flux.setup(Flux.Adam(), (xform, ref_pts_d))
    @showprogress "optimizing registraction" for K in 1:n_iter
        obsinfo = (k=K, pts=obs_pts, img=obs_img, warp=obs_warp)
        l, g = Flux.withgradient((args...) -> loss_func(args..., obsinfo), (xform, ref_pts_d))
        Flux.update!(optst, (xform, ref_pts_d), g[1])
        #println("$K $l $(g[1][1].warp[1].deltas)")
        #dump(g)
        #=
        if haskey(xform_pars, :warpfactor)
            println("mean w1: $(sum(xform_pars.warp[1].weights))")
            println("mean w2: $(sum(xform_pars.warp[2].weights))")
        end
        =#
        #GC.gc(false)
        #CUDA.memory_status()

        if K % 100 == 0
            #Main.xxxgrad = g[1]
            if g[1][1].warp !== nothing
                dd = g[1][1].warp.deltas[end]
                dd2 = reshape(sum(reshape(dd, 3, :) .^ 2, dims=1), size(dd)[2:end])
                xyz = findall(>(0), dd2)
                xyz = stack((map(v -> v[i], xyz) for i in 1:3), dims=1)
                dxyz = reshape(dd, 3, :)[:, findall(>(0), reshape(dd2, :))]
                dxyz = xyz + 100 * dxyz
                obs_deltas[] = reshape(vcat(xyz, dxyz), 3, :) |> Flux.cpu
            end

            println(log_io, "$K/$n_iter dist=$l")
            #CUDA.memory_status(log_io)
            flush(log_io)
            #XXX if haskey(xform, :warpfactor) && xform.warpfactor[]<1000f0
            #XXX xform.warpfactor[]+=50f0
            #XXX end

            #obs_pts[]=get_new_pts(ref_pts_d) |>Flux.cpu
            #if obs2!==nothing
            #obs2[]=get_image_tovis(showimg...) |> Flux.cpu
            #end
        end
    end

    ref_pts = get_new_pts(ref_pts_d) |> Flux.cpu
    for (g, grp) in groups
        pos = reshape(sum(ref_pts[:, grp], dims=2), :) ./ length(grp)
        for i in grp
            @assert prob.landmarks[i].group == g
        end
        prob.group_pos[g] = pos
    end

    return xform |> Flux.cpu
end
#const segreg_opt=optimize

function test3(prob::Problem{Tf,Cf,Nm,Tm,Cm}, ::Int, xform::Network2;
    # {Float16, 4, 2, UInt16, 4}
    # :SegReg.Network2{2, 3, 4,
    landmark_weights=(1.0, 1.0, 1.0),
    n_iter=10000,
    log_io=stderr,
    #bones=nothing,
    dpts_scale=10000
) where {Tf,Cf,Nm,Tm,Cm}

    prob = prob |> Flux.gpu
    xform = xform |> Flux.gpu

    for epoch in 1:5
        label_moving_imgs(xform, prob.moving, prob.fixed)
        # train_segment()
        # segment_moving_imgs()

        # uUUU
        prob(xform; landmark_weights, n_iter, log_io, dpts_scale)
    end
    # u
    # w
    # u
    # w
    # u
    # w
    # u
    # w
    # u
    # w


    return xform |> Flux.cpu
end

function get_warp_img(prob::Problem, xform_pars::Network)
    moving_imgs = deepcopy(prob.moving) |> Flux.gpu
    ref_img = deepcopy(prob.fixed) |> Flux.gpu
    xform_pars = deepcopy(xform_pars) |> Flux.gpu

    out_img = similar(ref_img.data, size(ref_img.data)[1:3]..., size(moving_imgs[1].data, 4))
    out_warp = similar(ref_img.data, size(ref_img.data)[1:3]..., 3)
    SegReg.prepare_xform!(xform_pars)
    @showprogress for z in 1:size(out_img, 3)
        inds = CartesianIndices(out_img)[:, :, z, 1] |> Flux.gpu
        inds = reshape(inds, size(inds)..., 1)
        pos1i = stack([map(i -> i[k], inds) for k in 1:3], dims=1)
        pos1 = SegReg.pix2position(pos1i, ref_img.resolution)
        #pos1e = expandpos(pos1; n=4, σ=1.0f0)
        #pos2e = execute_xform(xform_pars, :, pos1e)
        pos2, pos2d = execute_xform(xform_pars, :, pos1; retd=true)
        siz1, siz2, siz3p... = size(pos2)
        imgs2, mask2 = sample_image(xform_pars.slicers, moving_imgs, pos2)
        #=
        pos2e = reshape(pos2e, siz1, siz2, :, size(pos2, 3))
        # XXX dedup
        valsw3 = map(1:size(pos2, 2)) do B
            pos2b = view(pos2, :, B, :)
            mask, vals1, pos2be = prepare_block_img_avg(view(pos2e, :, B, :, :), moving_imgs[B])
            vals2 = sum(vals1 .* reshape(mask, size(mask)..., 1), dims=1)
            vals2 = reshape(vals2, size(vals2)[2:end]...)
            weight = reshape(sum(mask, dims=1), size(mask)[2:end]..., 1)
            vals2, weight
        end
        vals3, weight2 = reduce(.+, valsw3)
        imgs2 = reshape(vals3 ./ (weight2 .+ 0.001f0), siz3p..., :)
        =#
        for c in 1:size(out_img, 4)
            #@info "$(size(out_img))"
            #@info "$(size(imgs2))"
            # [ Info: (512, 624, 400, 4) [ Info: (512, 624, 4, 1)
            out_img[map(i -> i + CartesianIndex(0, 0, 0, c - 1), inds)] .= imgs2[:, :, c, :]
        end
        for c in 1:size(out_warp, 4)
            out_warp[map(i -> i + CartesianIndex(0, 0, 0, c - 1), inds)] .= pos2d[c, 1, :, :]
        end
    end
    (out_img, out_warp) |> Flux.cpu
end

function get_fw_warp_img(prob, xform_pars, src, src_shift)
    moving_imgs = deepcopy(prob.moving) |> Flux.gpu
    ref_img = deepcopy(prob.fixed) |> Flux.gpu
    xform_pars = deepcopy(xform_pars) |> Flux.gpu

    out_img = Dict{NTuple{3,Int16},UInt8}()
    SegReg.prepare_xform!(xform_pars)
    src_res = [250.0f0, 250.0f0, 250.0f0] |> Flux.gpu
    @showprogress for z in 1:size(src, 3)
        inds = filter(i -> src[i] > 0, CartesianIndices(src)[:, :, z])
        isempty(inds) && continue
        inds = map(inds) do i
            SVector(Tuple(i) .- src_shift)
        end
        inds = inds |> Flux.gpu
        inds = reshape(inds, size(inds)..., 1)
        pos1i = stack([map(i -> i[k], inds) for k in 1:3], dims=1)
        pos1 = SegReg.pix2position(pos1i, src_res)
        pos2, pos2d = execute_xform(xform_pars, :, pos1; retd=true)
        pos2i = SegReg.position2pix(pos2, moving_imgs[1].resolution, moving_imgs[1].offset)
        pos2i = reshape(pos2i, 3, :) |> Flux.cpu
        for v in eachcol(pos2i)
            push!(out_img, round.(Int16, (v...,)) => z % 30 + 1)
        end
    end
    inds = collect(keys(out_img))
    aa = min.((minimum(v -> v[i], inds) for i in 1:3) .- 5, 0)
    bb = [maximum(v -> v[i], inds) for i in 1:3] .+ 5
    oimg = zeros(UInt8, (bb - aa)...)
    aa = (aa...,)
    for (v, x) in out_img
        oimg[(v .- aa)...] = 200 + x
    end
    return oimg, .-aa
end

end # module SegReg
