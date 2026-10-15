module nurbs

import Flux

# XXX @functor/chainrule
# xbases and ybases, independent or what?
struct Basis{Ord}
    ord::Val{Ord}

    knots::AbstractVector
end
Flux.@layer Basis{3} trainable = (knots)

function Base.getproperty(bas::Basis{Ord}, prop::Symbol) where {Ord}
    if prop === :order
        return Ord
    elseif prop === :nbreak
        return length(bas.knots) - 2 - Ord
    else
        return getfield(bas, prop)
    end
end
function Basis(order::Int, n::Int)
    knots = Float32[clamp((i - order) / (n - order + 1), -1.0f-5, 1 + 1.0f-5) + min(i - order, 0) * 1.0f-6 + max(i - n - 1, 0) * 1.0f-6 for i in 0:n+order+1]
    Basis{order}(Val(order), knots)
end
####SegReg.nurbs.Basis(::CUDA.CuArray{Float32, 1, CUDA.DeviceMemory})

function gen_idxvec(tgt, Ord)
    idxvec = reshape(collect(1:2*Ord), :, 1)
    copyto!(similar(tgt, eltype(idxvec), size(idxvec)), idxvec)
end

function prepare_vals(bas::Basis{Ord}, ts::AbstractVector) where {Ord}
    diff = reshape(ts, 1, :) .- reshape(bas.knots, :, 1)
    diff[diff.<0.0f0] .= 1.1f0
    maxidx::Int = length(bas.knots) - 1 - 2 * Ord
    idx1 = map(i -> clamp(i[1] - Ord, 1, maxidx), findmin(diff, dims=1)[2])
    vals = similar(ts, 1, length(idx1))
    vals .= 0.0f0
    vals[1, :] .= 1.0f0
    cache_idx = gen_idxvec(idx1, Ord) .+ reshape(idx1, 1, :)
    knots_cache = map(v -> clamp(v, 0, 1), view(bas.knots, cache_idx))
    idx1, vals, knots_cache
end

function recursive_vals_impl(ts, vals, lower, upper, next)
    www = (ts .- lower) ./ (upper .- lower)
    wwwvals = www .* vals
    vcat(vals .- wwwvals, next) .+ vcat(next, wwwvals)
end
import ChainRulesCore
using ChainRulesCore: NoTangent
import CUDA
function ChainRulesCore.rrule(::typeof(recursive_vals_impl), ts, vals, lower, upper, next)
    dlu = (upper .- lower)
    www = (ts .- lower) ./ dlu
    ##############
    wwwvals = www .* vals
    res = vcat(vals .- wwwvals, next) .+ vcat(next, wwwvals)
    #println("asdf")
    function tang(dy)
        begin
            dvals = dy[1:(end-1), :]
            dwwwvals = dy[2:end, :] - dvals
            dwww = dwwwvals .* vals
            dvals .+= dwwwvals .* www
            dts = sum(dwww ./ dlu, dims=1)
            ##########
            #
            (NoTangent(), dts, dvals, NoTangent(), NoTangent(), NoTangent())
        end
    end
    (res, tang)
end
function recursive_vals(j, Ord, knots_cache, ts, vals0, next)
    vals = if j == 1
        vals0
    else
        recursive_vals(j - 1, Ord, knots_cache, ts, vals0, next)
    end
    lower = view(knots_cache, (Ord - j) .+ (1:j), :)
    upper = view(knots_cache, Ord .+ (1:j), :)

    recursive_vals_impl(ts, vals, lower, upper, next)
end
function (bas::Basis{Ord})(ts::AbstractVector) where {Ord}
    @assert eltype(ts) == Float32
    idx1, vals0, knots_cache = Flux.ignore_derivatives() do
        prepare_vals(bas, ts)
    end

    next = Flux.ignore_derivatives() do
        similar(ts, 1, length(ts)) .= 0.0f0
    end
    vals9 = recursive_vals(Ord, Ord, knots_cache, reshape(ts, 1, :), vals0, next)
    @assert eltype(vals9) == Float32
    #println("vals9: $vals9")
    #println("idx1: $idx1")
    vals9, idx1
end
#############################################################
#0:0
#####################################################################
##############################
###################################
##########################################
#################################
#ww[1]
#1. (1-ww[1]).*vals[1]
#2. ww[1]*vals[1]
#
#
#

##################################################
#int gsl_bseval (const double x, gsl_vector *B,
#          gsl_bspline_workspace *w)
#

#n = nbreak + k - 2
##################################################


#nbreak
#n = nbreak + k - 2

struct NurbsMap
    ##m::AbstractArray
    bases::Tuple
    weights::AbstractArray
end
Flux.@layer NurbsMap trainable = (bases)

function NurbsMap(gridsize::Tuple{Vararg{Integer,N}}) where {N}
    weights = ones(Float32, (gridsize .+ 1)...)
    NurbsMap(map(v -> Basis(3, v), gridsize), weights)
end

function grid()
    stack([map(v -> Tuple(v)[i] / 10, CartesianIndices((11, 11, 11)) .- CartesianIndex(1, 1, 1)) for i in 1:3], dims=1)
end

function gen_karray(tgt)
    karray = reshape(collect(CartesianIndices((4, 4, 4)) .- CartesianIndex(1, 1, 1)), 4, 4, 4, 1)
    copyto!(similar(tgt, eltype(karray), size(karray)), karray)
end

function nurbs_get_www(nurbs, points)
    pars = map(i->nurbs.bases[i](view(points, i, :)),((1:3)...,))
    idxes = Flux.ignore_derivatives() do
        li = LinearIndices(nurbs.weights)
        idx1s = map(v -> v[2], pars)
        t = reshape(CartesianIndex.(idx1s...), 1, 1, 1, :)
        k = gen_karray(t)
        broadcast((kk, tt) -> li[kk+tt], k, t)
        #broadcast((i, j, k)->CartesianIndex(i,j,k), reshape(0:3, :, 1, 1), reshape(0:3, 1, :, 1), reshape(0:3, 1, 1, :))
        #return map((x,y,z)->CartesianIndex(x, y, z), idx1s...)
        #map(p->p[2], pars)
        #par
    end
    #println("mean w: $(sum(weights))")
    #@assert !any(isnan.(weights))
    broadcast(*, reshape(pars[1][1], 4, 1, 1, :), reshape(pars[2][1], 1, 4, 1, :), reshape(pars[3][1], 1, 1, 4, :)), idxes
end

function apply_weights_impl(deltas, wij, idxes)
    #println("calc impl 4 4 4 1000")
    pts = view(deltas, :, idxes) .* reshape(wij, 1, size(wij)...)
    sum(pts, dims=(2, 3, 4))
end
function apply_weights(deltas, wij, idxes)
    apply_weights_impl(deltas, wij, idxes)
end

#function ChainRulesCore.rrule(config, ::typeof(apply_weights), deltas, wij, idxes)
#end
function ChainRulesCore.rrule(::typeof(apply_weights), deltasx, wij, idxes)
    function kernelfw(vals, deltasx, wij, idxes)
        idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        nn = size(idxes, 2)
        for j in idx:4096:nn
            for i in 1:3
                v = Float32(0)
                for k in 1:64
                    v += deltasx[i, idxes[k, j]] * wij[k, j]
                end
                vals[i, j] = v
            end
        end
        ################
        #####################
        ######################
        ####################
    end
    deltas = view(deltasx, :, idxes)
    vals = similar(deltas, size(deltas, 1), size(wij)[end])
    CUDA.@cuda threads = 512 blocks = 8 kernelfw(vals, deltasx, reshape(wij, 64, :), reshape(idxes, 64, :))
    ###############################
    #########################
    #######################

    function sin_pullback(dy)
        #println("rrule $(size(idxes)) $(size(dy)) $(size(deltas)) $(size(wij))")
        #t1=Flux.ignore_derivatives(()->time())
        #3 64 10000
        #3 10000
        #3 64 10000
        dx3 = reshape(sum(reshape(dy, 3, 1, 1, 1, :) .* deltas, dims=1), size(wij)...)
        #dx3=@time "dx3" similar(wij).=0.01
        function kernel(zzz, dyy, idxes)
            s1, s2 = size(idxes)
            idx = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            #println(idx)
            j = idx
            while j <= s2
                for i in 1:s1
                    k = idxes[i, j]
                    for q in 1:3
                        zzz[q, k, idx] += dyy[q, i, j]
                    end
                end
                j += 4096
            end
            #=
            #idxes: 64 10000
            #dyy 3 64 10000
            #zzz 3 NNN 4096
            #
            idx < length(dA.rowPtr) || return
            s = zero(Tnorm)
            for k in dA.rowPtr[idx]:dA.rowPtr[idx+1]-1
                s += abs(dA.nzVal[k])
            end
            out[idx] = s
            =#
            return
        end
        dx2 = begin
            zzz = similar(deltasx, size(deltasx)..., 4096)
            zzz .= 0.0
            dyy = reshape(dy, 3, 1, 1, 1, :) .* reshape(wij, 1, size(wij)...)
            CUDA.@cuda threads = 512 blocks = 8 kernel(zzz, reshape(dyy, 3, 64, :), reshape(idxes, 64, :))
            #=
            #      config = launch_configuration(kernel_f.fun)                                  
            #            threads = min(n, config.threads)                                             
            #                  blocks = cld(n, threads) 
            ui=unique(idxes|>Flux.cpu)
            dyy2=dyy[2, :, :, :, :]
            uip=map(ui) do i
                sum(dyy2[idxes.==i])
            end |>Flux.gpu
            zzz[2, ui].=uip
            #zzz[:, idxes].+=
            =#
            sum(zzz, dims=3)
        end
        #println("dx3: ", Flux.ignore_derivatives(()->time()-t1))
        #res=
        (NoTangent(), dx2, dx3, NoTangent())
    end
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    #####################################################
    return (vals, sin_pullback)
    ##
    #
    #
    #                               
    #                                 (Ω, (Ω̄₁, Ω̄₂, ...) -> (s̄elf, x̄₁, x̄₂, ...))
    #                                                                                                
    #                                                                                                  Where the second return value is the the propagation rule or pullback. It takes in cotangents corresponding to the outputs
    #                                                                                                    (x̄₁, x̄₂, ...), and s̄elf, the internal values of the function itself (for closures)
end
#function ChainRulesCore.frule(config, ::typeof(apply_weights), deltas, wij, idxes)
#end
#=
function ChainRulesCore.frule(::typeof(apply_weights), deltas, wij, idxes)
    println("frule")
    vals=apply_weights_impl(deltas, wij,idxes)
    return vals, 0
end
=#

# function ChainRulesCore.frule((_, Δx), ::typeof(sin), x::Number)
#     sinx, cosx = sincos(x)
#     return (sinx, cosx * Δx)
#     ##  frule([::RuleConfig,] (Δf, Δx...), f, x...)
#     #                               
#     #                                 Expressing the output of f(x...) as Ω, return the tuple:
#     #                                                                
#     #                                                                  (Ω, ΔΩ)
#     #                                                                                                 
#     #                                                                                                   The second return value is the tangent w.r.t. the output.
#     #
# end

function (nurbs::NurbsMap)(points::AbstractArray{T,N}) where {T,N}
    #@assert !any(isnan.(points))
    @assert eltype(points) == Float32
    siz1, sizes... = size(points)
    @assert size(points, 1) == 3
    #println("cccccccccccc")
    www, idxes = nurbs_get_www(nurbs, reshape(points, siz1, :))
    #println("dddddddddddddd")
    weights = Flux.ignore_derivatives(nurbs.weights)
    # 4 4 4 N
    wij = www#.*view(weights, idxes) # XXX
    #@assert !any(isnan.(wij))
    (; wij=wij, idxes=idxes)
end
function apply(deltas, coefs)
    #@assert !any(isnan.(deltas))
    # 3 4 4 4 N
    pts = apply_weights(deltas, coefs.wij, coefs.idxes)
    pts = apply_weights(deltas, coefs.wij, coefs.idxes)
    #println("aaaaaaaaaa")
    #println("bbbbbbbbbbbbb $(size(pts))")
    ###############################
    ############################################################
    pts9 = pts
    #println("qqqqqqqqqqqqqqqqq")
    @assert eltype(pts9) == Float32
    #@assert !any(isnan.(pts))
    pts9
end

#=
"""
	res=map() do k
		Pts=reshape(wij, 1, 4, 4).*surf.P[:, irng, jrng]
		pt=reshape(sum(Pts, dims=(2,3))./sum(wij), 3)
		pt
	end
	reduce(hcat, res)
end
{V} <: AbstractAffineMap
Translation(x,y,z) = Translation(SVector(x,y,z))
Base.show(io::IO, trans::Translation) = print(io, "Translation$((trans.translation...,))")


Base.inv(trans::Translation) = Translation(-trans.translation)

function compose(trans1::Translation, trans2::Translation)
    Translation(trans1.translation + trans2.translation)
end

transform_deriv(trans::Translation, x) = I
transform_deriv_params(trans::Translation, x) = I

function Base.isapprox(t1::Translation, t2::Translation; kwargs...)
    isapprox(t1.translation, t2.translation; kwargs...)
end


end
function nurbs_change(surf; shift=nothing, scale=nothing)
	newP=shift===nothing ? surf.P : shift#surf.P.+shift
	newW=scale===nothing ? surf.w : scale#surf.w.*scale
	(m=surf.m, n=surf.n, w=newW, P=newP, ubas=surf.ubas, vbas=surf.vbas)
end
function nurbs_surf_samp(surf, nsamp=3000)
	uvvec=Zygote.@ignore reduce(hcat, [rand(2) for i=1:nsamp])
	inside=Zygote.@ignore [c[1]<.8 && c[1]>.2 && c[2]<.8 && c[2]>.2 for c in eachcol(uvvec)]
	nurbs_surf_points(surf, uvvec), inside
end
function nurbs_to_mesh(surf)
	uvs=[]
	uv2vi=Dict()
	for u in 0:1:50, v in 0:1:50
		push!(uvs, [u/50; v/50])
		uv2vi[(u, v)]=length(uvs)
	end
	verts=nurbs_surf_points(surf, reduce(hcat, uvs))
	verts=[Point3(c...) for c in eachcol(verts)]
	function get_v(u, v)
		r=get(uv2vi, (u, v), nothing)
		#@info "$u $v"
		@assert r!==nothing
		r
	end
	faces=[]
	for u in 0:1:49, v in 0:1:49
		a=get_v(u, v)
		b=get_v(u+1, v)
		c=get_v(u+1, v+1)
		d=get_v(u, v+1)
		push!(faces, [a, b, c])
		push!(faces, [c, d, a])
	end
	faces=reduce(hcat, faces)
	faces=[NgonFace(c...) for c in eachcol(faces)]
	GeometryBasics.Mesh(verts, faces)
end
    Translation(v) <: AbstractAffineMap
    Translation(dx, dy)       (2D)
    Translation(dx, dy, dz)   (3D)

Construct the `Translation` transformation for translating Cartesian points by
an offset `v = (dx, dy, ...)`
=#

import Flux

function test_bspline()
    gpudev = Flux.CUDADevice(nothing)
    let b = Basis(3, 12) |> gpudev
        w, i = b(Float32[0.5] |> gpudev) |> Flux.cpu
        @assert w ≈ [0.16666672; 0.6666666; 0.16666666; 0.0;;]
        @assert i == [6;;]
        w, i = b(Float32[0.5001] |> gpudev) |> Flux.cpu
        @assert i == [6;;]
        @assert w ≈ [0.16616714; 0.6666657; 0.16716723; 1.6674961f-10;;]

        xx = repeat(Float32[0.5], 1000_000) |> gpudev
        l, g = @time Flux.withgradient((x) -> Flux.cpu(b(x)[1])[2], xx) |> Flux.cpu
        @assert l ≈ 0.6666666
        @assert g[1][1] ≈ 2.3841858f-7
        l, g = Flux.withgradient((x) -> Flux.cpu(b(x)[1])[2], Float32[0.5001] |> gpudev) |> Flux.cpu
        @assert l ≈ 0.6666657
        @assert g[1][1] ≈ -0.019987369
        l, g = Flux.withgradient((x) -> Flux.cpu(b(x)[1])[2], Float32[0.5003] |> gpudev) |> Flux.cpu
        println(g)
        @assert l ≈ 0.6666577
        @assert g[1][1] ≈ -0.059862472
    end

    let m = NurbsMap((24, 24, 24)) |> gpudev
        deltas = zeros(3,size(m.weights)...) |> gpudev
        deltas[:, 12, 12, 12] .= 3.0
        deltas=reshape(deltas, 3, :)
        xx = repeat(Float32[0.5001, 0.5001, 0.5001], 1, 2000_000) |> gpudev
        #println("hello")
        l, g = @time "gradient" Flux.withgradient((x, y,dd) -> sum(apply(dd, y(x))[[2]]), xx, m, deltas)
        l, g = l, g |> Flux.cpu
        @assert l ≈ 0.01361627
        @assert g[1][2] ≈ -0.900655
        @assert reshape(g[3], 3, size(m.weights)...)[2, 12, 12, 12] ≈ 0.004538757
        #println(g[1][1].deltas)
    end

    #
    #nurbs.
end

end
