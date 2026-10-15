include("0-bspline.jl")

struct WarpSpline{Nm}
    num_moving::Val{Nm}

    bsp::nurbs.NurbsMap
    deltas::NTuple{Nm,AbstractArray}
    factor::Float32
    domain::AbstractArray{Float32}
end
Flux.@layer WarpSpline trainable = (deltas)

function Base.show(io::IO, w::WarpSpline)
    print(io, "WarpSpline(", "grid=", size(w.bsp.weights) .- 1, ", domain=", w.domain, ")")
end

function WarpSpline{Nm}(domain=(64000.0f0, 78000.0f0, 50000.0f0); factor=512, deltas=zeros, grid=(32, 40, 24)) where {Nm}
    bsp = nurbs.NurbsMap(grid)
    deltas = ((deltas(Float32, 3, size(bsp.weights)...) for _ in 1:Nm)...,)
    domain = reshape([domain...], 3, :)
    WarpSpline{Nm}(Val(Nm), bsp, deltas, factor, domain)
end

function initialize!(warp::WarpSpline, prevwarp::WarpSpline)
    warp.domain != prevwarp.domain && return
    warp.num_moving != prevwarp.num_moving && return
    foreach(prevwarp.deltas, warp.deltas) do prevdeltas, curdeltas
        deltas = prevdeltas .* prevwarp.factor ./ warp.factor |> Flux.cpu
        sizin = size(prevdeltas)
        sizout = size(curdeltas)
        out = Array{Float32}(undef, sizout)
        for ci in CartesianIndices(out)
            xyz = (round(Int, clamp(ci[1+d] * 1.0 / sizout[1+d] * sizin[1+d], 1, sizin[1+d])) for d in 1:3)
            out[ci] = prevdeltas[ci[1], xyz...]
        end
        copyto!(curdeltas, out)
    end
end

function (bsp::WarpSpline)(x::AbstractArray{Float32}, blocks::Union{AbstractRange,Colon})
    @assert eltype(x) == Float32
    siz = size(x)
    @assert length(siz) >= 2 && siz[1] == 3

    xxx = Flux.ignore_derivatives() do
        xxxx1 = reshape(x, siz[1], :) ./ bsp.domain
        #println("xxxx $xxxx1")
        xx = min.(max.(xxxx1, 0.0f0), 1.0f0)
        #clamp!(xxxx1, 0f0, 1f0)
        a, b = extrema(xx)
        #println("extrema $a $b")
        @assert a >= 0
        @assert b <= 1
        xx
    end

    coefs = bsp.bsp(xxx)
    res = map(bsp.deltas[blocks]) do dd
        @assert LinearIndices(bsp.bsp.weights) == LinearIndices(view(dd, 1, :, :, :))
        dd = reshape(dd, siz[1], :)
        nurbs.apply(dd, coefs)
    end
    #@assert !any(isnan.(xxx))
    x = reshape(reduce(vcat, res), siz[1], :, siz[2:end]...) .* bsp.factor
    #@assert !any(isnan.(x))
    @assert eltype(x) == Float32
    #println(size(x, 2))
    #println("sxxxxxxxxxxxxxx $(size(x))")
    x
end

function loss_diff(a, b)
    Statistics.mean((a .- b) .^ 2)
end
function regularize(warp::WarpSpline)
    ll = map(warp.deltas) do deltas
        l0 = Statistics.mean(deltas .^ 2)
        l1 = loss_diff(@view(deltas[:, 1:end-1, :, :]), @view(deltas[:, 2:end, :, :]))
        l2 = loss_diff(@view(deltas[:, :, 1:end-1, :]), @view(deltas[:, :, 2:end, :]))
        l3 = loss_diff(@view(deltas[:, :, :, 1:end-1]), @view(deltas[:, :, :, 2:end]))
        0.0000000001f0 * l0 + l1 + l2 + l3
    end
    sum(ll) * (warp.factor / 1024.0f0)
end

