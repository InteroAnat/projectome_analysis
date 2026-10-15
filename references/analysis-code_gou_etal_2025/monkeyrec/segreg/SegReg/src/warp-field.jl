struct WarpField{Nm}
    num_moving::Val{Nm}

    deltas::NTuple{Nm,AbstractArray}
    factor::Float32
    domain::AbstractArray{Float32}
    gridsize::AbstractArray{Float32}
end
Flux.@layer WarpField trainable = (deltas)

function Base.show(io::IO, w::WarpField)
    print(io, "WarpField(", "grid=", size(w.deltas[1]) .- 2, ", domain=", w.domain, ")")
end

function WarpField{Nm}(gridsize, domain=(64000.0f0, 78000.0f0, 50000.0f0); factor=512, deltas=zeros) where {Nm}
    deltas = ((deltas(Float32, 3, (gridsize .+ 2)...) for _ in 1:Nm)...,)
    domain = reshape([domain...], 3, :)
    gridsize = reshape([gridsize...], 3, :)
    WarpField{Nm}(Val(Nm), deltas, factor, domain, gridsize)
end

function initialize!(warp::WarpField, prevwarp::WarpSpline)
    warp.domain != prevwarp.domain && return
    warp.num_moving != prevwarp.num_moving && return
    deltasize = size(view(warp.deltas[1], 1, :, :, :))
    domain = (Flux.cpu(warp.domain)...,)
    xyz = map(CartesianIndices(deltasize)) do ci
        (Tuple(ci) .- 0.5f0) ./ deltasize .* domain
    end
    xyz = stack((map(v -> v[i], xyz) for i in 1:3), dims=1)
    prevwarp = prevwarp |> Flux.gpu
    xyz = xyz |> Flux.gpu
    for (b, curdeltas) in enumerate(warp.deltas)
        blk = similar(xyz)
        blk .= 0
        @showprogress "initializing" for j in axes(xyz, 4)
            tmp = prevwarp(xyz[:, :, :, j], b) ./ warp.factor
            blk[:, :, :, j] .= tmp
        end
        copyto!(curdeltas, blk)
    end
end

function (warp::WarpField)(x::AbstractArray{Float32}, blocks::Union{AbstractRange,Colon})
    @assert eltype(x) == Float32
    siz = size(x)
    @assert length(siz) >= 2 && siz[1] == 3
    corner, cond = Flux.ignore_derivatives() do
        xxxflt = reshape(x, siz[1], :) ./ warp.domain .* warp.gridsize .+ 0.5f0
        xxxint = round.(Int32, xxxflt)
        dsize = size(view(warp.deltas[1], 1, :, :, :))
        if maximum(abs.(xxxflt .- xxxint)) < 0.0001f0
            xxxinds = map((view(xxxint, i, :) for i in 1:siz[1])...) do xi, yi, zi
                LinearIndices(dsize)[xi+1, yi+1, zi+1]
            end
            return xxxinds, :aligned
            ###############################
            rngs = extrema(xxxint, dims=2) |> Flux.cpu
            rngs = splat(UnitRange).(rngs)
            if Flux.gpu(LinearIndices(dsize)[(r .+ 1 for r in rngs)...]) == xxxinds
                rngs, :simple
            else
                xxxinds, :aligned
            end
        else
            xxxinds = map((view(xxxint, i, :) for i in 1:siz[1])...) do xi, yi, zi
                LinearIndices(dsize)[xi+1, yi+1, zi+1]
            end
            return xxxinds, :aligned
            @assert false
            0, :unaligned
        end
    end
    res = if cond === :simple
        map(warp.deltas[blocks]) do dd
            reshape(dd[:, corner...], siz[1], :)
        end
    elseif cond === :aligned
        map(warp.deltas[blocks]) do dd
            reshape(dd, siz[1], :)[:, corner]
        end
    elseif cond === :unaligned
    else
        @assert false
    end
    x = reshape(reduce(vcat, res), siz[1], :, siz[2:end]...) .* warp.factor
    @assert eltype(x) == Float32
    x
end


function regularize(warp::WarpField)
    function loss_diff(a, b)
        Statistics.mean(abs.(a .- b))
    end
    ll = map(warp.deltas) do deltas
        return 0
        l0 = 0#Statistics.mean(deltas .^ 2)
        xi = rand(1:size(deltas, 2)-2)
        yi = rand(1:size(deltas, 3)-2)
        zi = rand(1:size(deltas, 4)-2)
        l1 = loss_diff(@view(deltas[:, xi:xi+1, :, :]), @view(deltas[:, xi+1:xi+2, :, :]))
        l2 = loss_diff(@view(deltas[:, :, yi:yi+1, :]), @view(deltas[:, :, yi+1:yi+2, :]))
        l3 = loss_diff(@view(deltas[:, :, :, zi:zi+1]), @view(deltas[:, :, :, zi+1:zi+2]))
        0.0000000001f0 * l0 + l1 + l2 + l3
    end
    sum(ll) * (warp.factor / 1024.0f0)
end

function (net::Union{WarpSpline,WarpField})(x, B::Int)
    x = net(x, B:B)
    siz1, siz2, siz3p... = size(x)
    @assert siz2 == 1
    reshape(x, siz1, siz3p...)
end
########################################

# function xfunc_warp3(net, B::Union{AbstractRange,Colon}, x::AbstractArray, factor::Float32; gpudev)
#     if net isa Tuple
#         return xform_bsp(net, B, x, factor; gpudev)
#     elseif net isa NamedTuple
#         return xform_tetragrid(net, B, x, factor; gpudev)
#     end
#     @assert eltype(x) == Float32
#     siz1, siz2p..., siz9 = size(x)
#     @assert siz1 == 3
#     x = reshape(x, siz1, :, siz9) ./ 30000.0f0
#     x = net(permutedims(x, (2, 1, 3))) .* factor
#     x = reshape(x, size(x, 1), siz1, :, siz9)
#     x = permutedims(view(x, :, :, B, :), (2, 3, 1, 4))
#     x = reshape(x, siz1, :, siz2p..., siz9)
#     #@assert size(x, 2)==2
#     x
# end
#
########################################
