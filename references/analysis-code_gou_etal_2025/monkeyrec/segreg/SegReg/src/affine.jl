struct Affine{Nm}
    num_moving::Val{Nm}

    rotate::AbstractMatrix{Float32}
    scale::AbstractMatrix{Float32}
    translate::AbstractMatrix{Float32}

    #cache_mats::AbstractArray{Float32,3}
    #cache_vecs::AbstractMatrix{Float32}
    cache_mats::Ref{<:AbstractArray{Float32,3}}
    cache_vecs::Ref{<:AbstractMatrix{Float32}}
end
Flux.@layer Affine trainable = (rotate, scale, translate)
function Base.show(io::IO, a::Affine)
    print(io, "Affine(", ")")
end

function Affine{Nm}(; rotate=rand, scale=ones, translate=rand) where {Nm}
    rotate = rotate(Float32, 3, Nm)
    scale = scale(Float32, 3, Nm)
    translate = translate(Float32, 3, Nm)
    Affine{Nm}(Val(Nm), rotate, scale, translate,
        Ref(zeros(Float32, 0, 0, 0)),
        Ref(zeros(Float32, 0, 0)),
        #zeros(Float32, 3, 3, Nm),
        #zeros(Float32, 3, Nm),
    )
end

function (aff::Affine)(; translate_unit=10000.0f0)
    mats = prepare_rotate(aff.rotate)
    mats = prepare_scale(aff.scale) .* mats
    vecs = prepare_translate(aff.translate, translate_unit)
    aff.cache_mats[] = mats
    aff.cache_vecs[] = vecs
end
function (aff::Affine)(x::AbstractArray{Float32}, blocks::Union{AbstractRange,Colon})
    siz = size(x)
    @assert length(siz) >= 2 && siz[1] == 3
    if length(siz) == 2
        apply_affine(aff, reshape(x, siz..., 1), blocks)
    elseif length(siz) >= 3
        if siz[end] == 1
            apply_affine(aff, reshape(x, siz[1], :, 1), blocks)
        else
            apply_affine(aff, reshape(x, siz[1], :, siz[end]), blocks)
        end
    end
end

function apply_affine(aff::Affine, x::AbstractArray{Float32,3}, blocks::Union{AbstractRange,Colon})
    siz1, _, _ = size(x)
    @assert siz1 == 3
    mm = view(aff.cache_mats[], :, :, blocks)
    # 3x3xB
    x1 = Flux.batched_mul(mm, x)
    # 3xNxB
    vv = view(aff.cache_vecs[], :, blocks)
    # 3xB
    vv = reshape(vv, size(vv, 1), 1, :)
    # 3x1xB
    x1 .+ vv
    # 3xNxB
end



function prepare_scale(scale::AbstractMatrix{Float32})
    siz1, siz2 = size(scale)
    @assert siz1 == 3
    reshape(scale, siz1, 1, siz2)
end
function prepare_translate(translate::AbstractMatrix{Float32}, unit::Float32)
    siz1, _ = size(translate)
    @assert siz1 == 3
    translate .* unit
end

const _rotate_mat_indices = [3 6 10 9 3 10 10 10 11 2 10 8 10 11 10 5 10 2 11 10 10 10 1 4 10 7 1]
function prepare_rotate(rotate::AbstractMatrix{Float32})
    siz1, siz2 = size(rotate)
    @assert siz1 == 3
    cosine = cos.(rotate)
    sine = sin.(rotate)
    zero1, selection = Flux.ignore_derivatives() do
        zo = similar(cosine, (2, siz2))
        zo[1, :] .= 0.0f0
        zo[2, :] .= 1.0f0
        sel = similar(cosine, Int32, length(_rotate_mat_indices))
        copyto!(sel, _rotate_mat_indices)
        zo, sel
    end
    elems = vcat(cosine, sine, -sine, zero1)
    mats = reshape(elems[selection, :], 3, 3, 3, :)
    Flux.batched_mul(view(mats, :, :, 3, :), Flux.batched_mul(view(mats, :, :, 2, :), view(mats, :, :, 1, :)))
end

