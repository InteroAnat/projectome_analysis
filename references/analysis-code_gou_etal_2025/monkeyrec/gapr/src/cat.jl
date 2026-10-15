import IniFile
import IntervalSets

abstract type ImageChannel end

@kwdef struct _common_chan_info
    size::SVector{3,Int} = SVector(zeros(Int, 3)...)
    origin::SVector{3,Float64} = SVector(zeros(Float64, 3)...)
    direction::SMatrix{3,3,Float64} = SMatrix{3}(LinearAlgebra.I(3)...)
    range::IntervalSets.ClosedInterval{Float64} = IntervalSets.ClosedInterval(0.0, 1.0)
end

struct ImageCubes <: ImageChannel
    name::String
    pattern::String
    info::_common_chan_info
    cubesize::SVector{3,Int}
end

Base.getindex(cubes::Gapr.ImageCubes, x::Int64, y::Int64, z::Int64) = begin
    replace(cubes.pattern,
            r"<08u\$(z)>" => (s)->string(z, pad=8),
            r"<08u\$(y)>" => (s)->string(y, pad=8),
            r"<08u\$(x)>" => (s)->string(x, pad=8),
           )
end

struct ImageDownsample <: ImageChannel
    name::String
    location::String
    info::_common_chan_info
end

Base.getproperty(x::ImageChanT, f::Symbol) where {ImageChanT<:ImageChannel} = begin
    if hasfield(ImageChanT, f)
        (@inline; getfield(x, f))
    else
        (@inline; getproperty(x.info, f))
    end
end

struct ImageIndex
    channels::AbstractDict{String,ImageChannel}
end

Base.getindex(imgidx::Gapr.ImageIndex, nam::String) = imgidx.channels[nam]

function Base.parse(::Type{ImageIndex}, file::Union{IO,AbstractString}; base=nothing)
    ini = read(IniFile.Inifile(), file)
    chans = Dict{String,ImageChannel}()
    for (nam, sec) in ini.sections
        chan = _parse_channel(nam, sec, base)
        pchan = get!(chans, nam, chan)
        @assert pchan === chan
    end
    ImageIndex(chans)
end

function _parse_channel(nam, sec, base)
    if get(sec, "location", nothing) !== nothing
        _parse_chan_ds(nam, sec, base)
    elseif get(sec, "pattern", nothing) !== nothing
        _parse_chan_cubes(nam, sec, base)
    else
        @assert false
    end
end

function _parse_common(k, v)
    if k == "size"
        :size => _parse_as_vec(3, Int, v)
    elseif k == "origin"
        :origin => _parse_as_vec(3, Float64, v)
    elseif k == "direction"
        :direction => _parse_as_vec(9, Float64, v)
    elseif k == "range"
        :range => IntervalSets.ClosedInterval(_parse_as_vec(2, Float64, v)...)
    else
        nothing
    end
end

function _joinloc(base, v)
    if base === nothing
        return v
    else
        # XXX:
        return joinpath(base, v)
    end
end

function _parse_chan_ds(nam, sec, base)
    common = (;)
    loc = nothing
    for (k, v) in sec
        kv = _parse_common(k, v)
        if kv !== nothing
            common = (; common..., kv.first => kv.second)
        elseif k == "location"
            loc = _joinloc(base, v)
        else
            @assert false
        end
    end
    ImageDownsample(nam, loc, _common_chan_info(; common...))
end

function _parse_as_vec(N, T, s)
    vec = parse.(T, split(s, r"\s+"))
    @assert length(vec) == N
    (vec...,)
end

function _parse_chan_cubes(nam, sec, base)
    common = (;)
    loc = nothing
    cubsiz = nothing
    for (k, v) in sec
        kv = _parse_common(k, v)
        if kv !== nothing
            common = (; common..., kv.first => kv.second)
        elseif k == "pattern"
            loc = _joinloc(base, v)
        elseif k == "cubesize"
            cubsiz = _parse_as_vec(3, Int, v)
        else
            @assert false
        end
    end
    ImageCubes(nam, loc, _common_chan_info(; common...), cubsiz)
end

