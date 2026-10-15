module My


struct ProjHelper
    dat::Dict{String,Tuple{Float64,Float64}}
    function ProjHelper()
        new(Dict())
    end
end
Base.getindex(h::ProjHelper, id::String) = Base.getindex(h, id, :all)
function Base.getindex(h::ProjHelper, id::String, s::Symbol)
    v = get(h.dat, id, (0.0, 0.0))
    s != :left || return v[1]
    s != :right || return v[2]
    s != :all || return v[2] + v[1]
    s != :both || return v
    error("err $s")
end

end







function foreach_ds(f::Function)
    for ds in datasets()
        for stg in sort(collect(keys(ds.stgs)))
            f(ds.id, stg)
        end
    end
end

function check_dir(func, outdir)
    outdir = deepcopy(outdir)
    if !isdir(outdir)
        rm("$outdir.temp", force=true, recursive=true)
        mkdir("$outdir.temp")
        func("$outdir.temp")
        mv("$outdir.temp", outdir)
    end
    outdir
end



