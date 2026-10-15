# import Gapr_jll
# 
# const libgapr = Gapr_jll.libgaprcore
# 
# struct Repo end
# struct NodeData
#     id::Int64
#     pos::NTuple{3,Cdouble}
#     typ::Cint
#     rad::Cfloat
#     cov::Cuchar
# end
# struct LinkData
#     id::Int64
#     id2::Int64
# end
# 
# function openrepo(path::AbstractString)
#     @ccall libgapr.gapr_jl_model_create(path::Cstring)::Ptr{Repo}
# end
# 
# function closerepo(repo::Ptr{Repo})
#     @ccall libgapr.gapr_jl_model_destroy(repo::Ptr{Repo})::Cvoid
# end
# 
# function openrepo(f::Function, path::AbstractString)
#     repo = openrepo(path)
#     try
#         f(repo)
#     finally
#         closerepo(repo)
#     end
# end
# 
# function prepare_jump(repo::Ptr{Repo}, num::Integer)
#     @ccall libgapr.gapr_jl_model_prepare(repo::Ptr{Repo}, num::UInt64)::UInt64
# end
# 
# function jumpto(repo::Ptr{Repo}, num::Integer)
#     @ccall libgapr.gapr_jl_model_jump(repo::Ptr{Repo}, num::UInt64)::Cvoid
# end
# 
# function getnodes(repo::Ptr{Repo})
#     siz = Ref{Csize_t}()
#     ptr = @ccall libgapr.gapr_jl_model_nodes(repo::Ptr{Repo}, siz::Ref{Csize_t})::Ptr{NodeData}
#     Base.unsafe_wrap(Array, ptr, (siz[],))
# end
# 
# function getlinks(repo::Ptr{Repo})
#     siz = Ref{Csize_t}()
#     ptr = @ccall libgapr.gapr_jl_model_links(repo::Ptr{Repo}, siz::Ref{Csize_t})::Ptr{LinkData}
#     Base.unsafe_wrap(Array, ptr, (siz[],))
# end
# 
# 
# 
# 
