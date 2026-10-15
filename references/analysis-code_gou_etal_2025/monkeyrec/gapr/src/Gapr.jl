module Gapr

export swcplot, swcplot!
function swcplot end
function swcplot! end

include("swc.jl")
include("repo.jl")
include("cat.jl")

# C interfaces
#include("edge-model.jl")

end # module Gapr
