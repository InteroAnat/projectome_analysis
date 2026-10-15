using Gapr
using Test

@testset "swc" begin
    swcstr = raw"""
1 0 1 1 1 0.1 -1
2 0 2 2 2 0.2 1
3 0 3 3 3 0.3 2
4 0 4 4 4 0.4 3
5 0 5 5 5 0.5 3
6 0 6 6 6 0.6 5
"""
    swc = load_swc(IOBuffer(swcstr))
    saveandread(swc) = begin
        io = IOBuffer()
        Gapr.save_swc(io, swc)
        seekstart(io)
        Gapr.load_swc(io)
    end
    @test collect(swc.id) == 1:6
    @test collect(swc.parind) == [0, 1, 2, 3, 3, 5]
    f(n) = n.id in (2, 3, 4, 5)
    @test collect(filter(f, swc).id) == [1, 2, 3, 4, 5]
    @test collect(filter(f, swc; toterm=true).id) == 1:6
    @test collect(filter(f, swc; toroot=false).id) == [2, 3, 4, 5]
    @test collect(filter(f, swc; toroot=false, toterm=true).id) == [2, 3, 4, 5, 6]
    @test collect(saveandread(filter(f, swc; toroot=false)).parind) == [0, 1, 2, 2]
    @test collect(saveandread(filter(f, swc; toroot=false, toterm=true)).parind) == [0, 1, 2, 2, 4]
end
