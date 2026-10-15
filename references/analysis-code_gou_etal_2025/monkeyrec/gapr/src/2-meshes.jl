Make.@target meshes_from_allen () -> begin
    areas=[1097,693,997]
    function renorm(msh)
        msh=triangle_mesh(msh)
        y,z=GeometryBasics.pop_pointmeta(msh, :normals)
        y=normal_mesh(y)
    end
    function repos(msh)
        for (i,p) in pairs(msh.position)
            msh.position[i]=(p[3], p[2], p[1])
        end
        msh
    end
    function invnorm(msh)
        for (i,n) in pairs(msh.normals)
            msh.normals[i]=-n
        end
        msh
    end
    @showprogress "load allen objs" map(areas) do a
        objstr=Pipe()
        cmd=run(`curl -sf -Y1000 -y10 http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/$a.obj`, devnull, objstr, stderr, wait=false)
        close(objstr.in)
        o=load(Stream{format"OBJ"}(objstr))
        success(cmd) || error("err get allen obj")

        o=o |> repos |> renorm |> invnorm
        fn="extra/allen-ccf-$a.obj"
        save(fn, o)
        a=>fn
    end
end

Make.@target meshes_from_allen_xform (meshes_from_allen) -> begin
    meshes=deepcopy(meshes_from_allen)
    meshes=map(meshes) do (a,fn); a=>load(fn); end
    dats = [
            17391, 18133, 18708, 18709, 18925, 18926,
            192095, 192096, 192100, 192101, 192102,
            202813, 202814, 202815, 202816,
           ]
    function xform(pts, ds)
        pin=Pipe()
        pout=Pipe()
        cmd=run(`bash misc/antsxform-rev.sh $ds`, pin, pout, stderr, wait=false)
        close(pout.in)
        close(pin.out)
        tsk=@async begin
            for pt in pts
                x,y,z=pt
                println(pin, "$x $y $z")
            end
            close(pin.in)
        end
        res=[]
        for l in eachline(pout)
            f=split(l, " ")
            @assert length(f)==3
            x,y,z=map(v->parse(Float64, v), f)
            push!(res, (x,y,z))
        end
        @assert length(res)==length(pts)
        fetch(tsk)
        success(cmd) || error("failed to run xform-rev.sh")
        res
    end
	
    for d in dats
        points=[]
        for (a, m) in meshes
            @assert size(m.position)==size(m.normals)
            for (i, p) in pairs(m.position)
                push!(points, p)
                push!(points, p+100*m.normals[i])
            end
        end
        new_pts=xform(points, d)
        idx=1
        for (a, mm) in meshes
            m=deepcopy(mm)
            for (i, p) in pairs(m.position)
                p1=new_pts[idx]
                p2=new_pts[idx+1]
                idx+=2
                m.position[i]=p1
                n=p2 .-p1
                n=n./norm(n)
                m.normals[i]=n
            end
            save("extra/allen-ccfx-$a.$d.obj", m)
        end
    end
end
        #######
        ######################################
        ######################################
#=
#
extra/allen-ccfx-997.18133.obj
extra/allen-ccfx-997.18708.obj
extra/allen-ccfx-997.18709.obj
extra/allen-ccfx-997.18925.obj
extra/allen-ccfx-997.18926.obj
extra/allen-ccfx-997.192095.obj
extra/allen-ccfx-997.192096.obj
extra/allen-ccfx-997.192100.obj
extra/allen-ccfx-997.192101.obj
extra/allen-ccfx-997.192102.obj
extra/allen-ccfx-997.202813.obj
extra/allen-ccfx-997.202814.obj
extra/allen-ccfx-997.202815.obj
extra/allen-ccfx-997.202816.obj

# j j j j
extra/allen-ccf-%.obj:

# j j j j
extra/allen-ccfx-%.obj: extra/allen-ccf-997.obj extra/allen-ccf-1097.obj
=#

