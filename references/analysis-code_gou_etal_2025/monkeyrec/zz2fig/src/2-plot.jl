function set_width!(ax; force=false)
    return
    # 180 mm 510 pt 7.08 inch Nature
    # (467.71725, 566.93): Cell
    # (510, ): Nature
    !(ax.width[] isa Makie.Auto) && !force && return
    ax.width[] = pt(510)
end

set_width() = (;) # (; width=pt(510))
set_font_h1x() = (; font=:bold, fontsize=pt(8))
set_font_h1() = (; fontsize=pt(8))
set_font_h2x() = (; font=:bold, fontsize=pt(7))
set_font_h2() = (; fontsize=pt(7))

function relax_clipping(ax, lrbt)
    l, r, b, t = lrbt
    get_vp() = begin
        yy = ax.scene.viewport[]
        nw = yy.widths .- ((l, t) .+ (r, b))
        y = Makie.Rect2f(yy.origin .- (-l, -b), nw)
        y, yy
    end
    update_ps(ax31r, yy) = begin
        ax31r.scene.camera.pixel_space.val = [2/yy.widths[1] 0.0 0.0 -1.0+0.0; 0.0 2/yy.widths[2] 0.0 -1.0; 0.0 0.0 -0.0001 -0.0; 0.0 0.0 0.0 1.0]
    end
    blksig = Ref(0)
    projmod = Ref{Any}()
    Makie.on(ax.scene.camera.projectionview; priority=-1000) do vp
        tt = vp * Vec4f(0, 0, 0, 1)
        @warn "origin: $tt"
    end
    Makie.onany(ax.scene.camera.view, ax.scene.camera.projection; priority=-1000) do vmat, proj
        bs = blksig[]
        @warn "view proj $bs: $vmat $proj "
        blksig[] += 1
        y, yy = get_vp()
        u, v = y.widths ./ yy.widths
        prevproj, chkproj = if isassigned(projmod)
            if projmod[].first == proj
                projmod[]
            elseif projmod[].second == proj
                projmod[]
            else
                proj, nothing
            end
        else
            proj, nothing
        end
        proj = Makie.inv(prevproj)
        ss = proj * Vec4f(-1, -1, 0, 1)
        tt = proj * Vec4f(1, 1, 0, 1)
        #nextproj = prevproj * [u 0.0 0.0 (l*tt[1]/tt[4]+r*ss[1]/ss[4])/yy.widths[1]+0; 0.0 v 0.0 (tt[2]*b+t*ss[2])/yy.widths[2]; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
        nextproj = [(y.widths[1])/yy.widths[1] 0 0 (l-r)/yy.widths[1]; 0 (y.widths[2])/yy.widths[2] 0 (b-t)/yy.widths[2]; 0 0 1 0; 0 0 0 1] * prevproj
        #nextproj = # * prevproj
        proj = Makie.inv(nextproj)
        sss = proj * Vec4f(-1, -1, 0, 1)
        ttt = proj * Vec4f(1, 1, 0, 1)
        ttt = proj * Vec4f(1, 1, 0, 1)
        @info "$u/$v $y $yy\n$ss $tt\n$sss $ttt\n$prevproj\n$nextproj"
        if chkproj != nextproj || bs == 0
            projmod[] = prevproj => nextproj
            ax.scene.camera.projection[] = nextproj
            #ax.scene.camera.projectionview[] = nextproj * vmat
        end
        ###ax.scene.camera.view[] = [u 0.0 0.0 (l*tt[1]+r*ss[1])/yy.widths[1]; 0.0 v 0.0 (tt[2]*b+t*ss[2])/yy.widths[2]; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
        blksig[] -= 1
    end
    Makie.on(ax.scene.camera.pixel_space; priority=-1000) do ps
        y, yy = get_vp()
        update_ps(ax, yy)
    end
    avoidvp = Ref{Any}()
    Makie.onany(ax.scene.viewport, ax.scene.camera.resolution; update=true, priority=-1000) do y, _
        isassigned(avoidvp) && avoidvp[] == y && return
        @info "vp $y"
        nw = y.widths .+ (l, t) .+ (r, b)
        yy = Makie.Rect2f(y.origin .+ (-l, -b), nw)
        avoidvp[] = yy
        ax.scene.viewport[] = yy
        ax.scene.camera.resolution[] = yy.widths
        update_ps(ax, yy)
        notify(ax.scene.camera.projection)
    end
end


function align_axis_label(xy::Symbol, axvec...; fix=pt(0))
    ticksize_obs(a) = begin
        getproperty(a, Symbol(xy, :ticksize)), getproperty(a, Symbol(xy, :ticksvisible))
    end
    ticksize(a) = begin
        s, t = ticksize_obs(a)
        s[] * t[]
    end
    labelpad(a) = getproperty(a, Symbol(xy, :labelpadding))
    ticklabelspace(a) = begin
        getproperty(a, Symbol(xy, :ticklabelspace))
    end
    space = maximum(axvec) do a
        s1 = (x=tight_xticklabel_spacing!, y=tight_yticklabel_spacing!)[xy](a)
        @info "ttls $s1"
        s1 + ticksize(a)
    end
    for a in axvec
        ticklabelspace(a)[] = space - ticksize(a) + fix
    end
end

function theme_cell()
    # ga: 288pt 8pt-12pt
    a4 = (595, 842)
    sz = (493, 595)
    pad = (a4 .- sz) ./ 2
    Makie.Theme(
        size=a4,
        figure_padding=(pad[1], pad..., pad[2]),
        fontsize=pt(6),
        rowgap=pt(8 + 4),
        colgap=pt(8 + 4),
        backgroundcolor=:white,
        font=:regular,
        fonts=(;
            regular=to_font("Arial Regular"),
            bold=to_font("Arial Bold"),
            italic=to_font("Arial Italic"),
            bolditalic=to_font("Arial Bold Italic"),
        ),
        CairoMakie=(;
            pt_per_unit=1.0,
            px_per_unit=6.0,
            antialias=:best,
        ),
    )
    #pt-8pt
end

function theme_cell_ga()
end

function theme_custom_core()
    colormap = :devon
    colormap = :GnBu
    #colormap=:viridis,
    colors = Makie.Palette(:tab10).colors

    t = Makie.Theme(
        palette=(;
            color=colors,
            patchcolor=map(c -> Makie.to_color((c, 0.8)), colors),
        ),
        #padding = Vec3f(0.05),
        #figure_padding = 16,
        linewidth=pt(0.5),
        colormap=colormap,
        textcolor=:black,
        Heatmap=(;
            colormap,
            # axis=(;
            #     xticksvisible=false,
            #     yticksvisible=false,
            # ),
        ),
        Axis=(;
            xticklabelsize=pt(6),
            yticklabelsize=pt(6),
            xlabelsize=pt(7),
            ylabelsize=pt(7),
            xgridwidth=pt(0.5),
            ygridwidth=pt(0.5),
            spinewidth=pt(1.0),
            titlesize=pt(7),
            xtickwidth=pt(0.7),
            ytickwidth=pt(0.7),
            xticksize=pt(2.5),
            yticksize=pt(2.5),
            xminortickwidth=pt(0.5),
            yminortickwidth=pt(0.5),
            xminorticksize=pt(1.5),
            yminorticksize=pt(1.5),
            xticklabelpad=pt(1),
            yticklabelpad=pt(1),
            xlabelpadding=pt(2),
            ylabelpadding=pt(2),
            xgridvisible=false,
            ygridvisible=false,
            xticksvisible=true,
            yticksvisible=true,
            xminorticksvisible=true,
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(5),
            yminorticks=IntervalsBetween(5),
            bottomspinevisible=true,
            leftspinevisible=true,
            rightspinevisible=false,
            topspinevisible=false,
        ),
        Colorbar=(;
            size=pt(6),
            ticklabelsize=pt(6),
            labelsize=pt(6),
            labelfont=:bold,
            ticklabelpad=pt(0.5),
            labelpadding=pt(1),
        ),
        Legend=(;
            labelsize=pt(6),
            markersize=pt(6),
            titlesize=pt(8),
            titlegap=pt(2),
            patchsize=(pt(5), pt(5)),
            patchlabelgap=pt(2),
            colgap=pt(6),
            rowgap=pt(1),
            groupgap=pt(10),
            titlehalign=:left,
            gridhalign=:left,
            framewidth=pt(0.6),
            backgroundcolor=(:gray, 0.3),
            padding=pt.((2, 2, 2, 2)),
        ),
        Bracket=(;
            width=pt(3),
            style=:square,
        ),
        Scatter=(;
            markersize=pt(1.5),
        ),
        Box=(;
            strokewidth=pt(0.25),
            color=:transparent,
            strokecolor=(:red, 0.5),
            cornerradius=0,
        ),
        BarPlot=(;
            label_offset=pt(2),
        ),
        Mesh=(;
            inspectable=false,
        ),
    )
    merge!(t, Makie.theme_light())
    #t.Heatmap.
    # :Blues
    # :Dark2_7
    # :Paired_10
    # :RdYlBu
    #Makie.set_theme!(CairoMakie=(screen_config...,)).
end

function theme_custom()
    t = Theme(
        CairoMakie=(;
            antialias=:best,
            pt_per_unit=1.0,
            px_per_unit=6.0,
        ),
        GLMakie=(;
            antialias=:best,
            pt_per_unit=1.0,
            px_per_unit=6.0,
        ),
        RPRMakie=(;
            iterations=100,
            plugin=RPRMakie.Northstar,
            #max_recursion=10,
        ),
    )
    merge!(t, theme_custom_core())
end

function theme_custom_gl()
    t = Theme(
        CairoMakie=(;
            antialias=:best,
            pt_per_unit=1.0,
            px_per_unit=1.0,
        ),
        GLMakie=(;
            antialias=:best,
            pt_per_unit=1.0,
            px_per_unit=1.0,
        ),
        RPRMakie=(;
            iterations=100,
            plugin=RPRMakie.Northstar,
            #max_recursion=10,
        ),
    )
    merge!(t, theme_custom_core())
end

function calculate_rgba(rgb1, rgb2, rgba_bg)::GLMakie.RGBAf
    rgb1 == rgb2 && return GLMakie.RGBAf(rgb1.r, rgb1.g, rgb1.b, 1)
    c1 = Float64.((rgb1.r, rgb1.g, rgb1.b))
    c2 = Float64.((rgb2.r, rgb2.g, rgb2.b))
    alphas_fg = 1 .+ c1 .- c2
    alpha_fg = clamp(sum(alphas_fg) / 3, 0, 1)
    alpha_fg == 0 && return rgba_bg
    rgb_fg = clamp.((c1 ./ alpha_fg), 0, 1)
    rgb_bg = Float64.((rgba_bg.r, rgba_bg.g, rgba_bg.b))
    alpha_final = alpha_fg + (1 - alpha_fg) * rgba_bg.alpha
    rgb_final = @. 1 / alpha_final * (alpha_fg * rgb_fg + (1 - alpha_fg) * rgba_bg.alpha * rgb_bg)
    return GLMakie.RGBAf(rgb_final..., alpha_final)
end

function alpha_colorbuffer(figure::GLMakie.Figure)
    alpha_colorbuffer(figure.scene)
end
function alpha_colorbuffer(scene::GLMakie.AbstractScene)
    bg = scene.backgroundcolor[]
    scene.backgroundcolor[] = GLMakie.RGBAf(0, 0, 0, 1)
    b1 = copy(GLMakie.colorbuffer(scene))
    scene.backgroundcolor[] = GLMakie.RGBAf(1, 1, 1, 1)
    b2 = GLMakie.colorbuffer(scene)
    scene.backgroundcolor[] = bg
    global xxximg = deepcopy((b1, b2))
    return map(b1, b2) do b1, b2
        calculate_rgba(b1, b2, bg)
    end
end

function alpha_colorbuffer(scene::GLMakie.AbstractScene, scr)
    bg = scene.backgroundcolor[]
    scene.backgroundcolor[] = GLMakie.RGBAf(0, 0, 0, 1)
    b1 = copy(GLMakie.colorbuffer(scr))
    scene.backgroundcolor[] = GLMakie.RGBAf(1, 1, 1, 1)
    b2 = GLMakie.colorbuffer(scr)
    scene.backgroundcolor[] = bg
    global xxximg = deepcopy((b1, b2))
    return map(b1, b2) do b1, b2
        calculate_rgba(b1, b2, bg)
    end
end

function is_gen_pdf()
    try
        global gen_pdf
        if gen_pdf
            return true
        end
    catch
    end
    false
end
function pt(x::Number)
    is_gen_pdf() ? x : x * 3
end

function prep_mixed_scene(scene::Makie.AbstractScene, backend::Module; bg=nothing, rprscale=6, rpriterations=600, rprcontour=nothing, mixfunc=nothing)
    @assert getproperty(backend, :activate!) isa Function
    newviewport = map(scene.viewport) do rect
        Makie.Rect(rect.origin .* 0, rect.widths .* (backend === RPRMakie ? rprscale : 1))
    end
    if bg === nothing
        bg = scene.backgroundcolor[]
    end
    scene2 = if is_gen_pdf()
        Makie.Scene(
            viewport=newviewport,
            clear=true,
            backgroundcolor=bg,
            camera=Makie.camera(scene),
            camera_controls=Makie.cameracontrols(scene),
        )
    else
        scene
    end
    rprscreen = nothing
    if backend === RPRMakie
        dumb = RPRMakie.RPR.Context(; singleton=false)
        RPRMakie.RPR.release(dumb)
        rprscreen = RPRMakie.Screen(scene2; iterations=rpriterations)
        RPRMakie.RPR.ACTIVE_CONTEXT[] = dumb
    end
    Makie.lines!(scene, Vec3f[])
    (; scene, scene2, backend, rprscreen, kwargs=(; rprscale, rpriterations, rprcontour, mixfunc))
end
function prep_mixed_scene(axis::Makie.AbstractAxis, backend::Module; kwargs...)
    (; axis, prep_mixed_scene(axis.scene, backend; kwargs...)...)
end
# function prep_mixed_scene(backend::Module; kwargs...)
#     (scenelike) -> prep_mixed_scene(scenelike, backend; kwargs...)
# end

function rotflipfb(A)
    ind1, ind2 = axes(A)
    B = similar(A, (ind2, ind1))
    m = first(ind1) + last(ind1)
    for i = ind1, j = axes(A, 2)
        B[j, i] = A[i, j]
    end
    return B
end
function fini_mixed_scene_rpr(scene::Makie.AbstractScene, screen::RPRMakie.Screen, scene2::Makie.AbstractScene; rprscale, rpriterations, rprcontour, mixfunc)
    isempty(scene2.plots) && isempty(scene2.children) && return
    img = if rprcontour !== nothing
        contcol, contwid = if rprcontour isa Tuple
            rprcontour
        elseif rprcontour isa Symbol
            (rprcontour, 1.0f0)
        elseif rprcontour isa Real
            (:black, rprcontour)
        else
            @assert "Unknown rprcontour $(rprcontour)"
        end
        contcol = Makie.to_color(contcol)
        contour = zz1vis.showcontour(screen; linewidth=contwid * rprscale)
        w = Makie.to_color(:white)
        map(contour.color) do col
            Makie.RGBAf(contcol.r, contcol.g, contcol.b, sum((w.r - col.r, w.g - col.g, w.b - col.b)) / 3)
        end
    else
        img = zz1vis.mycolorbuffer(screen, withalpha=false)
        img.color
    end
    # XXX: img = zz1vis.mixcontour(img, contour)
    plt = image!(Makie.camrelative(scene), (0, 1), (1, 0), rotflipfb(img),)
    translate!(plt, 0, 0, -1)
end
function fini_mixed_scene(scene::Makie.AbstractScene, scene2::Makie.AbstractScene, backend::Module; mixfunc=nothing, kwargs...)
    isempty(scene2.plots) && isempty(scene2.children) && return
    scr = display(scene2, backend=backend, update=true, visible=false)
    img = alpha_colorbuffer(scene2, scr) |> rotflipfb
    if mixfunc === nothing
        plt = image!(Makie.camrelative(scene), (0, 1), (1, 0), img,)
        Makie.translate!(plt, 0, 0, -1)
    else
        mixfunc(Makie.camrelative(scene), img)
    end
end
function fini_mixed_scene(tuple::NamedTuple)
    if (pars = get(tuple, :cam, nothing)) !== nothing
        cam = Makie.cameracontrols(tuple.scene)
        for (k, v) in pairs(pars)
            getproperty(cam, k)[] = v
        end
        for scene in (tuple.scene, tuple.scene2)
            GLMakie.update_cam!(scene)
        end
    end
    tuple.backend === RPRMakie && return fini_mixed_scene_rpr(tuple.scene, tuple.rprscreen, tuple.scene2; tuple.kwargs...)
    fini_mixed_scene(tuple.scene, tuple.scene2, tuple.backend; tuple.kwargs...)
end

const _mixed_scenes_to_finish = NamedTuple[]
function with_mixed_scene(f::Function, axis_scene, backend::Module; kwargs...)
    tup = prep_mixed_scene(axis_scene, backend; kwargs...)
    if !(cameracontrols(axis_scene.scene) isa Makie.EmptyCamera)
        try
            GLMakie.update_cam!(axis_scene.scene)
        catch
        end
    end
    f(tup)
    cam = Makie.cameracontrols(tup.scene)
    camsetup = if cam isa Makie.Camera3D
        map((:eyeposition, :lookat, :upvector, :fov, :near, :far,)) do k
            k => getproperty(cam, k)[]
        end |> NamedTuple
    else
        nothing
    end
    push!(_mixed_scenes_to_finish, (; tup..., cam=camsetup))
end

function fini_mixed_scene_finalize()
    for tup in _mixed_scenes_to_finish
        if tup.backend === RPRMakie
            RPRMakie.RPR.release(tup.rprscreen.context)
        end
    end
    empty!(_mixed_scenes_to_finish)
end
function fini_mixed_scene()
    for tup in _mixed_scenes_to_finish
        is_gen_pdf() && fini_mixed_scene(tup)
    end
end

function get_camera_config_impl(cfg::NamedTuple)
    cfg
end
function get_camera_config_impl(cfg::Symbol)
    (
        h=(cent=:nmt, fov=9, deye=(0, 0, 5.0f5), up=(0, 1, 0)),
        hr=(cent=:nmt, fov=7, deye=(0, 0, 5.0f5), up=(1, 0, 0)),
        s=(cent=:nmt, fov=6, deye=(-5.0f5, 0, 0), up=(0, 0, 1)),
        c=(cent=:nmt, fov=6, deye=(0, 5.0f5, 0), up=(0, 0, 1)),
        hf=(fov=6.5, cent=:nmt, dcent=Vec3f(0, 20000, 0), deye=(0.0, -0.0, 1.0) .* 3.5f5, up=(0.0, 1.0, 0.0)),
        cf=(cent=:nmt, fov=6.5, dcent=Vec3f(0, 0, 10000), deye=(0.0, 1.0, 0.0) .* 3.5f5, up=(0, 0, 1)),
        xfl=(fov=6.5, cent=(34987.2, 49823.91, 24563.49), deye=(-0.5914952971239986, 0.53277545536815, 0.6052137041057921) .* 321509.49782211584, up=(0.45, -0.41, 0.8)),
        xf=(fov=6.5, cent=(36467.8, 49915.13, 22495.43), deye=(-0.37583546762456954, 0.5263208397134711, 0.7627149368923672) .* 365102.2636774412, up=(0.44, -0.62, 0.65)),
        hfrot=(cent=:nmt, fov=6.5, dcent=Vec3f(0, 15000, 0), deye=(0, 0, 5.0f5), up=(1, 0, 0)),
        hfl=(fov=9.0, cent=(21852.78, 58703.21, 25753.98), deye=(-0.0, 0.0, 1.0) .* 250000.28, up=(0.0, 1.0, 0.0)),
        intrinsic1=(fov=9, cent=(25289.0, 58107.0, 37150.0), deye=(0, 0, 1.0f5), up=(0, 1, 0)),
        intrinsic2=(fov=9, cent=(24195.205777069255, 53235.89927725902, 42279.898715883544), deye=(12875.584069595236, 67806.40874910026, -9037.35474454359), up=(-0.3777043956197972, 0.1924356999477527, 0.9057085021766518)),
        acgg1=(fov=9, cent=(32118.075023747708, 62697.610315434584, 34359.31056430382), deye=(13378.096846640456, -131656.696628869, 54902.89811658879), up=(0, 0, 1),),
        ofc1=(fov=9, cent=(22248.605629016514, 63591.86930451797, 33432.04329240536), deye=(-123298.81199313546, 114364.28670526658, 75974.1379135841), up=(0.30184785646449513, -0.2799752425838243, 0.9113186792159341),),
        ofc2=(fov=9.02, cent=(34198.23, 63026.41, 32688.21), deye=(-6005.48, -233921.67, -84277.86), up=(0.06, -0.34, 0.94)),
        allenh=(cent=:allen, fov=9, deye=(0, -1.0f5, 0), up=(-1, 0, 0)),
        allenhr=(cent=:allen, fov=7, deye=(0, -1.0f5, 0), up=(0, 0, 1)),
        allens=(cent=:allen, fov=6, deye=(0, 0, -1.0f5), up=(0, -1, 0)),
        allenc=(cent=:allen, fov=6),
        csoma=(fov=6.0, cent=(21000.00, 60000.0, 33000.0), deye=(0.00, 1, 0.0) .* 250000, up=(0.0, 0.0, 1.0)),
        ssoma=(fov=6.0, cent=(21000.00, 60000.0, 33000.0), deye=(-1, 0, 0) .* 250000, up=(0.0, 0.0, 1.0)),
        str1=(fov=9.0, cent=(32156.35, 49857.11, 27464.94), deye=(0, 0, 1) .* 210000.0, up=(0.0, 1.0, 0.0)),
        strtopoleft=(fov=9.0, cent=(26353.81, 49027.59, 30036.0), deye=(-0.288456217633492, -0.40198377102426347, 0.8690236235809081) .* 106818.91433226095, up=(-0.12, 0.92, 0.38)),
        strpuproj=(fov=9.0, cent=(25000.97, 48947.65, 28125.39), deye=(0.39413251888090817, -0.059619279137182686, -0.9171178218286641) .* 141706.67814617982, up=(0.1, 0.99, -0.02)),
        ptthalpons=(fov=9.0, cent=(27866.6, 33975.2, 17487.4), deye=(-0.978538919981206, -0.142756169337744, -0.14860100335471632) .* 183069.3561002567, up=(0, 0, 1)),
        zoominhpt=(fov=9.0, cent=(32002.41, 34197.74, 15454.73), deye=(0.0005136558953086275, -0.03700356726995618, 0.9993150014719678) .* 190730.80031746786, up=(-0.01, 1.0, 0.04)),
        zoominspt=(cent=:nmt, fov=6, deye=(-2.0f5, 0, 0), up=(0, 0, 1)),
        zoominctptthal=(fov=9.0, cent=(33846.24, 37318.71, 21891.73), deye=(-0.9988208751475038, -0.04388096768516392, 0.020768246064322336) .* 121553.83715030018, up=(0.02, 0.0, 1.0)),
        demoparitrunk=(fov=9.0, cent=(33772.01, 47157.67, 32256.75), deye=(-1.0, 0.0, 0.0) .* 302369.569957107, up=(0, 0, 1)),
        demotrunkshort=(fov=9.0, cent=(15897.74, 53422.28, 32722.62), deye=(0.29227122490108987, -0.6864123054487984, 0.6658946448375164) .* 30129.66211760132, up=(-0.26, 0.61, 0.75)),
    )[cfg]
end

function get_camera_config(cfg)
    cfg = get_camera_config_impl(cfg)
    cent = get(cfg, :cent, nothing)
    if cent isa Symbol
        cent = (; allen=Allen.center, nmt=NmtDat.center,)[cent]
        cfg = (; cfg..., cent=cent())
    end
    use = get(cfg, :use, nothing)
    if use !== nothing
        cfg = (; get_camera_config(use)..., cfg..., use=nothing)
    end
    patch = get(cfg, :patch, nothing)
    if patch !== nothing
        cfg = (; cfg..., patch=nothing, patch(cfg)...)
    end
    cfg
end

function setup_camera_impl(ax, cfg)
    cfg = get_camera_config(cfg)
    cent = cfg[:cent]
    cent = cent .+ get(cfg, :dcent, Vec3f(0))
    cam = Makie.cameracontrols(ax)
    cam.lookat[] = cent
    cam.lookat[] = cent
    cam.fov[] = cfg.fov
    cam.eyeposition[] = cent .+ cfg.deye
    cam.upvector[] = cfg.up
end
function setup_camera(axpairs...; link=true)
    for (ax, v) in axpairs
        setup_camera_impl(ax, v)
    end
    link && linkfov!(first.(axpairs)...)
end

function inspect_cameras(fig)
    print_camera(cam::Makie.Camera3D) = begin
        getval(v::AbstractVector) = (round.(v, digits=2)...,)
        getval(v::AbstractFloat) = round(v, digits=2)
        cent = getval(cam.lookat[])
        deye = getval(cam.eyeposition[] .- cent)
        deye = "$(deye./norm(deye)) .* $(norm(deye))"
        up = getval(cam.upvector[])
        fov = getval(cam.fov[])
        for (i, (k, v)) in enumerate(pairs((; fov, cent, deye, up)))
            print(i == 1 ? "(" : ", ")
            print("$k=$v")
        end
        println(")")
        #:far
        #:near
    end
    print_camera(cam) = begin
        #Main.xxxcam = deepcopy(cam)
        @warn "$(typeof(cam))"
    end
    Makie.on(Makie.events(fig).keyboardbutton, priority=2) do event
        if event.action == Makie.Keyboard.release && event.key == Makie.Keyboard.p
            for ax in fig.content
                ax isa LScene || continue
                cam = cameracontrols(ax)
                if cam.selected[]
                    print_camera(Makie.cameracontrols(ax))
                end
            end
            return Makie.Consume(true)
        end
        return Makie.Consume(false)
    end
end

calc_fov2(fov1, h1, h2) = begin
    asind(h2 * sind(fov1) / h1)
end
function linkfov!(ax1, ax2...)
    ax1.height[] === nothing && return
    cam1 = Makie.cameracontrols(ax1)
    tmp = map(cam1.fov, ax1.height) do fov1, h1
        sind(fov1) / h1
    end
    for ax in ax2
        cam2 = Makie.cameracontrols(ax)
        Makie.onany(tmp, ax.height; update=true) do t, h2
            if h2 !== nothing
                cam2.fov[] = asind(h2 * t)
            end
        end
    end
end

cls_temp_color(df; clscol=:cls) = begin
    subtyps = sort(unique(df[:, clscol]))
    f(c::Number) = Makie.wong_colors()[findfirst(==(c), subtyps)]
    f(r::DataFrameRow) = f(r[clscol])
    f
end

function isdebuglayout()
    try
        something(Main.xxxdebuglayout)
        return true
    catch
        return false
    end
end
function debug_layout(fig)
    dbg = isdebuglayout()
    todo = Set()
    function recurse_layout(layo)
        for c in layo.content
            if c.content isa Makie.GridLayout
                push!(todo, c.content => (1, :gray))
                recurse_layout(c.content)
            elseif c.content isa MakieTeX.LTeX
                push!(todo, c.content => (100, :green))
            elseif c.content isa Makie.AbstractAxis
                push!(todo, c.content => (10, :red))
            elseif c.content isa Makie.Legend
                push!(todo, c.content => (20, :green))
            elseif c.content isa Makie.Box
            elseif c.content isa Makie.Block
                push!(todo, c.content => (2, :blue))
            else
            end
        end
    end
    recurse_layout(fig.layout)
    for (g, (z, c)) in todo
        !dbg && continue
        GLB = Makie.GridLayoutBase
        o = GLB.layoutobservables(g)
        p = lines!(fig.scene, o.computedbbox, color=(c, 0.3), linewidth=0.3)
        translate!(p, 0, 0, z)
    end
    rct = map(theme(:size), theme(:figure_padding)) do (w, h), (l, r, b, t)
        tt = h - t
        bb = b
        ll = l
        rr = w - r
        Point2f[
            (0, bb), (w, bb),
            (0, tt), (w, tt),
            (ll, 0), (ll, h),
            (rr, 0), (rr, h),
        ]
    end
    p = linesegments!(fig.scene, rct, color=:red, linewidth=0.3)
    translate!(p, 0, 0, 2)
end

function focus!(x::Symbol)
    global zzzfocus = x
end
function focus!()
    global zzzfocus = nothing
end
function isfocus(x::Symbol)
    f = try
        zzzfocus
    catch
        nothing
    end
    f === nothing && return true
    f === x && return true
    false
end

function plot_all_grids(fig, x)
    ddd = "output/testout$x"
    mkpath(ddd)
    alllbls = []
    allcont = []
    func(g) = begin
        for c in g.content
            c = c.content
            if c isa GridLayout
                func(c)
            elseif c isa Label
                push!(alllbls, c)
            else
                push!(allcont, c)
            end
        end
    end
    func(fig.layout)
    imgpth = joinpath(ddd, "test.png")
    save(imgpth, fig, update=false, px_per_unit=3)
    img = load(imgpth)
    bbox(c) = c.layoutobservables.computedbbox[]
    spans(bb) = begin
        (bb.origin..., (bb.origin + bb.widths)...)
    end
    do_with(ci, bb) = begin
        a, b, c, d = round.(Int, 3 .* spans(bb))
        b, d = size(img, 1) .- (d, b)
        save(joinpath(ddd, "cc_$ci.png"), img[min(b, end):min(d, end), min(a, end):min(c, end)])
    end
    done = Set()
    for l in alllbls
        bb = bbox(l)
        p = bb.origin + bb.widths
        ci = findmin(c -> (bb2 = bbox(c); norm(bb2.origin + bb2.widths .* (0, 1) - p)), allcont)[2]
        bb2 = bbox(allcont[ci])
        a, b, c, d = min.(spans(bb), spans(bb2))
        a2, b2, c2, d2 = max.(spans(bb), spans(bb2))
        do_with(ci, Rect2f(a, b, c2 - a, d2 - b))
        push!(done, ci)
    end
    for (ci, c) in enumerate(allcont)
        ci in done && continue
        bb = bbox(c)
        do_with(ci, bb)
    end
end

function areaname(h::Symbol; kw...)
    areaname(string(h); kw...)
end
function areaname(h; capital=true)
    n = replace(h, r"^.*\|" => "")
    if startswith("area_")(n)
        return replace(n, r"^area_" => (capital ? "Area " : "area "))
    end
    if startswith("areas_")(n)
        return replace(n, r"^areas_" => (capital ? "Areas " : "areas "))
    end
    n
end
function areashort(h)
    n = replace(h, r"^.*\|" => "")
    startswith("area_")(n) && return n[6:end]
    startswith("areas_")(n) && return n[7:end]
    n
end
function compact_str(v::AbstractFloat; sigdigits=3)
    buf = IOBuffer()
    print(IOContext(buf, :compact => true), round(v; sigdigits))
    String(take!(buf))
end
function compact_pval(p::AbstractFloat; ns="(ns)")
    if p < 0.001
        "***"
    elseif p < 0.01
        "**"
    elseif p < 0.05
        "*"
    else
        ns
        #@assert false
    end
end
function meanandsem(x, unit=nothing)
    x = filter(v -> !isnan(v) && !isinf(v), x)
    m = mean(x)
    sd = std(x)
    sem = sd / sqrt(length(x))
    n = length(x)
    m = round(m, sigdigits=3)
    sem = compact_str(sem; sigdigits=2)
    u = unit === nothing ? "" : " $unit"
    "$m ± $sem$u, SEM, n = $n"
end

function venn_test()
    vals = rand(Random.MersenneTwister(0), Bool, 3, 1000)
    vals[1, 1:100] .= 0
    vals[2, 300:500] .= 0
    vals[3, 600:900] .= 0
    df = DataFrame(:a => vals[1, :], :b => vals[2, :], :c => vals[3, :])
    fig = Figure()
    test_venn!(Axis(fig[1, 1]), df, :a => :red, :b => :green, :c => :blue, scale=5)
    fig
end

import Graphs
function slink_dendro_pairs(h, dm)
    links = Pair{Int,Int}[]
    interms = Vector{Int}[]
    get_set(a) = a < 0 ? [-a] : interms[a]
    g = Graphs.SimpleGraph(length(h.order))
    valid = in(h.order[1:11])
    h75 = quantile(h.heights, 0.75)
    foreach(eachrow(h.merges), h.heights) do (a, b), h
        aa = get_set(a)
        bb = get_set(b)
        push!(interms, union(aa, bb))
        hh, ij = findmin(dm[aa, bb])
        @assert abs(hh - h) < 0.001
        a, b = aa[ij[1]] => bb[ij[2]]
        #valid(a) && valid(b) && push!(links, a => b)
        hh <= h75 && push!(links, a => b)
    end
    return links
    for (a, b) in links
        Graphs.add_edge!(g, a, b)
    end
    # for u in 1:length(h.order)
    # for v in 1:length(h.order)
    #     e=Graphs.all_simple_paths(g, u,v)
    # end
    # end
    ccc = Graphs.johnson_shortest_paths(g)
    pred = map(ccc.dists, dm) do m, n
        m > 1e9 && return -Inf
        n / (sign(m + 1))
    end
    cc = findmax(pred)[2]
    a, b = cc[1], cc[2]
    p = Int[]
    while b != 0
        push!(p, b)
        b = ccc.parents[a, b]
    end
    @warn "cc $cc"
    links
end

function draw_scale_bar!(ax, len, refpts::AbstractMesh; kw...)
    draw_scale_bar!(ax, len, collect(Point3f.(refpts.position)); kw...)
end

function cacl_scale_bar_pos(c, len, ref, offset, a0, b0)
    pt2st(pt) = [pt[1:3]...]
    st2pt(x) = SVector(x...)

    aaaref = Makie.project(c, :data, :pixel, ref)
    f1(x) = begin
        a = st2pt(x)
        aaa = Makie.project(c, :data, :pixel, a)
        norm(aaa[1:2] .- offset)^2 + norm(aaa[3] .- aaaref[3])^2
    end
    opt = Optim.optimize(f1, fill(-Inf, 2), fill(Inf, 2), pt2st(@something(a0, ref)), Optim.NelderMead())
    @warn "opt $opt"
    a = st2pt(Optim.minimizer(opt))

    noneg(x) = x >= 0 ? 0 : x^2
    aa = Makie.project(c, :data, :relative, a)
    f2(x) = begin
        b = st2pt(x)
        bb = Makie.project(c, :data, :relative, b)
        l = 0.1 * abs(norm(a - b) - len)^2 + 10000 * norm(aa[2:3] - bb[2:3])^2 + 10000000 * noneg(bb[1] - aa[1] - 0.01) + 10000000 * norm(aa[3] - bb[3])^2
        l
    end
    opt = Optim.optimize(f2, fill(-Inf, 2), fill(Inf, 2), pt2st(@something(b0, ref)), Optim.NelderMead())
    opt = Optim.minimizer(opt)
    @warn "opt $opt"
    b = st2pt(opt)
    a, b
end
function draw_scale_bar!(ax, len, refpts; nolabel=false, offset=(3, 6))
    c = Makie.camera(ax)
    ref = SVector(median(reduce(hcat, refpts), dims=2)...)
    a0, b0 = nothing, nothing
    bar = Point3f[ref, ref]
    Makie.on(c.projectionview, update=true) do _
        a0, b0 = cacl_scale_bar_pos(c, len, ref, offset, a0, b0)
        aa = Makie.project(c, :data, :pixel, a0)
        bb = Makie.project(c, :data, :pixel, b0)
        @warn "$aa -> $bb"
        bar[1:2] .= a0, b0
    end

    Makie.lines!(ax, bar, color=:black, linewidth=2)
    !nolabel && text!(ax, bar, text=["", " $(len)μm"]; align=(:left, :center))
end

calc_xrng(pars, y) = begin
    y2, x2 = max((pars.y1, pars.x1), (pars.y2, pars.x2))
    y1, x1 = min((pars.y1, pars.x1), (pars.y2, pars.x2))
    d2 = ((x2 - x1)^2 + (y2 - y1)^2) / 4
    r2 = (d2 + sqrt(d2^2 + 4 * pars.r^4)) / 2
    if y > y2
        y2 = 2y - y2
    end
    if y < y1
        y1 = 2y - y1
    end
    (y2 - y1)^2 + (x2 - x1)^2 > 4 * r2 && return nothing

    x0 = if y2 - y1 > 0.001
        (y - y1) / (y2 - y1) * (x2 - x1) + x1
    else
        (x2 + x1) / 2
    end
    det2(x) = begin
        a2 = (x - x1)^2 + (y - y1)^2
        b2 = (x - x2)^2 + (y - y2)^2
        a2 + b2 + 2 * sqrt(a2 * b2)
    end
    det2(x0) == 4 * r2 && return x0, x0
    narrowdown(ll, rr, condrr) = begin
        d2ll = det2(ll)
        d2rr = det2(rr)
        @assert sign(d2ll - 4 * r2) == -condrr "d2ll $d2ll $(4*r2) -1"
        @assert sign(d2rr - 4 * r2) == condrr "d2rr $d2rr $(4*r2) +1 $condrr"
        @assert rr > ll
        while rr - ll > 0.0005
            m = (rr + ll) / 2
            c = sign(det2(m) - 4 * r2)
            c == condrr && (rr = m)
            c == -condrr && (ll = m)
            c == 0 && return m
        end
        (rr + ll) / 2
    end
    a = narrowdown(x0 - 2 * sqrt(r2) - 0.001, x0, -1)
    b = narrowdown(x0, x0 + 2 * sqrt(r2) + 0.001, +1)
    a, b
end

import NearestNeighbors
function test_venn!(ax, df, specs...; tagname=nothing, scale=1.0, patcolor=nothing)
    #specs=sort(specs, by=p->sum(df[:, p.first]),rev=true)
    dat = reduce(hcat, [df[:, s] for s in first.(specs)])'
    lbls = string.(first.(specs))
    #dat = hcat([true, true, false, false,false,], [false, true, true, true, true])'

    N = size(dat, 1)

    rads = sqrt.(vec(sum(dat, dims=2)))
    rads = rads ./ norm(rads)
    prec = 300

    pars2vec(pars) = begin
        v1 = reduce(vcat, (SVector(p.y1, p.x1) for p in pars))
        v2 = reduce(vcat, (SVector(p.y2, p.x2) for p in pars))
        vcat(v1, v2)[4:end]
    end
    vec2pars(v; trimed=true) = begin
        v = if trimed
            vcat([0, 0, 0,], vec(v))
        else
            v[1:3] .= 0
            vec(v)
        end
        v = reshape(v, 2, :, 2)
        map(axes(v, 2)) do j
            (; x1=v[2, j, 1], y1=v[1, j, 1], x2=v[2, j, 2], y2=v[1, j, 2], r=rads[j])
        end
    end

    gen_set(par) = begin
        s = Dict{Int,UnitRange{Int}}()
        y1 = min(par.y1, par.y2)
        y2 = max(par.y1, par.y2)
        for yy in round(Int, y1 * prec - par.r * prec):round(Int, y2 * prec + par.r * prec)
            xrng = calc_xrng(par, yy / prec)
            xrng === nothing && continue
            ###@warn "$yy"
            ab = round(Int, xrng[1] * prec):round(Int, xrng[2] * prec)
            s[yy] = ab
        end
        s
    end

    set_manip(a, b, op) = begin
        c = Dict{Int,AbstractVector{Int}}()
        for (k, v) in pairs(a)
            v = op(v, get(b, k, 0:-1))
            if !isempty(v)
                c[k] = v
            end
        end
        c
    end
    set_and(::Nothing, b::Dict) = b
    set_and(a::Dict, b::Dict) = set_manip(a, b, intersect)
    set_not(a::Dict, b::Dict) = set_manip(a, b, setdiff)
    set_size(a::Dict) = begin
        n = 0
        for (k, v) in pairs(a)
            n += length(v)
        end
        n
    end

    showset(s) = begin
        pts = reduce(vcat, [[(j, i) for j in jj] for (i, jj) in s])
        scatter(Point2f.(pts)) |> display
    end
    # s1=gen_set((; x1=2, y1=2, x2=0, y2=1, r=1))
    #     s1s=set_size(s1)
    #     showset(s1)
    #     @warn "s1 $s1s"
    #     return


    patcnt = Dict{SVector,Int}()
    for c in eachcol(dat)
        k = SVector(c...)
        patcnt[k] = get(patcnt, k, 0) + 1
    end
    patcnt = collect(patcnt)
    @warn "patcnt: $patcnt"
    filter!(p -> sum(p.first) > 0, patcnt)

    calcsets(pars) = begin
        sss = map(pars) do par
            gen_set(par)
        end
        map(patcnt) do (pat, cnt)
            foreach(sss) do s
                #@warn "ppp $(set_size(s))"
            end
            s = nothing
            #@warn "asdf $pat"
            for (i, b) in enumerate(pat)
                #@warn "asdf /$i/ $pat"
                if b
                    s = set_and(s, sss[i])
                end
                #@warn "asdf $i/ $pat"
            end
            s === nothing && return s
            for (i, b) in enumerate(pat)
                if !b
                    s = set_not(s, sss[i])
                end
            end
            #@warn "asdf $pat $(set_size(s))"
            s
        end
    end
    fff(pars) = begin
        sss = calcsets(pars)
        map(patcnt, sss) do (pat, cnt), s
            sum(pat) == 0 && return 0 => 0
            set_size(s) => cnt
        end
    end
    fimp(x, ddd) = begin
        x = vec2pars(x)
        vals = fff(x)
        a = first.(vals)
        b = last.(vals)
        #yield()
        aa = a ./ sum(a)
        bb = b ./ sum(b)
        o = sum(abs.(aa .- bb))
        o2 = ddd * sum(map(par -> ((par.x1 - par.x2)^2 + (par.y1 - par.y2)^2), x))
        @warn " $o $o2 $x $rads $aa $bb"
        o + o2
    end
    f1(x) = begin
        fimp(x, 0.001)
    end
    f2(x) = begin
        fimp(x, 0.01)
    end

    mmm = zz1data.cached_result(:asdf36, dat) do _
        Random.seed!(0)
        x0 = zeros(Float32, 2, N)
        while true
            vals = fff(vec2pars(hcat(x0, x0); trimed=false))
            bad = false
            for (a, b) in vals
                b == 0 && continue
                a < 0.03 && (bad = true)
            end
            !bad && break
            @warn "bad $x0"
            x0 = rand(Float32, 2, N) .- 0.5f0
        end
        x0 = pars2vec(vec2pars(hcat(x0, x0); trimed=false))
        opt0 = Optim.optimize(f1, vec(Float32.(fill(-0.5, 2, N * 2))[4:end]), vec(Float32.(fill(0.5, 2, N * 2))[4:end]), x0, Optim.NelderMead())
        opt = Optim.optimize(f2, vec(Float32.(fill(-0.5, 2, N * 2))[4:end]), vec(Float32.(fill(0.5, 2, N * 2))[4:end]), Optim.minimizer(opt0), Optim.NelderMead())
        vec2pars(Optim.minimizer(opt))
    end

    prec = round(prec / scale)
    rads .= rads .* scale
    mmm .= vec2pars(pars2vec(mmm) .* scale)
    sss = calcsets(mmm)
    cents = map(sss, patcnt) do s, (pat, _)
        s === nothing && return Vec2f(NaN)
        pts = [map(j -> SVector(j, i), jj) for (i, jj) in s]
        isempty(pts) && return Vec2f(NaN)
        pts = reduce(vcat, pts)
        isempty(pts) && return Vec2f(NaN)
        pts = reduce(hcat, pts) ./ prec

        iii = argmax(pat)
        s0 = gen_set((; mmm[iii]..., r=mmm[iii].r * 1.1))
        s0 = set_not(s0, s)
        pts0 = [map(j -> SVector(j, i), jj) for (i, jj) in s0]
        pts0 = reduce(vcat, pts0)
        pts0 = reduce(hcat, pts0) ./ prec
        tree = NearestNeighbors.KDTree(pts0)
        k = argmax(first.(NearestNeighbors.knn(tree, pts, 1)[2]))
        c = Vec2f(pts[:, k]...)
        @warn "cent $c"
        c
    end

    if patcolor === nothing
        patcolor = pat -> begin
            (sum(pat) == 1 ? first : :black) => :transparent
        end
    end
    ccc = last.(specs)
    decodecolor(pat) = begin
        c = patcolor(pat)
        if !(c isa Pair)
            c = c => :transparent
        end
        getc(c) = begin
            c isa Function && return c(ccc[pat])
            c
        end
        (; text=getc(c[1]), fill=getc(c[2]))
    end

    io = IOBuffer()
    strpt(x, y) = "($x cm, $y cm) circle(0pt) "
    strcent(i) = "($(cents[i][1])cm, $(cents[i][2])cm)"
    strcirc(x, y, xr, yr, rot) = "($x cm, $y cm) circle [x radius=$xr cm, y radius=$yr cm, rotate=$rot]"
    writepathcirc(cmd, i) = begin
        par = mmm[i]
        x = (par.x1 + par.x2) / 2
        y = (par.y1 + par.y2) / 2
        d2 = ((par.x2 - par.x1)^2 + (par.y2 - par.y1)^2) / 4
        r2 = (d2 + sqrt(d2^2 + 4 * par.r^4)) / 2
        r = sqrt(r2)
        yr = sqrt(r2 - d2)
        dd = atand(par.y2 - par.y1, par.x2 - par.x1)
        print(io, "\\path [$cmd]",
            strpt(par.x1, par.y1), strpt(par.x2, par.y2),
            strcirc(x, y, r, yr, dd),
            ";\n")
    end
    writefill(pat) = begin
        c = decodecolor(pat).fill
        c === :transparent && return ""
        nclip = 1
        print(io, "\\begin{scope}\n")
        for (i, cc) in enumerate(ccc)
            !pat[i] && continue
            if nclip < sum(pat)
                writepathcirc("clip", i)
                nclip += 1
                continue
            end
            writepathcirc("fill=$c", i)
        end
        print(io, "\\end{scope}\n")
    end
    writecnt(i, pat, cnt) = begin
        col = decodecolor(pat).text
        print(io, "\\node [color=$col,font={\\fontsize{6}{0}\\selectfont\\sffamily}] at ",
            strcent(i), "{$cnt}; \n")
    end
    writetikz() = begin
        print(io, "\\begin{tikzpicture}\n")
        for (pat, _) in sort(patcnt, by=pc -> sum(first(pc)))
            writefill(pat)
        end
        for (i, cc) in enumerate(ccc)
            writepathcirc("draw=$cc", i)
        end
        for (i, (pat, cnt)) in enumerate(patcnt)
            writecnt(i, pat, cnt)
        end
        print(io, "\\end{tikzpicture}\n")
    end
    writetikz()
    teximg!(ax, compile_tikz_str(String(take!(io))))
    @warn "$patcnt $mmm $rads"
end

import MathTeXEngine
function add_arial_font()
    get_tex_font(fn) = read(`kpsewhich $fn`, String) |> rstrip
    get_fc_font(pat) = read(`fc-match -f'%{file}\n' $pat`, String) |> rstrip
    push!(MathTeXEngine.default_font_families, "Arial" => MathTeXEngine.FontFamily(Dict(
        :regular => get_fc_font("Arial"),
        :italic => get_fc_font("Arial:italic"),
        :bold => get_fc_font("Arial:bold"),
        :bolditalic => get_fc_font("Arial:italic:bold"),
        :math => get_tex_font("FiraMath-Regular.otf"),
    )))
end
function del_arial_font()
    pop!(MathTeXEngine.default_font_families, "Arial", nothing)
end

const _fig_cur_figure = Pair{Ref{Char},Dict{Char,AbstractString}}[]
_fig_cur_fignum = Ref{Union{Number,Symbol}}()
function with_figure(f::Function, num, output::AbstractString, env=nothing; kwargs...)
    try
        GLMakie.closeall()
    catch
    end
    _fig_cur_fignum[] = num
    @info "start figure $num"
    push!(_fig_cur_figure, Ref('a') => Dict{Char,AbstractString}())
    global gen_pdf = true
    try
        add_arial_font()
        with_theme(merge!(theme_cell(), theme_custom())) do
            fig = Figure(; kwargs...)
            set_width!(fig.layout)
            if length(_fig_cur_figure) == 1
                global xxxfig = fig
                global xxxenv = env
            end
            @time "plot" f(fig, env)
            strs = map(sort(collect(_fig_cur_figure[end].second))) do (c, s)
                if c == '0'
                    num_s = replace("$num", r"S0+" => "S")
                    LaTeXString("Figure $num_s")
                else
                    LaTeXString("\\textbf{$c}, $s")
                end
            end
            if !isempty(strs) && false
                capt = LaTeXString(string("\\sffamily ", join(strs[1:min(1, end)], " ")))
                if length(_fig_cur_figure) == 1
                    global xxxcapt = String(capt)
                end
                @info "capt $capt"
                LTeX(fig[end+1, :], TEXDocument(capt, true, preamble=read("figs/preamble.tex", String)))
            end
            #Label(fig[end+1, :], capt, fontsize=10, tellwidth=false)
            ensure_width(fig)
            Makie.resize_to_layout!(fig)
            debug_layout(fig)
            @time "fini scenes" fini_mixed_scene()
            save(output, fig, update=false, backend=CairoMakie)
        end
    finally
        pop!(_fig_cur_figure)
        fini_mixed_scene_finalize()
        @info "finish figure $num"
        gen_pdf = false
        del_arial_font()
        _fig_cur_fignum[] = 0
    end
end
function with_subfig(f::Function, g, env, args...; kwargs...)
    cnt = _fig_cur_figure[end].first
    lbl = cnt[] = uppercase(cnt[])
    try
        skip = try
            !(lbl in Main.xxxsubfigs)
        catch
            false
        end
        if !skip
            grid = GridLayout(g, args...; valign=:top, halign=:left, kwargs...)
            let gg = grid.parent
                while gg isa GridLayout
                    gg.valign[] = :top
                    gg.halign[] = :left
                    gg = gg.parent
                end
            end
            gridx = GridLayout(grid[1, 1])
            gridx.alignmode[] = Outside()
            r = @time "plot $lbl" f(gridx, env)
            lblpad = map(p -> (0, p.left, p.top, 0), gridx.layoutobservables.protrusions)
            lbl = Label(grid[1, 1, TopLeft()], "$lbl", halign=:right, valign=:bottom, tellwidth=false, tellheight=false, padding=lblpad; set_font_h1x()...)
            # on(
            #     lbl.padding = 
            #     # if p.left > pt(10)
            #     #     lbl.halign = :left
            #     # end
            #     # if p.top > pt(10)
            #     #     lbl.valign = :top
            #     # end
            # end
            r
        end
    finally
        cnt[] = lowercase(cnt[] + 1)
    end
end
function with_caption(str::AbstractString)
    isempty(_fig_cur_figure) && return
    cnt, dict = _fig_cur_figure[end]
    dict[(lbl = cnt[]; islowercase(lbl) ? '0' : lbl)] = str
end
function with_caption(str::LaTeXString)
    str = String(str)
    if count(==('$'), str) == 2 && str[1] == '$' && str[end] == '$'
        str = str[2:end-1]
    end
    with_caption(str)
end

function plot_figure(f::Function, env, args...;)
    try
        GLMakie.closeall()
    catch
    end
    with_theme(theme_custom_gl()) do
        push!(_fig_cur_figure, Ref('a') => Dict{Char,AbstractString}())
        try
            add_arial_font()
            fig = Makie.Figure()
            grid = GridLayout(fig[1, 1];)
            @time "plot" f(grid, env, args...; env...)
            set_width!(grid)
            Box(fig[1, 1], color=:transparent, strokecolor=:red)
            Makie.resize_to_layout!(fig)
            debug_layout(fig)
            fini_mixed_scene()
            Makie.DataInspector()
            inspect_cameras(fig)
            fig
        finally
            pop!(_fig_cur_figure)
            fini_mixed_scene_finalize()
            del_arial_font()
        end
    end
end

function ensure_width(fig)
    for r in 1:20
        cc = contents(fig[r, 1])
        if length(cc) == 1 && cc[1] isa GridLayout
            set_width!(content(fig[r, 1]))
            if isdebuglayout()
                Makie.Box(fig[r, 1])
            end
        end
    end
end

import Revise
function repeat_pdf(fff, file, out="test.pdf")
    global xxxstop = Ref(false)
    func() = begin
        while !xxxstop[]
            Revise.retry()
            plot_figure_pdf(fff, xxxenv, out)
            w = Revise.FileWatching.watch_file(file, 600)
            while !ispath(file)
                sleep(0.2)
            end
            @info "got $w"
        end
        @warn "Stop"
    end
    schedule(Task(func))
end

function plot_figure_pdf(f::Function, env, out="test.pdf")
    try
        GLMakie.closeall()
    catch
    end
    out = joinpath("output", out)
    global gen_pdf = true
    push!(_fig_cur_figure, Ref('a') => Dict{Char,AbstractString}())
    try
        add_arial_font()
        with_theme(merge!(theme_cell(), theme_custom())) do
            fig = Figure(;)
            set_width!(fig.layout)
            ff = GridLayout(fig[1, 1])
            @time "plot" invokelatest(f, ff, env; env...)
            ensure_width(fig)
            Makie.resize_to_layout!(fig)
            debug_layout(fig)
            @time "fini scenes" fini_mixed_scene()
            save(out, fig, update=false, backend=CairoMakie)
        end
    catch e
        e isa InterruptException && rethrow(e)
        Base.showerror(stderr, e, Base.catch_backtrace())
        @warn "ploterr $e"
    finally
        pop!(_fig_cur_figure)
        fini_mixed_scene_finalize()
        gen_pdf = false
        del_arial_font()
    end
end

function inspect_nmt_hier()
    nmt = NmtDat.load_images()
    getter = NmtDat.getter(nmt)
    (p, i, x) -> begin
        h = getter(reshape(x, :, 1))[2][1]
        "$x $h"
    end
end
function inspect_allen_hier()
    nmt = Allen.load_images()
    getter = Allen.getter(nmt)
    (p, i, x) -> begin
        h = getter(reshape(x, :, 1))[2][1]
        "$h"
    end
end

parseaxspec((ax, cam)::Pair) = begin
    ax, gl, rprscreen = if ax isa NamedTuple
        ax.axis, ax.scene2, get(ax, :rprscreen, nothing)
    else
        ax, ax.scene, nothing
    end
    (; ax, gl, cam, rprscreen)
end
parseaxspec(ax) = begin
    parseaxspec(ax => nothing)
end

function load_swc_as_linesx(fn::AbstractString; mirrorby=nothing, xformby=nothing)::AbstractVector
    swc = Gapr.load_swc(fn)
    if mirrorby !== nothing
        if !mirrorby.isleft(swc[1].pos)
            for n in swc.nodes
                n.pos = mirrorby.mirror(n.pos)
            end
        end
    elseif xformby !== nothing
        for n in swc.nodes
            n.pos = xformby(n)
        end
    end
    push!(reduce(vcat, [push!([n.pos for n in seg], Point3f(NaN)) for seg in Gapr.eachsegment(swc)]), swc[1].pos)
end

function load_swc_as_linesx(f::Function, swc::AbstractString)
    load_swc_as_linesx(f, Gapr.load_swc(swc))
end

function load_swc_as_linesx(f::Function, swc::Gapr.SwcTable)
    decode_col(c) = begin
        Makie.to_color(c)
    end
    decode_col_lw(v::Pair) = begin
        decode_col(v[1]), v[2]
    end
    decode_col_lw(v) = begin
        decode_col(v), 0.3
    end
    ptscollw = reduce(vcat, [push!(Tuple{Any,Any,Any}[(n.pos, decode_col_lw(f(n))...) for n in seg], (Point3f(NaN), missing, NaN)) for seg in Gapr.eachsegment(swc)])
    lws = last.(ptscollw)
    cols = [v[2] for v in ptscollw]
    T = eltype(skipmissing(cols))
    cols = if T <: Real
        collect(Float32, map(c -> Float32(c === missing ? NaN : c), cols))
    else
        map(c -> c === missing ? :transparent : c, cols)
    end
    pts = map(Point3f ∘ first, ptscollw)
    pts, map(Makie.to_color, cols), lws
end



function rotatexlabel!(ax, rot=π / 3)
    ax.xticklabelalign[] = (:left, :center)
    ax.xticklabelrotation[] = -rot
end

function rotatexlabelalt!(ax, rot=π / 3)
    ax.xticklabelalign[] = (:right, :top)
    ax.xticklabelrotation[] = rot
end

function rotatexlabeltop!(ax)
    ax.xticklabelalign[] = (:left, :center)
    ax.xticklabelrotation[] = π / 6
end

get_bilevel_img(i, bg=nothing) = begin
    bg = if bg === nothing
        cnt = Dict{Any,Int}()
        for v in i
            cnt[v] = get(cnt, v, 0) + 1
        end
        sort(collect(cnt), by=last)[end].first
    else
        to_color(bg)
    end
    map(v -> norm(v - bg), i)
end
function mixfunc_contour(cfgs, cols)
    (s, i) -> begin
        for (cfg, tgt) in zip(cfgs, cols)
            plt = Makie.contour!(s, range(0, 1, length=size(i, 1)), range(1, 0, length=size(i, 2)), -get_bilevel_img(i, tgt) .+ 0.1, labels=get(cfg, :text, nothing) !== nothing, levels=[0.000], color=get(cfg, :color, :black), linewidth=0.5, linestyle=:solid, labelsize=5, labelformatter=(x) -> string(get(cfg, :text, nothing)))
            Makie.translate!(plt, 0, 0, 2)
        end
    end
end
function mixfunc_contour_behind(cfg)
    (s, i) -> begin
        #plt = Makie.contour!(s, range(0, 1, length=size(i, 1)), range(1, 0, length=size(i, 2)), get_bilevel_img(i, nothing) .- 0.01, labels=get(cfg, :text, nothing) !== nothing, levels=[0.000], color=get(cfg, :color, :black), linewidth=0.5, linestyle=:dash, labelsize=5, labelformatter=(x) -> string(get(cfg, :text, nothing)))
        #Makie.translate!(plt, 0, 0, 2)
    end
end

function draw_contours_impl(objs::AbstractVector{<:Pair{<:AbstractMesh,<:NamedTuple}}, axspecs::Makie.AbstractAxis...)
    cols = [:red, :blue, :green, :cyan, :magenta, :white]
    for ax in axspecs
        mixfunc = mixfunc_contour(last.(objs), cols)
        with_mixed_scene(ax, GLMakie; mixfunc) do mix
            for (k, (obj, _)) in enumerate(objs)
                Makie.mesh!(mix.scene2, obj, color=cols[k], shading=Makie.NoShading)
            end
        end
        for (k, (obj, cfg)) in enumerate(objs)
            with_mixed_scene(ax, GLMakie; mixfunc=mixfunc_contour_behind(cfg)) do mix
                Makie.mesh!(mix.scene2, obj, color=cols[k], shading=Makie.NoShading)
            end
        end
    end
end
function draw_contours(objs, axspecs...)
    objs = parse_objs_spec(objs)
    if !is_gen_pdf()
        for (obj, cfg) in objs
            ccc = Makie.to_color(get(cfg, :color, :white))
            color, alpha = Makie.RGBf(ccc), ccc.alpha
            for ax in axspecs
                #Makie.mesh!(ax, obj, color=(color, 0.1), transparency=true, inspectable=false)
            end
        end
        return
    end
    objs = map(vec(objs)) do obj
        obj, cfg = obj isa Union{Pair,Tuple} ? obj : obj => (;)
        if obj isa AbstractString
            obj = load(obj)
        end
        @assert obj isa AbstractMesh
        cfg = if cfg isa AbstractString
            (; text=cfg)
        elseif cfg isa Number
            (; linewidth=cfg)
        elseif cfg isa ColorTypes.Colorant
            (; color=cfg)
        elseif cfg isa Symbol
            (; color=cfg)
        elseif cfg isa NamedTuple
            cfg
        else
            @assert false
        end
        obj => cfg
    end
    draw_contours_impl(objs, axspecs...)
end

function mixfunc_overlay(color, alpha)
    iii(i) = begin
        cnt = Dict{Any,Int}()
        for v in i
            cnt[v] = get(cnt, v, 0) + 1
        end
        vv = sort(collect(cnt), by=last)[end].first
        ii = Images.imfilter(i, Images.Kernel.gaussian((3, 3)))
        mina = 0.03
        global xxximg = map(i, ii) do v, iv
            v0 = Makie.RGBf(v)
            # fga=(1 - norm(v - color))
            # fgb=min(iv.alpha * 5, 1)
            bga = (norm(v - vv) * 1.0)^2
            #bga=1
            fga = (norm(v - vv) * 1.0)^2
            fgb = 1
            Makie.RGBAf(v0, min(bga + mina, 1 - fga * fgb + mina, v.alpha))
        end
    end
    (s, i) -> begin
        #plt = Makie.contour!(s, range(0, 1, length=size(i, 1)), range(0, 1, length=size(i, 2)), get_bilevel_img(i), labels=false, levels=[0.000], color=:black, linewidth=0.3, labelsize=5, labelformatter=(x) -> string("asdf"))
        #Makie.translate!(plt, 0, 0, 2)
        plt = Makie.image!(s, (0, 1), (1, 0), iii(i); alpha, transparency=true)
        Makie.translate!(plt, 0, 0, 2)
    end
end
function parse_objs_spec(objs)
    objs = map(vec(objs)) do obj
        obj, cfg = obj isa Union{Pair,Tuple} ? obj : obj => (;)
        if obj isa AbstractString
            obj = load(obj)
        end
        @assert obj isa AbstractMesh
        cfg = if cfg isa AbstractString
            (; text=cfg)
        elseif cfg isa Number
            (; linewidth=cfg)
        elseif cfg isa ColorTypes.Colorant
            (; color=cfg)
        elseif cfg isa Symbol
            (; color=cfg)
        elseif cfg isa Tuple{Symbol,<:Number}
            (; color=cfg)
        elseif cfg isa NamedTuple
            cfg
        else
            @assert false "$(typeof(cfg))"
        end
        obj => cfg
    end
end

function draw_mesh_overlay(objs, axspecs...)
    #=
    #
    HSVA-> value
    =#
    objs = parse_objs_spec(objs)
    if !is_gen_pdf()
        for (obj, cfg) in objs
            ccc = Makie.to_color(get(cfg, :color, :white))
            color, alpha = Makie.RGBf(ccc), ccc.alpha
            for ax in axspecs
                Makie.mesh!(ax, obj, color=(color, 0.1), transparency=true, inspectable=false)
            end
        end
        return
    end
    for ax in axspecs
        mixfunc = mixfunc_overlay(0, 1)
        with_mixed_scene(ax, GLMakie; mixfunc, bg=:white) do mix
            for (obj, cfg) in objs
                ccc = Makie.to_color(get(cfg, :color, :lightgray))
                color, alpha = Makie.RGBf(ccc), ccc.alpha
                Makie.mesh!(mix.scene2, obj; color)
            end
        end
    end
end

function draw_neurons_rawxyz!(axspec, df, rawinfo)
    spec = parseaxspec(axspec)
    #mesh!(spec.gl, obj, color=(:gray, 0.05), transparency=true, fxaa=true)
    if !isfocus(:rawg97)
        df = shuffle(df)[1:min(10, end), :]
    end
    for r in eachrow(df)
        swc = Gapr.load_swc(r.path)
        zz1data.set_pos_rawxyz!(swc, rawinfo(r.sample, r.channel))
        pts, _ = load_swc_as_linesx(n -> :red, swc)
        Makie.lines!(spec.gl, pts, transparency=true, linewidth=0.3)
    end
end

function draw_nmr_data(site, val, axspec...; obj, sitecolor=:red)
    specs = parseaxspec.(axspec)
    for spec in specs
        Makie.mesh!(spec.gl, obj, color=val)
        Makie.meshscatter!(spec.gl, [site], markersize=1200, color=sitecolor)
    end
end


function draw_brain_bricks(mix, df, obj, rawinfo, sampinfo)
    !isfocus(:brainbricks) && return
    mix = parseaxspec(mix)
    matsys = mix.rprscreen.matsys

    thinglass = zz1vis.glass(matsys;
        refraction_color=Vec4f(1.0f0),
        refraction_roughness=Vec4f(0.0000001),
        refraction_caustics=true,
        refraction_ior=1.01f0,
        reflection_ior=1.30f0,
        reflection_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
        refraction_absorption_color=Vec4f(0.999999, 0.999999, 0.999999, 0.999999),
        refraction_absorption_distance=Vec4f(5000.0f0),
        coating_color=Vec4f(0.9f0, 0.9f0, 0.9f0, 1),
        coating_weight=0.7f0,
        coating_ior=1.9f0,
        coating_roughness=0.5f0,
        coating_mode=RPRMakie.RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR,
        coating_metalness=0.3f0,
        coating_thickness=0.1f0,
    )
    objx = GeometryBasics.normal_mesh(map(p -> p ./ 1.0f5, obj.position), GeometryBasics.faces(obj))
    zz1vis.meshvis!(
        mix.gl,
        objx,
        color=nothing,
        material=thinglass,
        #primary_only_flag=true,
        shadow=false,
        transparent=false,
    )
    colors = Makie.wong_colors()

    www = Makie.to_color(:white)
    for (i, r) in enumerate(eachrow(df))
        #@assert i <= length(colors)
        while i > length(colors)
            i -= length(colors)
        end
        colorsi = colors[i]
        swc = Gapr.load_swc(r.path)
        zz1data.set_pos_rawxyz!(swc, rawinfo(r.sample, r.channel))
        mat() = zz1vis.plastic(matsys,
            reflection_roughness=Vec4f(0.7), reflection_weight=Vec4f(0.3),
            # emission_weight=Vec3f(0.9),
            # emission_color=colors[i],
        )
        plt = zz1vis.fibers!(mix.gl, swc, color=colorsi, material=mat(), linewidth=30.0f0 * 1.0f3)
        Makie.scale!(plt, fill(1.0f-5, 3)...)
        # pts = reduce(vcat, [push!([n.pos ./ 1.0f4 for n in seg], Vec3f(NaN)) for seg in Gapr.eachsegment(swc)])
        # zz1vis.lines!(mix.scene2, pts, color=colors[i], linewidth=2.0f0)
        mat2 = RPRMakie.RPR.Chrome(matsys, color=colorsi)
        Makie.meshscatter!(mix.gl, [swc[1].pos ./ 1.0f5], color=0.7 * colorsi + 0.3www, material=mat(), markersize=0.003f0)
    end
    scale_cam(cam) = begin
        (; fov=cam.fov, cent=cam.cent ./ 1.0f5, up=cam.up, deye=cam.deye ./ 1.0f5, dcent=cam.dcent ./ 1.0f5)
    end
    setup_camera(mix.ax => scale_cam(mix.cam))
    #########
    #cam = Makie.cameracontrols(mix.axis)
    #cam.lookat[] = sampinfo.center ./ 1.0f5
    #cam.fov[] = 13
    #cam.upvector[] = normalize(Vec3f(-1, 0, -0.1))
    #cam.eyeposition[] = sampinfo.center ./ 1.0f5 .+ normalize(Vec3f(-0.7, 0.5, 0.2)) .* 3

    radiance = 700
    lights = [
        Makie.EnvironmentLight(1.0, load(RPRMakie.RPR.assetpath("studio026.exr"))),
        Makie.PointLight(Vec3f(-10, 10, 10), Makie.RGBf(radiance, radiance, radiance * 1.1)),
        Makie.PointLight(Vec3f(-10, 8, 11), Makie.RGBf(radiance, radiance, radiance * 1.1)),
        Makie.PointLight(Vec3f(-10, 2, -8), Makie.RGBf(radiance, radiance, radiance * 1.1)),
    ]
    #append!(mix.scene2.lights, lights)
    empty!(mix.gl.lights)
    push!(mix.gl.lights, Makie.AmbientLight(Makie.to_color(:white) * 0.7f0))
    zz1vis.spherelights!(mix.rprscreen, [
            normalize(Vec3f(-0.9, 0.6, 0.4)) .* 7 => 1.9,
            normalize(Vec3f(-0.9, -0.6, -0.4)) .* 7 => 1.9,
        ], intensity=5.7, center=sampinfo.center ./ 1.0f5)
    #push!(mix.scene2.lights, Makie.PointLight(Makie.RGBf(1, 1, 1) .* 1.0f2, sampinfo.center ./ 1.0f5.+Vec3f(-3,3,3)))
end

function draw_arbor_dist_along!(ax, df, arbors, dir; scene=nothing, rowcolor, rowlabel)
    arbors = leftjoin(arbors, df, on=:path) |> dropmissing!
    filter!(r -> contains("Temporal")(r.arborhier), arbors)
    arbors.path = arbors.arborpath
    segs = zz1data.load_swcs_as_segs(arbors.arborpath, arbors.soma_pos)
    dats = []
    l2c = Dict()
    for s in segs
        c = rowlabel(arbors[s.nrn, :])
        cc = rowcolor(arbors[s.nrn, :])
        l2c[c] = Makie.RGBAf(Makie.RGBf(cc), 0.7)
        for i in 1:ceil(Int, s.len)
            push!(dats, (c, s.pos))
        end
    end
    nodepos = reduce(hcat, last.(dats))

    # argmax(v->StatsBase.corspearman(first.(x), (v'*reduce(hcat, last.(x)))[:]), eachrow(rand(1000,3)))
    x = first.(dats), vec(reshape(dir, 1, :) * nodepos) ./ 1000
    kkk = sort(unique(x[1]))
    vals1, vals2 = [x[2][x[1].==kk] for kk in kkk]
    global xxxvals1 = deepcopy(vals1)
    global xxxvals2 = deepcopy(vals2)
    t = HypothesisTests.MannWhitneyUTest(vals1, vals2)
    pval = HypothesisTests.pvalue(t)
    Makie.density!(ax, vals1, color=l2c[kkk[1]], label="vlPFC", strokewidth=pt(0.5), direction=:y)
    Makie.density!(ax, vals2, color=l2c[kkk[2]], label="dlPFC", strokewidth=pt(0.5), direction=:y)
    #ax.title[] = "p=$(pval) ?!"
    if pval < 0.05
        text!(ax, 1, 0, space=:relative, text=compact_pval(pval), align=(:right, :bottom), offset=(-2, 2))
    end
    ax.xlabel = "Density"
    xlims!(ax, 0, nothing)
    if scene !== nothing
        v1 = SVector(median(nodepos, dims=2)...)
        lines!(scene, [v1 - dir * 5000, v1 + dir * 5000], color=:red, linewidth=1.3)
        v2 = v1 + dir * 5000
        arrows!(scene, ([v2[d]] for d in 1:3)..., ([dir[d]] for d in 1:3)..., color=:red, arrowsize=1500, linewidth=5)
    end
end

function draw_trunks!(ax, dfarbors; trunkcolor=nothing)
    draw_trunks(dfarbors, ax => :h, linewidth=2, arborcolor=(:gray, 0.2); trunkcolor)
end
function draw_arbors!(ax, dfarbors; kw...)
    draw_trunks(dfarbors, ax => :h; trunkcolor=:gray, arborcolor=nothing, kw...)
end

function get_inspector(inspect)
    if inspect === nothing
        Makie.automatic
    elseif inspect === :nmt
        inspect_nmt_hier()
    elseif inspect === :allen
        inspect_allen_hier()
    else
        @assert false
    end
end

function draw_subtrees(dfarbors, axspecs...; rowlabel=nothing, autotrunk=true, per_neuron=nothing, per_arbor=nothing, per_node=nothing, inspect=nothing, mirrorby=nothing, arbornodes=nothing, drawsoma=true, kw...)
    specs = parseaxspec.(axspecs)
    path2i = Dict{String,Int}()
    for (i, r) in enumerate(eachrow(dfarbors))
        get!(path2i, r.path, i)
    end
    dfnrn = dfarbors[last.(collect(path2i)), :]
    if !isfocus(:draw_subtrees)
        dfnrn = shuffle(Random.MersenneTwister(3), dfnrn)[1:min(60, end), :]
    end
    sort!(dfnrn, :path)

    if per_neuron === nothing
        per_neuron = (r) -> nothing
    end
    if per_arbor === nothing
        per_arbor = (pnrn, rr, cond) -> nothing
    end
    if per_node === nothing
        per_node = (pnrn, parb, n) -> :gray => pt(0.3)
    end
    if arbornodes === nothing
        arbornodes = rr -> begin
            fixid(id) = id % 100_000_000_000
            swcx = Gapr.load_swc(rr.arborpath)
            (fixid(n.id) for n in swcx.nodes)
        end
    end

    somas = []
    inspector_label = get_inspector(inspect)
    per_arbor_cache = Dict{Pair{Int,Int},Any}()
    @showprogress for (ri, r) in enumerate(eachrow(dfnrn))
        pnrn = per_neuron(r)
        swc = Gapr.load_swc(r.path)
        if mirrorby !== nothing
            if !mirrorby.isleft(swc[1].pos)
                for n in swc.nodes
                    n.pos = mirrorby.mirror(n.pos)
                end
            end
        end
        swc = Gapr.addattr(swc; typpp=Int)

        for n in swc.nodes
            n.typpp = 0
        end
        arbors = filter(rr -> rr.path == r.path, dfarbors)
        depths = collect(Gapr.nodedepth(swc).depth)
        todo = []
        for (rri, rr) in enumerate(eachrow(arbors))
            ids = arbornodes(rr)
            for id in ids
                swc[!, id].typpp = -rri
            end
            push!(todo, (swc[!, first(ids)].ind, rri))
        end
        sort!(todo, by=p -> depths[p[1]])
        ii = findmax(depths)[2]
        if autotrunk isa Function
            ii = findmax(splat(autotrunk), collect(zip([n.pos for n in swc], depths)))[2]
            push!(todo, (ii, 100000))
        elseif autotrunk && swc[ii].typpp == 0
            push!(todo, (ii, 100000))
        end
        for (idx, rri) in todo
            while idx != 0
                swc[idx].typpp = rri
                idx = swc[idx].parind
            end
        end
        somacol = nothing
        pts, col, lw = load_swc_as_linesx(swc) do n
            parb = get!(per_arbor_cache, ri => n.typpp) do
                rr, cond = if n.typpp > size(arbors, 1)
                    nothing, true
                elseif n.typpp > 0
                    arbors[n.typpp, :], true
                elseif n.typpp < 0
                    arbors[-n.typpp, :], false
                else
                    nothing, false
                end
                per_arbor(pnrn, rr, cond)
            end
            pn = per_node(pnrn, parb, n)
            if n.ind == 1
                somacol = first(pn)
            end
            pn
        end
        labelkw = (;)
        if rowlabel !== nothing
            labelkw = (; label=rowlabel(r))
        end
        for spec in specs
            lines!(spec.gl, pts, color=col, linewidth=lw, joinstyle=:round, linecap=:round, transparency=true, fxaa=true; inspector_label, labelkw..., kw...)
        end
        push!(somas, swc[1].pos => somacol)
    end
    crng = []
    @warn " crng $kw"
    for (k, v) in kw
        if k in (:colorrange, :colormap)
            push!(crng, k => v)
        end
    end
    crng = NamedTuple(crng)
    drawsoma && draw_somas(first.(somas), last.(somas), specs...; crng)
end

function draw_pathways(df, axspecs...; rowcolor=nothing, kw...)
    if rowcolor === nothing
        rowcolor = r -> log(r.w)
    end
    specs = parseaxspec.(axspecs)

    points = map(eachrow(df)) do r
        pts = deepcopy(r.trunk)
        #pushfirst!(pts,r.tailxyz)
        #push!(pts, r.headxyz)
        push!(pts, Vec3f(NaN))
        pts, fill(rownumber(r), length(pts)), fill(rowcolor(r), length(pts))
    end
    pp = reduce(vcat, first.(points))
    cc = reduce(vcat, last.(points))
    points = reduce(vcat, [v[2] for v in points])

    inspector_label(p, i, x) = begin
        r = df[points[i], :]
        a, b = r.trunkcls, r.w
        "$a $b"
    end
    for spec in specs
        lines!(spec.gl, pp, color=cc; linewidth=0.5, joinstyle=:round, linecap=:round, transparency=true, fxaa=true, inspector_label, kw...)
    end
end

function draw_neurons(df, axspecs...; pickby=nothing, picklimit=10, rowlabel=nothing, rowcolor=nothing, inspect=nothing, mirrorby=nothing, xformby=nothing, kw...)
    if !isfocus(:drawneurons)
        df = shuffle(Random.MersenneTwister(3), df)[1:min(end, 20), :]
    end
    if pickby !== nothing
        df = pick_neurons_with_limit(df, pickby, picklimit)
    end
    if rowlabel === nothing
        rowlabel = (r) -> nothing
    end
    if rowcolor === nothing
        rowcolor = (r) -> Makie.Cycled(rownumber(r))
    end
    specs = parseaxspec.(axspecs)
    inspector_label = get_inspector(inspect)
    somapos = Pair{Point3f,Any}[]
    colors = map(eachrow(df)) do r
        cc = rowcolor(r)
        if cc isa Pair
            cc
        else
            cc => pt(0.3)
        end
    end
    lw = last.(colors)
    colors = first.(colors)
    crng = if eltype(colors) <: Number
        (; colorrange=extrema(colors))
    else
        (;)
    end

    foreach(eachrow(df), colors, lw) do r, ccc, lw0
        for spec in specs
            lll = load_swc_as_linesx(r.path; mirrorby, xformby)
            plt = lines!(spec.gl, lll, color=ccc, linewidth=lw0, joinstyle=:round, linecap=:round, transparency=true, fxaa=true, label=rowlabel(r); inspector_label, crng..., kw...)
            !get(r, :qroot, false) && push!(somapos, lll[end] => plt.color[])
        end
    end
    draw_somas(first.(somapos), last.(somapos), specs...; crng)
end

draw_somas(somapos, somacol, specs...; crng=(;)) = begin
    isempty(somapos) && return
    for spec in specs
        cols = [to_color(c) for c in somacol]
        cols = map(cols) do c
            if c isa Number
                return c => :black
            end
            c * 0.7 + 0.3 * to_color(:white) => c * 0.7 + 0.3 * to_color(:black)
        end
        Makie.scatter!(spec.ax, somapos; color=first.(cols), strokecolor=last.(cols), strokewidth=pt(0.3), crng..., markersize=pt(2.5), transparency=true, fxaa=true)
    end
end

function plot_subtyp_inj_matrix!(ax1, df)
    n2ms(n) = sqrt(n)
    subtyps = sort(unique(df.cls))
    hierof(r) = r.qroot ? "Unknown" : (brain_region_props(r.somahier)[1] <= 2 ? r.somahier : "Other")
    ba2cnt = get_sorted_ba_cnt(df, hierof)
    bas = filter(r -> r != "Unknown", first.(ba2cnt))
    vals = []
    for (row, reg) in enumerate(bas)
        for (col, subt) in enumerate(subtyps)
            n = size(filter(r -> hierof(r) == reg && r.cls == subt, df), 1)
            push!(vals, (; row, col, n))
        end
    end
    Makie.scatter!(ax1, [t.col for t in vals], [t.row for t in vals], markersize=[n2ms(t.n) for t in vals], color=:black)
    ax1.height[] = length(bas) * 6
    basticks = (axes(bas, 1), areaname.(bas))
    ax1.yticks[] = basticks
    ax1.yreversed[] = true
    ax1.topspinevisible[] = true
    ax1.bottomspinevisible[] = false
    ax1.ylabel[] = "Brain areas"
    hidedecorations!(ax1, label=false, ticklabels=false)
    ax1.xgridvisible[] = true
    ax1.ygridvisible[] = true
    ax1.xticks = axes(subtyps, 1)
    ax1.xticks[] = (axes(subtyps, 1), ["#$c" for c in subtyps])
    ax1.xticklabelrotation = π / 3
    ax1.xticklabelalign = (:left, :center)
    ax1.xlabel[] = "Subtypes"
    ax1.xaxisposition = :top
end

function plot_subtyp_soma_depth!(ax, df; select=nothing)
    dfx = filter(r -> !r.qroot && !isnan(r.somaw), df)
    if select !== nothing
        filter!(select, dfx)
    end
    Makie.rainclouds!(ax, dfx.cls, dfx.somaw, violin_limits=(0, 1), show_median=true, markersize=1,
        boxplot_width=0.4,
        clouds=nothing,
        strokewidth=0.5,
        center_boxplot=false,
        boxplot_nudge=-0.1,
    )
    ax.ylabel[] = "Soma depth"
    ax.height[] = 40
    ax.yreversed[] = true
end

function plot_subtyp_spans!(ax3, df; linkto=nothing, gridto=nothing, rowgrp=nothing, grplabel=nothing, grpcolor=nothing, label="Subtype")
    if rowgrp === nothing
        rowgrp = r -> r.cls
    end
    grps = rowgrp.(eachrow(df))
    gridto = @something(gridto, [])
    if grpcolor === nothing
        grpcolor = (i, g) -> iseven(i) ? :gray80 : :gray90
    end
    for (i, c) in enumerate(unique(grps))
        a, b = extrema(findall(==(c), grps))
        if i > 1
            for ax2 in gridto
                plt = vlines!(ax2, a - 0.5, color=:gray)
                translate!(plt, 0, 0, 1)
            end
        end
        vspan!(ax3, a - 0.5, b + 0.5, color=grpcolor(i, c))
        l = grplabel(i, c)
        if l !== nothing
            plt = text!(ax3, (a + b) / 2, 0, text="$l", align=(:center, :center))
            translate!(plt, 0, 0, 1)
        end
    end
    ax3.height = pt(7)
    ax3.yaxisposition = :right
    ax3.yticks = ([-1, 0, 1], ["", isempty(label) ? "" : L"\fontfamily{Arial}$⇐$%$label", ""])
    ax3.yticklabelfont = :bold
    ylims!(ax3, -1, 1)
    if linkto !== nothing
        linkxaxes!(ax3, linkto)
    end
    hideydecorations!(ax3, ticklabels=false)
    hidexdecorations!(ax3)
    hidespines!(ax3)
end

function plot_global_heatmap!(ax0, ax1, ax2, ax31, ax32, ax4, df, hclust; skipic=false, ax1x=nothing, ax2x=nothing)
    cols = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-all-neurons.jl")
    zz1vis.dendrogram!(ax0, hclust)
    hidespines!(ax0)
    hidedecorations!(ax0)
    linkxaxes!(ax0, ax1)
    plot_subtyp_spans!(ax31, df; linkto=ax1, gridto=[ax1, ax2], rowgrp=r -> r.cls, grplabel=(i, g) -> isodd(i) ? "#$g" : "")
    plot_subtyp_spans!(ax32, df; linkto=ax1, rowgrp=r -> r.cls, grplabel=(i, g) -> iseven(i) ? "#$g" : "")
    typ2c = typ2color_func(df)
    ax32.xautolimitmargin = (0, 0.01)
    plot_subtyp_spans!(ax4, df; linkto=ax1, rowgrp=r -> r.typ, grplabel=(i, g) -> "$g", grpcolor=(i, g) -> typ2c(g), label="Type")
    draw_neuron_proj_hm!(ax1, ax2, df, cols; skipic, ax1x, ax2x, rowh=pt(6))
    ax1.width = nothing
end

function draw_neuron_proj_hm!(ax1, ax2, df, cols; ax3=nothing, marksig=true, skipic=false, ax1x=nothing, ax2x=nothing, clscol=:cls, longic=true, yticks=nothing, rowh=pt(7), rowpad=(pt(2)))
    subtyps = unique(df[:, clscol])
    projofba(ic) = (cols) -> begin
        filter(h -> match(r"^\w+(\w)\|", h).captures[1] == string(ic), cols)
    end
    nameofba(ba) = begin
        ic = replace(ba, r"^\w+(\w)\|.*\|(.*)$" => s"\1")
        ba = areaname(ba)
        skipic ? "$ba" : "$ba ($ic)"
    end
    getprojmat(cols) = begin
        projs = filter!(zz1data.is_projcol, names(df))
        getcol(c) = begin
            c in projs ? df[:, c] : zeros(size(df, 1))
        end
        reduce(hcat, [getcol(c) for c in cols])
    end
    vals = []
    plts = []
    for (ax, ic) in ((ax1, :I), (ax2, :C))
        colsic = projofba(ic)(cols)
        isempty(colsic) && continue
        mat = getprojmat(colsic)
        if marksig
            for r in axes(mat, 2)
                rr = mat[:, r]
                for c in subtyps
                    cc = rr[df[:, clscol].==c]
                    t = HypothesisTests.MannWhitneyUTest(cc, rr)
                    if HypothesisTests.pvalue(t) < 0.05 && median(cc) > median(rr)
                        a, b = extrema(findall(df[:, clscol] .== c))
                        plt = Makie.poly!(ax, Rect(a - 0.5, r - 0.5, b - a + 1, 1), color=:transparent, strokecolor=:red, strokewidth=0.3)
                        Makie.translate!(plt, (0, 0, 3))
                    end
                end
            end
        end
        push!(vals, extrema(log.(mat .+ 1))...)
        plt = Makie.heatmap!(ax, log.(mat .+ 1))
        push!(plts, plt)
        if yticks === nothing
            ax.yticks[] = (axes(colsic, 1), nameofba.(colsic))
        elseif yticks === :ignore
            ax.yticklabelsvisible = false
        else
            ticks = map(projofba(ic)(yticks), colsic) do a, b
                a = nameofba(a)
                b = nameofba(b)
                lowercase(a) == lowercase(b) && return a
                "$a ($b)"
            end
            ax.yticks[] = (axes(colsic, 1), ticks)
        end
        ax.yaxisposition[] = :right
        ax.height[] = length(colsic) * rowh + rowpad
        ax.yreversed[] = true
        foreach(println, colsic)
        ylims!(ax, size(mat, 2) + 0.5 + 1 / 7, 0.5 - 1 / 7)
        hidespines!(ax)
        hidexdecorations!(ax)
        hideydecorations!(ax, ticklabels=false)

        let (axr, txt, txtlong) = (; I=(ax1x, "Ipsi.", "Ipsilateral"), C=(ax2x, "Contra.", "Contralateral"))[ic]
            if axr !== nothing
                hidedecorations!(axr)
                hidespines!(axr)
                bracket!(axr, 0, 1 - 0.3, 0, size(mat, 2) + 0.3, width=1, text=longic ? txtlong : txt, orientation=:down; set_font_h2()...)
                linkyaxes!(axr, ax)
                xlims!(axr, 0, 1)
                axr.yautolimitmargin[] = (0, 0)
                #, 0, 1)
                axr.width[] = 10
            end
        end
    end
    ax1.width[] = size(df, 1) * 0.2
    if ax2 !== nothing
        for ax in plts
            ax.colorrange[] = (extrema(vals)...,)
        end
        Makie.linkxaxes!(ax1, ax2)
    end
end

function bracket_in_3d!(ax, x, y; kw...)
    proj(c, p) = begin
        Makie.project(c, :data, :relative, p)[1:2]
    end
    xx = Makie.Observable(Point2f(0.0f0))
    yy = Makie.Observable(Point2f(0.0f0))
    cam = Makie.camera(ax)
    p = Makie.bracket!(Makie.camrelative(ax.scene), xx, yy; kw...)
    Makie.on(cam.projectionview, update=true) do _
        xx.val = proj(cam, x)
        yy[] = proj(cam, y)
        p.rotation[] = Makie.automatic
    end
end

function linear_reg_scatter!(ax, df; rowx, rowy, rowcolor, legpos=nothing, force=false, linecolor=nothing, noleg=false, legoff=nothing, legcol=nothing, kw...)
    x = [rowx(r) for r in eachrow(df)]
    y = [rowy(r) for r in eachrow(df)]
    sel = .!(isnan.(x) .|| isnan.(y))
    x = x[sel]
    y = y[sel]
    Makie.scatter!(ax, x, y, color=rowcolor.(eachrow(df)), markersize=pt(2.0); kw...)
    tt = HypothesisTests.CorrelationTest(x, y)
    p = HypothesisTests.pvalue(tt)
    r0 = StatsBase.cor(x, y)
    r1 = StatsBase.corspearman(x, y)
    @warn "pval $p $r0 $r1"
    if p < 0.05 || force
        rr = LinearRegression.linregress(x, y)
        if linecolor === nothing
            linecolor = :gray
        end
        lbl = "r=$(compact_str(r0))$(compact_pval(p))"
        plt = ablines!(ax, rr.coeffs[[2, 1]]..., color=linecolor, label=lbl, linestyle=p < 0.05 ? :solid : :dot, linewidth=pt(0.75))
        @warn "lrlr $(rr.coeffs)"
        translate!(plt, 0, 0, -1)
        @warn "abline $(rr.coeffs)"
        if legpos === nothing
            text!(ax, 0, 1, space=:relative, text=lbl, align=(:left, :top), offset=@something(legoff, (2, -2)), color=@something(legcol, :black))
            #!noleg && axislegend(ax, ax, position=:rb)
        else
            Label(legpos, lbl, halign=:right, tellwidth=false,)
            #!noleg && Legend(legpos, ax, tellheight=true, tellwidth=false)
        end
    end
end

function ranked_scatter!(ax, df; rowx, rowy, rowcolor, rowtext=nothing)
    calc_rank(y) = invperm(sortperm(vec(y)))
    x = [rowx(r) for r in eachrow(df)]
    y = [rowy(r) for r in eachrow(df)]
    sel = .!(isnan.(x) .|| isnan.(y))
    x = x[sel]
    y = y[sel]
    xx = calc_rank(x)
    yy = calc_rank(y)
    Makie.scatter!(ax, xx, yy, color=rowcolor.(eachrow(df)[sel]), markersize=pt(2))
    tt = HypothesisTests.CorrelationTest(xx, yy)
    p = HypothesisTests.pvalue(tt)
    r = StatsBase.corspearman(x, y)
    @warn "pval $p"
    if p < 0.05
        rr = LinearRegression.linregress(xx, yy)
        Makie.ablines!(rr.coeffs[[2, 1]]..., color=:black)
        Makie.text!(ax, 0, 1, text="r=$(round(r,sigdigits=3))$(compact_pval(p))", space=:relative, align=(:left, :top), offset=(2, -2))
    end
    hidedecorations!(ax, label=false, ticklabels=false, ticks=false)
    if rowtext !== nothing
        text!(ax, xx, yy, color=rowcolor.(eachrow(df)[sel]), text=[rowtext(r) for r in eachrow(df)][sel])
    end
end

function plot_flatmap_base!(ax, highlights, obj, objphy; canvassize=1024, label=true, debug=false)
    if !debug
        hidedecorations!(ax)
        ax.topspinevisible[] = true
        ax.rightspinevisible[] = true
    else
        ax.xminorgridvisible[] = true
        ax.yminorgridvisible[] = true
        ax.xgridvisible[] = true
        ax.ygridvisible[] = true
    end
    highlights = map(highlights) do hl
        hl isa Pair ? hl : (hl => 0.5)
    end
    highlight_pars = map(last, highlights)
    highlights = map(first, highlights)

    images = zz1data.cached_result(:cachedcanvas2, highlights, canvassize, obj, objphy) do highlights, canvassize, obj, objphy
        nmt = NmtDat.load_images()
        annots = Main.mesh_annotate_vert(objphy, nmt, highlights)
        canvas = fill(Vecf{length(highlights)}(0.0f0), canvassize, canvassize)
        Main.sample_mesh(obj, obj.position, annots; perface=16) do vec
            ###@assert length(vec) == 3
            pos = vec[1:2]
            xy = round.(Int, (pos .+ 1) ./ 2 .* (size(canvas)[1:2] .- 1) .+ 0.51)
            if checkbounds(Bool, canvas, xy..., 1)
                canvas[xy...] = Vecf{length(highlights)}(vec[2+1:end])
            end
        end

        smooth(img) = Images.imfilter(img, Images.Kernel.gaussian((3, 3)))
        canvas = smooth(canvas)

        calc_image(hl) = begin
            findpar(h) = begin
                ####@info "find par:$h"
                while count('|', h) > 0
                    h = replace(h, r"\|[^|]+$" => "")
                    i = findfirst(==(h), highlights)
                    ######@info "find par $h $i"
                    i !== nothing && return i
                end
                0
            end

            avoid = Set{String}()
            let h = highlights[hl]
                push!(avoid, h)
                for K in 1:length(highlights)
                    for hh in highlights
                        hi = findpar(hh)
                        hi == 0 && continue
                        h = highlights[hi]
                        #######@info "par: $hh $h"
                        if h in avoid
                            push!(avoid, hh)
                        end
                    end
                end
                h = highlights[hl]
                while true
                    h = replace(h, r"\|[^|]+$" => "")
                    h in avoid && break
                    push!(avoid, h)
                end
            end
            ###@info "avoid: $(collect(avoid))"
            par = findpar(highlights[hl])
            hll = filter(axes(highlights, 1)) do hi
                !(highlights[hi] in avoid)
            end
            hlll = filter(hll) do hi
                findpar(highlights[hi]) in hll
            end
            @info "$(highlights[[hl, hlll...]])"
            hll = setdiff(hll, hlll)
            diffs = []
            parr = par
            while parr != 0
                push!(diffs, parr => filter(hi -> findpar(highlights[hi]) == parr, axes(highlights, 1)))
                parr = findpar(highlights[parr])
            end
            
            @info "$(highlights[[hl, hll...]])"
            @info ": $(highlights[[hlll...]])"
            map(canvas) do vals
                a = 0.0
                for (parr, inds) in diffs
                    a = max(a, vals[parr] - sum(vals[inds], init=0.0))
                end
                vals[hl] - max(maximum(vals[hll], init=0.0), a)
            end
        end

        map(enumerate(highlights)) do (l, hl)
            hl => calc_image(l)
        end |> Dict
    end |> deepcopy


    xyrng = range(-1.0f0, 1.0f0, canvassize)
    foreach(highlights, highlight_pars) do hl, par
        l = findfirst(==(hl), highlights)
        if par isa Number
            contour!(ax, xyrng, xyrng, images[hl], labels=false, levels=[-0.0], color=:black, linewidth=pt(par), labelformatter=(x) -> string(all_lobes[l], ": ", x), inspectable=false)
        else
            contourf!(ax, xyrng, xyrng, images[hl], levels=[-0.0, 1.5], colormap=[Makie.to_color(par)], inspectable=false)
        end
    end
    markinfos = include("monkeytemp/wyz-upload/julia/flatmap-brainarea-coords.jl") |> Dict
    for h in highlights
        !label && continue
        info = get(markinfos, h, nothing)
        info === nothing && continue
        t, p, s, f = info
        Makie.text!(ax, p[1:2]..., text=t, align=(p[3:4]...,), font=f, fontsize=s, color=:gray)
    end
    Makie.onany(ax.width, ax.height, update=true) do a, b
        #if a isa Number && b isa Number
        @warn "asp $a $b"
        ax.autolimitaspect[] = 1
        #end
    end
end

function plot_flatmap_soma!(ax, df; rowcolor)
    dfx = filter(r -> !r.qroot, df)
    scatter!(ax, dfx.somauv, color=rowcolor.(eachrow(dfx)), transparency=true, overdraw=true, markersize=2)
    limits!(ax, -0.45, 0.45, -0.55, 0.25)
end

function plot_inj_flatmap!(ax, df)
    inj2c = inj2color_func(df)
    lblinfos = map(sort(unique(df.inj))) do inj
        dfx = filter(r -> r.inj == inj && !r.qroot, df)
        scatter!(ax, dfx.somauv, markersize=2.0, color=inj2c(inj), inspector_label=(plt, idx, pos) -> Tuple(dfx[idx, :]))
        (Point2f(median(reduce(hcat, dfx.somauv), dims=2)...), inj)
    end
    sort!(lblinfos, by=t -> t[1][2])
    yval = -0.33
    for (cent, inj) in lblinfos
        dx, xalign = if cent[1] > 0
            0.2, :right
        else
            cent[2] > -0.1 ? 0 : -0.15, :left
        end
        plt2 = if minimum(t -> (t[2] == inj ? Inf : norm(t[1] .- cent)), lblinfos) < 0.10
            cent2 = Vec2f(-0.00 + dx, yval)
            yval += 0.044
            plt = lines!(ax, [cent, cent2], color=:black)
            translate!(plt, 0, 0, 3)
            text!(ax, cent2, text=string(inj), color=inj2c(inj), align=(xalign, :center); set_font_h2()...)
        else
            text!(ax, cent + (cent[1] > 0.2 ? Vec2f(0.00, -0.07) : Vec2f(0.03, 0)), text=string(inj), color=inj2c(inj), align=(:left, :center); set_font_h2()...)
        end
        Makie.translate!(plt2, 0, 0, 1)
    end
    limits!(ax, -0.47, 0.47, -0.55, 0.35)
    #Makie.hidedecorations!(ax)
end

function plot_inj_in3d!(mix, df, obj)
    #obj2 = load("interm/mesh-nmt-OFC-l.obj")
    #obj3 = load("interm/mesh-nmt-motor-l.obj")
    # for (o, c) in zip([obj1, obj2, obj3], [:red, :green, :blue])
    #     o2 = normal_mesh(Point3f.(map(p -> NmtDat.mirror(p), o.position)), faces(o))
    #     for oo in (o, o2)
    #         Makie.mesh!(mix.scene2, oo, color=(c, 0.09), transparency=true, fxaa=true)
    #     end
    # end
    inj2c = inj2color_func(df)
    for inj in sort(unique(df.inj))
        dfx = filter(r -> r.inj == inj && !r.qroot, df)
        Makie.scatter!(mix.axis, dfx.soma_pos, color=inj2c(inj), markersize=1.0)
    end
end

typ_order(typ) = begin
    (istypit(typ), typ)
end

function typ2color_func(df)
    typs = sort(unique((df.typ)), by=typ_order)
    pal = Makie.Palette(:tab10)
    t2col = map(typs) do typ
        typ => Makie.to_color(pal)
    end |> Dict
    t -> t2col[t]
end
function inj2color_func(df)
    colors = vec([
        "#000080", "#0000ff", "#006400", "#00bfff", "#00ced1",
        "#00ff7f", "#3cb371", "#483d8b", "#696969", "#7fff00",
        "#808000", "#9932cc", "#9acd32", "#a0522d", "#b03060",
        "#dc143c", "#ff0000", "#ff00ff", "#ff8c00", "#ffd700",
    ])
    injs = sort(unique(df.inj))
    cols = map(enumerate(injs)) do (i, inj)
        inj => colors[i]
    end |> Dict
    inj -> cols[inj]
end

function pick_neurons_with_limit(df, rowgroup, limit)
    grps = unique([rowgroup(r) for r in eachrow(df)])
    sels = map(grps) do grp
        dfx = filter(r -> rowgroup(r) == grp, df)
        size(dfx, 1) <= limit && return dfx
        dfxx = filter(r -> !get(r, :qroot, false), dfx)
        n1 = size(dfxx, 1)
        n1 >= limit && return sort!(dfxx, :len, rev=true)[1:limit, :]
        vcat(dfxx, sort!(filter(r -> r.qroot, dfx), :len, rev=true)[1:limit-n1, :])
    end
    df = reduce(vcat, sels)
end

function plot_all_fibers_by_inj!(mix1, mix2, df, obj; limit=100000)
    inj2col = inj2color_func(df)
    df = pick_neurons_with_limit(df, r -> r.inj, limit)
    @showprogress for r in eachrow(df)
        isfocus(:fig2fibers) || (rand(1:50) != 1 && continue)
        pts = load_swc_as_linesx(r.path)
        for mix in (mix1, mix2)
            Makie.lines!(mix.scene2, pts, linewidth=0.2, color=inj2col(r.inj), transparency=true)
        end
    end
end

function axis_as_layer(ax, xy, a, b)
    setattr(a, v) = setproperty!(ax, Symbol(xy, a), v)
    setattr(:label, "Laminar depth")
    setattr(:ticks, ([a, (a + b) / 2, b], ["", L"\fontfamily{Arial}L6 $⟷$ L1", ""]))
    setattr(:minorticksvisible, false)
    if xy === :y
        setattr(:ticklabelrotation, π / 2)
    end
    (; x=hidexdecorations!, y=hideydecorations!)[xy](ax, ticklabels=false, label=false)
    (; x=xlims!, y=ylims!)[xy](ax, b, a)
end
yaxis_as_layer(ax, a, b) = axis_as_layer(ax, :y, a, b)
xaxis_as_layer(ax, a, b) = axis_as_layer(ax, :x, a, b)

function plot_depth_itptct!(ax, df)
    typ2c = typ2color_func(df)
    dfx = filter(r -> !r.qroot, df)
    alltypes = sort(unique(dfx.typ), by=t -> (istypit(t), t))
    Makie.rainclouds!(ax, something.(indexin(dfx.typ, alltypes)), dfx.somaw, violin_limits=(0, 1), color=typ2c.(dfx.typ))
    ax.xticks = (axes(alltypes, 1), alltypes)
    ax.xlabel = "Neuron type"
    hidexdecorations!(ax, ticklabels=false, label=false)
    yaxis_as_layer(ax, 0, 1)
end



function boxplot_and_compare!(ax, df, ax2=nothing; rowx, xname, rowy, rowdodge=nothing, rowcolor=nothing, statif=nothing, legendpos=nothing, kw...)
    if ax2 === nothing
        ax2 = ax
    end
    if rowdodge === nothing
        rowdodge = r -> nothing
    end
    if rowcolor === nothing
        rowcolor = r -> nothing
    end
    vals = filter!(v -> !isnan(v.y) && !isinf(v.y), map(r -> (; x=rowx(r), y=rowy(r), c=rowcolor(r), d=rowdodge(r)), eachrow(df)))
    x = [t.x for t in vals]
    d = [t.d for t in vals]
    xdvals = sort(unique([(; t.x, t.d) for t in vals]))
    xvals = sort(unique(x))
    dvals = sort(unique(d))
    grps = map(xdvals) do xd
        xd => map(t -> t.y, filter(t -> (; t.x, t.d) == xd, vals))
    end
    copt = (;)
    cc = [t.c for t in vals]
    cvals = unique(cc)
    if length(cvals) != 1
        copt = (; copt..., color=cc)
    end
    dint = something.(indexin(d, dvals))
    if length(dvals) != 1
        copt = (; copt..., dodge=dint)
    end
    Makie.rainclouds!(ax, something.(indexin(x, xvals)), [t.y for t in vals]; copt..., show_median=false, kw...)

    brackety = []
    bracketx(xx, dd, sss) = begin
        off = (
            v -> 0,
            v -> 0.4 * v - 0.2,
            v -> 0.3 * v - 0.3,
        )[length(dvals)](findfirst(==(dd), dvals) - 1)
        findfirst(==(xx), xvals) + off + sss / 16
    end
    prev_brks = []
    getbrks(x1, x2) = begin
        hh = 0
        for ((xx1, xx2), h) in prev_brks
            if xx2 > x1 && xx1 < x2
                hh = max(hh, h + 1)
            end
        end
        push!(prev_brks, (x1, x2) => hh)
        hh
    end
    for (xd1, g1) in grps
        for (xd2, g2) in grps
            xd1 >= xd2 && continue
            statif === nothing && continue
            !statif(xd1, xd2) && continue
            t = HypothesisTests.MannWhitneyUTest(g1, g2)
            p = HypothesisTests.pvalue(t)
            m1 = mean(g1)
            m2 = mean(g2)
            @warn "stat: $xd1 $xd2 $p $m1 $m2"
            if p < 0.05
                sgn = m1 < m2 ? "<" : ">"
                x1 = bracketx(xd1.x, xd1.d, 1)
                x2 = bracketx(xd2.x, xd2.d, -1)
                hh = getbrks(x1, x2)
                Makie.bracket!(ax2, x1, hh, x2, hh, text="$sgn$(compact_pval(p))")
                push!(brackety, hh)
            end
        end
    end
    if isempty(brackety)
        ax2.height[] = 0
    else
        my = maximum(brackety)
        @warn "$my"
        ax2.height[] = 12 * (my + 1)
        ylims!(ax2, 0, my + 0.5)
        # if 2more
        ylims!(ax2, 0, my + 1.0)
    end
    hidedecorations!(ax2)
    hidespines!(ax2)
    linkxaxes!(ax, ax2)
    if length(cvals) != 1
        for v in (:x, :d)
            acc = Dict([c => Set() for c in cvals])
            for vv in vals
                push!(acc[vv.c], vv[v])
            end
            acc = collect(acc)
            if extrema(length.(last.(acc))) == (1, 1) && legendpos !== nothing
                @warn "use as color  $v $cvals"
                Makie.axislegend(ax, [MarkerElement(marker=:rect, color=c) for (c, v) in acc], [string(first(v)) for (c, v) in acc], position=legendpos, orientation=:horizontal)
                break
            end
        end
    end
    ax.xticks[] = (axes(xvals, 1), xname.(xvals))
    hidexdecorations!(ax, ticklabels=false)
end

function get_vals_x_and_dodge(df; rowx, rowy, rowdodge=nothing, rowcolor=nothing)#df, rowx, rowdodge=nothing)
    if rowdodge === nothing
        rowdodge = r -> nothing
    end
    if rowcolor === nothing
        rowcolor = r -> nothing
    end
    vals = filter!(v -> !isnan(v.y) && !isinf(v.y), map(r -> (; x=rowx(r), y=rowy(r), c=rowcolor(r), d=rowdodge(r)), eachrow(df)))
    x = [t.x for t in vals]
    d = [t.d for t in vals]
    xdvals = sort(unique([(; t.x, t.d) for t in vals]))
    xvals = sort(unique(x))
    dvals = sort(unique(d))
    grps = map(xdvals) do xd
        xd => map(t -> t.y, filter(t -> (; t.x, t.d) == xd, vals))
    end
    dint = something.(indexin(d, dvals))
    xint = something.(indexin(x, xvals))
    y = [t.y for t in vals]
    c = [t.c for t in vals]
    cvals = unique(c)
    (; vals, x, d, c, xvals, dvals, xdvals, cvals, grps, dint, xint, y)
end

function plot_foldchange!(ax, df::AbstractDataFrame; rowx, rowy, rowdodge=nothing, rowcolor=nothing, kw...)
    dat = get_vals_x_and_dodge(df; rowx, rowy, rowdodge, rowcolor)
    plot_foldchange!(ax, dat; kw...)
end
function plot_foldchange!(ax, dat; colorA=:red, colorB=:blue, color=:black)
    # drawmv(x, d, mv) = begin
    #     ab = map(plt.gap) do g
    #         [Point2f(x + (d - 1.5) * (1 - g) - 0.3, mv), Point2f(x + (d - 1.5) * (1 - g) + 0.3, mv)]
    #     end
    #     lines!(ab, linewidth=pt(2))
    #     map(first, ab), map(last, ab)
    # end
    xs = Int[]
    ys = Float32[]
    for (xd1, vv1) in dat.grps
        ##@warn " asdfasdf $xd1"
        d1 = indexin(Ref(xd1.d), dat.dvals)[]
        x1 = indexin(Ref(xd1.x), dat.xvals)[]
        if d1 == 1
            mv1 = mean(vv1)
            for (xd2, vv2) in dat.grps
                d2 = indexin(Ref(xd2.d), dat.dvals)[]
                x2 = indexin(Ref(xd2.x), dat.xvals)[]
                if x1 == x2 && d2 > 1
                    mv2 = mean(vv2)
                    push!(xs, x1)
                    push!(ys, mv1 / mv2)
                end
            end
        end
    end
    ylims!(ax, 0.2, 5)
    ax.yscale = log2
    plt = hlines!(ax, [1], linewidth=pt(0.75), color=color)
    translate!(plt, 0, 0, -5)
    #barplot!(ax, xs, ys, fillto=1, color=Pattern("x"), strokewidth=pt(0.0))
    barplot!(ax, xs, ys, fillto=1, color=[v > 1 ? colorA : colorB for v in ys], strokewidth=pt(0.0))
end

function violin_and_compare!(ax, df::AbstractDataFrame, ax2=nothing; compact=false, rowx, xname, rowy, rowdodge=nothing, rowcolor=nothing, statif=nothing, legendpos=nothing, kw...)
    dat = get_vals_x_and_dodge(df; rowx, rowy, rowdodge, rowcolor)
    violin_and_compare!(ax, dat, ax2; compact, xname, statif, legendpos, kw...)
end
compare_ranksum(t::HypothesisTests.HypothesisTest) = sign(t.U - t.nx * t.ny / 2)

function violin_and_compare!(ax, dat, ax2=nothing; compact=false, xname, statif=nothing, legendpos=nothing, kw...)
    copt = (;)
    if length(dat.cvals) != 1
        copt = (; copt..., color=dat.c)
    end
    if length(dat.dvals) != 1
        copt = (; copt..., dodge=dat.dint)
    end
    show_median = true
    plt = violin!(ax, dat.xint, dat.y; copt..., show_median, kw...)

    brackety = []
    bracketx(xx, dd, sss) = begin
        off = (
            v -> 0,
            v -> 0.4 * v - 0.2,
            v -> 0.3 * v - 0.3,
        )[length(dat.dvals)](findfirst(==(dd), dat.dvals) - 1)
        findfirst(==(xx), dat.xvals) + off + sss / 16
    end
    
    prev_brks = []
    getbrks(x1, x2) = begin
        hh = 0
        for ((xx1, xx2), h) in prev_brks
            if xx2 > x1 && xx1 < x2
                hh = max(hh, h + 1)
            end
        end
        push!(prev_brks, (x1, x2) => hh)
        hh
    end
    for (xd1, g1) in dat.grps
        for (xd2, g2) in dat.grps
            xd1 >= xd2 && continue
            statif === nothing && continue
            !statif(xd1, xd2) && continue
            t = HypothesisTests.MannWhitneyUTest(g1, g2)
            p = HypothesisTests.pvalue(t)
            m1 = mean(g1)
            m2 = mean(g2)
            @warn "stat: $xd1 $xd2 $p $m1 $m2"
            if p < 0.05
                sgn = compare_ranksum(t) < 0 ? "<" : ">"
                if compact
                    sgn = ""
                end
                x1 = bracketx(xd1.x, xd1.d, 1)
                x2 = bracketx(xd2.x, xd2.d, -1)
                hh = getbrks(x1, x2)
                bracket!(ax2, x1, hh, x2, hh; text="$sgn$(compact_pval(p))", (!compact ? (; textoffset=2) : (; textoffset=1))...)
                push!(brackety, hh)
            end
        end
    end
    if isempty(brackety)
        ax2.height[] = 0
    else
        my = maximum(brackety)
        @warn "$my"
        ax2.height[] = (compact ? 8 : 10) * (my + 1)
        ylims!(ax2, 0, my + 0.5)
        # if 2more
        ylims!(ax2, 0, my + 1.0)
    end
    hidedecorations!(ax2)
    hidespines!(ax2)
    linkxaxes!(ax, ax2)
    if length(dat.cvals) != 1
        for v in (:x, :d)
            acc = Dict([c => Set() for c in dat.cvals])
            for vv in dat.vals
                push!(acc[vv.c], vv[v])
            end
            acc = collect(acc)
            if extrema(length.(last.(acc))) == (1, 1) && legendpos !== nothing
                Makie.axislegend(ax, [MarkerElement(marker=:rect, color=c) for (c, v) in acc], [string(first(v)) for (c, v) in acc], position=legendpos, orientation=:horizontal)
                break
            end
        end
    end
    ax.xticks[] = (axes(dat.xvals, 1), xname.(dat.xvals))
    hidexdecorations!(ax, ticklabels=false, ticks=false)
end

function plot_recon_imaging(grid, env; df, objg97, inj_vert_colors, caminj, sampslices, sampinfo, _...)
    ax1 = Makie.LScene(grid[1, 1][1, 1], width=96, show_axis=false)
    content(grid[1, 1]).alignmode[] = Outside()
    Label(grid[1, 1][1, 1, Top()], "Virus injection"; set_font_h2x()...)
    ax2 = Makie.Axis(grid[1, 2])
    ax3 = Makie.LScene(grid[1, 3], width=175, height=80, show_axis=false)
    [colgap!(grid, c, 0) for c in 1:2]

    with_mixed_scene(ax1, GLMakie) do mix1
        axspec = parseaxspec(mix1)
        !isfocus(:injsite) && return
        vertcolors = inj_vert_colors()
        Makie.mesh!(axspec.gl, objg97, color=vertcolors)
        teximg!(axspec.ax, TEXDocument(read("figs/injpos.tex", String), true, preamble=read("figs/preamble.tex", String)), position=(0.36, 0.38), tellwidth=false, space=:relative, scale=1)
        setup_camera(axspec.ax => (; use=caminj, patch=x -> (; deye=x.deye .* 0.95)))
    end

    arrows!(ax2, [Point2f(0, 0)], [Point2f(3, 0)], linewidth=2, arrowsize=6)
    text!(ax2, 1.5, 0, text="fMOST\nimaging", align=(:center, :bottom), offset=(0, 3); set_font_h2x()...)
    ax2.xautolimitmargin[] = (0.03, 0.09)
    hidedecorations!(ax2)
    hidespines!(ax2)

    with_mixed_scene(ax3, GLMakie) do mix2
        axspec = parseaxspec(mix2)
        !isfocus(:slicestack) && return
        coords = []
        pool(meth, img, scale) = meth(reshape(img, (size(img)..., 1, 1)), (scale, scale))[:, :, 1, 1]
        for f in readdir(sampslices.path, join=true)
            m = match(r"(\d+).tif", f)
            m === nothing && continue
            z = parse(Int, m.captures[1])
            img = load(f)
            x, y = size(img) .* 65.0
            #img = pool(Flux.maxpool, img, 3)
            #img = pool(Flux.meanpool, img, 5)
            img = map(img) do p
                v = clamp(p.val * 50, 0, 1)
                GLMakie.RGBAf(0, v, 0, 1.0)
            end
            plt = GLMakie.image!(axspec.gl, (0, x), (0, y), img; alpha=1.0, fxaa=false) #transparency=true)
            zz = z * 3.6
            Makie.translate!(plt, (0, 0, zz))
            plt = GLMakie.lines!(axspec.gl, Rect2f(0, 0, x, y), color=:gray, linewidth=0.3, fxaa=true)
            Makie.translate!(plt, (0, 0, zz))
            push!(coords, (x, y, zz))
        end
        camslices = (fov=9, cent=((sampinfo.center.*(1, 1, 3.6))[1:2]..., mean(extrema(last.(coords))) + 9000) .+ (-3000, 0, 0), deye=deye = normalize(Vec3f(-0.5, -0.7, -0.3)) .* 7.5f5, up=up = normalize(Vec3f(-1, 0, -0.18)))
        setup_camera(axspec.ax => camslices)
        let (x, y, z) = coords[end], (x0, y0, z0) = coords[1]
            bracket_in_3d!(axspec.ax, Point3f(0, y0, z0), Point3f(0, y, z); offset=1, text="23000 slices (432TB)", set_font_h2()...)
            bracket_in_3d!(axspec.ax, Point3f(0, 0, z), Point3f(0, y, z); offset=1, text="112005 px", set_font_h2()...)
            bracket_in_3d!(axspec.ax, Point3f(0, 0, z), Point3f(x, 0, z); offset=1, text="83988 px", orientation=:down, set_font_h2()...)
        end
    end
end

function plot_recon_proc(grid, env; df, rawinfo, objg97, caminj, _...)

    ax1 = Axis(grid[1, 1][1, 1], height=38)
    ax1x = Axis(grid[1, 1][2, 1])
    #ax2 = LScene(grid[1, 2], width=180, height=140, show_axis=false)
    hidedecorations!.([ax1, ax1x])
    hidespines!.([ax1, ax1x])
    Makie.rowgap!(content(grid[1, 1]), 1, 0)
    #Makie.colgap!(grid, 1, 6)

    teximg!(ax1, TEXDocument(read("figs/recon-proc.tex", String), true, preamble=read("figs/preamble.tex", String)))
    teximg!(ax1x, PDFDocument(read("monkeytemp/wyz-upload/fig1b-regions.pdf", String)), scale=1.0)

end

function plot_recon_single_neurons(grid, env; caminj, objg97, rawinfo, sampinfo, df, _...)
    Random.seed!(0)
    dfx = filter(r -> r.sample == "234197" && !r.qroot, df)
    neuronspick = include("monkeytemp/wyz-upload/julia/select-single-neurons.jl")
    if false
        append!(neuronspick, shuffle(sort(dfx, :axonlen)[1:200, :])[1:160, :path])
    end
    filter!(r -> r.path in neuronspick, dfx)

    axrpr = Makie.LScene(grid[1, 1], width=320 * 0.56, height=240 * 0.56, show_axis=false, scenekw=(; backgroundcolor=:black))
    with_mixed_scene(axrpr, RPRMakie, rprscale=6, rpriterations=300) do mix
        draw_brain_bricks(mix => caminj, dfx, objg97, rawinfo, sampinfo)
    end

    dfx = filter(r -> r.sample == "234197", df)
    ax2 = LScene(grid[1, 1, BottomRight()], width=58, height=45, tellwidth=false, tellheight=false, valign=:bottom, halign=:right, scenekw=(; backgroundcolor=:ivory), show_axis=false)
    setup_camera(ax2 => caminj)
    with_mixed_scene(ax2, GLMakie) do mix2
        draw_neurons_rawxyz!(mix2, dfx, rawinfo)
    end
    draw_mesh_overlay([objg97 => :lightgray], ax2)
end

function plot_recon_compare(grid, env; df, nmr, nmrdat, objinfl, _...)
    gridx = GridLayout(grid[1, 1], alignmode=Outside())
    inj = "G97-1bL"
    cam = (; use=:s, patch=x -> (; deye=x.deye .* 0.85 .+ 120000 .* x.up))
    dfx = filter(r -> r.inj == inj, df)

    ax1 = LScene(gridx[1, 1][1, 1], height=56, width=90, show_axis=false)
    ax2 = LScene(gridx[1, 1][1, 2], height=56, width=90, show_axis=false)
    colgap!(content(gridx[1, 1]), pt(0))
    Legend(gridx[2, 1][1, 1], [MarkerElement(marker=:circle, color=c) for c in (:red, :orange)][1:1], ["Injection/stimulation site", "site"][1:1], tellwidth=false, telheight=false)
    Makie.rowgap!(gridx, 1, 6)

    injpos = filter(r -> r.inj == inj, nmr.injs)[1, :injxyzm]
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        draw_nmr_data(injpos, log.(nmr.vert_conn[:, inj] .+ 1), mix1; obj=objinfl, sitecolor=:red)
    end
    Label(gridx[1, 1][1, 1, Makie.Top()], "Single-neuron projectomes", alignmode=Inside(); set_font_h2x()...)
    cb = Colorbar(gridx[2, 1][1, 2], limits=extrema(log.(nmr.vert_conn[:, inj] .+ 0.01)), vertical=false, tellwidth=false, flipaxis=false)
    cb.ticklabelsvisible = false
    cb.width = pt(80)
    #cb.tic=false
    #Label(grid[3, 1][2, 1][1, 2], "Projection/activity strength")
    cb.label = "Projection/activity strength"
    #rowgap!(content(gridx[2, 1][1,2]), 1, 4)

    cols = sortperm([norm(p .- injpos) for p in nmrdat.sites.sitepos])[1:9]
    c = argmax(c -> StatsBase.corspearman(nmrdat.vert_conn[:, nmrdat.sites[c, :site]], nmr.vert_conn[:, inj]), cols)
    site, sitepos = nmrdat.sites[c, [:site, :sitepos]]
    setup_camera(ax2 => cam)
    with_mixed_scene(ax2, GLMakie) do mix2
        draw_nmr_data(sitepos, log.(nmrdat.vert_conn[:, site] .+ 0.01), mix2; obj=objinfl, sitecolor=:red)
    end
    Label(gridx[1, 1][1, 2, Makie.Top()], "fMRI mapping", alignmode=Inside(); set_font_h2x()...)
end


function plot_all_injs(fig, env; df, nmt, obj, _...)
    with_caption(L"The results of single-neuron projectomes for each virus injection site.")
    grid = GridLayout(fig[1, 1])
    getter = NmtDat.getter(nmt)
    hiers = getter(reduce(hcat, df.soma_pos))[2]
    injid(r) = (; r.inj, r.injname)
    injs = sort(unique(injid.(eachrow(df))))
    function calc_heatmap(inj)
        stats = Dict()
        stats_soma = Dict()
        stats_ba = Dict()
        soma_src(r) = begin
            return "IGNORE"
            r.qroot && return "Qroot"
            f = hiers[rownumber(r)]
            f != "" && return f
            "NA"
        end
        for r in eachrow(df)
            injid(r) == inj || continue
            f = soma_src(r)
            stats_soma[f] = get(stats_soma, f, 0) + 1
            for ba in names(df)
                zz1data.is_projcol(ba) || continue
                count('|', ba) != 4 && continue
                v = r[ba]
                v == 0.0 && continue
                key = (f, ba)
                stats[key] = get(stats, key, 0.0) + v
                stats_ba[ba] = get(stats_ba, ba, 0.0) + v
            end
        end
        areas = map(first, sort(collect(stats_ba), lt=(a, b) -> a[2] > b[2]))
        if length(areas) > 10
            areas = areas[1:10]
        end
        lt_soma(a, b) = begin
            fixname(x) = begin
                if x[1] == "Qroot"
                    ("ZZZ2", 0)
                elseif x[1] == "Unknown"
                    ("ZZZ11", 0)
                elseif x[1] == "NA"
                    ("ZZZ13", 0)
                elseif x[1] == ""
                    ("ZZZ10", 0)
                else
                    ("AAAA", -x[2])
                end
            end
            a = fixname(a)
            b = fixname(b)
            a < b
        end
        somas_p = sort(collect(stats_soma), lt=lt_soma)
        somas = map(first, somas_p)
        somas_cnt = map(last, somas_p)
        mat = zeros(Float64, length(somas), length(areas))
        for idx in CartesianIndices(mat)
            key = somas[idx[1]], areas[idx[2]]
            mat[idx] = get(stats, key, 0.0)
        end
        (; mat, areas, somas, somas_cnt)
    end
    for (k, inj) in enumerate(injs)
        row, col = divrem(k - 1, 3) .+ 1
        gridx = Makie.GridLayout(grid[row, col])
        if row == 1
            col > 1 && colgap!(grid, col - 1, 16)
        else
            rowgap!(grid, row - 1, 24)
        end
        ax1 = Makie.LScene(gridx[2, 1][1, 1], width=pt(116), height=(150), show_axis=false)
        ax2 = Makie.LScene(gridx[1, 1], width=pt(150), height=pt(100), show_axis=false)
        setup_camera(ax1 => :h, ax2 => :s)
        with_mixed_scene(ax1, GLMakie) do mix1
            with_mixed_scene(ax2, GLMakie) do mix2
                draw_neurons(filter(r -> injid(r) == inj, df), mix1, mix2)
            end
        end
        draw_mesh_overlay([obj], ax1, ax2)
        rowgap!(gridx, 1, 0)
        Label(gridx[1, :, Makie.Top()], string(inj.inj, ": ", inj.injname), halign=:left; set_font_h1x()...)

        ax3 = Makie.Axis(gridx[2, 1][1, 2][2, 1][1, 1])
        content(gridx[2, 1][1, 2][2, 1]).alignmode[] = Outside()
        colgap!(content(gridx[2, 1]), 1, 2)
        hm = calc_heatmap(inj)
        cnt = sum(hm.somas_cnt)
        Label(gridx[2, 1][1, 2][1, 1], "N=$cnt", halign=:left, tellwidth=false; set_font_h2()...)
        rowgap!(content(gridx[2, 1][1, 2]), 1, 2)
        plt = heatmap!(ax3, log.(hm.mat .+ 1), colormap=[:white, :green])
        translate!(plt, 0, 0, -1000)
        ax3.yreversed[] = true
        ax3.yaxisposition = :right
        areanameic(n) = begin
            #@warn "xxx $n"
            m = match(r"^\w+([IC])\|", n)
            @assert m !== nothing
            string(areaname(n), '(', m.captures[1], ')')
        end
        ax3.yticks = (axes(hm.areas, 1), map(areanameic, hm.areas))
        ax3.yticklabelpad[] = -32
        ax3.yticklabelspace[] = 0.0
        hidexdecorations!(ax3)
        hideydecorations!(ax3, ticklabels=false)
        hidespines!(ax3)
    end
end

function plot_all_subtypes(fig, env; df, umap, obj, _...)
    with_caption(L"The results of single-neuron projectomes for each projectome-based neuron subtypes.")
    grid = with_subfig(fig[1, 1], env) do grid, env
        for K in sort(unique(df.cls))
            row, col = divrem(K - 1, 4) .+ 1
            gridx = GridLayout(grid[row, col])
            col > 1 && colgap!(grid, col - 1, 4)
            row > 1 && rowgap!(grid, row - 1, 12)
            ax1 = LScene(gridx[1, 1]; width=pt(120), height=pt(96), show_axis=false)
            ax2 = LScene(gridx[2, 1]; width=pt(120), height=pt(80), show_axis=false)
            Label(gridx[1, 1, Top()], "Subtype $K"; set_font_h1x()...)

            dfx = df[df.cls.==K, :]
            setup_camera(ax1 => :hr, ax2 => :s)
            with_mixed_scene(ax1, GLMakie) do mix1
                with_mixed_scene(ax2, GLMakie) do mix2
                    draw_neurons(dfx, mix1, mix2; mirrorby=NmtDat)
                end
            end
            draw_mesh_overlay([obj], ax1, ax2)

            ax3 = Makie.Axis(gridx[3, 1], height=pt(100))
            rowgap!(gridx, 1, 0)
            rowgap!(gridx, 2, 0)
            scatter!(ax3, umap[:, df.cls.!=K], color=:black, markersize=1, rasterize=6)
            scatter!(ax3, umap[:, df.cls.==K], color=:red, markersize=2.5, rasterize=6)
            hidedecorations!(ax3)
            ax3.topspinevisible[] = true
            ax3.rightspinevisible[] = true
        end
        grid
    end
end

function plot_arbor_detect_meth(grid, env; arbors, arbors_train, arbors_pred, _...)
    egsel(r) = endswith("monkeytemp/split-sample234197-ch1/root53280756-root255-good.swc.gz")(r.path)
    for (col, t) in [(1, "Input"), (3, "Output")]
        ax = LScene(grid[1, col], width=pt(160), height=pt(120), show_axis=false, alignmode=Outside(0, -25, 0, 0))
        with_mixed_scene(ax, GLMakie) do mix
            arb2col = Dict()
            per_arbor(pn, r, cond) = begin
                col == 1 && return :black => pt(0.3)
                cond && return :black => pt(0.3)
                r === nothing && return :black => 0.3
                c = get!(arb2col, r.arborpath) do
                    to_colormap(:tab10)[length(arb2col)%7+1]
                end
                c => pt(0.5)
            end
            per_node(pn, pa, n) = pa
            draw_subtrees(filter(egsel, arbors), mix; autotrunk=false, per_arbor, per_node)
            center!(mix.scene2)
            cam = cameracontrols(mix.scene2)
            cam.eyeposition[] = (cam.eyeposition[] - cam.lookat[]) * 0.8 + cam.lookat[]
        end
        #ax.title[] = t
    end

    grid2 = GridLayout(grid[1, 2])
    Random.seed!(0)
    for (i, r) in enumerate(eachrow(shuffle!(arbors_train)[1:4, :]))
        row, col = divrem(i - 1, 2) .+ 1
        ax = Axis(grid2[row, col])
        Main.draw_dend!(ax, r.dend)
        hidedecorations!(ax)
        hidespines!(ax)
    end
    Label(grid2[end+1, :], "Training examples"; set_font_h2()...)
    for (col, ff) in [(3, (xx) -> nothing), (1, (xx) -> xx[1, :dend].tgt .= 0)]
        ax2 = Axis(grid[2, col],)
        arbors_train = deepcopy(filter(egsel, arbors_pred))
        ff(arbors_train)
        Main.draw_dend!(ax2, arbors_train[1, :dend])
        hidedecorations!(ax2)
        hidespines!(ax2)
    end
    ax5 = Axis(grid[2, 2])
    arrows!(ax5, [Point2f(0, 0.5)], [Point2f(1, 0.0)], space=:data, linewidth=pt(2), arrowsize=pt(6))
    text!(ax5, 0.5, 0.5, space=:data, text="GNN"; align=(:center, :bottom), offset=(0, 1), set_font_h1x()...)
    hidespines!(ax5)
    hidedecorations!(ax5)
    Makie.colgap!(grid2, 1, 2)
    Makie.rowgap!(grid2, 1, 2)
    Makie.rowgap!(grid2, 2, 4)
    Makie.colgap!(grid, 1, 4)
    Makie.colgap!(grid, 2, 4)
    #Makie.colsize!(grid, 2, 100)
    Makie.rowgap!(grid, 1, 4)
    Makie.rowsize!(grid, 1, 80)
    #Makie.rowsize!(grid, 2, 30)
end
function plot_arbor_by_cls(grid, env; df, arbors, obj, cortarbors, _...)
    p2c = Dict(zip(cortarbors.arborpath, cortarbors.arborcls))
    dfx = filter(r -> haskey(p2c, r.arborpath), leftjoin(arbors, df, on=:path))

    ax = Makie.LScene(grid[2, 1], height=pt(145), width=pt(220), show_axis=false)
    ax2 = Makie.LScene(grid[1, 1], height=pt(165), show_axis=false)
    rowgap!(grid, 1, -8)
    setup_camera(ax => :s, ax2 => :hr)
    with_mixed_scene(ax, GLMakie) do mix
        with_mixed_scene(ax2, GLMakie) do mix2
            arb2col = Dict()
            per_arbor(pn, r, cond) = begin
                cond && return :transparent => 0.0
                r === nothing && return :transparent => 0.0
                c = get!(arb2col, r.arborpath) do
                    c = get(p2c, r.arborpath, 0)
                    c == 0 && return to_color(:black)
                    to_colormap(:Set1_8)[c]
                    ###Makie.wong_colors()[length(arb2col)%7+1]
                end
                c => 0.3
            end
            per_node(pn, pa, n) = pa
            draw_subtrees(dfx, mix, mix2; per_arbor, per_node, drawsoma=false, mirrorby=NmtDat) #= , mix2, mix3 =#
        end
    end
    draw_mesh_overlay([obj], ax, ax2) #= , ax2,ax3 =#
end

function plot_arbor_depth_dist(grid, env; df, cortarbors, hclust1, _...)
    cortarbors = cortarbors[hclust1.order, :]
    ax1 = Axis(grid[1, 1], height=13)
    ax2 = Axis(grid[2, 1], height=35)
    ax3 = Axis(grid[3, 1], height=7)
    gridx = GridLayout(grid[4, 1])
    rowgap!(grid, 1, 1)
    rowgap!(grid, 2, 1)
    rowgap!(grid, 3, 2)
    linkxaxes!(ax1, ax2)
    zz1vis.dendrogram!(ax1, hclust1, color=:black, rasterize=6)
    ylims!(ax1, 0, nothing)
    depths = reduce(hcat, cortarbors.fv_whist)
    depths = depths ./ (maximum(depths, dims=1) .+ 1)
    Makie.heatmap!(ax2, permutedims(depths[:, :], (2, 1)),
        inspector_label=(p, i, x) -> string(hclust1.order[i[1]]),
        rasterize=6,
    )
    hidedecorations!(ax1)
    hidexdecorations!(ax2)
    hideydecorations!(ax2, ticklabels=false, label=false)
    ax2.ylabel = "Laminar depth"
    yaxis_as_layer(ax2, 0.5, 101.5)
    hidespines!(ax1)
    hidespines!(ax2)
    cols = Makie.to_colormap(:Set1_8)
    for c in sort(unique(cortarbors.arborcls))
        a, b = extrema(findall(cortarbors.arborcls .== c))
        p = vspan!(ax3, a - 0.5, b + 0.5, color=(cols[c], 1.0))
        translate!(p, 0, 0, -100)
        text!(ax3, (a + b) / 2, 0, text="#$c", align=(:center, :center))
        LTeX(gridx[1, c], PDFDocument(read("monkeytemp/wyz-upload/figs5-example-arbor-cluster-$c.pdf", String)), scale=0.9667)
        Label(gridx[1, c, Top()], "#$c"; color=:white, tellheight=false, valign=:top, set_font_h2x()...)
        c > 1 && colgap!(gridx, c - 1, 0)
    end
    linkxaxes!(ax2, ax3)
    hidespines!(ax3)
    hidedecorations!(ax3)
    xlims!(ax3, 0.5, size(cortarbors, 1) + 0.5)
end


function plot_all_subdomains_intra_pfc_proj_flat(grid, env; df, subregs, dfarbor, highlights, objflat, objphy, _...)
    inj2c = inj2color_func(df)
    kkk = 0
    for rrr in eachrow(subregs)
        length(rrr.neurons) < 10 && continue
        kkk += 1
        rowcol = divrem(kkk - 1, 3) .+ 1
        ax = Axis(grid[rowcol...], aspect=Makie.DataAspect(), width=130, height=130, title="$(rrr.subreg)")
        plot_flatmap_base!(ax, highlights, objflat, objphy)
        xlims!(ax, -0.85, 0.45)
        ylims!(ax, -0.65, 0.65)

        dfx = filter(r -> r.path in rrr.neurons && r.ipsi, dfarbor)
        Makie.scatter!(ax, dfx.arboruvcent2, color=inj2c.(dfx.inj))
    end
end

function get_sorted_ba_cnt(df, hierof)
    ba2cnt = Dict()
    for r in eachrow(df)
        ba = hierof(r)
        ba2cnt[ba] = get(ba2cnt, ba, 0) + 1
    end
    sortby((ba, cnt)) = begin
        (brain_region_props(ba)[1], -cnt)
    end
    sort(collect(ba2cnt), by=sortby)
end

function plot_recon_injtable(grid, env; df, _...)
    grid.alignmode[] = Outside()
    ax1 = Makie.Axis(grid[2, 1], height=pt(90))
    ax1r = Makie.Axis(grid[2, 2])
    ax1t = Makie.Axis(grid[1, 1])
    Makie.linkxaxes!(ax1, ax1t)
    Makie.linkyaxes!(ax1, ax1r)
    colgap!(grid, 1, 1)
    rowgap!(grid, 1, 1)
    hideydecorations!(ax1r)
    hidexdecorations!(ax1t)
    hidedecorations!(ax1, label=false, ticklabels=false)

    n2ms(n) = sqrt(n)
    hierof(r) = r.qroot ? "Unknown" : (brain_region_props(r.somahier)[1] <= 2 ? r.somahier : "Other")
    ba2cnt = get_sorted_ba_cnt(df, hierof)
    vals = []
    injs = sort(unique(df.inj))
    allrows = filter(r -> r != "Unknown", first.(ba2cnt))
    qrootrow = []
    for (col, inj) in enumerate(injs)
        for (row, ba) in enumerate(allrows)
            n = size(filter(r -> hierof(r) == ba && r.inj == inj, df), 1)
            push!(vals, (; row, col, n))
        end
        n = size(filter(r -> hierof(r) == "Unknown" && r.inj == inj, df), 1)
        push!(qrootrow, (; col, n))
    end
    Makie.scatter!(ax1, [t.col for t in vals], [t.row for t in vals], markersize=[n2ms(t.n) for t in vals], color=:black)
    ax1.yreversed[] = true
    ax1.yticks[] = (axes(allrows, 1), areaname.(allrows))
    ax1.xticks[] = (axes(injs, 1), injs)
    ax1.ylabel = "Brain regions"
    ax1.xlabel = "Injection sites"
    #ax1.height[] = length(unique([t.row for t in vals])) * 7
    rotatexlabel!(ax1, π / 4)
    ax1.xgridvisible[] = true
    ax1.ygridvisible[] = true

    rowsum = [sum([t.row == r ? t.n : 0 for t in vals]) for r in axes(allrows, 1)]
    Makie.barplot!(ax1r, axes(allrows, 1), rowsum, color=:black, direction=:x)
    Makie.xlims!(ax1r, 0, nothing)
    #ax1r.xlabel[] = "No. of Neurons"
    Label(grid[2, 2, Right()], "No. of neurons"; set_font_h2x()..., tellwidth=false, halign=:right, valign=:bottom, padding=(0, 0, 4, 0), rotation=-0.5π)
    ax1r.yticklabelsvisible[] = false
    ax1r.width[] = 25
    ax1r.yreversed[] = true
    ax1r.xminorticksvisible = false
    rotatexlabel!(ax1r, π / 4)

    colsum = [sum([t.col == c ? t.n : 0 for t in vals]) for c in axes(injs, 1)]
    Makie.barplot!(ax1t, axes(injs, 1), colsum, dodge=1, n_dodge=2, color=:black, label="Soma complete")
    Makie.barplot!(ax1t, axes(injs, 1), [sum([t.col == c ? t.n : 0 for t in qrootrow]) for c in axes(injs, 1)], dodge=2, n_dodge=2, color=:gray, label="Soma missing")
    Makie.ylims!(ax1t, 0, nothing)
    ax1t.height[] = 25
    ax1t.xticklabelsvisible[] = false
    ax1t.yminorticksvisible = false
    Label(grid[1, 1, Top()], "No. of neurons"; set_font_h2x()..., tellheight=false, halign=:left, valign=:top, padding=(4, 0, 0, 0))
    #ax1t.ylabel[] = 

    #align_axis_label(:x, ax1, ax1r)
    #align_axis_label(:y, ax1, ax1t)
    axislegend(ax1t, ax1t, position=:rt, margin=(2, -20, 2, 2))
end

function plot_recon_somadist(grid, env; df, obj, objflat, objphy, highlights, _...)
    ax1 = Axis(grid[1, 1][1, 1], height=126, width=126)
    ax2 = LScene(grid[1, 1][2, 1][1, 1], height=90, width=126, show_axis=false, valign=:top, tellheight=false)
    rowgap!(content(grid[1, 1]), 1, -8)
    #content(grid[1,1][2,1]).height=72

    plot_flatmap_base!(ax1, highlights, objflat, objphy)
    plot_inj_flatmap!(ax1, df)

    setup_camera(ax2 => :hf)
    with_mixed_scene(ax2, GLMakie) do mix
        plot_inj_in3d!(mix, df, obj)
    end
    obj1 = load("interm/mesh-nmt-lat_PFC-l.obj") |> Main.add_mirror_mesh(NmtDat.mirror)
    #obj2 = load("interm/mesh-nmt-OFC-l.obj")
    #obj3 = load("interm/mesh-nmt-motor-l.obj")
    draw_contours([obj1 => (; color=:black)], ax2)
    draw_mesh_overlay([obj], ax2)
end

function brain_region_props(ba)
    if contains("|lat_PFC|")(ba)
        (0, :black)
    elseif contains("|OFC|")(ba)
        (1, :gray)
    elseif contains("|Frontal|")(ba)
        (2, :gray)
    else
        (3, :gray)
    end
end

function plot_recon_res_by_inj(grid, env; df, obj, _...)
    ax1 = LScene(grid[1, 1], height=pt(90), show_axis=false)
    ax2 = LScene(grid[2, 1], height=pt(80), show_axis=false)
    rowgap!(grid, 1, -2)
    setup_camera(ax1 => :hr, ax2 => :s)
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            plot_all_fibers_by_inj!(mix1, mix2, df, obj; limit=100)
        end
    end
    draw_mesh_overlay([obj => :white], ax1, ax2)
end

function plot_result_itptct(grid, env; df, obj, _...)
    ax11 = LScene(grid[1, 1][1, 1], width=100, height=130, show_axis=false)
    ax12 = LScene(grid[1, 2][1, 1], width=126, height=90, show_axis=false)
    ax3 = Axis(grid[1, 2][3, 1][1, 1])
    content(grid[1, 2][3, 1]).alignmode[] = Outside()
    ax21 = LScene(grid[1, 1][2, 1], width=100, height=130, show_axis=false)
    ax22 = LScene(grid[1, 2][2, 1], width=126, height=90, show_axis=false)
    Makie.colgap!(grid, 1, 2)
    Makie.rowgap!(content(grid[1, 1]), 1, 2)
    Makie.rowgap!(content(grid[1, 2]), 1, 4)
    Makie.rowgap!(content(grid[1, 2]), 2, 4)
    setup_camera(ax11 => :h, ax12 => :s)
    setup_camera(ax21 => :h, ax22 => :s)

    typ2c = typ2color_func(df)
    rowlabel(r) = string(r.typ)
    rowcolor(r) = typ2c(r.typ)
    with_mixed_scene(ax11, GLMakie) do mix1
        with_mixed_scene(ax12, GLMakie) do mix2
            dfx = pick_neurons_with_limit(filter(istypit, df), r -> r.typ, 10)
            draw_neurons(dfx, mix1, mix2; rowlabel, rowcolor, mirrorby=NmtDat, linewidth=0.2)
        end
    end
    draw_mesh_overlay([obj], ax11, ax12)
    objstr = load("interm/mesh-nmt-Str-l.obj")
    draw_contours([objstr => (; color=:black)], ax11, ax12)

    with_mixed_scene(ax21, GLMakie) do mix1
        with_mixed_scene(ax22, GLMakie) do mix2
            dfx = pick_neurons_with_limit(filter(!istypit, df), r -> r.typ, 10)
            draw_neurons(dfx, mix1, mix2; rowlabel, rowcolor, mirrorby=NmtDat, linewidth=0.2)
        end
    end
    draw_mesh_overlay([obj], ax21, ax22)
    draw_contours([load("interm/mesh-nmt-$ba-lr.obj") => (; color=:black) for ba in [:Mid, :MD]], ax21, ax22)

    for typ in unique(df.typ)
        ecdfplot!(ax3, push!(filter(r -> r.typ == typ, df).axonlen ./ 1000, 1000), color=typ2c(typ), label=string(typ), linewidth=0.7)
    end
    ax3.xscale[] = log10
    ax3.xlabel = "Axon length (mm)"
    ax3.ylabel = "Empirical CDF (%)"
    ax3.xticks[] = [10, 100, 1000]
    #ax3.xtickformat[] = (vv) -> [string(Int(v)) for v in vv]
    ax3.ytickformat[] = (vv) -> [string(Int(v * 100)) for v in vv]
    ylims!(ax3, 0, 1)
    axislegend(ax3, ax3, position=:rb) |> fix_line_legs
end

function fix_line_legs(leg)
    for (_, grp) in leg.entrygroups[]
        for ent in grp
            if length(ent.elements) == 1 && ent.elements[1] isa Makie.LineElement
                e = pop!(ent.elements)
                push!(ent.elements, Makie.PolyElement(color=e.linecolor[]))
            end
        end
    end
    notify(leg.entrygroups)
end

function plot_result_depth(grid, env; df, _...)
    ax = Makie.Axis(grid[1, 1], alignmode=Outside())
    plot_depth_itptct!(ax, df)
end

function plot_result_depth_g97(grid, env; df, _...)
    ax1 = Makie.Axis(grid[1, 1], height=200)
    teximg!(ax1, PDFDocument(read("monkeytemp/wyz-upload/soma-G97-layer-plot.pdf", String)), scale=1.0)
    hidedecorations!(ax1)
    hidespines!(ax1)
end

function plot_result_depth_g97_dist(fig, env; df, _...)
    with_caption(L"Soma laminar depth distribution of neuron types in injection site G97-1bL")
    grid = GridLayout(fig[1, 1])
    ax = Makie.Axis(grid[1, 2])
    dfx = filter(r -> r.inj == "G97-1bL", df)
    plot_depth_itptct!(ax, dfx)
    ax2 = Makie.Axis3(grid[1, 1], elevation=0, azimuth=0, zticklabelpad=0, zlabeloffset=20, protrusions=10)
    filter!(r -> !isnan(r.somaw) && !r.qroot, dfx)
    typ2c = typ2color_func(df)
    Makie.scatter!(ax2, splat(vcat).(zip(dfx.somauv, dfx.somaw)), color=typ2c.(dfx.typ))
    ax2.zlabel[] = "Depth"
    Makie.zlims!(ax2, 1, 0)
end

function plot_result_clust(grid, env; df, hclust, _...)
    gridx = GridLayout(grid[1, 1], alignmode=Outside())
    ax1 = Makie.Axis(gridx[1, 1], height=pt(32))
    ylims!(ax1, 0, nothing)
    ax2 = Axis(gridx[2, 1])
    ax3 = Axis(gridx[3, 1])
    ax4 = Axis(gridx[4, 1])
    ax5 = Axis(gridx[5, 1])
    ax6 = Axis(gridx[6, 1])
    for i in 1:5
        Makie.rowgap!(gridx, i, pt(1.2))
    end
    ax4x = Axis(gridx[4, 2])
    ax5x = Axis(gridx[5, 2])
    colgap!(gridx, 1, 0.5)
    plot_global_heatmap!(ax1, ax4, ax5, ax3, ax6, ax2, df, hclust, skipic=true, ax1x=ax4x, ax2x=ax5x)
end

function plot_result_umap(grid, env; df, obj, umap, _...)
    grid.tellwidth = false
    grid.halign = :left
    #grid.width = pt(170)
    ax = Makie.Axis(grid[1, 1])

    typ2c = typ2color_func(df)
    Makie.scatter!(ax, umap, color=typ2c.(df.typ), markersize=pt(2), inspector_label=(p, i, x) -> df[i, :path])
    for c in unique(df.cls)
        Makie.text!(ax, Point2f(median(umap[:, df[:, :cls].==c], dims=2)...), text="#$c", fontsize=pt(6), align=(:center, :center), color=:black, glowcolor=(:white, 0.3), glowwidth=0.2)
    end
    typs = sort(unique(df.typ), by=typ_order)
    leg = Legend(grid[1, 2], [Makie.MarkerElement(marker=:circle, color=typ2c(t)) for t in typs], typs; tellheight=false)
    colgap!(grid, 1, 2)
    leg.orientation[] = :vertical
    hidedecorations!(ax, label=false)
    ax.topspinevisible[] = true
    ax.rightspinevisible[] = true
    ax.xlabel[] = "UMAP/1"
    ax.ylabel[] = "UMAP/2"
    return
    df = pick_neurons_with_limit(df, r -> r.cls, 10)
    filter!(zz1data.select_neurons(:all_demo), df)
    for (col, c) in enumerate(sort(unique(df.cls)))
        row, col = divrem(col - 1, 2) .+ 1
        gridx = grid[2, 1][row, col]
        ax1 = Makie.LScene(gridx[1, 1], width=pt(80), height=pt(100), show_axis=false)
        ax2 = Makie.LScene(gridx[2, 1], width=pt(80), height=pt(50), show_axis=false)
        col > 1 && colgap!(content(grid[2, 1]), col - 1, 3)
        row > 1 && rowgap!(content(grid[2, 1]), row - 1, 3)
        setup_camera(ax1 => :h, ax2 => :s, link=false)
        with_mixed_scene(ax1, GLMakie) do mix1
            with_mixed_scene(ax2, GLMakie) do mix2
                dfx = filter(r -> r.cls == c, df)
                draw_neurons(dfx, mix1, mix2, mirrorby=NmtDat)
                typ = zz1data.vote(dfx, :typ, :path)[end].first
                Makie.Label(gridx[1, 1, Makie.Top()], "#$c ($typ)", fontsize=7)
                Makie.rowgap!(Makie.content(gridx), 1, 0)
            end
        end
        draw_mesh_overlay([obj], ax1, ax2)
    end
    rowgap!(grid, 1, 6)
    content(grid[2, 1]).alignmode[] = Outside()
end

function plot_result_table_inj(grid, env; df, _...)
    ax1 = Makie.Axis(grid[1, 1])
    ax1d = Makie.Axis(grid[1, 1, Bottom()])
    ax1r = Makie.Axis(grid[1, 1, Right()], tellheight=false)
    n2ms(n) = sqrt(n)
    vals = []
    subtyps = sort(unique(df.cls))
    injs = sort(unique(df.inj))
    for (col, subt) in enumerate(subtyps)
        for (row, inj) in enumerate(injs)
            n = size(filter(r -> r.cls == subt && r.inj == inj, df), 1)
            push!(vals, (; row, col, n))
        end
    end
    scatter!(ax1, [t.col for t in vals], [t.row for t in vals], markersize=[n2ms(t.n) for t in vals], color=:black)
    ax1.xticks[] = (axes(subtyps, 1), ["#$c" for c in subtyps])
    ax1.xticklabelrotation = π / 3
    ax1.xticklabelalign = (:left, :center)
    ax1.yticks[] = (axes(injs, 1), string.(injs))
    ax1.height[] = length(injs) * 6
    ax1.yreversed[] = true
    ax1.xaxisposition[] = :top
    ax1.xlabel[] = "Subtypes"
    ax1.topspinevisible[] = true
    ax1.bottomspinevisible[] = false
    hidedecorations!(ax1, label=false, ticklabels=false)
    ax1.xgridvisible[] = true
    ax1.ygridvisible[] = true
    ax1.ylabel[] = "Injection sites"

    subtypents = map(enumerate(subtyps)) do (col, subtyp)
        pp = map(v -> v.n, filter(v -> v.col == col, vals))
        col => StatsBase.entropy(pp ./ sum(pp))
    end

    if ax1d !== nothing
        Makie.barplot!(ax1d, first.(subtypents), last.(subtypents), color=:black, width=0.5)
        ax1d.height[] = 25
        ax1d.ylabel[] = "Entropy"
        ylims!(ax1d, 0, nothing)
        hidexdecorations!(ax1d)
        hideydecorations!(ax1d, label=false, ticklabels=false, ticks=false)
        ax1d.yreversed[] = true
        ax1d.topspinevisible[] = true
        ax1d.bottomspinevisible[] = false
        Makie.linkxaxes!(ax1, ax1d)
    end

    injents = map(enumerate(injs)) do (row, inj)
        vals = map(subtyps) do cls
            size(filter(r -> r.cls == cls && r.inj == inj, df), 1)
        end
        row => StatsBase.entropy(vals ./ sum(vals))
    end
    if ax1r !== nothing
        Makie.barplot!(ax1r, first.(injents), last.(injents), color=:black, direction=:x)
        ax1r.yreversed[] = true
        hideydecorations!(ax1r)
        hidexdecorations!(ax1r, label=false, ticklabels=false, ticks=false)
        ax1r.xlabel[] = "Entropy"
        ax1r.width[] = 30
        ax1r.tellheight[] = false
        ax1r.xaxisposition = :top
        ax1r.topspinevisible = true
        ax1r.bottomspinevisible = false
        xlims!(ax1r, 0, nothing)
        Makie.linkyaxes!(ax1, ax1r)
    end
    align_axis_label(:y, ax1, ax1d)
    align_axis_label(:x, ax1, ax1r)
    ax1, ax1d
end

function plot_result_table(grid, env; df, _...)
    ax1 = Axis(grid[1, 1])
    ax2 = Axis(grid[2, 1])
    Label(grid[1, 1, Right()], "", width=pt(30))
    rowgap!(grid, 1, 3)
    plot_subtyp_inj_matrix!(ax1, df)
    plot_subtyp_soma_depth!(ax2, df)
    Makie.linkxaxes!(ax2, ax1)
    hidexdecorations!(ax2)
    yaxis_as_layer(ax2, 0, 1)
    align_axis_label(:y, ax1, ax2)
    ax1, ax2
end

function draw_soma_dist3d(df, axspecs...; rowcolor)
    axspecs = parseaxspec.(axspecs)
    dfxx = filter(r -> !r.qroot, df)
    pts = [NmtDat.isleft(p) ? p : NmtDat.mirror(p) for p in dfxx.soma_pos]
    for ax in axspecs
        meshscatter!(ax.gl, pts, color=rowcolor.(eachrow(dfxx)), markersize=500)
        mesh!(ax.gl, load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.white_surface.rsl.gii.obj"), color=:gray)
    end
end

function plot_longproj_demo(grid, env; dfx, obj, highlights, objflat, objphy, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfx)

    ax1 = LScene(grid[1, 2][1, 1], width=pt(120), height=pt(80), show_axis=false)
    ax2 = LScene(grid[1, 1], width=pt(96), height=pt(120), show_axis=false)
    gridx = GridLayout(grid[1, 2][2, 1])
    # ax3 = LScene(grid[1, 1][2, 1][1, 1], width=pt(45), show_axis=false)
    # ax4 = LScene(grid[1, 1][2, 1][1, 2], width=pt(50), show_axis=false)
    colgap!(grid, 1, -pt(10))
    rowgap!(content(grid[1, 2]), 1, 0)

    setup_camera(ax1 => :s, ax2 => :h)
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            draw_neurons(dfx, mix1, mix2; pickby=r -> r.cls, rowlabel, rowcolor, mirrorby=NmtDat)
            axislegend(ax2, mix1.scene2, unique=true, tellwidth=true) |> fix_line_legs
        end
    end
    draw_mesh_overlay([obj], ax1, ax2)
    #draw_contours([load("interm/mesh-nmt-$x-l.obj") for x in [:PMC, :IPL, :ITC, Symbol("STG/STSd")]], ax1, ax2)
    draw_contours([load("interm/mesh-nmt-$x-l.obj") => (; color=c) for (x, c) in [:Temporal => :blue, :Parietal => :red]], ax1, ax2)
    #axislegend(ax2, [PolyElement(strokecolor=c, color=:transparent, strokewidth=1) for c in [:blue, :red]], ["Temporal", "Parietal"], position=:rb)

    plot_longproj_types(gridx, env; env...)

    # setup_camera(ax3 => :csoma, ax4 => :ssoma)
    # with_mixed_scene(ax3, GLMakie) do mix3
    #     with_mixed_scene(ax4, GLMakie) do mix4
    #         draw_soma_dist3d(dfx, mix3, mix4; rowcolor)
    #     end
    # end
    # draw_mesh_overlay([obj], ax3, ax4)
end

function draw_sankey_vert!(ax1, ax2, df; linkto=nothing, rowgrp0=nothing, rowgrp=nothing, grplabel=nothing, grpcolor=nothing, label="Soma location")
    grp0s = rowgrp0.(eachrow(df))
    allgrp0s = unique(grp0s)
    grps = rowgrp.(eachrow(df))
    off = 0
    grp02off = Dict{Any,Int}()
    getgrp0(g0, n0) = begin
        p = get(grp02off, g0) do
            minimum(findall(==(g0), grp0s))
        end
        grp02off[g0] = p + n0
        p
    end
    mix(a, b, t) = a * (1 - t) + b * t
    for (i, g) in enumerate(sort(unique(grps)))
        n = count(==(g), grps)
        vspan!(ax2, off + 1 - 0.5, off + n + 0.5, color=grpcolor(i, g))
        text!(ax2, off + (n + 1) / 2, 0, text=grplabel(i, g), align=(:center, :center))
        offx = 0
        for g0 in allgrp0s
            n0 = count(==((g0, g)), zip(grp0s, grps))
            o0 = getgrp0(g0, n0)
            pl = [Point2f(mix(off + offx + 1 - 0.5, o0 - 0.5, t), t) for t in 0:1]
            pr = [Point2f(mix(off + offx + n0 + 1 - 0.5, o0 - 0.5 + n0, t), t) for t in 0:1]
            band!(ax1, pl, pr, color=(grpcolor(i, g), 0.6))
            offx += n0
        end
        off += n
    end
    hidedecorations!(ax1)
    hidexdecorations!(ax2)
    tightlimits!(ax1)
    tightlimits!(ax2)
    hideydecorations!(ax2, ticklabels=false)
    ax2.yticks = ([-1, 0, 1], ["", "$label", ""])
    ax2.yaxisposition = :right
    hidespines!(ax2)
    hidespines!(ax1)
    if linkto !== nothing
        linkxaxes!(ax1, linkto)
    end
    linkxaxes!(ax2, ax1)
    ylims!(ax2, -1, 1)
    ax2.height = 7
end
function plot_longproj_heatmap(grid, env; dfx, obj, highlights, objflat, objphy, _...)
    cols = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-long-projection.jl")
    ax30 = Makie.Axis(grid[1, 1])
    ax31 = Makie.Axis(grid[2, 1])
    ax32 = Makie.Axis(grid[3, 1])
    ax33 = Makie.Axis(grid[4, 1])
    ax31r = Axis(grid[2, 1, Right()], width=pt(40))
    rowgap!(grid, 1, 0)
    rowgap!(grid, 2, 0)
    rowgap!(grid, 3, 0)
    plot_subtyp_spans!(ax30, dfx; linkto=ax31, gridto=[ax31], rowgrp=r -> r.cls, grplabel=(i, g) -> "#$g")
    draw_neuron_proj_hm!(ax31, nothing, dfx, cols; skipic=true)
    ax31.width[] = nothing

    for (o, tgt) in [(0.73, "Parietal"), (0.45, "Temporal")]
        a, b = extrema(findall(zz1data.matchhier(tgt), cols))
        bracket!(ax31r, o, a, o, b, text="$tgt Lobe"; set_font_h2()...)
    end
    ax31r.yreversed[] = true
    linkyaxes!(ax31, ax31r)
    ax31r.yautolimitmargin = (0.01, 0.01)
    hidedecorations!(ax31r)
    hidespines!(ax31r)
    ### Makie.lines!(ax31r, [(0, 4), (1, 5), (0, 6), (1, 7), (-0.1, 10.3)], color=:red)
    xlims!(ax31r, 0, 1)
    relax_clipping(ax31r, (9, 9, 9, 9))

    rowgrp(r) = begin
        zz1data.matchhier("dlPFC")(r.somahier) && return :dlPFC
        zz1data.matchhier("vlPFC")(r.somahier) && return :vlPFC
        return :zzz
    end
    draw_sankey_vert!(ax32, ax33, dfx; linkto=ax31, rowgrp0=r -> r.cls, rowgrp, grplabel=(i, g) -> g === :zzz ? "" : "$g", grpcolor=(i, g) -> get((; dlPFC=:red, vlPFC=:lightblue), g, :gray))
end

function plot_longproj_flatmap(grid, env; dfx, obj, highlights, objflat, objphy, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfx)

    ax4 = Makie.Axis(grid[1, 1], width=120, height=120)
    plot_flatmap_base!(ax4, highlights, objflat, objphy)
    plot_flatmap_soma!(ax4, dfx; rowcolor)
    #limits!(ax4, -0.35, 0.00, -0.2, 0.28)
end

function plot_longproj_types(grid, env; dfx, obj, highlights, objflat, objphy, _...)
    grid.alignmode[] = Outside()
    ax2 = Makie.Axis(grid[1, 1])
    rowx(r) = (; r.Temporal, r.Parietal)
    xname(r) = begin
        r.Temporal && r.Parietal && return "to Both"
        r.Temporal && return "to Temporal"
        r.Parietal && return "to Parietal"
        "to Neither"
    end
    test_venn!(ax2, dfx, :Temporal => :blue, :Parietal => :red; tagname=nothing, scale=0.8)
    hidedecorations!(ax2)
    hidespines!(ax2)
    Legend(grid[1, 2], [PolyElement(color=c) for c in [:blue, :red]], ["to Temporal", "to Parietal"])
    colgap!(grid, 1, 0)
end

function plot_longproj_diverg(grid, env; dfx, nodesuvw, obj, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfx)
    dfxx = filter(zz1data.select_neurons(:long_it_ipsi_g97), dfx)

    ax1 = LScene(grid[1, 1], width=140, height=96, show_axis=false)
    Label(grid[1, 1, Top()], "Injection site G97-1bL", tellheight=false, valign=:top; set_font_h1x()...)

    setup_camera(ax1 => :s)
    with_mixed_scene(ax1, GLMakie) do mix1
        draw_neurons(dfxx, mix1; pickby=r -> r.cls, rowcolor, rowlabel)
        axislegend(ax1, mix1.scene2, unique=true, position=:lb) |> fix_line_legs
    end
    draw_mesh_overlay([obj], ax1)
    draw_contours([load("interm/mesh-nmt-$x-l.obj") => (; color=:black) for x in [:LIP, :TE]], ax1)
end

function plot_longproj_diverg_soma(grid, env; dfx, nodesuvw, obj, _...)
    grid.alignmode[] = Outside()
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfx)
    dfxx = filter(zz1data.select_neurons(:long_it_ipsi_g97), dfx)
    for r in eachrow(dfxx)
        nodesinfo = nodesuvw(r.sample, r.channel)
        r.somaw = nodesinfo[r.root].rawdepth
    end

    ax1 = Axis(grid[1, 1][1, 1])
    ax1t = Axis(grid[1, 1][1, 1, Top()], valign=:bottom)
    #ax1.width[] = pt(40)
    ylims!(ax1t, 0, 1)
    violin_and_compare!(ax1, filter!(r -> !r.qroot, dfxx), ax1t; rowx=rowlabel, xname=identity, rowy=r -> r.somaw, rowcolor, statif=(a, b) -> a.x != b.x, datalimits=(0, 1), compact=true)
    #ax2.titlegap[]=12
    yaxis_as_layer(ax1, 0, 1)
    ax1.ylabel = "Soma laminar depth"
end

function plot_longproj_diverg_soma_demo(grid, env; dfx, nodesuvw, obj, _...)
    LTeX(grid[1, 1], PDFDocument(read("monkeytemp/wyz-upload/fig3-soma-G97-layer.pdf", String)), width=pt(220))
end

function plot_longproj_arbors(grid, env; dfx, arbors, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfx)
    arbors = leftjoin(arbors, dfx, on=:path) |> dropmissing!

    ax3 = Axis(grid[1, 1], height=36)

    dats = []
    for (tgt, subtgt) in [:LIP => :lip, :TE => :te, :PCC => :pcc, :STSd => :stsd]
        dfxxx = filter(r -> r.ipsi && r.sample in ("234197", "232717") && zz1data.select_neurons(Symbol(:long_it_ipsi_, subtgt))(r) && zz1data.matchhier("$tgt")(r.arborhier), arbors)
        cls = unique(dfxxx.cls)
        @assert length(cls) == 1
        cls = first(cls)
        nnn = length(unique(dfxxx.path))
        @warn "nnn $nnn"
        xx, yy = calc_fvhist(dfxxx.fv_whist)
        lines!(ax3, xx, yy, color=rowcolor(cls), linestyle=nnn >= 5 ? :solid : :solid, label="#$cls ($tgt)")
        push!(dats, (tgt, cls) => yy)
    end
    Legend(grid[1, 1, Right()], ax3, halign=:left, valign=:bottom, tellwidth=false, tellheight=false, nbanks=2, margin=(4, 2, -16, 2)) |> fix_line_legs

    xaxis_as_layer(ax3, 0, 1)
    #axislegend(ax3, ax3)
    ylims!(ax3, 0, nothing)
    ax3.ylabel[] = "Density"
    hideydecorations!(ax3, label=false)
    ax3
end

function plot_longproj_arbors_demo(grid, env; dfx, _...)
    for (i, (tgt, subtgt)) in enumerate([:LIP => :lip, :TE => :te, :PCC => :pcc, :STSd => :stsd])
        dfxxx = filter(r -> r.sample == "234197" && zz1data.select_neurons(Symbol(:long_it_ipsi_, subtgt))(r), dfx)
        row, col = divrem(i - 1, 2) .+ 1
        LTeX(grid[row, col], PDFDocument(read("monkeytemp/wyz-upload/fig3-capture-arbors-$subtgt.pdf", String)))
        col > 1 && colgap!(grid, col - 1, 2)
        row > 1 && rowgap!(grid, row - 1, 2)
        cls = unique(dfxxx.cls)
        @assert length(cls) == 1
        cls = first(cls)
        Label(grid[row, col, Top()], "#$cls ($tgt)"; tellheight=false, color=:white, valign=:top, set_font_h2()...)
    end
end

arborcenter(arbors, path, cond) = begin
    dftmp = filter(rr -> rr.path == path && cond(rr), arbors)
    size(dftmp, 1) == 0 && return Vec3f(NaN)
    SVector(median(reduce(hcat, dftmp.center), dims=2)...)
end

function plot_longproj_converg(grid, env; dfx, obj, highlights, objflat, objphy, arbors, _...)
    rowcolor = cls_temp_color(dfx)
    rowlabel(r) = string("#", r.cls)
    dfxx = filter(zz1data.select_neurons(:long_it_ipsi_pari), dfx)

    grid.width[] = 105
    ax1 = LScene(grid[1, 1], width=120, height=150, show_axis=false, halign=:left, tellwidth=false)
    setup_camera(ax1 => :h)
    with_mixed_scene(ax1, GLMakie) do mix1
        draw_neurons(dfxx, mix1; rowcolor, rowlabel, mirrorby=NmtDat)
        axislegend(ax1, mix1.scene2, unique=true) |> fix_line_legs
    end
    draw_mesh_overlay([obj], ax1)
    draw_contours([load("interm/mesh-nmt-$x-l.obj") => (; color=:black) for x in [:LIP, :PCC]], ax1)
end

function plot_longproj_converg_topo(grid, env; dfx, arbors, _...)
    rowcolor = cls_temp_color(dfx)
    rowlabel(r) = string("#", r.cls)
    dfxx = filter(zz1data.select_neurons(:long_it_ipsi_pari), dfx)

    ax3 = Makie.Axis(grid[1, 1])
    ax3.width = 72
    #ax3.height[] = 80

    insertcols!(dfxx, :somax => 0.0, :arborx => 0.0)
    for r in eachrow(dfxx)
        sa = r.soma_pos, arborcenter(arbors, r.path, rr -> rr.ipsi && contains("Parietal")(rr.arborhier))
        if !NmtDat.isleft(sa[1])
            sa = NmtDat.mirror.(sa)
        end
        r.somax = sa[1][1]
        r.arborx = sa[2][1]
    end
    filter!(r -> !isnan(r.somax) && !isnan(r.arborx), dfxx)
    linear_reg_scatter!(ax3, dfxx; rowx=r -> r.somax / 1000, rowy=r -> r.arborx / 1000, rowcolor)
    ax3.xlabel[] = "Soma position"
    ax3.ylabel[] = "Arbor center"
    setup_axis_by_orient!(ax3, :x, dfxx.somax ./ 1000, (1, 0, 0))
    setup_axis_by_orient!(ax3, :y, dfxx.arborx ./ 1000, (1, 0, 0))
    ax3
end

function plot_longproj_converg_temp(grid, env; dfx, obj, arbors, objflat, objphy, highlights, nmr, nmrdat, _...)
    rowcolor = cls_temp_color(dfx)
    rowlabel(r) = string("#", r.cls)
    dfxx = filter(zz1data.select_neurons(:long_it_ipsi_temp), dfx)

    ax1 = LScene(grid[1, 1], width=140, height=96, show_axis=false)
    setup_camera(ax1 => :s)
    with_mixed_scene(ax1, GLMakie) do mix1
        draw_neurons(dfxx, mix1; rowcolor, rowlabel)
        axislegend(ax1, mix1.scene2, unique=true) |> fix_line_legs
    end
    draw_mesh_overlay([obj], ax1)
    draw_contours([load("interm/mesh-nmt-$x-l.obj") => (; color=:black) for x in [:TE, :STSd]], ax1)
    global xxxaxtemp = ax1
end


function plot_longproj_converg_temp_topo(grid, env; dfx, obj, arbors, objflat, objphy, highlights, nmr, nmrdat, _...)
    grid.alignmode[] = Outside()
    rowcolor = cls_temp_color(dfx)
    rowlabel(r) = string("#", r.cls)
    dfxx = filter(zz1data.select_neurons(:long_it_ipsi_temp), dfx)

    ax2 = Axis(grid[1, 2])
    ax4 = Axis(grid[1, 1])
    #ax4 = Makie.Axis(grid[1, 2][2, 1][1, 1])
    vec = SVector(0.0, 0.676, 0.737)
    draw_arbor_dist_along!(ax2, dfxx, arbors, vec; scene=xxxaxtemp, rowlabel, rowcolor)
    ax2.width[] = 35
    linkyaxes!(ax2, ax4)
    hideydecorations!(ax2)
    hidexdecorations!(ax2, label=false)
    colgap!(grid, 1, 3)

    insertcols!(dfxx, :somax => 0.0, :arborxxx => 0.0)
    for r in eachrow(dfxx)
        sa = r.soma_pos, arborcenter(arbors, r.path, rr -> rr.ipsi && contains("Temporal")(rr.arborhier))
        if !NmtDat.isleft(sa[1])
            sa = NmtDat.mirror.(sa)
        end
        r.somax = sa[1][1]
        r.arborxxx = dot(sa[2], vec)
    end
    filter!(r -> !isnan(r.somax) && !isnan(r.arborxxx), dfxx)
    linear_reg_scatter!(ax4, dfxx; rowx=r -> r.somax / 1000, rowy=r -> r.arborxxx / 1000, rowcolor)
    ax4.xlabel[] = "Soma position"
    ax4.ylabel[] = "Arbor center"
    setup_axis_by_orient!(ax4, :x, dfxx.somax ./ 1000, (1, 0, 0))
    setup_axis_by_orient!(ax4, :y, dfxx.arborxxx ./ 1000, vec)
    ax4.yticks = ([mean(extrema(dfxx.arborxxx ./ 1000))], [rich("Arrow direction ", rich("⟶", color=:red))])
    ax4.yticklabelrotation = π / 2
    hideydecorations!(ax4, label=false, ticklabels=false)
end

function plot_longproj_nmr(grid, env; dfx, obj, arbors, objflat, objphy, highlights, nmr, nmrdat, _...)
    ax3 = Makie.LScene(grid[1, 1], height=80, show_axis=false)
    ax4 = Makie.LScene(grid[2, 1], height=80, show_axis=false)
    obj = nmrdat.surf
    for (ax, inj, lblpos, dv) in [(ax3, "G97-1bL", grid[1, 1, Makie.Top()], "vlPFC"), (ax4, "G97-1aL", grid[2, 1, Makie.Top()], "dlPFC")]
        rr = argmax(r -> StatsBase.corspearman(nmr.vert_conn[:, inj], nmrdat.vert_conn[:, r.site]), eachrow(nmrdat.sites))
        setup_camera(ax => (; use=:s, patch=x -> (; deye=x.deye .+ 120000 .* x.up)))
        with_mixed_scene(ax, GLMakie) do mix
            draw_nmr_data(rr.sitepos, log.(nmrdat.vert_conn[:, rr.site] .+ 0.01), mix; obj)
        end
        Makie.Label(lblpos, "fMRI ($dv)"; set_font_h2()..., valign=:top, tellheight=false)
    end

    Legend(grid[3, 1][1, 1], [MarkerElement(marker=:circle, color=c) for c in (:red, :orange)][1:1], ["Stimulation  site", "site"][1:1], tellwidth=false, telheight=false)
    Makie.rowgap!(grid, 1, 8)
    Makie.rowgap!(grid, 2, 8)
    cb = Colorbar(grid[3, 1][2, 1], limits=extrema([0, 1, 2]), vertical=false, tellwidth=false, flipaxis=false)
    cb.ticklabelsvisible = false
    cb.width = pt(80)
    #cb.tic=false
    #Label(grid[3, 1][2, 1][1, 2], "Projection/activity strength")
    cb.label = "Activity strength"
    rowgap!(content(grid[3, 1]), 1, 4)
    GridLayout(grid[4, 1])
end

function compile_tikz_str(cont; dump=true)
    if dump
        open("/tmp/xxxtesttikz.tex", "w") do io
            write(io, cont)
        end
    end
    d = TEXDocument(cont, true, preamble=read("figs/preamble.tex", String), class="article", classoptions="12pt")
    MakieTeX.CachedTEX(d; tex_engine="lualatex", options="--interaction=batchmode --halt-on-error")
end
function compile_tikz(path)
    compile_tikz_str(read(path, String), dump=false)
end

function plot_longproj_diagram(grid, env; _...)
    LTeX(grid[1, 1], compile_tikz("figs/fig4-long-proj-diagram.tex"), halign=:center, tellwidth=false)
end

function plot_itc_demo(grid, env; dfxitc, obj, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfxitc)
    clipy = 20

    ax1 = LScene(grid[1, 1], width=pt(96), height=pt(126), show_axis=false)
    setup_camera(ax1 => :h)
    with_mixed_scene(ax1, GLMakie) do mix1
        dfx = pick_neurons_with_limit(dfxitc, r -> r.cls, 10)
        draw_neurons(dfx, mix1; rowlabel, rowcolor, mirrorby=NmtDat)
        axislegend(ax1, mix1.scene2, unique=true, orientation=:horizontal, position=:rb, margin=(0, 4, 4 + clipy, 0)) |> fix_line_legs
    end
    draw_mesh_overlay([obj], ax1)

    plot_itc_demo_flatmap(grid[2, 1], env; env...)
    rowgap!(grid, 1, -clipy)
end

function plot_itc_demo_hm(grid, env; dfxitc, _...)
    cols = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-itc.jl")

    ax30 = Makie.Axis(grid[1, 1], height=7)
    hidexdecorations!(ax30)
    ax31 = Makie.Axis(grid[2, 1])
    ax32 = Makie.Axis(grid[3, 1])
    ax31x = Axis(grid[2, 2])
    ax32x = Axis(grid[3, 2])
    grid.alignmode = Outside()
    colgap!(grid, 1, 1)
    rowgap!(grid, 1, 2)
    rowgap!(grid, 2, 2)
    plot_subtyp_spans!(ax30, dfxitc; linkto=ax31, gridto=[ax31, ax32], rowgrp=r -> r.cls, grplabel=(i, g) -> "#$g")
    draw_neuron_proj_hm!(ax31, ax32, dfxitc, cols; ax3=ax30, skipic=true, ax1x=ax31x, ax2x=ax32x)
    ax31.width[] = pt(65)
    ax31.xautolimitmargin = (0, 0.005)
end

function plot_itc_demo_flatmap(grid, env; dfxitc, highlights, objflat, objphy, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfxitc)

    ax4 = Makie.Axis(grid[1, 1], height=pt(76), backgroundcolor=:white)
    ax4.width[] = 96
    plot_flatmap_base!(ax4, highlights, objflat, objphy)
    plot_flatmap_soma!(ax4, dfxitc; rowcolor)
    limits!(ax4, -0.45, 0.45, -0.30, 0.20)
    translate!(ax4.scene.parent, 0, 0, 200)
end


function plot_itc_homotopy(grid, env; dfxitc, arbors, _...)
    itcarbor(r) = !r.ipsi && zz1data.matchhier("charm")(r.arborhier)
    arbors = filter(itcarbor, arbors)
    contra_strict = Set(arbors.path)
    dfx = filter(r -> r.path in contra_strict, dfxitc)
    needmirror = Set{String}()
    for r in eachrow(dfx)
        if !NmtDat.isleft(r.soma_pos)
            r.soma_pos = NmtDat.mirror(r.soma_pos)
            push!(needmirror, r.path)
        end
    end
    intersect!(contra_strict, dfx.path)
    arbors = filter(r -> r.path in contra_strict, arbors)
    for r in eachrow(arbors)
        if r.path in needmirror
            r.center = NmtDat.mirror(r.center)
        end
    end
    calc_min_dist_to_mirrored_soma(paths, somas) = begin
        map(paths, somas) do path, soma
            soma = NmtDat.mirror(soma)
            extrema(p -> norm(p - soma), filter(r -> r.path == path, arbors).center)
        end
    end

    ax = Makie.Axis(grid[1, 1])
    vals1 = first.(calc_min_dist_to_mirrored_soma(dfx.path, dfx.soma_pos))
    #Makie.density!(ax, vals1 ./ 1000, boundary=(0, 5e1), label="Matched")
    vals2 = first.(calc_min_dist_to_mirrored_soma(dfx.path, shuffle(dfx.soma_pos)))
    ecdfplot!(ax, vals1 ./ 1000, color=:black, label="Real data")
    vlines!(ax, quantile(vals1 ./ 1000, [0.9]), linestyle=:dash, color=:black)
    ecdfplot!(ax, vals2 ./ 1000, color=:red, label="Shuffled")
    vlines!(ax, quantile(vals2 ./ 1000, [0.9]), linestyle=:dash, color=:red)
    #rainclouds!(ax, vcat(fill(1, length(vals1)), fill(2, length(vals2))), vcat(vals1./1000, vals2./1000), )
    @warn "med vals1 $(median(vals1))"
    @warn "med vals2 $(median(vals2))"
    t = HypothesisTests.MannWhitneyUTest(vals1, vals2)
    p = HypothesisTests.pvalue(t)
    @warn "pvalue $p"
    #Makie.density!(ax, vals2 ./ 1000, boundary=(0, 5e1), label="Permutated")
    ax.xlabel[] = "Minimum distance to\nmirrored soma (mm)"
    ax.ylabel[] = "ECDF (%)"
    ax.ytickformat[] = (vv) -> [string(Int(v * 100)) for v in vv]
    axislegend(ax, ax, position=:rb, margin=(4, 4, 4, 4)) |> fix_line_legs
    ylims!(ax, 0, 1)
    xlims!(ax, 0, 30)
end

function plot_itc_ipsicontra_len(grid, env; dfxitc, arbors, _...)
    cortexarbor(r) = zz1data.matchhier("charm")(r.arborhier)
    itc_paths = Set(dfxitc.path)
    arborx = filter(r -> cortexarbor(r) && r.path in itc_paths, arbors)
    stats = cached_result(:lencontraxx, dfxitc, arborx) do dfx, arborx
        zz1data.calc_ipsicontra_len(dfx, arborx)
    end
    leftjoin!(stats, dfxitc, on=:path) |> dropmissing!
    filter!(r -> r.arboripsi + r.arborcontra > 0, stats)
    rowcolor = cls_temp_color(dfxitc)

    ax = Axis(grid[1, 1], width=pt(52))
    ddd = (0, 0, 1)
    symm(r) = (r.arborcontra - r.arboripsi) / (r.arborcontra + r.arboripsi)
    insertcols!(stats, :xx => [dot(zz1data.mirrorby(NmtDat)(p), ddd) for p in stats.soma_pos])
    linear_reg_scatter!(ax, stats; rowy=r -> symm(r), rowx=r -> r.xx / 1000, rowcolor, legpos=grid[1, 1, Top()])
    #rowgap!(grid, 1, 2)
    setup_axis_by_orient!(ax, :x, stats.xx ./ 1000, ddd)
    ax.yticks = -1:1
    ax.ylabel[] = "Hemispheric bias"
    hideydecorations!(ax, label=false, ticklabels=false, ticks=false)
    ax.xlabel[] = "Soma position"

    ax1r = Axis(grid[1, 2], width=pt(16))
    colgap!(grid, 1, 2)
    hist!(ax1r, symm.(eachrow(stats)), direction=:x, bins=range(-1.01, 1.01, length=10))
    hideydecorations!(ax1r)
    #ax1r.xlabel = "Density"
    hidexdecorations!(ax1r, label=false)
    linkyaxes!(ax, ax1r)
    Label(grid[1, 2, Right()], "Density", rotation=-π / 2, tellwidth=false, tellheight=false, halign=:right, valign=:center)
    ylims!(ax1r, -1.01, 1.01)
    xlims!(ax1r, 0, nothing)
    Label(grid[1, 1, Top()], "", height=pt(10))
end

function plot_itc_ipsicontra_arbors(grid, env; dfxitc, arbors, _...)
    arbors = leftjoin(arbors, dfxitc[:, Not(:subreg)], on=:path) |> dropmissing!
    ax1 = Axis(grid[1, 1][1, 1][1, 1])
    content(grid[1, 1][1, 1]).alignmode[] = Outside()
    gridx = GridLayout(grid[1, 2][1, 1])
    colgap!(grid, 1, 2)

    dats = Dict()
    for (hh, col) in ["vlPFC" => :blue, "dlPFC" => :red]
        for (ic, lin) in (:C => :solid,) #= :I => :dot,  =#
            dfx = filter(r -> (r.ipsi ? :I : :C) == ic && zz1data.matchhier("Frontal")(r.arborhier) && r.sample in ("234197", "232717") && zz1data.matchhier(hh)(r.somahier), arbors)
            dats[ic=>hh] = dfx
            xx, yy = calc_fvhist(dfx.fv_whist)
            ics = (; I="Ipsi.", C="Contra.")[ic]
            lines!(ax1, yy, xx, color=col, linestyle=lin, label="$hh → $ics", linewidth=pt(0.7))
        end
    end
    #test_depth_dist(:DlIpsiVsDlContra, dats[:I=>"dlPFC"].fv_whist, dats[:C=>"dlPFC"].fv_whist)
    #test_depth_dist(:VlIpsiVsVlContra, dats[:I=>"vlPFC"].fv_whist, dats[:C=>"vlPFC"].fv_whist)
    #test_depth_dist(:DlIpsiVsVlIspi, dats[:I=>"dlPFC"].fv_whist, dats[:I=>"vlPFC"].fv_whist)
    test_depth_dist(:DlContraVsVlContra, dats[:C=>"dlPFC"].fv_whist, dats[:C=>"vlPFC"].fv_whist)
    yaxis_as_layer(ax1, 0, 1)
    xlims!(ax1, 0, nothing)
    ax1.xlabel[] = "Density"
    hidexdecorations!(ax1, label=false)
    Legend(grid[1, 2][2, 1], ax1, position=:rb, tellheight=true, nbanks=2) |> fix_line_legs
    rowgap!(content(grid[1, 2]), 1, 2)
    for (c, (a, n)) in enumerate((45 => "vlPFC", 10 => "dlPFC"))
        LTeX(gridx[1, c], PDFDocument(read("monkeytemp/wyz-upload/fig4-capture-arbors-area-$a-contra.pdf", String)))
        c > 1 && colgap!(gridx, c - 1, 0)
        Label(gridx[1, c], "$n → Contra.", valign=:top, halign=:center, color=:white)
    end
    rowsize!(gridx, 1, 65)
    Label(grid[0, :], "Contralateral frontal projection arbors in dataset G97-1")
    rowgap!(grid, 1, 1)
end

function plot_subreg_flat_conn_contra(grid, env; df, subregs, arbors, trunks, obj, objflat, objphy, highlights2, _...)
    pathways = vote_connections(subregs, arbors; thr=3)
    filter!(r -> abs(r.from) != abs(r.to), pathways)

    ax1 = Makie.Axis(grid[1, 1], width=pt(106), height=pt(80))

    plot_flatmap_base!(ax1, highlights2, objflat, objphy)
    isboth = in(Set([r.from => r.to for r in pathways]))
    pwboth = filter(r -> r.to < 0 && isboth(r.from => -r.to), pathways)
    pwonce = filter(r -> r.to < 0 && !isboth(r.from => -r.to), pathways)
    l1, l2 = length(pwonce), length(pwboth)
    rrr = l2 / (l1 + l2) * 100
    write_stat(:ContraHeteroMirrorPercent, rrr)
    draw_pathway_flat_alt!(ax1, subregs, pwboth)
    draw_pathway_flat_alt!(ax1, subregs, pwonce, linestyle=:dot)
    xlims!(ax1, -0.45, 0.30)
    ylims!(ax1, -0.50, 0.30)
end

function plot_itc_acgg_demo(grid, env; df, subregs, arbors, dfxitc, obj, _...)
    nrnspec(rr, from, to) = begin
        arborsxx = filter(r -> r.path == rr.path, arbors)
        heteroipsi = !isempty(filter(r -> r.subreg == to && r.ipsi, arborsxx))
        heterocontra = !isempty(filter(r -> r.subreg == to && !r.ipsi, arborsxx))
        homocontra = !isempty(filter(r -> r.subreg == from && !r.ipsi, arborsxx))
        (; heteroipsi, heterocontra, homocontra)
    end

    fff = 4
    ttt = 43
    # dfxitc = filter(r -> r.subreg == fff, dfxitc)
    # dfxitc = vcat(dfxitc, filter(r -> r.subreg == fff && r.typ == "ITi", df))
    # insertcols!(dfxitc, :spec => nrnspec.(eachrow(dfxitc), fff, ttt))
    # ####filter!(r -> sum(r.spec) > 0, dfxitc)
    dfxitc = filter(r -> r.typ[1:2] == "IT" && r.subreg == fff, df)
    insertcols!(dfxitc, :spec => nrnspec.(eachrow(dfxitc), fff, ttt))

    ccc = zz1data.vote(dfxitc, :spec, :path)
    specs = unique(dfxitc.spec)
    spec2col = Dict(zip(specs, Makie.to_colormap(:tab10)))
    @warn "votes: $ccc"

    clipy = 30
    ax5 = LScene(grid[1, 1], width=pt(110), height=pt(95), show_axis=false)
    #setup_camera(ax5 => (; use=:c, patch=cfg -> (; cfg..., deye=(0, 3.0f5, 3.0f5))))
    setup_camera(ax5 => :hf)
    with_mixed_scene(ax5, GLMakie) do mix2
        path2spec = Dict(zip(dfxitc.path, dfxitc.spec))
        trunkcolor(rr) = begin
            rr === nothing && return :transparent
            cc = spec2col[path2spec[rr.path]]
            if rr.ipsi && rr.subreg == ttt
                return cc
            elseif !rr.ipsi && rr.subreg == ttt
                return cc
            elseif !rr.ipsi && rr.subreg == fff
                return cc
            end
            :gray
        end
        arborcolor(rr) = begin
            return trunkcolor(rr)
            return :transparent
        end
        spec2nrns = Dict()
        for r in eachrow(dfxitc)
            r.qroot && continue
            push!(get!(() -> Set(), spec2nrns, r.spec), r.path)
        end
        nrns = map(p -> shuffle!(Random.MersenneTwister(0), collect(p.second))[1], collect(spec2nrns)) |> Set
        draw_subtrees(filter(r -> r.path in nrns && r.subreg in (fff, ttt, -ttt), arbors), mix2; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...)
    end
    draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:MD => :red]], ax5)
    draw_mesh_overlay([obj], ax5)

    plot_contra_proj_pat(GridLayout(grid[2, 1]), dfxitc)
    rowgap!(grid, 1, 4 - clipy)
    colgap!(content(grid[2, 1]), 1, 2)
end

function plot_contra_proj_pat(grid, dfxitc, homoarea, heteroarea)
    specs = unique(dfxitc.spec)
    spec2col = Dict(zip(specs, Makie.to_colormap(:tab10)))
    spec2cnt = Dict([spec => count(r -> r.spec == spec, eachrow(dfxitc)) for spec in specs])
    sort!(specs, by=r -> (-spec2cnt[r], -sum(r), r...))
    totcnt = sum(p -> p.second, spec2cnt)
    probs = map(keys(first(specs))) do k
        p = sum(p -> p.first[k] * p.second, spec2cnt)
        k => p / totcnt
    end

    #  [0.01, 0.02, 0.03, 0.04])

    tests = map(specs) do spec
        n = spec2cnt[spec]
        prob = mapreduce(p -> spec[p.first] ? p.second : 1 - p.second, *, probs, init=1.0)
        (; n, p=collect(RCall.R"prop.test($n, $totcnt, $prob)$p.value")[1], s=sign(n / totcnt - prob))
    end
    pvals = map(t -> t.p, tests)
    tests = map(tests, collect(RCall.R"""p.adjust($pvals, "fdr")""")) do t, padj
        (; t..., p=padj)
    end
    spec2t = Dict(zip(specs, tests))

    for (i, spec) in enumerate(filter(s -> sum(s) != 0, specs))
        sum(spec) == 0 && continue
        ax = Axis(grid[i, 1])
        ccc(s) = begin
            s.heteroipsi && return :blue
            s.heterocontra && return :red
            :black
        end
        ls(c, cc) = (; linestyle=c ? :solid : :dot, linewidth=c ? pt(1.5) : pt(0.1), color=spec2col[spec])
        lines!(ax, [Point2f(-1, 0), Point2f(-1, 1.3)]; ls(spec.heteroipsi > 0, :blue)..., joinstyle=:round, linecap=:round)
        lines!(ax, [Point2f(1, 0), Point2f(1, 1.3)]; ls(spec.heterocontra > 0, :red)..., joinstyle=:round, linecap=:round)
        arc!(ax, Point2f(-1, 1), 1, π, 1.5π; ls(sum(spec) > 0, :black)..., joinstyle=:round, linecap=:round)
        lines!(ax, [Point2f(-1, 0), Point2f(1, 0)]; ls(spec.homocontra + spec.heterocontra > 0, :black)..., joinstyle=:round, linecap=:round)
        arc!(ax, Point2f(1, 1), 1, 1.5π, 2π; ls(spec.homocontra > 0, :black)..., joinstyle=:round, linecap=:round)
        scatter!(ax, Point2f(-2, 1), markersize=pt(5), color=:black)
        hidedecorations!(ax)
        hidespines!(ax)
        ax.yautolimitmargin[] = (0.3, 0.3)
        #zz1data.prop_test(n, totcnt, prob)
        t = spec2t[spec]
        n = t.n
        @warn "$spec $t"
        sfx = t.p < 0.05 ? string(t.s > 0 ? "↑" : "↓", compact_pval(t.p)) : ""
        Label(grid[i, 2], text="N=$n$sfx", halign=:left, tellheight=false)
        if i > 1
            rowgap!(grid, i - 1, 2)
        end
        if i == 1
            ax.xticks = ([-2, -1, 1, 2], ["$homoarea", "$heteroarea", "$heteroarea (C)", "$homoarea (C)"])
            ax.xaxisposition = :top
            rotatexlabeltop!(ax)
            ax.xticklabelsvisible = true
        end
    end
    colgap!(grid, 1, 2)
    for a in keys(first(specs))
        for b in keys(first(specs))
            a === b && continue
            v1 = Int[s[a] for s in dfxitc.spec]
            v2 = Int[s[b] for s in dfxitc.spec]
            t = HypothesisTests.ChisqTest(v1, v2, (0:1, 0:1))
            p = HypothesisTests.pvalue(t)
            N = length(v1)
            p1 = sum(v1) / N * sum(v2) / N
            p2 = sum(v1 .* v2) / N
            s = p1 < p2 ? '<' : '>'
            if p < 0.05
                @warn "sig: $a $b $p $p1 $s $p2"
            end
        end
    end
end


function plot_itc_acgg_combo(grid, env; df, arbors, obj, _...)
    dfx = filter(r -> r.typ[1:2] == "IT" && zz1data.matchhier("dlPFC")(r.somahier), df)
    projto(path, tgt, ipsi) = size(filter(rr -> rr.path == path && zz1data.matchhier(tgt)(rr.arborhier) && rr.ipsi == ipsi, arbors), 1) > 0
    stats = map(eachrow(dfx)) do r
        acgg = projto(r.path, "ACgG", true)
        acgg2 = projto(r.path, "ACgG", false)
        dlpfc2 = projto(r.path, "dlPFC", false)
        (; r.path, acgg, acgg2, dlpfc2, spec=(; heteroipsi=acgg, heterocontra=acgg2, homocontra=dlpfc2),)
    end
    leftjoin!(dfx, DataFrame(stats), on=:path) |> dropmissing!
    plot_contra_proj_pat(grid, dfx, "dlPFC", "ACgG")
end

function plot_itc_str_combo(grid, env; df, arbors, obj, _...)
    dfx = filter(r -> r.typ[1:2] == "IT" && zz1data.matchhier("lat_PFC")(r.somahier), df)
    projto(path, tgt, ipsi) = size(filter(rr -> rr.path == path && zz1data.matchhier(tgt)(rr.arborhier) && rr.ipsi == ipsi, arbors), 1) > 0
    stats = map(eachrow(dfx)) do r
        acgg = projto(r.path, "Str", true)
        acgg2 = projto(r.path, "Str", false)
        dlpfc2 = projto(r.path, "lat_PFC", false)
        (; r.path, acgg, acgg2, dlpfc2, spec=(; heteroipsi=acgg, heterocontra=acgg2, homocontra=dlpfc2),)
    end
    leftjoin!(dfx, DataFrame(stats), on=:path) |> dropmissing!
    plot_contra_proj_pat(grid, dfx, "lPFC", "Str")
end

function plot_its_demo(grid, env; dfxits, obj, highlights, objflat, objphy, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfxits)

    ax1 = LScene(grid[1, 1], width=pt(98), height=pt(130), show_axis=false)
    setup_camera(ax1 => :h)
    with_mixed_scene(ax1, GLMakie) do mix1
        draw_neurons(dfxits, mix1; rowlabel, rowcolor, mirrorby=NmtDat)
        axislegend(ax1, mix1.scene2, unique=true, position=:rb) |> fix_line_legs
    end
    draw_mesh_overlay([obj], ax1)

    plot_its_demo_flatmap(grid[2, 1], env; env...)
    rowgap!(grid, 1, 2)
end

function plot_its_demo_hm(grid, env; dfxits, obj, highlights, objflat, objphy, _...)
    grid.alignmode[] = Outside()
    cols = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-its.jl")
    ax30 = Makie.Axis(grid[1, 1])
    ax31 = Makie.Axis(grid[2, 1])
    ax32 = Makie.Axis(grid[3, 1])
    ax31x = Makie.Axis(grid[2, 2])
    ax32x = Makie.Axis(grid[3, 2])
    colgap!(grid, 1, -6)
    plot_subtyp_spans!(ax30, dfxits; linkto=ax31, gridto=[ax31, ax32], rowgrp=r -> r.cls, grplabel=(i, g) -> "#$g")
    draw_neuron_proj_hm!(ax31, ax32, dfxits, cols; ax3=ax30, ax1x=ax31x, ax2x=ax32x, skipic=true, longic=false)
    ax31.width[] = nothing
    rowgap!(grid, 1, 2)
    rowgap!(grid, 2, 2)
end

function plot_its_demo_flatmap(grid, env; dfxits, obj, highlights, objflat, objphy, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfxits)

    ax4 = Makie.Axis(grid[1, 1], width=pt(98), height=pt(85))
    plot_flatmap_base!(ax4, highlights, objflat, objphy)
    plot_flatmap_soma!(ax4, dfxits; rowcolor)
    limits!(ax4, -0.45, 0.45, -0.58, 0.25)
end

function setup_axis_by_orient!(ax, xy, vals, dir; spe=:mac)
    dat = (;
        mac=(;
            dorsal=[0, 0, 1],
            rostral=[0, 1, 0],
            right=[1, 0, 0],
            cent=32000.0f0,
        ),
        mus=(;
            dorsal=[0, -1, 0],
            rostral=[-1, 0, 0],
            right=[0, 0, 1],
            cent=5700.0f0,
        ),
    )[spe]
    ticks = getproperty(ax, Symbol(xy, :ticks))
    ticksv = getproperty(ax, Symbol(xy, :ticksvisible))
    fixrot() = begin
        if xy === :y
            tickrot = getproperty(ax, Symbol(xy, :ticklabelrotation))
            tickrot[] = π / 2
        end
    end
    f, t, lr = if norm(dir .- dat.dorsal) < 0.01
        :Ventral, :Dorsal, false
    elseif norm(dir .+ dat.dorsal) < 0.01
        :Dorsal, :Ventral, false
    elseif norm(dir .- dat.rostral) < 0.01
        :Caudal, :Rostral, false
    elseif norm(dir .+ dat.rostral) < 0.01
        :Rostral, :Caudal, false
    elseif norm(dir .- dat.right) < 0.01
        :Medial, :Lateral, true
    elseif norm(dir .+ dat.right) < 0.01
        :Medial, :Lateral, true
    else
        return
    end
    mval = median(vals)
    a, b = extrema(vals)
    midl = dat.cent / 1000
    cent = (a + b) / 2 * sum(dir)
    if !lr
        ticks[] = ([(a + b) / 2], [L"\fontfamily{Arial}%$f $⟷$ %$t"])
        ticksv[] = false
        return fixrot()
    end
    vals = vals .* sum(dir)
    if midl <= a
        @warn "ZZZ"
        if sum(dir) > 0
            ticks[] = ([(a + b) / 2], [L"\fontfamily{Arial}%$f $⟷$ %$t"])
        else
            ticks[] = ([(a + b) / 2], [L"\fontfamily{Arial}%$t $⟷$ %$f"])
        end
    elseif midl >= b
        @warn "YYY"
        if sum(dir) > 0
            ticks[] = ([(a + b) / 2], [L"\fontfamily{Arial}%$t $⟷$ %$f"])
        else
            ticks[] = ([(a + b) / 2], [L"\fontfamily{Arial}%$f $⟷$ %$t"])
        end
        ticksv[] = false
    else
        @warn "XXX"
        ticks[] = ([midl * sum(dir)], [L"\fontfamily{Arial}%$t $⟷$ %$t"])
    end
    @warn "$(ticks[]) $a $b $(sum(dir)) $midl"
    fixrot()
end

function draw_topo_map_xy!(ax3, dfx, topodat, (xpars, ypars); rowcolor=nothing)
    joinpm(a, b) = b[1] == '-' ? string(a, " - ", b[2:end]) : string(a, " + ", b)
    r = StatsBase.corspearman(xx, yy)
    setup_axis_by_orient!(ax3, :x, xx, xpars)
    setup_axis_by_orient!(ax3, :y, yy, ypars)
    ax3.xlabel[] = string("Soma (", reduce(joinpm, join.(zip(xpars, 'X':'Z'))), ")")
    ax3.ylabel[] = string("Target (", reduce(joinpm, join.(zip(ypars, 'X':'Z'))), ")")
    ax3.title[] = "r=$(round(r; digits=2)) N=$(size(dfxx, 1))"
end

function plot_its_topo(grid, env; df, dfxits, arbors, topo_its_str, _...)
    ax1 = Makie.Axis(grid[1, 2][1, 1][1, 1])
    content(grid[1, 2][1, 1]).alignmode[] = Outside()
    ax3 = LScene(grid[1, 1], width=pt(40), show_axis=false)
    colgap!(grid, 1, 12)
    per_neuron(r) = r.inj == "G97-1bL" ? :red : :blue

    dfx = leftjoin(dfxits, topo_its_str, on=:path) |> dropmissing!
    let dfxx = filter(r -> r.sample == "234197", dfx)
        xpars, ypars = zz1data.find_best_coor_axis(dfxx.srcpos, dfxx.tgtpos, aligned_x=1, aligned_y=1)
        @warn "$xpars $ypars"
        ##xpars = (1, 0, 0)
        insertcols!(dfxx, :xx => dot.(dfxx.srcpos, Ref(xpars)), :yy => dot.(dfxx.tgtpos, Ref(ypars)))
        linear_reg_scatter!(ax1, dfxx; rowx=r -> r.xx / 1000, rowy=r -> r.yy / 1000, rowcolor=per_neuron, legpos=grid[1, 2][0, 1])
        ax1.xlabel[] = "Soma position"
        ax1.ylabel[] = "Arbor center"
        setup_axis_by_orient!(ax1, :x, dfxx.xx ./ 1000, xpars)
        setup_axis_by_orient!(ax1, :y, dfxx.yy ./ 1000, ypars)
    end
    rowgap!(content(grid[1, 2]), 1, 2)

    cam = (fov=9.0, cent=(26353.81, 48027.59, 30036.0), deye=(-0.288456217633492, -0.40198377102426347, 0.8690236235809081) .* 130000, up=(-0, 1.0, 0.0))
    setup_camera(ax3 => cam)
    with_mixed_scene(ax3, GLMakie) do mix
        itsnrns = Set(dfxits.path)
        dfxarbors = leftjoin(arbors, df[:, Not(:subreg)], on=:path) |> dropmissing!
        filter!(r -> r.path in itsnrns && r.sample == "234197" && r.ipsi && zz1data.matchhier("Str")(r.arborhier), dfxarbors)
        per_arbor(pn, r, cond) = begin
            cond && return :transparent
            return pn
        end
        per_node(pn, pa, n) = pa => pt(0.3)
        draw_subtrees(dfxarbors, mix; per_neuron, per_arbor, per_node, mirrorby=NmtDat)
        axislegend(ax3, [Makie.PolyElement(color=c) for c in (:red, :blue)], ["vlPFC", "dlPFC"], position=:rb, unique=true, margin=(0, -10, 0, 0))
    end
    draw_mesh_overlay([load("interm/mesh-nmt-Str-l.obj")], ax3)
    draw_contours([load("interm/mesh-nmt-$x-l.obj") for x in [:CdH]], ax3)
end

function find_clu_pars(ll, hh, clus; arborsg97, arbordistg97, _...)
    arborcluster = let arbors = arborsg97, dist = arbordistg97
        h = Clustering.hclust(Symmetric(dist + dist') / 2, linkage=ll, branchorder=:barjoseph)
        Clustering.cutree(h, h=hh)
    end
    #insertcols!(arborsg97, :arborcluster => 0)
    @info "$(size(arbordistg97)) $(size(arborsg97))"
    foreach(eachrow(arborsg97), arborcluster) do r, c
        r.arborcluster = c
    end
    @info "$(size(arbordistg97)) $(size(arborsg97))"
    get_arbor(pt) = begin
        pt = SVector(pt)
        argmin(r -> norm(r.center - pt), eachrow(arborsg97)).arborcluster
    end
    clus = map(v -> map(get_arbor, v), clus)
    @warn "testclu $ll $hh $clus"
end


select_srcnrns_tgtclus(arbors) = begin
    clu2nrns = Dict{Int,Set{String}}()
    clu2strs = Set{Int}()
    for r in eachrow(arbors)
        nrns = get!(() -> Set{String}(), clu2nrns, r.arborcluster)
        push!(nrns, r.path)
        if zz1data.matchhier("CdH")(r.arborhier) && r.ipsi
            push!(clu2strs, r.arborcluster)
        end
    end
    clu2nrns = filter!(>=(3) ∘ length ∘ last, collect(clu2nrns))
    clu2nrns = filter!(in(clu2strs) ∘ first, collect(clu2nrns))
    nrns = collect(mapreduce(last, union, clu2nrns))
    nrn2pos = Dict(zip(arbors.path, arbors.soma_pos))
    mat = [n in nn for (c, nn) in clu2nrns, n in nrns]
    @warn "clus $(length(clu2nrns))"
    @warn "nrns $(length(nrns))"
    #return GLMakie.heatmap(mat) |> display
    #dm1 = [sum(abs.(view(mat, i, :) .- view(mat, j, :))) for i in axes(mat, 1), j in axes(mat, 1)]
    dm1 = 1 .- cor(mat')
    h1 = Clustering.hclust(dm1, linkage=:ward, branchorder=:barjoseph)
    dm2 = [sum(abs.(view(mat, :, i) .- view(mat, :, j))) + norm(nrn2pos[nrns[i]] - nrn2pos[nrns[j]]) / 1000 for i in axes(mat, 2), j in axes(mat, 2)]
    h2 = Clustering.hclust(dm2, linkage=:ward, branchorder=:barjoseph)
    sep = maximum(findall(Clustering.cutree(h1, k=2)[h1.order] .== 1))
    return nrns[h2.order], first.(clu2nrns[h1.order]), mat[h1.order, h2.order], sep
    cls = Clustering.cutree(h1, k=2)
    cls2 = Clustering.cutree(h2, k=4)
    #return scatter(arbors.center, color=map(v -> v === nothing ? NaN : cls[v], indexin(arbors.arborcluster, first.(clu2nrns))))
    #return scatter(something.(arbors.soma_pos), color=map(v-> v===nothing ? NaN : cls2[v], indexin(arbors.path, nrns)))
    #GLMakie.heatmap(mat) |> display
    #Vis.dendrogram(h2) 
end

function plot_arbor_cluster2(grid, env; df, arborsg97, obj, arbordistg97, _...)
    findclu(tgt) = begin
        argmin(unique(arborsg97.arborcluster)) do clu
            dfx = filter(r -> r.arborcluster == clu, arborsg97)
            size(dfx, 1) < 3 && return Inf
            mean((norm(r.center .- tgt) for r in eachrow(dfx)))
        end
    end

    srcnrns, tgtclus = select_srcnrns_tgtclus(arborsg97)

    cam1 = (fov=6.5, cent=(20461.08, 54798.17, 22735.65), deye=(-0.05991377976767829, 0.005895412412107087, 0.998186146520999) .* 250693.37334043803, up=(0.08, 1.0, -0.0))
    cam2 = (fov=6.5, cent=(20676.67, 49164.69, 28850.04), deye=(-0.18111815766242498, -0.41011857459970247, -0.8938674217876369) .* 180898.85369032912, up=(-0.95, 0.3, 0.06))

    ax2 = LScene(grid[1, 1], width=pt(130), height=pt(130), show_axis=false)
    #ax1 = LScene(grid[1, 1, Makie.BottomLeft()], width=pt(40), height=pt(40), show_axis=false, tellwidth=false, tellheight=false, halign=:left, valign=:bottom)
    #setup_camera(ax1 => cam1)
    setup_camera(ax2 => cam2)

    clu2col = Dict()
    #with_mixed_scene(ax1, GLMakie) do mix1
    with_mixed_scene(ax2, GLMakie) do mix2
        dfx = filter(r -> r.arborcluster in tgtclus && r.inj == "G97-1bL" && r.path in srcnrns, arborsg97)
        selnrns = srcnrns[1:end]

        nrn2col = Dict()
        per_neuron(rr) = begin
            rr.path in selnrns || return to_color(:transparent)
            get!(() -> to_colormap(:tab10)[length(nrn2col)%10+1], nrn2col, rr.path)
        end
        #text!(mix1.scene2, tgtcluspp, text=map(p -> string(p), enumerate(tgtclus)))
        per_arbor = (pn, rr, cond) -> begin
            nodisp = pn == to_color(:transparent)
            cond && return :gray
            cond && (rr === nothing || !(rr.arborcluster in tgtclus)) && return pn
            cond && nodisp && return pn
            if rr !== nothing
                #rr.arborcluster in tgtclus || return to_color(:gray)
                # !nodisp && return pn
                c = get!(clu2col, rr.arborcluster) do
                    0.9 * to_colormap(:tab10)[length(clu2col)%10+1] + 0.1 * to_color(rr.inj == "G97-1aL" ? :blue : :red)
                end
                c = clu2col[rr.arborcluster]
                !nodisp && return c
                return 0.6 * c + 0.4 * to_color(:white)
            end
            nodisp && return pn
            to_color(:transparent)
        end
        per_node = (pn, pa, n) -> pa => pt(0.3)
        draw_subtrees(dfx, mix2; autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=NmtDat)
        p2c = Dict()
        cc = [get!(() -> to_colormap(:tab10)[length(p2c)%10+1], p2c, p) for p in dfx.path]
        #scatter!(mix1.scene2, dfx.center, markersize=pt(3), color=cc)

        for c in tgtclus
            xxxx = filter(r -> r.arborcluster == c, arborsg97)
            ccc = SVector(median(reduce(hcat, xxxx.center), dims=2)...)
            !any([zz1data.matchhier("Str")(h) for h in xxxx.arborhier]) && continue
            if length(unique(xxxx.inj)) > 1
                #scatter!(mix1.scene2, [ccc], color=:magenta, markersize=pt(10))
            end
            ppp = unique(zip(xxxx.path, xxxx.inj))
            @warn "tgtclux $(length(ppp)) $ccc $c $ppp"
        end
        ccc = Gapr.load_swc("monkeytemp/wyz-upload/C1-entropy-density-G97-1bL.arbor-center-str.swc")
        dm = [norm(a.pos - b.pos) for a in ccc.nodes, b in ccc.nodes]
        h = Clustering.hclust(dm, linkage=:single, branchorder=:barjoseph)
        lll = slink_dendro_pairs(h, dm)
        #scatter!(ax2, [n.pos for n in ccc], markersize=pt(3), color=:red)
        #linesegments!(ax2, [(ccc[a].pos, ccc[b].pos) for (a, b) in lll], color=:red, linewidth=pt(1))
        write_stat(:ArborPatchDiameterMeanAndSem, meanandsem([norm(ccc[a].pos - ccc[b].pos) / 1000 for (a, b) in lll], "mm"))
        draw_scale_bar!(ax2, 2000, [n.pos for n in ccc.nodes],)
    end
    write_stat(:StrArborSizeMeanAndSem, meanandsem(zz1data.arbor_size.(filter(r -> zz1data.matchhier("Str")(r.arborhier), arborsg97).f_volume) / 1000, "mm"))
    #end
    #draw_mesh_overlay([obj], ax1)
    draw_mesh_overlay(["interm/mesh-nmt-$ba-l.obj" => col for (ba, col) in [(:CdH, :gray)]], ax2)
    return
end

function plot_arbor_cluster_conn(grid, env; df, arborsg97, obj, arbordistg97, _...)
    srcnrns, tgtclus, mat, vsep = select_srcnrns_tgtclus(arborsg97)
    path2inj = Dict(zip(df.path, df.inj))
    nnrn = length(srcnrns)
    npat = length(tgtclus)

    ax3 = Axis(grid[1, 1][1, 1], width=pt(80))
    content(grid[1, 1]).alignmode = Outside()
    #ax3.aspect[] = Makie.DataAspect()
    #colsize!(grid, 1, Aspect(1, nnrn / npat))
    dat = Point2f[]
    for i in axes(mat, 1)
        for j in axes(mat, 2)
            if mat[i, j]
                push!(dat, (i, j))
            end
        end
    end
    #scatter!(ax3, dat, marker=:circle, markersize=1.5, color=:black, markerspace=:data)
    heatmap!(ax3, mat')
    #ax3.autolimitaspect[] = 1
    ax3.xlabel[] = "Neurons (N=$nnrn)"
    ax3.ylabel[] = "Patches (N=$npat)"

    # ax3r = Axis(grid[1, 1])
    # rowgap!(grid, 1, 0)
    for (i, inj) in enumerate(unique([path2inj[p] for p in srcnrns]))
        a, b = extrema(findall(p -> path2inj[p] == inj, srcnrns))
        if i == 1
            plt = vlines!(ax3, [b], color=:red, linestyle=:dash)
            translate!(plt, 0, 0, 1)
        end
        #bracket!(ax3r, b, 0, a, 0, text="$inj")
    end
    plt = hlines!(ax3, [vsep], color=:red, linestyle=:dash)
    translate!(plt, 0, 0, 1)
    # linkxaxes!(ax3, ax3r)
    # ax3r.height[] = pt(11)
    # ylims!(ax3r, 0, 1)
    # hidedecorations!(ax3r)
    # hidespines!(ax3r)
    hidedecorations!(ax3, label=false)
    hidespines!(ax3)
    #ylims!(ax3, 0.5, npat + 0.5)
    #xlims!(ax3, 0.5, nnrn + 0.5)
end

function plot_arbor_cluster_numtgt(grid, env; df, arborsg97, obj, arbordistg97, _...)
    srcnrns, tgtclus, mat = select_srcnrns_tgtclus(arborsg97)
    path2inj = Dict(zip(df.path, df.inj))
    cnts = vec(sum(mat, dims=1))
    injs = sort(unique([path2inj[p] for p in srcnrns]))
    inj2stk(inj) = begin
        findfirst(==(inj), injs) |> something
    end

    dats = map(enumerate(injs)) do (rrr, inj)
        sel = findall(p -> path2inj[p] == inj, srcnrns)
        cnt = cnts[sel]
        xx = sort(unique(cnt))
        yy = [count(==(x), cnt) for x in xx]
        ss = fill(rrr, length(xx))
        xx, yy, ss
    end
    xx, yy, ss = vcat.(dats...)

    ax = Axis(grid[1, 1])
    ax = content(grid[1, 1])
    barplot!(ax, xx, yy, gap=0, color=[:blue, :red][ss], stack=ss, label=injs[ss])
    xlims!(ax, 0.5, nothing)
    ylims!(ax, 0, nothing)
    ax.xticks[] = 1:2:maximum(cnts)
    ax.xlabel[] = "No. of target patches\nper neuron"
    ax.ylabel[] = "No. of neurons     "
    hideydecorations!(ax, label=false, ticks=false, ticklabels=false)
    hidexdecorations!(ax, label=false, ticks=false, ticklabels=false)
    axislegend(ax, [PolyElement(color=c) for c in [:red, :blue]], ["vlPFC", "dlPFC"], position=:rt)
end

function plot_arbor_cluster_diagram(grid, env; df, arborsg97, obj, arbordistg97, _...)
    srcnrns, tgtclus, mat = select_srcnrns_tgtclus(arborsg97)

    # XXX: select clus lined up in a row, so distance between is valid
    selclus = map(1:length(tgtclus)÷2) do i
        vvv = vec(sum(mat[:, findall(mat[i, :] .> 0)], dims=2))
        sortperm(vvv, rev=true)[1:6]
    end
    ooo = argmax(selclus) do sel
        cc = count(vec(sum(mat[sel, :], dims=1)) .> 2)
        @warn "cnt $sel $cc"
        cc
    end
    tgtclus = tgtclus[ooo]
    mat = mat[ooo, :]
    sel = shuffle(findall(vec(sum(mat, dims=1)) .> 2))[1:min(7, end)]
    srcnrns = srcnrns[sel]
    mat = mat[:, sel]

    ax = Axis(grid[1, 1], width=pt(60))
    for (i, tgtclus) in enumerate(tgtclus)
        scatter!(ax, [Point2f(0, i)], markersize=1.25, markerspace=:data, color=:gray)
    end

    clu2off = Dict()
    fixtgt(t) = begin
        c = get(clu2off, t, 0)
        clu2off[t] = c + 1
        iseven(c) && return t + c ÷ 2 * 0.1
        t - (c + 1) ÷ 2 * 0.1
    end
    for (j, srcnrns) in enumerate(srcnrns)
        cc = Makie.wong_colors()[j]
        scatter!(ax, [Point2f(-j, 0)], color=cc, markersize=pt(5))
        tgts = findall(mat[:, j])
        tgts = fixtgt.(tgts)
        lines!(ax, Point2f[(-j, 0), (-j, maximum(tgts) - 0.2)], color=cc)
        for t in tgts
            arc!(ax, Point2f(-j + 0.2, t - 0.2), 0.2, 0.5π, π, color=cc)
            lines!(ax, Point2f[(-j + 0.2, t), (0, t)], color=cc)
        end
    end
    ax.xlabel[] = "Neurons"
    ax.xaxisposition = :top
    ax.ylabel[] = "Patches"
    ax.yaxisposition[] = :right
    hidedecorations!(ax, label=false)
    hidespines!(ax)
    ylims!(ax, -0.2, nothing)
    xlims!(ax, -0.2 - length(srcnrns), nothing)
    ax.yreversed[] = true
    tightlimits!(ax)
end

function barplot_with_chisq!(ax, df; rowx=nothing, xname=nothing, skipfirst=false)
    stats = rowx.(eachrow(df))
    cnt = reduce(mergewith!(+), Dict.(stats .=> 1), init=Dict{NamedTuple,Int}())
    cnt = sort(collect(cnt), by=p -> (sum(p.first), p.first))
    @warn "$cnt"
    xvals = first.(cnt)[skipfirst+1:end]
    yvals = last.(cnt)[skipfirst+1:end]
    barplot!(ax, something.(indexin(xvals, xvals)), yvals, bar_labels=:y, flip_labels_at=60, label_formatter=v -> string(Int(v)))
    for (xiiii, (x, xcnt)) in enumerate(zip(xvals, yvals))
        if sum(x) == 2
            v1, v2 = map(filter(v -> x[v], keys(x))) do v
                map(s -> Int(s[v]), stats)
            end
            ##@assert sum(yvals) == length(v1) == length(v2)
            pred = sum(v1) / length(v1) * sum(v2) / length(v2) * sum(yvals)
            t = HypothesisTests.ChisqTest(v1, v2, (0:1, 0:1))
            p = HypothesisTests.pvalue(t)
            @warn "$x $p $pred $xcnt"
            if p < 0.05
                #@assert xcnt == sum(v1 .* v2)
                sgn = sum(v1 .* v2) > pred ? "+" : "-"
                text!(ax, xiiii, yvals[xiiii], text="$sgn (p=$(round(p, sigdigits=3)))", align=(:left, :bottom), offset=(0, 8))
            end
        end
    end
    ax.xticks[] = (axes(xvals, 1), xname.(xvals))
    ylims!(ax, 0, nothing)
    hidexdecorations!(ax, ticklabels=false)
end


function node_color_trunks(; selection=nothing, linewidth=pt(0.5), trunkcolor=nothing, arborcolor=:transparent, arborwidth=pt(0.3))
    if trunkcolor === nothing
        trunkcolor = (rr) -> rand(Makie.to_colormap(:tab10))
    end
    if !(arborcolor isa Function)
        ac = deepcopy(arborcolor)
        arborcolor = (rr) -> ac
    end
    add_lw(v, lw) = begin
        v isa Pair ? v : v => lw
    end
    per_arbor(pnrn, rr, cond) = begin
        if cond
            if selection !== nothing && (rr === nothing || !selection(rr))
                add_lw(arborcolor(rr), linewidth)
            else
                add_lw(trunkcolor(rr), linewidth)
            end
        else
            add_lw(arborcolor(rr), arborwidth)
        end
    end
    per_node(pnrn, parb, n) = begin
        parb
    end
    (; per_arbor, per_node)
end

function calc_trunks_midpoints_cc(trunks, mod=NmtDat)
    pts = map(eachrow(trunks)) do r
        #r.trunk[round(Int, rat*(end-1))+1]
        argmin(p -> norm(p - mod.mirror(p)), r.trunk[end÷3:end*2÷3])
    end

    pt = SVector(median(reduce(hcat, pts), dims=2)...)
    nn = map(eachrow(trunks)) do r
        r.trunk[end] .- r.trunk[1]
    end
    nn = reduce(+, nn)

    nn = nn ./ norm(nn)

    v1 = SVector(0, -1, 0)
    v2 = SVector(0, 0, 1)
    uvs = map(eachrow(trunks), pts) do r, ptx
        k = findmin(p -> norm(p - ptx), r.trunk)[2]
        pp = (r.trunk[k] - pt)
        dot(pp, v1), dot(pp, v2)
    end
    [pt .+ v1 * uv[1] .+ v2 * uv[2] for uv in uvs], uvs
end

function calc_trunks_midpoints2(trunks; rat=0.501, initu=nothing)
    pts = map(eachrow(trunks)) do r
        r.trunk[round(Int, rat * (end - 1))+1]
        #r.trunk[1] .* (1 - rat) .+ r.trunk[end] .* rat
    end
    projondir(d) = begin
        vv = [dot(p, d) for p in pts]
        mv = median(vv)
        sum([abs(v - mv) for v in vv])
    end
    if false
        ddd = SVector(argmin(projondir, eachcol(rand(3, 1000)))...)
        ddd = ddd / norm(ddd)
        mvvv = median([dot(p, ddd) for p in pts])
        #dot(pt, ddd)-mvvv==0
        pt = ddd / norm(ddd)^2 * mvvv
        @assert abs(dot(pt, ddd) - mvvv) < 1e-5
        nn = SVector(ddd...)
    else
        pt = SVector(median(reduce(hcat, pts), dims=2)...)
        nn = map(eachrow(trunks)) do r
            r.trunk[end] .- r.trunk[1]
        end
        nn = reduce(+, nn)
    end
    nn = nn ./ norm(nn)
    v1 = if initu === nothing
        argmax(p -> norm(p - pt), pts) - pt
    else
        SVector(initu...)
    end
    v1 = v1 .- dot(v1, nn) * nn
    v2 = cross(v1, nn)
    v1 = v1 ./ norm(v1)
    v2 = v2 ./ norm(v2)
    uvs = map(eachrow(trunks)) do r
        k = findmin(p -> abs(dot(p - pt, nn)), r.trunk)[2]
        pp = (r.trunk[k] - pt)
        dot(pp, v1), dot(pp, v2)
    end
    uvs, (uv -> pt .+ v1 * uv[1] .+ v2 * uv[2])
end

function calc_f1_for_offset_imp(cls, vals)
    calcf1(ntp, nfp, nfn) = 2 * ntp / (2 * ntp + nfp + nfn)
    ppp = sortperm(vals)
    ntp, nfp = 0, 0
    nfn = count(!=(0), cls)
    ntn = length(cls) - nfn
    best = (calcf1(ntp, nfp, nfn), Inf, (ntp, nfp, nfn, ntn))
    i = 0
    add_i(i, pred) = begin
        if pred != 0
            ntp += 1
            nfn -= 1
        else
            nfp += 1
            ntn -= 1
        end
        i + 1
    end
    ######@warn "$best"
    while i < length(ppp)
        ii = ppp[i+1]
        v1 = vals[ii]
        i = add_i(i, cls[ii])
        while i < length(ppp)
            ii = ppp[i+1]
            vals[ii] > v1 && break
            i = add_i(i, cls[ii])
        end
        v2 = i + 1 <= length(ppp) ? vals[ppp[i+1]] : Inf
        c = -(v1 + v2) / 2
        @assert ntp + nfp + nfn + ntn == length(cls)
        f1 = calcf1(ntp, nfp, nfn)
        @assert !isnan(f1)
        best = max(best, (f1, c, (ntp, nfp, nfn, ntn)))
        #####@warn "$best"
    end
    best
end
function calc_f1_for_offset(cls, vals)
    max(calc_f1_for_offset_imp(cls, vals), calc_f1_for_offset_imp([v == 0 ? 1 : 0 for v in cls], vals))
end

function calc_classify_f1_score(uvs, cls)
    best = maximum(0:0.1:2π) do t
        a, b = sincos(t)
        vals = [dot(SVector(a, b), uv) for uv in uvs]
        f1, c, tbl = calc_f1_for_offset(cls, vals)
        (f1, a, b, c, tbl)
    end
    best[1], (best[2:end-1]...,)
end

function calc_topo_map_index(uvs, xyzs)
    ddd = randn(1000, 3)
    dirs = ddd * reduce(hcat, xyzs)
    rrr = map((0:0.05:1) * 2π) do rad
        a, b = sincos(rad)
        vals0 = [dot(SVector(a, b), uv) for uv in uvs]
        cls = vals0 .> median(vals0)
        maximum(eachrow(dirs)) do vals
            f1, c, tbl = calc_f1_for_offset(cls, vals)
            f1
        end
    end
    # rrr = map(eachrow(dirs)) do vals0
    #     cls = vals0 .> median(vals0)
    #     maximum((0:0.05:1) * 2π) do rad
    #         a, b = sincos(rad)
    #         vals = [dot(SVector(a, b), uv) for uv in uvs]
    #         f1, c, tbl = calc_f1_for_offset(cls, vals)
    #         f1
    #     end
    # end
    rr = quantile(rrr, (0.25, 0.5, 0.75))
    rr
end


function plot_intra_pfc_tgt_numberx(grid, env; df, trunks, subregs, obj, _...)
    trunk2ft(rr) = begin
        ff = findfirst(nn -> rr.path in nn, subregs.neurons)
        tt = findfirst(nn -> rr.arborpath in nn, subregs.arbors)
        if tt !== nothing && !rr.ipsi
            tt = -tt
        end
        ff => tt
    end

    p2tgts = Dict([p => Set() for p in df.path])
    for r in eachrow(trunks)
        ff, tt = trunk2ft(r)
        (ff === nothing || tt === nothing) && continue
        ff == tt && continue
        subregs[abs(tt), :frontal] || continue
        push!(p2tgts[r.path], tt)
    end
    typ2c = typ2color_func(df)
    mm = maximum(length.(last.(collect(p2tgts))))
    for (k, t) in enumerate(sort(filter(r -> r == "ITi", unique(df.typ))))
        ax = Axis(grid[k, 1])
        if k > 1
            linkxaxes!(content(grid[k-1, 1]), ax)
            linkyaxes!(content(grid[k-1, 1]), ax)
            rowgap!(grid, k - 1, 0)
        end
        v = [length(p2tgts[p]) for p in filter(r -> r.typ == t, df).path]
        mm = maximum(v)
        filter!(!=(0), v)
        hist!(v, bins=0.5:1:mm+0.5, color=typ2c(t), label="$t", normalization=:probability, direction=:x)
        ax.width[] = 40
        ax.yticks = 1:mm
        ax.yminorticksvisible = false
        ylims!(ax, 0.5, nothing)
        if t == "ITi"
            ax.ylabel[] = "No. of target\nsubdomains"
        else
            hidexdecorations!(ax, ticks=false)
        end
        xlims!(ax, 0, nothing)
        ax.xlabel[] = "Proportion"
    end
end

function draw_intra_pfc_modular!(ax; dfarbor, subregs, _...)
    pathways = vote_connections(subregs, dfarbor; thr=1)
    subregs = filter(r -> r.lpfc, subregs)

    mm = zeros(size(subregs, 1), size(subregs, 1))
    for (i, r1) in enumerate(eachrow(subregs))
        for (j, r2) in enumerate(eachrow(subregs))
            for p in pathways
                if p.from == r1.subreg && p.to == r2.subreg
                    mm[i, j] += length(p.neurons)
                    if i != j
                        mm[j, i] += length(p.neurons)
                    end
                end
            end
        end
    end

    clu = Clustering.assignments(Clustering.mcl(mm, inflation=1.8, expansion=2.0))
    @warn "n clu $(length(unique(clu)))"
    for c in unique(clu)
        hh = subregs[clu.==c, :parhier]
        @warn "$hh"
    end
    scatter!(ax, subregs.centuv, color=[:cyan, :orange,][clu], markersize=4)
end

function plot_intra_pfc_reciprocal(grid, env; df, dfxreciprocal, dfarbor, trunks, subregs, obj, highlights, objflat, objphy, _...)
    ax1 = LScene(grid[1, 1], width=pt(126), height=pt(100), show_axis=false)

    cam = (fov=9.0, cent=(32804.94, 61155.57, 33295.17), deye=(-0.9992451011239071, 0.03821380837224695, -0.006995193318408471) .* 137378.39072532358, up=(0.0, 0.0, 1.0))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        trunkcolor(rr) = begin
            rr === nothing && return :transparent
            rr.trunkdir > 0 && return :red
            rr.trunkdir < 0 && return :blue
            :transparent
        end
        dfxxx = deepcopy(dfxreciprocal)
        draw_subtrees(dfxxx, mix1 => :hr; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor=trunkcolor)...)
        #o = load("/data/dat/NMT_v2.0_sym/NMT_v2.0_sym_surfaces/lh.white_surface.rsl.gii.obj")
        #mesh!(mix1.scene2, o, color=(:gray, 0.2), transparency=true)
    end
    draw_mesh_overlay(["interm/mesh-nmt-$ba-l.obj" => col for (ba, col) in [(:area_12r, :red), (:area_45b, :blue)]], ax1)
    draw_mesh_overlay([obj => :gray], ax1)
    names = [areaname(filter(r -> r.subreg == subreg, subregs)[1, :parhier]) for subreg in sort(unique(dfxreciprocal.subreg), rev=true)]
    axislegend(ax1, [Makie.PolyElement(color=c) for c in (:red, :blue)], [string(ff, " → ", tt) for (ff, tt) in (names, reverse(names))])
end

import Loess

calc_fvhist(hh; smoothn=9, smoothw=0.3, trimextream=true) = begin
    hh = reduce(hcat, filter(r -> sum(r[2:end-1]) > 10, hh))
    if trimextream
        hh[1, :] .= 0
        hh[end, :] .= 0
    end
    yy = if false
        vec(sum(hh ./ sum(hh, dims=1), dims=2))
    else
        vec(sum(hh, dims=2))
    end
    xx = axes(yy, 1) .- 1
    for _ in 1:smoothn
        uu = copy(yy)
        uu[1] += smoothw * yy[1]
        uu[1:end-1] .+= smoothw .* yy[2:end]
        uu[end] += smoothw * yy[end]
        uu[2:end] .+= smoothw .* yy[1:end-1]
        yy[:] .= uu[:]
    end
    xx = xx ./ maximum(xx)
    yy = yy ./ sum(yy)
    return xx, yy
end

function plot_intra_pfc_arbor_depth(grid, env; dfxdlvl, dfxreciprocal, dfarbor, trunks, subregs, obj, highlights, objflat, objphy, _...)
    ax1 = Axis(grid[2, 1], width=pt(48))
    ax2 = Axis(grid[2, 2], width=pt(48))
    colgap!(grid, 1, 2)

    dfx1 = filter(r -> r.trunkdir > 0, dfxreciprocal)
    dfx2 = filter(r -> r.trunkdir < 0, dfxreciprocal)
    for n in filter(startswith("f_"), names(dfxreciprocal))
        v1, v2 = dfx1[:, n], dfx2[:, n]
        t = HypothesisTests.MannWhitneyUTest(v1, v2)
        p = HypothesisTests.pvalue(t)
        if p < 0.05
            @warn "$n $p $(median(v1)), $(median(v2))"
        end
    end

    for (dfx, col, lbl) in [(dfx1, :red, "Area 12r → 45b"), (dfx2, :blue, "Area 45b → 12r")]
        xx, yy = calc_fvhist(dfx.fv_whist)
        lines!(ax1, yy, xx, color=col, label="$lbl")
    end
    test_depth_dist(:Area12rTo45bVsArea45bTo12r, dfx1.fv_whist, dfx2.fv_whist)

    dfx21 = filter(r -> r.trunkdir > 0, dfxdlvl)
    dfx22 = filter(r -> r.trunkdir < 0, dfxdlvl)
    for (dfx, col, lbl) in [(dfx21, :red, "dlPFC → vlPFC"), (dfx22, :blue, "vlPFC → dlPFC")]
        filter!(r -> r.sample in ("234197", "232717"), dfx)
        xx, yy = calc_fvhist(dfx.fv_whist)
        lines!(ax2, yy, xx, color=col, label="$lbl")
    end
    foreach(r -> @assert(r.sample in ("234197", "232717")), eachrow(dfx21))
    foreach(r -> @assert(r.sample in ("234197", "232717")), eachrow(dfx22))
    test_depth_dist(:DlToVlVsVlToDl, dfx21.fv_whist, dfx22.fv_whist)
    hideydecorations!(ax2)
    linkyaxes!(ax1, ax2)
    linkxaxes!(ax1, ax2)
    for (i, ax) in enumerate((ax1, ax2))
        ylims!(ax, 0, nothing)
        yaxis_as_layer(ax, 0, 1)
        ax.xlabel[] = "Density"
        hidexdecorations!(ax, label=false)
        Legend(grid[1, i], ax, i == 1 ? "(in vlPFC)" : "", tellheight=true, titlefont=:regular, titlesize=5, valign=:bottom, tellwidth=false) |> fix_line_legs
    end
    rowgap!(grid, 1, 2)
end

function plot_intra_pfc_dlvl(grid, env; df, dfarbor, dfxdlvl, subregs, obj, highlights, objflat, objphy, _...)
    ax1 = LScene(grid[1, 1], width=pt(126), height=pt(100), show_axis=false)
    ax2 = Axis(grid[1, 2][1, 1][2, 1],)
    ax3 = Axis(grid[1, 2][2, 1][2, 1],)
    colgap!(grid, 1, 2)
    rowgap!(content(grid[1, 2]), 1, 2)
    content(grid[1, 2]).alignmode[] = Outside()

    cam = (fov=9.0, cent=(29774.94, 59450.98, 24854.45), deye=(-0.6650354807252158, 0.059138288747787704, 0.7444665688804062) .* 197378.3862732528, up=(0.74, -0.07, 0.67))
    cam = (fov=9.0, cent=(38191.5, 58354.19, 30538.97), deye=(-0.9566801486319952, 0.18231915525071954, 0.22698638470650026) .* 179434.90334305336, up=(0.22, -0.05, 0.97))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        trunkcolor(rr) = begin
            rr === nothing && return :transparent
            rr.trunkdir > 0 && return :red
            rr.trunkdir < 0 && return :blue
            :transparent
        end
        draw_subtrees(pick_neurons_with_limit(dfxdlvl, r -> r.trunkdir, 100), mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor=trunkcolor)...)
    end
    draw_mesh_overlay(["interm/mesh-nmt-$ba-l.obj" => col for (ba, col) in [(:dlPFC, :red), (:vlPFC, :blue)]], ax1)
    draw_mesh_overlay([obj => :gray], ax1)
    axislegend(ax1, [Makie.PolyElement(color=c) for c in (:red, :blue)], ["dlPFC → vlPFC", "vlPFC → dlPFC"], position=:lb)

    if false
        plot_flatmap_base!(ax2, highlights, objflat, objphy)
        scatter!(ax2, filter(r -> r.trunkdir > 0, dfxdlvl).arboruvcent2, color=:red, marker=:x)
        scatter!(ax2, unique(filter(r -> r.trunkdir > 0, dfxdlvl).somauv), color=:red, markersize=2)
        scatter!(ax2, filter(r -> r.trunkdir < 0, dfxdlvl).arboruvcent2, color=:blue, marker=:x)
        scatter!(ax2, unique(filter(r -> r.trunkdir < 0, dfxdlvl).somauv), color=:blue, markersize=2)
        xlims!(ax2, -0.45, 0.45)
        ylims!(ax2, -0.45, 0.45)
    else
        dd = Vec3f(0, -1, 0)
        vals1 = [dot(p, dd) / 1000 for p in filter(r -> r.trunkdir > 0, dfxdlvl).center]
        vals2 = [dot(p, dd) / 1000 for p in unique(filter(r -> r.trunkdir < 0, dfxdlvl)).soma_pos]
        density!(ax2, vals1, color=(:red, 0.5), label="dlPFC → vlPFC arbors")
        density!(ax2, vals2, color=(:blue, 0.5), label="vlPFC → dlPFC soma")
        hidexdecorations!(ax2)
        let t = HypothesisTests.MannWhitneyUTest(vals1, vals2)
            write_stat(:Dl2VlArborVsVl2DlSomaPval, compact_str(HypothesisTests.pvalue(t)))
            write_stat(:Dl2VlArborVsVl2DlSomaMethod, :MannWhitneyUTestTwoSided)
        end

        vals1 = [dot(p, dd) / 1000 for p in filter(r -> r.trunkdir < 0, dfxdlvl).center]
        vals2 = [dot(p, dd) / 1000 for p in unique(filter(r -> r.trunkdir > 0, dfxdlvl)).soma_pos]
        density!(ax3, vals1, color=(:blue, 0.5), label="vlPFC → dlPFC arbors")
        density!(ax3, vals2, color=(:red, 0.5), label="dlPFC → vlPFC soma")
        linkxaxes!(ax2, ax3)
        let t = HypothesisTests.MannWhitneyUTest(vals1, vals2)
            write_stat(:Vl2DlArborVsDl2VlSomaPval, compact_str(HypothesisTests.pvalue(t)))
            write_stat(:Vl2DlArborVsDl2VlSomaMethod, :MannWhitneyUTestTwoSided)
        end
        for (i, ax) in enumerate((ax2, ax3))
            ax.ylabel[] = "Density"
            hideydecorations!(ax, label=false)
            ylims!(ax, 0, nothing)
            setup_axis_by_orient!(ax, :x, vcat(vals1, vals2), dd)
            Legend(grid[1, 2][i, 1][1, 1], ax, tellheight=true, tellwidth=false)
            rowgap!(content(grid[1, 2][i, 1]), 1, 0)
        end
    end
end

function plot_intra_pfc_acgg_ofc_bp(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    acggs = Set(filter(r -> contains("ACgG")(r.arborhier) && r.ipsi, dfarbor).path)
    ofcs = Set(filter(r -> contains("OFC")(r.arborhier) && r.ipsi, dfarbor).path)
    cnt = Dict()
    for rr in eachrow(subregs)
        length(rr.neurons) >= 10 || continue
        set0 = rr.neurons
        proj1 = intersect(set0, acggs)
        proj2 = intersect(set0, ofcs)
        set0 = union(proj1, proj2)
        proj3 = intersect(proj1, proj2)
        for (proj, sss) in enumerate((setdiff(proj1, proj3), setdiff(proj2, proj3), proj3))
            isempty(set0) && continue
            ##rr.subreg in (16, 19) && continue
            key = (; rr.subreg, proj)
            cnt[key] = get(cnt, key, 0) + length(sss) / length(set0)
        end
        #t = HypothesisTests.FisherExactTest(length(proj3), length(setdiff(proj1, proj2)), length(proj2), length(setdiff(set0, proj2)))
        #@info "$(rr.subreg) $t"
    end
    cnt = collect(cnt)

    ax = Axis(grid[1, 2], width=pt(52))
    hitregs = [p.first.subreg for p in cnt]
    dodge = [p.first.proj for p in cnt]
    hitregssort = sort(unique(hitregs), by=s -> filter(r -> r.subreg == s, subregs)[1, :centxyz][3])
    barplot!(ax, something.(indexin(hitregs, hitregssort)), [p.second for p in cnt], stack=dodge, color=[:red, :blue, :darkmagenta][dodge], direction=:x)
    ax.yticks[] = (axes(hitregssort, 1), string.(hitregssort))
    xlims!(ax, 0, 1)
    hideydecorations!(ax, ticklabels=false, ticks=false)
    # for rr in eachrow(subregs)
    #     vvv = filter(p -> p.first.subreg == rr.subreg, cnt)
    #     pie!(ax, rr.centuv..., [p.second for p in vvv], color=[:red, :blue, :darkmagenta][[p.first.proj for p in vvv]])
    # end
    ax2 = Axis(grid[1, 1], width=pt(12))
    bracket!(0, 4, 0, 1, text="vlPFC", orientation=:down)
    bracket!(0, 12, 0, 5, text="dlPFC", orientation=:down)
    ax2.yticks = ([13 / 2], [L"\fontfamily{Arial}Ventral $⟷$ Dorsal"])
    ax2.yticklabelrotation = π / 2
    hidedecorations!(ax2, ticklabels=false, label=false)
    hidexdecorations!(ax2)
    hidespines!(ax2)
    xlims!(ax2, -2, 0)
    linkyaxes!(ax, ax2)
    colgap!(grid, 1, 2)
    ax2.ylabel[] = "Subdomains"
    ax.xlabel[] = "Proportion"
    hidexdecorations!(ax, label=false, ticklabels=false, ticks=false)
    Legend(grid[1, 2, BottomRight()], [PolyElement(color=c) for c in (:red, :blue, :darkmagenta)], ["to ACgG", "to OFC", "to Both"], halign=:left, valign=:bottom, tellwidth=false, padding=(6, 0, 0, 0))
end

function plot_acgg_ofc_soma_venn(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    acggs = Set(filter(r -> contains("ACgG")(r.arborhier) && r.ipsi, dfarbor).path)
    ofcs = Set(filter(r -> contains("OFC")(r.arborhier) && r.ipsi, dfarbor).path)
    acgg_ofc_pat(r) = begin
        (; acgg=r.path in acggs, ofc=r.path in ofcs)
    end
    dfx = insertcols(df, :pat => acgg_ofc_pat.(eachrow(df)))
    filter!(r -> sum(r.pat) > 0 && zz1data.matchhier("lat_PFC")(r.somahier), dfx)

    ax = Axis(grid[1, 1], width=pt(90))
    test_venn!(ax, DataFrame(dfx.pat), :acgg => :red, :ofc => :blue; tagname=t -> (; acgg="to ACgG", ofc="to OFC")[t], scale=1.3)
    hidedecorations!(ax)
    hidespines!(ax)
    Legend(grid[2, 1], [PolyElement(color=c) for c in (:red, :blue, :darkmagenta)], ["to ACgG", "to OFC", "to Both"], orientation=:horizontal)
    rowgap!(grid, 1, 2)
end

function plot_acgg_ofc_soma_flat(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    acggs = Set(filter(r -> contains("ACgG")(r.arborhier) && r.ipsi, dfarbor).path)
    ofcs = Set(filter(r -> contains("OFC")(r.arborhier) && r.ipsi, dfarbor).path)
    acgg_ofc_pat(r) = begin
        (; acgg=r.path in acggs, ofc=r.path in ofcs)
    end
    rowcolor(r) = begin
        r.pat.acgg && r.pat.ofc && return :darkmagenta
        r.pat.acgg && return :red
        r.pat.ofc && return :blue
    end
    dfx = insertcols(df, :pat => acgg_ofc_pat.(eachrow(df)))
    filter!(r -> sum(r.pat) > 0 && !r.qroot && zz1data.matchhier("lat_PFC")(r.somahier), dfx)

    ax = Axis(grid[1, 1], height=pt(100), width=pt(130))
    plot_flatmap_base!(ax, highlights, objflat, objphy)
    scatter!(dfx.somauv, color=rowcolor.(eachrow(dfx)), markersize=pt(2))
    xlims!(ax, -0.35, 0.40)
    ylims!(ax, -0.40, 0.25)

    Legend(grid[2, 1], [PolyElement(color=c) for c in (:red, :blue, :darkmagenta)], ["to ACgG", "to OFC", "to Both"], position=:rb, orientation=:horizontal)
    rowgap!(grid, 1, 2)
end

function plot_intra_pfc_ofc_acgg_somadepth(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    acggs = Set(filter(r -> contains("ACgG")(r.arborhier) && r.ipsi, dfarbor).path)
    ofcs = Set(filter(r -> contains("OFC")(r.arborhier) && r.ipsi, dfarbor).path)

    ax1t = Axis(grid[1, 1])
    ax2 = Axis(grid[2, 1], width=38, height=56)
    rowgap!(grid, 1, 0)
    rowx(r) = begin
        r.path in acggs && r.path in ofcs && return 3
        r.path in acggs && return 1
        r.path in ofcs && return 2
        0
    end
    dfx = filter(r -> r.inj == "G97-1bL" && !r.qroot && rowx(r) in (1, 2), df)

    boxplot_and_compare!(ax2, dfx, ax1t; rowx, xname=r -> ("to ACgG", "to OFC", "to Both")[r], rowy=r -> r.somaw, rowcolor=r -> (:red, :blue, :darkmagenta)[rowx(r)], statif=(a, b) -> a.x != b.x, violin_limits=(0, 1),)
    let t = HypothesisTests.MannWhitneyUTest(filter(r -> rowx(r) == 1, dfx).somaw, filter(r -> rowx(r) == 2, dfx).somaw)
        write_stat(:ToAcggVsToOfcSomaDepthPval, HypothesisTests.pvalue(t))
        write_stat(:ToAcggVsToOfcSomaDepthMethod, :MannWhitneyUTestTwoSided)
    end
    yaxis_as_layer(ax2, 0, 1)
    ax2.ylabel = "Soma laminar depth"
    rotatexlabel!(ax2, π / 3)
    ax2.title = "Injection site\nG97-1bL"
end

function plot_intra_pfc_area_hm(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    acggs = Set(filter(r -> contains("ACgG")(r.arborhier) && r.ipsi, dfarbor).path)
    ofcs = Set(filter(r -> contains("OFC")(r.arborhier) && r.ipsi, dfarbor).path)
    temps = Set(filter(r -> contains("Temporal")(r.arborhier) && r.ipsi, dfarbor).path)
    paris = Set(filter(r -> contains("Parietal")(r.arborhier) && r.ipsi, dfarbor).path)
    dlpfcs = Set(filter(r -> contains("dlPFC")(r.arborhier) && r.ipsi, dfarbor).path)
    area12rs = Set(filter(r -> endswith("area_12r")(r.arborhier) && r.ipsi, dfarbor).path)
    subregtgts = map(rr -> Set(filter(r -> r.subreg == rr.subreg && r.ipsi, dfarbor).path), eachrow(subregs))

    cnt = Dict()
    for rr in eachrow(subregs)
        length(rr.neurons) >= 10 || continue
        set0 = rr.neurons
        proj1 = intersect(set0, acggs)
        proj2 = intersect(set0, ofcs)
        set0 = union(proj1, proj2)
        proj3 = intersect(proj1, proj2)
        for (proj, sss) in enumerate((setdiff(proj1, proj3), setdiff(proj2, proj3), proj3))
            key = (; rr.subreg, proj)
            cnt[key] = get(cnt, key, 0) + length(sss) / length(set0)
        end
        #t = HypothesisTests.FisherExactTest(length(proj3), length(setdiff(proj1, proj2)), length(proj2), length(setdiff(set0, proj2)))
        #@info "$(rr.subreg) $t"
    end
    cnt = collect(cnt)


    ax3 = Axis(grid[1, 1])
    tgts = [acggs, ofcs, temps, paris, dlpfcs, area12rs]
    #tgts = [subregtgts...]
    dfx = filter(r -> r.inj == "G97-1bL" && r.path in union(tgts...), df)
    mat = zeros(length(tgts), size(dfx, 1))
    for (i, tgt) in enumerate(tgts)
        for (j, r) in enumerate(eachrow(dfx))
            mat[i, j] = r.path in tgt
        end
    end
    o2 = Clustering.hclust(1 .- cor(mat), linkage=:ward, branchorder=:barjoseph).order
    o1 = Clustering.hclust(1 .- cor((mat .+ rand(size(mat)) .* 0.001)'), linkage=:ward, branchorder=:barjoseph).order
    o1 = 1:6
    cc = cor((mat .+ rand(size(mat)) .* 0.001)')
    global xxxcc = cc
    @warn "$cc"
    @warn "yyy $(sum(mat[1, :].*mat[5, :]))"
    heatmap!(ax3, mat[o1, o2]')
    ax3.yticks[] = (axes(tgts, 1), ["ACgG", "OFC", "Temporal", "Parietal", "dlPFC", "Area 12r"][o1])

    ax4 = Axis(grid[1, 2], width=60, height=60)
    @warn "$(subregs[o1, :subreg])"
    Makie.heatmap!(ax4, cc[o1, o1])

end

function plot_intra_pfc_subdomain_hm(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    pathways = vote_connections(subregs, dfarbor; thr=3)
    conns = find_intra_pfc_combo(subregs, pathways)

    srcsubreg = 3
    set0 = Set(filter(r -> r.subreg == srcsubreg, subregs)[1, :neurons])
    tgts = map(eachrow(subregs)) do r
        r.subreg => intersect(set0, Set(filter(rr -> rr.arborpath in r.arbors, dfarbor).path))
    end
    filter!(!isempty ∘ last, tgts)
    tgts_id = first.(tgts)
    tgts = last.(tgts)
    dfx = filter(r -> r.path in set0, df)

    mat = zeros(length(tgts), size(dfx, 1))
    for (i, tgt) in enumerate(tgts)
        for (j, r) in enumerate(eachrow(dfx))
            mat[i, j] = r.path in tgt
        end
    end

    ax3 = Axis(grid[1, 1])
    global xxxmat = deepcopy(mat)
    o2 = Clustering.hclust(1 .- cor(mat .+ randn(size(mat)) .* 0.01), linkage=:ward, branchorder=:barjoseph).order
    o1 = Clustering.hclust(1 .- cor((mat .+ randn(size(mat)) .* 0.01)'), linkage=:ward, branchorder=:barjoseph).order
    pvalue(t1, t2) = begin
        v12 = map(eachrow(dfx)) do r
            Int.((r.path in t1, r.path in t2))
        end
        t = HypothesisTests.ChisqTest(first.(v12), last.(v12), (0:1, 0:1))
        pv = HypothesisTests.pvalue(t)
        if pv < 0.05
            #scatter!(ax3, [jjjjjjjjjjj])
            @warn "pvvvv   $t"
        end
        pv
    end
    #-log(pv)
    #cc = [pvalue(t1, t2) for t1 in tgts, t2 in tgts]
    cc = cor((mat .+ rand(size(mat)) .* 0.001)')
    #global xxxcc = cc
    #@warn "$cc"
    @warn "yyy $(sum(mat[1, :].*mat[5, :]))"
    heatmap!(ax3, mat[o1, o2]')
    ax3.xlabel[] = "Neurons in subdomain #$srcsubreg"
    ax3.ylabel[] = "Target subdomains"
    hidexdecorations!(ax3, label=false)
    #ax3.yticks[] = (axes(tgts, 1), ["ACgG", "OFC", "Temporal", "Parietal", "dlPFC", "Area 12r"][o1])

    ax4 = Axis(grid[1, 2], width=pt(85), height=pt(85))
    colgap!(grid, 1, 2)
    #Label(grid[1, 2, Right()], "Co-innervation", rotation=-π / 2; set_font_h2x()...)
    @warn "$(subregs[o1, :subreg])"
    Makie.heatmap!(ax4, cc[o1, o1])
    findord(reg) = begin
        findfirst(i -> tgts_id[o1[i]] == reg, axes(o1, 1))
    end
    for cc in conns
        !(cc.from == srcsubreg && cc.p < 0.05) && continue
        a = findord(cc.to)
        b = findord(cc.to2)
        scatter!(ax4, [Point2f(a, b)], markersize=pt(1.5), color=:red)
    end
    for a in axes(o1, 1), b in axes(o1, 1)
        if a < b && pvalue(tgts[o1[a]], tgts[o1[b]]) < 0.05
            #scatter!(ax4, [Point2f(a, b)], markersize=pt(0.7), color=:red)
        end
    end
    a = findord(76)
    b = findord(83)
    lines!(ax4, Makie.Rect2f(a - 0.5, b - 0.5, 1, 1), color=:red)
    linkyaxes!(ax3, ax4)
    hidedecorations!(ax4, label=false)
    hidedecorations!(ax3, label=false)
    ax4.xlabel = "Co-innervation\ntarget subdomains"
end

function plot_intra_pfc_ofc_acgg(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    ax1 = LScene(grid[1, 1], width=pt(130), height=pt(130), show_axis=false)
    #ax2 = LScene(grid[1, 2], width=pt(130), height=pt(130), show_axis=false)
    #colgap!(grid, 1, 0)
    nrns1 = Set(filter(r -> contains("dlPFC")(r.somahier), df).path)
    nrns2 = Set(filter(r -> contains("vlPFC")(r.somahier), df).path)
    set0 = union(nrns1, nrns2)
    dfx = filter(r -> r.path in set0, dfarbor)
    filter!(r -> contains("OFC")(r.arborhier) && r.ipsi, dfx)

    cam = (fov=9.0, cent=(37241.47, 61715.64, 30481.07), deye=(-0.0992451000390354, 0.938213830967666146, -0.006995224853613485) .* 352710.64025207126, up=(-0.01, -0.0, 1.0))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        trunkcolor(rr) = begin
            rr === nothing && return :transparent
            rr.path in nrns1 && return :red
            rr.path in nrns2 && return :blue
            :transparent
            :transparent
        end
        #draw_subtrees(pick_neurons_with_limit(filter(r -> r.trunkto == 2, dfxacggofc), r -> r.trunkfrom, 10000), mix1; autotrunk=false, inspect=:nmt, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor=trunkcolor)...)
        draw_subtrees(dfx, mix1; autotrunk=false, inspect=:nmt, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor=trunkcolor)...)
    end
    draw_mesh_overlay([obj => :gray], ax1)
    draw_contours(["interm/mesh-nmt-$ba-l.obj" for ba in [:ACgG, :OFC]], ax1)
    axislegend(ax1, [Makie.PolyElement(color=c) for c in (:red, :blue)], ["dlPFC → ACgG", "dlPFC → OFC"])

    return

    cam = (fov=9.0, cent=(37241.47, 61715.64, 30481.07), deye=(-0.9992451000390354, 0.038213830967666146, -0.006995224853613485) .* 262710.64025207126, up=(-0.01, -0.0, 1.0))
    setup_camera(ax2 => cam)
    with_mixed_scene(ax2, GLMakie) do mix1
        trunkcolor(rr) = begin
            rr === nothing && return :transparent
            rr.trunkto == 1 && return :red
            rr.trunkto == 2 && return :blue
            :transparent
        end
        draw_subtrees(pick_neurons_with_limit(filter(r -> r.trunkfrom == 2, dfxacggofc), r -> r.trunkfrom, 100), mix1; autotrunk=false, inspect=:nmt, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor=trunkcolor)...)
    end
    draw_mesh_overlay([obj => :gray], ax2)
    draw_contours(["interm/mesh-nmt-$ba-l.obj" for ba in [:ACgG, :OFC]], ax2)
    axislegend(ax2, [Makie.PolyElement(color=c) for c in (:red, :blue)], ["vlPFC → ACgG", "vlPFC → OFC"])

    draw_ratio(gg, dfx) = begin
        ax = Axis(gg, width=pt(30), height=pt(30))
        a, b = [Set(filter(r -> r.trunkto == t, dfx).path) for t in 1:2]
        c = intersect(a, b)
        barplot!(ax, length.([a, b, c]), color=[:red, :blue, :magenta])
        #hidedecorations!(ax)
        #hidespines!(ax)
    end
    draw_ratio(grid[1, 3][1, 1], filter(r -> r.trunkfrom == 1, dfxacggofc))
    draw_ratio(grid[1, 3][2, 1], filter(r -> r.trunkfrom == 2, dfxacggofc))
end

function plot_intra_pfc_arbor_cond(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    nrns1 = Set(filter(r -> contains("dlPFC")(r.somahier), df).path)
    nrns2 = Set(filter(r -> contains("vlPFC")(r.somahier), df).path)

    ax = Axis(grid[1, 1], width=130, height=100) #= aspect=Makie.AxisAspect(1),  =#
    plot_flatmap_base!(ax, highlights, objflat, objphy)
    xlims!(ax, -0.35, 0.55)
    ylims!(ax, -0.65, -0.10)

    set0 = union(nrns1, nrns2)
    dfx = filter(r -> r.path in set0, dfarbor)
    filter!(r -> contains("OFC")(r.arborhier) && r.ipsi, dfx)
    arbcolor(r) = begin
        r.path in nrns1 && return :red
        r.path in nrns2 && return :blue
        :gray
    end
    v1 = zz1data.vote(filter(r -> r.path in nrns1, dfx), :arborhier, :path)
    v2 = zz1data.vote(filter(r -> r.path in nrns2, dfx), :arborhier, :path)
    @warn "vvv $v1"
    @warn "vvv $v2"
    Makie.scatter!(ax, dfx.arboruvcent2, color=arbcolor.(eachrow(dfx)))
    Legend(grid[2, 1], [PolyElement(color=c) for c in (:red, :blue)], ["from dlPFC", "from vlPFC"], orientation=:horizontal, halign=:right)
    Label(grid[2, 1], "Arbor center", halign=:left)
    rowgap!(grid, 1, 0)
end

function plot_intra_pfc_arbor_conn(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    dfx = filter(r -> r.inj == "G97-1bL" && r.ipsi, dfarbor)
    #filter!(r -> match(r"ACgG|OFC|lat_PFC", r.arborhier) !== nothing, dfx)
    #filter!(r -> match(r"area_45b", r.arborhier) === nothing, dfx)
    nrns = [
    # "monkeytemp/split-sample234197-ch1/root21013897-root049-good.swc.gz",
    # "monkeytemp/split-sample234197-ch1/root51727435-root103-good.swc.gz",
    # "monkeytemp/split-sample234197-ch1/root51920035-root026-good.swc.gz",
    # "monkeytemp/split-sample234197-ch1/root52261080-root105-good.swc.gz",
    # "monkeytemp/split-sample234197-ch1/root52266595-root106-good.swc.gz",
    # "monkeytemp/split-sample234197-ch1/root52337557-root141-good.swc.gz",
    # "monkeytemp/split-sample234197-ch1/root52342844-root144-good.swc.gz",
    #"monkeytemp/split-sample234197-ch1/root53414085-root280-good.swc.gz",
    ]
    #filter!(r->r.path in nrns, dfx)
    pts = []
    nrns = Set()
    for r in eachrow(dfx)
        contains("ACgG")(r.arborhier) || continue
        for rr in eachrow(dfx)
            r.arborpath == rr.arborpath && continue
            contains("dlPFC")(rr.arborhier) || continue
            if r.path == rr.path
                push!(pts, r.arboruvcent2)
                push!(pts, rr.arboruvcent2)
                push!(nrns, r.path)
            end
        end
    end
    nrns = sort(collect(nrns))[[3, 8]]
    @warn "xxx $(length(nrns))"
    @warn "xxx $(nrns)"
    filter!(r -> r.path in nrns, dfx)
    filter!(r -> match(r"(ACgG|dlPFC)", r.arborhier) !== nothing, dfx)

    ###
    ax = Axis(grid[1, 1], aspect=Makie.DataAspect(), width=130, height=130)
    plot_flatmap_base!(ax, highlights, objflat, objphy)
    xlims!(ax, -0.75, 0.45)
    ylims!(ax, -0.65, 0.35)
    linesegments!(ax, pts, color=(:blue, 0.3), linewidth=0.5)
    scatter!(ax, filter(r -> r.path in nrns && !r.qroot, df).somauv)

    ax5 = LScene(grid[1, 2], width=pt(120), height=pt(120), show_axis=false)
    ax6 = LScene(grid[1, 3], width=pt(120), height=pt(120), show_axis=false)
    setup_camera(ax5 => :hfrot, ax6 => :s)
    with_mixed_scene(ax5, GLMakie) do mix
        with_mixed_scene(ax6, GLMakie) do mix2
            trunkcolor(rr) = begin
                rr === nothing && return :transparent
                contains("ACgG")(rr.arborhier) && return :red
                contains("dlPFC")(rr.arborhier) && return :blue
                :transparent
            end
            draw_subtrees(dfx, mix, mix2; autotrunk=false, inspect=:nmt, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor=trunkcolor)...)
        end
    end
    draw_mesh_overlay([obj => :gray], ax5, ax6)
end

function plot_intra_pfc_arbor_test(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    nrns1 = Set(filter(r -> contains("dlPFC")(r.somahier), df).path)

    ax = Axis(grid[1, 1], width=90, height=100)
    plot_flatmap_base!(ax, highlights, objflat, objphy)
    xlims!(ax, -0.85, 0.05)
    ylims!(ax, -0.55, 0.20)

    set0 = union(nrns1, ())
    dfx = filter(r -> r.path in set0, dfarbor)
    filter!(r -> r.path in nrns1 && r.ipsi && !r.qroot && match(r"ACgG", r.arborhier) !== nothing, dfx) #= contains("ACgG")(r.arborhier) &&  =#
    #filter!(r -> r.path in nrns1 && r.ipsi && !r.qroot && match(r"ACgG|OFC", r.arborhier)!==nothing, dfx) #= contains("ACgG")(r.arborhier) &&  =#
    shuffle!(dfx)
    arbcolor(r) = begin
        r.path in nrns1 && return r.soma_pos[2]
        0#:gray
    end
    v1 = zz1data.vote(filter(r -> r.path in nrns1, dfx), :arborhier, :path)
    #v2 = zz1data.vote(filter(r -> r.path in nrns2, dfx), :arborhier, :path)
    @warn "vvv $v1"
    #@warn "vvv $v2"
    #Makie.scatter!(ax, dfx.arboruvcent2, color=arbcolor.(eachrow(dfx)), colormap=[:blue, :red])
    Makie.linesegments!(ax, collect(zip(dfx.somauv, dfx.arboruvcent2)), color=arbcolor.(eachrow(dfx)), colormap=[(:blue, 0.3), (:red, 0.3)])
    normit(v) = v ./ norm(v) / 1000
    Makie.arrows!(ax, Point2f.(dfx.arboruvcent2), (normit ∘ Point2f).(dfx.arboruvcent2 .- dfx.somauv), color=arbcolor.(eachrow(dfx)), colormap=[(:blue, 0.9), (:red, 0.9)], arrowsize=3)
    Legend(grid[2, 1], [PolyElement(color=c) for c in (:red, :blue)], ["Rostral dlPFC", "Caudal dlPFC"], position=:rb, orientation=:horizontal)
    rowgap!(grid, 1, 0)
end


function plot_intra_pfc_ofc_acgg_arbors(grid, env; dfxacggofc, _...)
    ax3 = Axis(grid[1, 1])
    ax4 = Axis(grid[1, 2],)
    colgap!(grid, 1, 4)

    dats = Dict()
    for (dlvl, col) in [(1, :red), (2, :blue)]
        dfx = filter(r -> r.trunkfrom == dlvl && r.trunkto == 1 && r.sample in ("234197", "232717"), dfxacggofc)
        xx, yy = calc_fvhist(dfx.fv_whist, trimextream=false)
        dats[dlvl] = dfx
        lines!(ax3, yy, xx, color=col, label=string(("dlPFC", "vlPFC")[dlvl], " → ACgG"))
    end
    test_depth_dist(:ToAcggDlVsVl, dats[1].fv_whist, dats[2].fv_whist)
    yaxis_as_layer(ax3, 0, 1)
    Legend(grid[2, 1], ax3, tellheight=true, tellwidth=false) |> fix_line_legs
    ax3.xlabel = "Density"
    hidexdecorations!(ax3, label=false)
    # t = test_depth_dist(filter(r -> r.trunkfrom == 2 && r.trunkto == 1, dfxacggofc).fv_whist, filter(r -> r.trunkfrom == 2 && r.trunkto == 2, dfxacggofc).fv_whist)
    # @info "test: $t"

    dats = Dict()
    for (dlvl, col) in [(1, :red), (2, :blue)]
        dfx = filter(r -> r.trunkfrom == dlvl && r.trunkto == 2 && r.sample in ("234197", "232717"), dfxacggofc)
        xx, yy = calc_fvhist(dfx.fv_whist, trimextream=false)
        dats[dlvl] = dfx
        lines!(ax4, yy, xx, color=col, label=string(("dlPFC", "vlPFC")[dlvl], " → OFC"))
    end
    test_depth_dist(:ToOfcDlVsVl, dats[1].fv_whist, dats[2].fv_whist)
    yaxis_as_layer(ax4, 0, 1)
    Legend(grid[2, 2], ax4, tellheight=true, tellwidth=false) |> fix_line_legs
    linkyaxes!(ax3, ax4)
    linkxaxes!(ax3, ax4)
    ax4.xlabel = "Density"
    hidexdecorations!(ax4, label=false)
    hideydecorations!(ax4)
    rowgap!(grid, 1, 2)
    colgap!(grid, 1, 2)
end

function plot_intra_pfc_dl_acgg_topo(grid, env; df, dfarbor, dfxacggofc, trunks, subregs, obj, highlights, objflat, objphy, _...)
    Random.seed!(0)
    dfx = filter(r -> r.trunkfrom == 1 && r.trunkto == 1, dfxacggofc)
    nrn2arb = Dict()
    for r in eachrow(dfx)
        aa = get!(() -> [], nrn2arb, r.path)
        push!(aa, r.center)
    end
    nrn2cent = Dict(zip(df.path, df.soma_pos))
    na = map(collect(nrn2arb)) do (n, aa)
        a = shuffle!(aa)[1]
        n = nrn2cent[n]
        if !NmtDat.isleft(n)
            n = NmtDat.mirror(n)
        end
        if !NmtDat.isleft(a)
            a = NmtDat.mirror(a)
        end
        (n, a)
    end
    dfxx = DataFrame(:x => first.(na), :y => last.(na))
    xpars, ypars = zz1data.find_best_coor_axis(dfxx.x, dfxx.y, aligned_x=2, aligned_y=3)
    @warn "$xpars $ypars"
    insertcols!(dfxx, :xx => dot.(dfxx.x, Ref(xpars)), :yy => dot.(dfxx.y, Ref(ypars)))

    ax1 = Axis(grid[1, 1])
    linear_reg_scatter!(ax1, dfxx; rowx=r -> r.xx / 1000, rowy=r -> r.yy / 1000, rowcolor=r -> r.xx, colormap=[:blue, :red])
    ax1.xlabel[] = "Soma position"
    ax1.ylabel[] = "Arbor center"
    setup_axis_by_orient!(ax1, :x, dfxx.xx ./ 1000, xpars)
    setup_axis_by_orient!(ax1, :y, dfxx.yy ./ 1000, ypars)
    Label(grid[1, 1, Top()], "", height=pt(10))
end

function find_intra_pfc_combo(subregs, pathways)
    conns = []
    posof(reg) = filter(r -> r.subreg == reg, subregs)[1, :centuv]
    hierof(reg) = filter(r -> r.subreg == reg, subregs)[1, :parhier]
    allnrns = collect(reduce(union, [t.neurons for t in pathways]))
    for p1 in pathways
        for p2 in pathways
            p1.from == p1.to && continue
            p2.from == p2.to && continue
            (p1.to <= 0 || p2.to <= 0) && continue
            p1.to >= p2.to && continue
            v12 = map(n -> (n in p1.neurons, n in p2.neurons), allnrns)
            v1, v2 = first.(v12) .+ 0, last.(v12) .+ 0
            t = HypothesisTests.ChisqTest(v1, v2, (0:1, 0:1))
            pa = sum(v1) / length(v1) * sum(v2) / length(v2) * length(v2)
            pb = sum(v1 .* v2)

            p = HypothesisTests.pvalue(t)
            if p < 0.05 && t.observed[2, 2] >= 2 && norm(posof(p1.to) - posof(p2.to)) > 0.2 && norm(posof(p1.to) - posof(p1.from)) > 0.2 && norm(posof(p2.from) - posof(p2.to)) > 0.2
                @warn "$p $pa $pb $(p1.from) $(p2.from) $(hierof(p1.from)) $(p1.to) $(p2.to) $(hierof(p1.to)) $(hierof(p2.to))"
                @assert p1.from == p2.from
                push!(conns, (; p1.from, p1.to, to2=p2.to, nrns=intersect(p1.neurons, p2.neurons), p))
            end
        end
    end
    conns
end

function plot_intra_pfc_combo(grid, env; df, dfarbor, trunks, subregs, obj, highlights, objflat, objphy, _...)
    pathways = vote_connections(subregs, dfarbor; thr=3)
    #@info "$([(subregs[t.from, :parhier], subregs[t.to, :parhier], length(t.neurons)) for t in dfx])"

    conns = Tuple.(filter(r -> r.p < 0.001, find_intra_pfc_combo(subregs, pathways)))
    @warn "xx conns $conns"
    filter!(v -> v[2:3] == (76, 83), conns)
    @warn "xx conns $conns"

    ax1 = LScene(grid[1, 1], width=pt(80), height=pt(100), show_axis=false)

    cam = (fov=9.0, cent=(19877.05, 60653.16, 29783.93), deye=(0, 0, 1) .* 150000.0, up=(1.0, 0, 0))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        nrn2col = Dict()
        trunkcolor(rr) = begin
            rr === nothing && return :transparent
            nrncol = get!(() -> to_colormap(:tab10)[length(nrn2col)+1], nrn2col, rr.path)
            rr.subreg == conns[1][2] && return nrncol
            rr.subreg == conns[1][3] && return nrncol
            :transparent
        end
        dfx = filter(r -> r.subreg in conns[1][2:3] && r.path in conns[1][4], dfarbor)
        draw_subtrees(dfx, mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor=trunkcolor)...)
    end
    draw_mesh_overlay(["interm/mesh-nmt-$ba-l.obj" => :red for ba in [:area_24c, :area_12l, :area_46d]], ax1)
    draw_mesh_overlay([obj => :gray], ax1)
end

function plot_intra_pfc_combo_flatmap(grid, env; df, dfarbor, trunks, subregs, obj, highlights, objflat, objphy, _...)
    pathways = vote_connections(subregs, dfarbor; thr=3)

    posof(reg) = filter(r -> r.subreg == reg, subregs)[1, :centuv]
    conns = Tuple.(filter(r -> r.p < 0.001, find_intra_pfc_combo(subregs, pathways)))

    for (iii, src) in enumerate(unique(sort(first.(conns))))
        connsx = filter(r -> r[1] == src, conns)
        cc = sort(unique([sort([c[2:3]...]) for c in connsx]))

        row, col = divrem(iii - 1, 3) .+ 1
        ax = Axis(grid[row, col], width=pt(160), height=pt(160))
        col > 1 && colgap!(grid, col - 1, 6)
        row > 1 && rowgap!(grid, row - 1, 8)
        plot_flatmap_base!(ax, highlights, objflat, objphy)
        xlims!(ax, -0.85, 0.45)
        ylims!(ax, -0.65, 0.65)

        c2c = map(cc) do c
            p = lines!(ax, posof.(c))
            c => p.color[]
        end
        c2c = Dict(c2c)
        done = Set()
        for c in connsx
            x = sort([c[2:3]...])
            m = +(posof.(x)...) / 2
            for c1 in c[1:3]
                c1 in done && continue
                push!(done, c1)
                plt = if c1 == c[1]
                    scatter!(ax, [posof(c1)], color=:red, markersize=:10)
                    text!([posof(c1)], text=["#$c1"]; set_font_h2x()...)
                else
                    text!([posof(c1)], text=["#$c1"])
                end
                translate!(plt, 0, 0, 1)
            end
            #lines!(ax, [posof(c[1]), m], color=c2c[x])
        end
    end
end

function plot_intra_pfc_grp_combo(grid, env; df, dfarbor, trunks, subregs, obj, highlights, objflat, objphy, _...)
    pathways = vote_connections(subregs, trunks; thr=3)
    get_grp(i) = begin
        sfx, i = if i > 0
            "", i
        else
            "(C)", -i
        end
        pfx = if subregs[i, :dlpfc]
            "dlPFC"
        elseif subregs[i, :vlpfc]
            "vlPFC"
        elseif contains("ACgG")(subregs[i, :parhier])
            "ACgG"
        elseif contains("OFC")(subregs[i, :parhier])
            "OFC"
        else
            return nothing
        end
        "$pfx$sfx"
    end
    conns = Dict()
    for pw in pathways
        pw.from == pw.to && continue
        ff = get_grp(pw.from)
        tt = get_grp(pw.to)
        (ff === nothing || tt === nothing) && continue
        nrns = get!(() -> Set(), conns, ff => tt)
        for n in pw.neurons
            push!(nrns, n)
        end
    end
    pathways = map(collect(conns)) do ((from, to), neurons)
        (; from, to, neurons)
    end
    allnrns = collect(reduce(union, [t.neurons for t in pathways]))

    conns = []
    for p1 in pathways
        for p2 in pathways
            p1.from == p1.to && continue
            p2.from == p2.to && continue
            p1.to >= p2.to && continue
            v12 = map(n -> (n in p1.neurons, n in p2.neurons), allnrns)
            v1, v2 = first.(v12) .+ 0, last.(v12) .+ 0
            t = HypothesisTests.ChisqTest(v1, v2, (0:1, 0:1))
            pa = sum(v1) / length(v1) * sum(v2) / length(v2) * length(v2)
            pb = sum(v1 .* v2) / length(v2)

            p = HypothesisTests.pvalue(t)
            if p < 0.001 && t.observed[2, 2] >= 5
                @warn "$p $pa $pb $(t.expected) $(t.observed) $(p1.from) $(p2.from) $(p1.to) $(p2.to)"
                @assert p1.from == p2.from
                push!(conns, (p1.from, p1.to, p2.to))
            end
        end
    end
    # cc = sort(unique([sort([c[2:end]...]) for c in conns]))
    #
    # ax = Axis(grid[1, 1], width=pt(120), height=pt(120))
    # plot_flatmap_base!(ax, highlights, objflat, objphy)
    # xlims!(ax, -0.85, 0.45)
    # ylims!(ax, -0.65, 0.65)
    #
    # c2c = map(cc) do c
    #     p = lines!(ax, subregs[abs.(c), :centuv])
    #     c => p.color[]
    # end
    # c2c = Dict(c2c)
    # for c in conns
    #     x = sort([c[2:end]...])
    #     m = +(subregs[abs.(x), :centuv]...) / 2
    #     lines!(ax, [subregs[c[1], :centuv], m], color=c2c[x])
    # end

end

function plot_bundle_demo(grid, env; trunks, pathways, obj, _...)
    for (row, (cls, cam)) in enumerate([(97, :demoparitrunk), (697, :demotrunkshort)])
        ax1 = Makie.LScene(grid[row, 1], width=pt(100), height=pt(65), show_axis=false)
        if row > 1
            rowgap!(grid, row - 1, 0)
        end
        setup_camera(ax1 => cam)
        selection = r -> r.trunkcls in (cls,)
        paths = Set(filter(selection, trunks).path)
        with_mixed_scene(ax1, GLMakie) do mix1
            draw_subtrees(filter(r -> r.path in paths && selection(r), trunks), mix1 => :hr; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; arborcolor=(:gray, 0.3), selection)...)
        end
        draw_mesh_overlay([obj => :gray], ax1)
    end
end

function draw_pathways_flatmap!(ax, df; rowcolor=nothing, hollow=false, short=false, kw...)
    inspector_label(p, i, x) = begin
        cls, src, tgt, ints, deps = df[i, [:trunkcls, :src_hier, :tgt_hier, :intensitys, :trunkdepths]]
        "$cls $src -> $tgt ; $(extrema(ints));$(extrema(map(v->isnan(v) ? 0 : v, deps)))"
    end
    if rowcolor === nothing
        rowcolor = r -> log(r.w)
    end
    #Makie.linesegments!(ax, collect(zip(df.tailuv, df.headuv)), color=log.(df.w), colorrange=(0, 5), colormap=:Blues)
    arrows!(ax, Point2f.(df.tailuv), Point2f.(df.headuv .- df.tailuv), color=[rowcolor(r) for r in eachrow(df)], arrowsize=short ? pt(2) : pt(5), linewidth=short ? pt(0.3) : pt(1.0); inspector_label, kw...)
    if hollow
        Makie.arrows!(ax, Point2f.(df.tailuv), Point2f.(df.headuv .- df.tailuv), color=:white, arrowsize=pt(3), linewidth=pt(0.0); inspector_label, kw...)
    end

    Makie.limits!(ax, -0.8, 0.5, -0.6, 0.4)
end

select_intrinsic(r) = maximum(map(v -> isnan(v) ? 1.1 : v, r.trunkw)) < 0.9 && NmtDat.isleft(r.trunk[end])

function vote_connections(subregs, arbors; thr=3)
    p2reg = Dict()
    for r in eachrow(subregs)
        for p in r.neurons
            ii = get!(p2reg, p, r.subreg)
            @assert ii == r.subreg
        end
    end
    regs = Set(subregs.subreg)
    connmat = Dict()
    for r in eachrow(arbors)
        i = get(p2reg, r.path, 0)
        j = r.subreg
        (i == 0 || j == 0) && continue
        j in regs || continue
        if !r.ipsi
            j = -j
        end
        conn = get!(() -> Set(), connmat, i => j)
        push!(conn, r.path)
    end
    connmat = filter(v -> length(last(v)) > thr, collect(connmat))
    map(connmat) do ((from, to), neurons)
        (; from, to, neurons)
    end
end

function draw_pathway_flat!(ax, subregs, pathways)
    xlims!(ax, -0.85, 0.45)
    ylims!(ax, -0.65, 0.65)
    scatter!(ax, subregs.centuv, markersize=subregs.radxyz ./ 10000 / 5, markerspace=:data, inspector_label=(p, i, x) -> "$i $(subregs[i, :subreg])")
    text!(ax, subregs.centuv, text=[replace(v, r"^.*\|" => "") for v in subregs.parhier], fontsize=2, color=[contains("Frontal")(v) ? :red : :blue for v in subregs.parhier])
    iii = [findfirst(==(t.from), subregs.subreg) for t in pathways]
    jjj = [findfirst(==(abs(t.to)), subregs.subreg) for t in pathways]
    vvv = [length(t.neurons) for t in pathways]
    linesegments!(ax, collect(zip(subregs.centuv[iii], subregs.centuv[jjj])), linewidth=log.(vvv) / 3, color=reduce(vcat, fill([:black, :red], size(vvv, 1))), linecap=:round)
end

function draw_pathway_flat_alt!(ax, subregs, pathways; linecolor=nothing, linestyle=nothing)
    xlims!(ax, -0.85, 0.45)
    ylims!(ax, -0.65, 0.65)
    #scatter!(ax, subregs.centuv, markersize=subregs.radxyz ./ 10000 / 5, markerspace=:data, inspector_label=(p, i, x) -> "$i $(subregs[i, :subreg])")
    #text!(ax, subregs.centuv, text=[replace(v, r"^.*\|" => "") for v in subregs.parhier], fontsize=2, color=[contains("Frontal")(v) ? :red : :blue for v in subregs.parhier])
    iii = [findfirst(==(t.from), subregs.subreg) for t in pathways]
    jjj = [findfirst(==(abs(t.to)), subregs.subreg) for t in pathways]
    vvv = [length(t.neurons) for t in pathways]
    if linecolor === nothing
        linecolor = r -> begin
            r.dlpfc ? :red : (r.vlpfc ? :blue : :black)
        end
    end
    fff, ttt = Point2f.(subregs.centuv[iii]), Point2f.(subregs.centuv[jjj])
    getcolors(a) = [(r = subregs[i, :]; (linecolor(r), a)) for i in iii]
    #linesegments!(ax, collect(zip(subregs.centuv[iii], subregs.centuv[jjj])), linewidth=log10.(vvv), color=[(r = subregs[i, :]; linecolor(r)) for i in iii], linecap=:round)
    arrows!(ax, fff, ttt .- fff, linewidth=log10.(vvv), color=getcolors(0.5), arrowsize=pt(3.0) .* log10.(vvv), arrowcolor=getcolors(1.0); linestyle)
end

import Optim

function remove_gray(colors; thr=0.1)
    filter(colors) do c
        cc = [c.r, c.g, c.b]
        a, b = extrema(cc)
        b - a > thr
    end
end
function arrange_color_a(positions, colors, adjs)
    ii = fill(0, length(positions))
    i2adjf = Dict([i => [] for i in axes(ii, 1)])
    while !isempty(i2adjf)
        for (i, v) in i2adjf
            empty!(v)
        end
        put(i, j) = begin
            ii[i] != 0 && return
            ii[j] == 0 && return
            push!(i2adjf[i], ii[j])
        end
        for a in adjs
            put(a[1], a[2])
            put(a[2], a[1])
        end
        i, v = sort(collect(i2adjf), by=length ∘ last)[end]
        y = setdiff(axes(colors, 1), v)
        @warn "$i $v"
        y = if isempty(y)
            rand(v)
        else
            rand(y)
        end
        ii[i] = y
        pop!(i2adjf, i)
    end
    colors[ii]
end
function arrange_color(positions, colors, K=6; adjs=nothing)
    #
    #
    N = length(positions)
    M = length(colors)
    dm = [norm(positions[i] - positions[j]) for i in 1:N, j in 1:N]
    if adjs === nothing
        adjs = SVector[]
        for i in 1:N
            pp = sortperm(view(dm, i, :))
            for p in pp[1:M]
                push!(adjs, sort(SVector(i, p)))
            end
        end
    else
        adjs = unique([sort(SVector(a...)) for a in adjs])
        return arrange_color_a(positions, colors, adjs)
    end
    f(x) = begin
        d = 0.0
        for (a, b) in adjs
            d += norm(colors[round(Int, x[a])] - colors[round(Int, x[b])])
        end
        #@info "$d"
        -d
    end
    opt = Optim.optimize(f, Float32.(fill(1, N)), Float32.(fill(M, N)), Float32.(rand(1:M, N)), Optim.SAMIN(verbosity=1), Optim.Options(iterations=10000))
    colors[round.(Int, Optim.minimizer(opt))]
end

function plot_subreg_arbors_all(grid, env; df, subregs, dfarbor, trunks, obj, objflat, objphy, highlights, _...)
    ax1 = LScene(grid[1, 1], width=pt(124), height=pt(135), show_axis=false)
    subreg2idx = Dict(reverse.(enumerate(subregs.subreg)))

    setup_camera(ax1 => :xfl)
    with_mixed_scene(ax1, GLMakie) do mix1
        per_arbor = (pn, rr, cond) -> begin
            if !cond && rr !== nothing
                return subregs[subreg2idx[rr.subreg], :color]
            end
            :transparent
        end
        per_node = (pn, pa, n) -> pa => 0.3
        dfx = filter(r -> get(subreg2idx, r.subreg, 0) != 0 && r.ipsi, dfarbor)
        draw_subtrees(dfx, mix1; per_arbor, per_node, mirrorby=NmtDat, drawsoma=false)
    end
    draw_mesh_overlay([obj], ax1)
end

function plot_subreg_arbors(grid, env; df, subregs, dfarbor, trunks, obj, objflat, objphy, highlights, _...)
    ax1 = LScene(grid[1, 1], width=pt(120), height=pt(120), show_axis=false)
    subreg2idx = Dict(reverse.(enumerate(subregs.subreg)))

    setup_camera(ax1 => :xfl)
    with_mixed_scene(ax1, GLMakie) do mix1
        Random.seed!(3)
        reg2col = Dict()
        per_arbor = (pn, rr, cond) -> begin
            !cond && rr !== nothing && return get!(() -> to_colormap(:Set1_8)[length(reg2col)+1], reg2col, rr.subreg)
            :transparent
        end
        per_node = (pn, pa, n) -> pa => 0.3
        dfx = filter(r -> get(subreg2idx, r.subreg, 0) != 0 && r.ipsi, dfarbor)
        selsubregs = shuffle(unique(dfx.subreg))[1:8]
        filter!(r -> r.subreg in selsubregs && r.ipsi, dfx)
        tgtarbs, tgtnrns = filter(r -> r.subreg == selsubregs[1], subregs)[1, [:arbors, :neurons]]
        draw_subtrees(dfx, mix1; per_arbor, per_node, mirrorby=NmtDat)

        selnrns = Set(filter(r -> r.arborpath in tgtarbs && !(r.path in tgtnrns) && r.ipsi, dfarbor).path)
        dfxx = filter(r -> r.path in selnrns && !r.qroot && r.typ == "ITi", df)
        selnrns = dfxx = sort(dfxx, :axonlen)[end-5:end, :].path
        dfxx = filter(r -> r.arborpath in tgtarbs && r.path in selnrns && r.ipsi, dfarbor)

        p2c = Dict()
        per_neuron = (r) -> begin
            get!(() -> to_colormap(:Set1_8)[length(p2c)+1], p2c, r.path)
        end
        per_arbor = (pn, rr, cond) -> begin
            #cond && return pn
            rr !== nothing && rr.arborpath in tgtarbs && return pn
            :transparent
        end
        per_node = (pn, pa, n) -> pa => 0.7
        draw_subtrees(dfxx, mix1; autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=NmtDat)
    end
    draw_mesh_overlay([obj], ax1)
end

function gen_canvas_from_flatmap_points(points)
    canvas = zeros(Float32, 1024, 1024, length(points) + 1)
    canvas .= 0
    canvas[:, :, end] .= 0.03
    @showprogress for (i, pts) in enumerate(points)
        for pt in pts
            u, v = clamp.(round.(Int, (view(pt, 1:2) .+ 1) ./ 2 * 1024), 1, 1024)
            canvas[u, v, i] += 9
        end
    end
    @showprogress for i in axes(canvas, 3)
        canvas[:, :, i] .= Images.imfilter(canvas[:, :, i], Images.Kernel.gaussian((10, 10)))
    end
    #return canvas
    ss = maximum(canvas, dims=3)[:, :, 1]
    canvas2 = similar(canvas)
    @showprogress for i in axes(canvas2, 3)
        canvas2[:, :, i] .= 1.1 .* canvas[:, :, i] .- ss#, Images.Kernel.gaussian((10, 10)))
    end
    global xxxcanvas2 = canvas2
    canvas2
end

function plot_subreg_flat_conn(grid, env; subregs, dfarbor, trunks, obj, objflat, objphy, highlights, _...)
    pathways = vote_connections(subregs, dfarbor; thr=3)

    ax1 = Makie.Axis(grid[1, 1], width=pt(130))

    plot_flatmap_base!(ax1, highlights, objflat, objphy)
    draw_pathway_flat_alt!(ax1, subregs, filter(r -> r.to > 0, pathways))
    draw_intra_pfc_modular!(ax1; env...)
    xlims!(ax1, -0.85, 0.45)
    ylims!(ax1, -0.75, 0.55)
end

function sort_matrix_cols(mat; dist=:cor)
    dm = if dist === :cor
        1 .- cor(mat)
    else
        @assert false
    end
    h = Clustering.hclust(dm, linkage=:ward, branchorder=:barjoseph)
    h.order
end
sort_matrix_rows(mat; dist=:cor) = sort_matrix_cols(transpose(mat); dist)

function plot_subreg_conn_hm(grid, env; subregs, dfarbor, trunks, obj, objflat, objphy, highlights, _...)
    pathways = vote_connections(subregs, dfarbor; thr=1)
    rows = [r.subreg for r in eachrow(filter(r -> length(r.neurons) >= 10, subregs))]
    rowsw = [length(r.neurons) for r in eachrow(filter(r -> length(r.neurons) >= 10, subregs))]
    cols = [r.subreg for r in eachrow(subregs)]
    mat = zeros(length(cols), length(rows))
    for r in pathways
        ff = findfirst(==(r.from), rows)
        tt = findfirst(==(r.to), cols)
        (ff === nothing || tt === nothing) && continue
        mat[tt, ff] += length(r.neurons) / rowsw[ff]
    end

    ax1 = Makie.Axis(grid[1, 1]) #= , width=pt(60), height=pt(120) =#
    ax1x = Axis(grid[1, 1])
    linkaxes!(ax1, ax1x)
    conn = mat[axes(rows, 1), axes(rows, 1)]
    conn = (1 .- conn) .* (1 .- conn')
    #conn=
    h1 = Clustering.hclust(conn, linkage=:ward, branchorder=:barjoseph)
    o1 = h1.order
    g1 = maximum(findall(Clustering.cutree(h1, k=2)[h1.order] .== 1))
    plt = vlines!(ax1, [g1 + 0.5], color=:gray, linestyle=:dash)
    translate!(plt, 0, 0, 1)
    plt = hlines!(ax1, [g1 + 0.5, length(o1) + 0.5], color=:gray, linestyle=:dash)
    translate!(plt, 0, 0, 1)

    cols2 = setdiff(cols, rows)
    o22 = sort_matrix_rows(mat[something.(indexin(cols2, cols)), :], dist=:cor)
    o2 = something.(indexin(vcat(rows[o1], cols2[o22]), cols))
    heatmap!(ax1, mat[o2, o1]')
    ax1.xlabel = "Source subdomains"
    ax1.xticklabelrotation = -π / 2
    hidedecorations!(ax1, ticklabels=false, label=false)
    hidedecorations!(ax1x, ticklabels=false, label=false)
    #hidexdecorations!(ax1x)
    ax1.yticks = (axes(cols, 1)[1:2:end], [(d = cols[i]; "$d") for i in o2[1:2:end]])
    ax1x.yticks = (axes(cols, 1)[2:2:end], [(d = cols[i]; "$d") for i in o2[2:2:end]])
    ax1.xticks = (axes(rows, 1)[1:2:end], [(d = rows[i]; "$d") for i in o1][1:2:end])
    ax1x.xticks = (axes(rows, 1)[2:2:end], [(d = rows[i]; "$d") for i in o1][2:2:end])
    ax1x.yaxisposition = :right
    ax1x.xaxisposition = :top
    ax1x.xticklabelrotation = π / 2
    ax1.ylabel = "Target subdomains"
    tightlimits!(ax1)
    tightlimits!(ax1x)
    hidespines!(ax1)
    hidespines!(ax1x)
    @warn "$(rows[o1])"
end

function plot_subreg_flat(grid, env; subregs, dfarbor, trunks, obj, objflat, objphy, highlights, _...)
    canvas = zz1data.cached_result(:canvas3893, subregs.uvvals) do uvvals
        gen_canvas_from_flatmap_points(uvvals)
    end

    # adjs = vec([(i, j) => sum(canvas[:, :, i] .+ 0.01 .> 0 .* canvas[:, :, j] .+ 0.01 .> 0) for i in axes(subregs, 1), j in axes(subregs, 1)])
    # adjs = first.(filter(v -> v[2] > 10, sort(vec(adjs), by=last)))
    # @warn "$(adjs)"

    ax1 = Makie.Axis(grid[1, 1], width=pt(130))

    plot_flatmap_base!(ax1, highlights, objflat, objphy)
    xyrng = range(-1.0f0, 1.0f0, size(canvas, 1))
    for r in eachrow(subregs)
        contourf!(ax1, xyrng, xyrng, canvas[:, :, rownumber(r)], levels=[0, 100], colormap=[(r.color, 0.3)])
        #scatter!(ax1, Makie.Point2f.(r.uvvals), markersize=0.5, color=subregs[rownumber(r), :color], inspector_label=(p, i, x) -> "$(r.subreg)")
        if length(r.neurons) >= 10
            contour!(ax1, xyrng, xyrng, canvas[:, :, rownumber(r)], levels=[0], color=:red, linewidth=0.5)
        end
    end
    text!(ax1, subregs.centuv; text=["$t" for t in subregs.subreg], color=subregs[:, :color], align=(:center, :center))
    xlims!(ax1, -0.85, 0.45)
    ylims!(ax1, -0.75, 0.55)
end

function plot_intra_network(grid, env; subregs, dfarbor, _...)
    pathways = vote_connections(filter(r -> length(r.neurons) >= 10 || true, subregs), dfarbor; thr=3)
    mktempdir() do path
        dotf = joinpath(path, "test.dot")
        pdff = joinpath(path, "test.pdf")
        @warn "$dotf"
        open(dotf, "w") do io
            done = Dict()
            node(subreg) = begin
                r = filter(r -> r.subreg == subreg, subregs)[1, :]
                x, y = (r.centuv .+ 1) .* 1000
                lcolxx = if r.dlpfc
                    :red, :pink
                elseif r.vlpfc
                    :blue, :lightblue
                else
                    :black, :gray
                end
                lcol = first(lcolxx)
                fcol = length(r.neurons) >= 10 ? last(lcolxx) : :white
                get!(() -> (println(io, "n$subreg[width=0.15,height=0.15,fixedsize=true,penwidth=\"0.5\", label=\"$subreg\",shape=\"circle\",style=\"filled\",color=\"$lcol\",fillcolor=\"$fcol\",pos=\"$x,$y\",fontsize=\"6\",fontname=\"Arial\"];"); 0), done, subreg)
            end
            edge(a, b, w) = begin
                ra = filter(r -> r.subreg == a, subregs)[1, :]
                rb = filter(r -> r.subreg == b, subregs)[1, :]
                col = if ra.dlpfc && rb.vlpfc
                    :red
                elseif ra.vlpfc && rb.dlpfc
                    :blue
                else
                    :black
                end
                l = log(norm(ra.centuv - rb.centuv) * 100)
                println(io, "n$a -> n$b [len=\"$l\", weight=$w,penwidth=$(log10(w)),arrowsize=0.3,color=\"$col\"];")
            end
            println(io, "digraph pfc {")
            #println(io, "size=\"2,2\";")
            println(io, "overlap=\"false\";")
            println(io, "splines=\"true\";")
            println(io, "fontsize=\"6\";")
            #println(io, "K=\"0.3\";")
            #println(io, "model=\"circuit\";")
            for r in pathways
                r.from == r.to && continue
                @assert r.from > 0
                r.to < 0 && continue
                node(r.from)
                node(r.to)
                @warn "to $(r.to)"
                edge(r.from, r.to, length(r.neurons))
            end
            println(io, "}")
        end
        cp(dotf, "/tmp/xxx.dot", force=true)
        run(`fdp -Tpdf -Kneato -LC0.9 -o$pdff $dotf`)
        run(`fdp -n2 -Txdot -Kneato -o/tmp/xxx.xdot $dotf`)
        cp(pdff, "/tmp/xxx.pdf", force=true)
        LTeX(grid[1, 1], PDFDocument(read(pdff, String)), tellwidth=false, tellheight=false, scale=1.15)
        grid.height = 323
    end
end

import Clustering

function draw_int_dist_with_clamp!(ax, vals, ub)
    a, b = extrema(vals)
    x2cnt = Dict{Int,Int}()
    for x in vals
        k = x > ub ? ub + 1 : x
        x2cnt[k] = get(x2cnt, k, 0) + 1
    end
    x2cnt = sort(collect(x2cnt))
    barplot!(ax, first.(x2cnt)[1:end-1], last.(x2cnt)[1:end-1], color=:black)
    barplot!(ax, first.(x2cnt)[end:end], last.(x2cnt)[end:end], label="$(ub+1)-$b")
    ylims!(ax, 0, nothing)
    ax.xticks[] = (a:ub, ["$v" for v in a:ub])
    hidexdecorations!(ax, ticks=false, ticklabels=false, label=false)
    axislegend(ax, ax, position=:rt)
end

function write_stat(key::Symbol, value)
    open("output/stats.tex", "a") do io
        num = _fig_cur_fignum[]
        lbl = uppercase(_fig_cur_figure[end].first[])
        println(io, "\\def\\Fig$num$lbl@$key{$value}")
    end
end

function plot_bundle_num_stat(grid, env; trunks, pathways, obj, objflat, objphy, highlights, _...)
    ax = Axis(grid[1, 1])
    ax2 = Axis(grid[2, 1][1, 1])
    ax3 = LScene(grid[2, 1][1, 2], width=pt(50), show_axis=false)

    write_stat(:MaximumNeuronsPerBundle, maximum(pathways.w))
    draw_int_dist_with_clamp!(ax, [r.w for r in eachrow(pathways)], 15)
    ax.xlabel[] = "No. of neurons sharing bundle"
    ax.ylabel[] = "Count"

    #trunks=filter(r->r.typ=="ITi", trunks)
    # CT->max 5
    # PT->max 15
    cls2w = Dict(zip(pathways.trunkcls, pathways.w))
    path2pws = Dict([p => Set{Int}() for p in trunks.path])
    for r in eachrow(trunks)
        pws = path2pws[r.path]
        if get(cls2w, r.trunkcls, 2) >= 5
            push!(pws, r.trunkcls)
        end
    end
    for (a, b) in path2pws
        if length(b) > 10
            @warn "$a $b"
        end
    end
    draw_int_dist_with_clamp!(ax2, length.(last.(collect(path2pws))), 8)
    ax2.xlabel[] = "No. of bundles per neuron"
    ax2.ylabel[] = "Count"

    cam = (fov=9.0, cent=(23871.92, 56211.66, 39332.96), deye=(-0.09669584324233581, -0.12902331047751098, 0.9869158521642448) .* 103469.28745350186, up=(0.59, 0.79, 0.16))
    setup_camera(ax3 => cam)
    with_mixed_scene(ax3, GLMakie) do mix1
        cls2col = Dict()
        trunkcolor(rr) = begin
            rr === nothing && return :black
            get(cls2w, rr.trunkcls, 2) >= 5 || return :black
            get!(() -> rand(to_colormap(:tab10)), cls2col, rr.trunkcls)
        end
        draw_subtrees(filter(r -> r.path == "monkeytemp/split-sample232717-ch2/root1056976249-root236-good.swc.gz", trunks), mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor)...)
    end
end

function draw_trunklen_dist!(ax, df, sel)
    df = cached_result(:trunklen, df) do df
        df = deepcopy(df)
        insertcols!(df, :trunklen => 0.0)
        for r in eachrow(df)
            r.trunklen = mapreduce(norm ∘ -, +, @view(r.trunk[1:end-1]), @view(r.trunk[2:end]))
        end
        df
    end
    #@warn df.trunklen
    #boxplot_and_compare!(ax, df; rowx=r -> !(r.trunkcls in sel), xname=x -> !x ? "Intrinsic" : "Other", rowy=r -> r.trunklen / 1000, statif=(a, b) -> a[1] != b[1])
    vals1 = filter(r -> r.arborpath in sel, df).trunklen
    vals2 = filter(r -> !(r.arborpath in sel), df).trunklen
    ecdfplot!(ax, vals1 ./ 1000, color=:red, label="Intrinsic")
    ecdfplot!(ax, vals2 ./ 1000, color=:black, label="Intrinsic")
    ax.xscale[] = log10
    #ax.xticks[]=
    #boxplot_and_compare!(ax, df; , xname=x -> !x ?  : "Other", rowy=r -> r.trunklen / 1000, statif=(a, b) -> a[1] != b[1])
end

function plot_bundle_demo_intrinsic2(grid, env; df, trunks, obj, _...)
    dfx = filter(select_intrinsic, trunks)
    dfxarborpaths = Set(dfx.arborpath)
    trunks = leftjoin(trunks, df, on=:path) |> dropmissing!

    ax1 = LScene(grid[1, 1], width=pt(140), height=pt(140), show_axis=false)
    ax2 = LScene(grid[1, 2], width=pt(170), height=pt(140), show_axis=false)
    colgap!(grid, 1, 2)

    cam1 = (fov=9.0, cent=(24051.52, 57347.18, 36702.89), deye=(-0.03770469172850035, 0.07320041519144625, 0.9966042622011299) .* 167114.21606020417, up=(-0.43, 0.9, -0.08))
    cam2 = (fov=9.0, cent=(23549.14, 58781.51, 40040.42), deye=(0.4870186940958119, -0.7734462319343565, 0.4057015133170636) .* 125555.38574042972, up=(-0.26, 0.31, 0.91))
    setup_camera(ax1 => cam1)
    setup_camera(ax2 => cam2)
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            c2c = Dict()
            per_neuron(r) = nothing
            per_arbor(pn, rr, cond) = begin
                rr === nothing && return :transparent => 0.0
                rr.arborpath in dfxarborpaths || return :transparent => 0.0
                cc = get!(c2c, rr.trunkcls) do
                    #rand(to_colormap(:Set1_8))
                    rand(Makie.wong_colors())
                end
                cc => (cond ? pt(0.5) : pt(0.3))
            end
            per_node(pn, pa, n) = pa
            draw_subtrees(filter(r -> r.typ == "ITi" && r.arborpath in dfxarborpaths && r.inj == "D17-2bL", trunks), mix1, mix2; autotrunk=false, mirrorby=NmtDat, per_neuron, per_arbor, per_node)
        end
    end
    draw_mesh_overlay([obj => :gray], ax1, ax2)
end

function plot_bundle_intrinsic_length(grid, env; df, trunks, obj, _...)
    dfx = filter(select_intrinsic, trunks)
    dfxarborpaths = Set(dfx.arborpath)

    ax3 = Makie.Axis(grid[1, 1])
    draw_trunklen_dist!(ax3, trunks, dfxarborpaths)
end

function plot_bundle_intrinsic_stats(grid, env; df, trunks, obj, _...)
    dfx = filter(select_intrinsic, trunks)
    arbint = Set(dfx.arborpath)
    arbass = setdiff(Set(trunks.arborpath), arbint)
    nrntyp = Dict([p => 0 for p in unique(trunks.path)])
    for r in eachrow(trunks)
        if r.arborpath in arbint
            #@warn "xxx"
            nrntyp[r.path] = nrntyp[r.path] + 1
        end
        if r.arborpath in arbass
            #@warn "yyy"
            nrntyp[r.path] = nrntyp[r.path] + 1000
        end
    end
    global xxxcnts = deepcopy(nrntyp)
    cnts = Dict()
    vals = map(eachrow(df)) do r
        cnt = get(nrntyp, r.path, 1000)
        k = (r.typ, (cnt ÷ 1000 > 0) * 2 + (cnt % 1000 > 0))
        cnts[k] = get(cnts, k, 0) + 1
    end
    cnts = collect(cnts)

    ax1 = Axis(grid[1, 1])
    dvals = last.(first.(cnts))
    xvals = first.(first.(cnts))
    xx = sort(unique(xvals))
    for d in sort(unique(dvals))
        barplot!(ax1, something.(indexin(xvals, xx))[dvals.==d], dodge=something.(indexin(dvals, sort(unique(dvals))))[dvals.==d], last.(cnts)[dvals.==d], n_dodge=4)
    end
    ax1.xticks[] = (axes(xx, 1), xx)
end

function test_depth_dist_ks1(hh1, hh2)
    dist_to_samp(hh, siz=10000) = begin
        xx, yy = calc_fvhist(hh)
        res = Float32[]
        foreach(xx, yy) do x, y
            for _ in 1:round(Int, y * siz)
                push!(res, x)
            end
        end
        shuffle!(res)[1:siz÷2]
    end
    HypothesisTests.ApproximateTwoSampleKSTest(dist_to_samp(hh1), dist_to_samp(hh2))
end
function test_depth_dist_mwu1(hh1, hh2; avg=mean)
    calccenter(x) = begin
        sum(x) < 10 && return missing
        avg((axes(x, 1) .- 1) ./ (length(x) - 1), StatsBase.Weights(x))
    end
    calccenterv(hh) = collect(skipmissing(calccenter.(hh)))
    hh1, hh2 = calccenterv(hh1), calccenterv(hh2)
    global xxxdepthdata = deepcopy((hh1, hh2))
    HypothesisTests.MannWhitneyUTest(hh1, hh2)
end
function test_depth_dist(name::Symbol, hh1, hh2)
    t = test_depth_dist_ks1(hh1, hh2)
    write_stat(Symbol(name, :KS, :Pval), compact_str(HypothesisTests.pvalue(t)))
    write_stat(Symbol(name, :KS, :Method), :ApproximateTwoSampleKSTest)
    t = test_depth_dist_mwu1(hh1, hh2; avg=mean)
    write_stat(Symbol(name, :MWUofMean, :Pval), compact_str(HypothesisTests.pvalue(t)))
    write_stat(Symbol(name, :MWUofMean, :Method), :MannWhitneyUTestTwoSided)
    write_stat(Symbol(name, :MWUofMean, :detail), "$(t.nx)/$(t.ny)/$(t.median)")
    t = test_depth_dist_mwu1(hh1, hh2; avg=median)
    write_stat(Symbol(name, :MWUofMedian, :Pval), compact_str(HypothesisTests.pvalue(t)))
    write_stat(Symbol(name, :MWUofMedian, :Method), :MannWhitneyUTestTwoSided)
    write_stat(Symbol(name, :MWUofMedian, :detail), "$(t.nx)/$(t.ny)/$(t.median)")
end

function plot_bundle_intrinsic_arbor_cmp(grid, env; df, dfarbor, trunks, obj, _...)
    dfx = filter(select_intrinsic, trunks)
    arbint = Set(dfx.arborpath)
    dfx1 = filter(r -> r.arborpath in arbint && r.ipsi && contains("Frontal")(r.arborhier), dfarbor)
    dfx2 = filter(r -> !(r.arborpath in arbint) && r.ipsi && contains("Frontal")(r.arborhier), dfarbor)

    ax1 = Axis(grid[1, 1])
    ax1.height[] = pt(50)
    ax1.width[] = pt(80)
    for (dfxx, l) in [(dfx1, "Intrinsic"), (dfx2, "Associate")]
        @warn "$(size(dfxx, 1))"
        xx, yy = calc_fvhist(dfxx.fv_whist)
        lines!(ax1, xx, yy, label="$l")
    end
    xaxis_as_layer(ax1, 0, 1)
    axislegend(ax1, ax1) |> fix_line_legs
end

function plot_intra_pfc_diagram(grid, env; _...)
    ax = Axis(grid[1, 1])
    hidedecorations!(ax)
    hidespines!(ax)
    ax.width[] = pt(300)
    ax.height[] = pt(300)
    teximg!(ax, compile_tikz("figs/intra-pfc-diagram.tex"))
end



function asdfasdf()


    # nrntyp = filter(v -> last(v) != 0, collect(nrntyp))
    # cnts = [count(last.(nrntyp) .== k) for k in 1:3]
    # barplot!(ax5, 1:3, cnts)
    # @warn "$nrntyp"
    # ax5.xticks[] = (1:3, ["Intrinsic only", "Associate only", "Both"])
    # write_stat(:IntrinsicOnlyNeurons, count(last.(nrntyp) .== 1))
    # write_stat(:AssociateOnlyNeurons, count(last.(nrntyp) .== 2))
    # write_stat(:AssociateAndIntrinsicNeurons, count(last.(nrntyp) .== 3))
end

function plot_bundle_demo_contra(grid, env; trunks, pathways, obj, _...)
    sel1 = [664]
    sel2 = [592, 540, 590]
    sel3 = [672, 629]

    nrns = intersect([Set(filter(r -> r.trunkcls in s, trunks).path) for s in (sel1, sel2, sel3)]...)

    ax1 = LScene(grid[1, 1], width=pt(100), height=pt(100), show_axis=false)
    # ax2 = LScene(grid[1, 3], width=160, height=160, show_axis=false)
    #ax3 = Makie.Axis(grid[2, 1][1, 1])

    cam = (fov=9.0, cent=(34487.06, 50749.2, 28156.62), deye=(-0.051982864880287534, -0.8774569781538932, 0.47683019330562076) .* 332050.7640306208, up=(0.03, 0.48, 0.88))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        cls2w = Dict(zip(pathways.trunkcls, pathways.w))
        cls2col = Dict()
        trunkcolor(rr) = begin
            rr === nothing && return :black
            get(cls2w, rr.trunkcls, 2) >= 5 || return :black
            rr.trunkcls in sel1 && return :red
            rr.trunkcls in sel2 && return :blue
            rr.trunkcls in sel3 && return :green
            :black
        end
        xxx = Set(filter(r -> r.trunkcls in union(sel1, sel2, sel3), trunks).path)
        draw_subtrees(filter(r -> r.path in xxx, trunks), mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor)..., inspect=:nmt)
    end
    draw_mesh_overlay([obj], ax1)

    ax5 = Axis(grid[1, 2])
    pwccc = Set(filter(r -> r.w >= 5 && r.ic == :C, pathways).trunkcls)
    @warn " pwint $pwccc"
    pwiii = setdiff(Set(filter(r -> r.w >= 5, pathways).trunkcls), pwccc)
    @warn " pwass $pwiii"
    nrntyp = Dict([p => 0 for p in unique(trunks.path)])
    for r in eachrow(trunks)
        if r.trunkcls in pwccc
            @warn "xxx"
            nrntyp[r.path] = nrntyp[r.path] | 1
        end
        if r.trunkcls in pwiii
            @warn "yyy"
            nrntyp[r.path] = nrntyp[r.path] | 2
        end
    end
    nrntyp = filter(v -> last(v) != 0, collect(nrntyp))
    cnts = [count(last.(nrntyp) .== k) for k in 1:3]
    barplot!(ax5, 1:3, cnts)
    @warn "$nrntyp"
    ax5.xticks[] = (1:3, ["Contra only", "Ipsi only", "Bilateral"])
    write_stat(:ContraOnlyNeurons, cnts[1])
    write_stat(:IpsiOnlyNeurons, cnts[2])
    write_stat(:BilateralNeurons, cnts[3])
end

function plot_bundle_demo_acggx(grid, env; trunks, pathways, df, dfarbor, _...)
    dfx = filter(r -> r.w >= 5 && contains("ACgG")(r.tgt_hier), pathways)
    sel = filter(r -> r.ic === :I, dfx).trunkcls
    sel2 = filter(r -> r.ic === :C, dfx).trunkcls
    sel = union(sel, sel2)

    ax1 = LScene(grid[1, 1], width=pt(100), height=pt(100), show_axis=false)
    # ax = Makie.Axis(grid[1, 1][1, 1], width=100, height=100)
    ax2 = Makie.Axis(grid[1, 2], width=pt(50), height=pt(50))

    # plot_flatmap_base!(ax, highlights, objflat, objphy, label=false)
    # draw_pathways_flatmap!(ax, dfx)

    obj = load("interm/mesh-nmt-lat_PFC-l.obj")
    cam = (fov=9.0, cent=(32927.81, 67194.83, 34036.5), deye=(-0.11065735699909447, -0.9537823912608822, 0.2793816376619057) .* 124766.03792473536, up=(0.03, 0.28, 0.96))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        cls2w = Dict(zip(pathways.trunkcls, pathways.w))
        cls2col = Dict()
        trunkcolor(rr) = begin
            rr === nothing && return :black
            get(cls2w, rr.trunkcls, 2) >= 5 || return :black
            rr.trunkcls in sel2 && return :red
            return :green
        end
        draw_subtrees(filter(r -> r.trunkcls in sel, trunks), mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor)...)
    end
    #draw_contours(["interm/mesh-nmt-$ba-l.obj" for ba in [:ACgG]], ax1)

    acggi = filter(r -> r.ipsi && contains("ACgG")(r.arborhier), dfarbor).path |> Set
    acggc = filter(r -> !r.ipsi && contains("ACgG")(r.arborhier), dfarbor).path |> Set
    rowx(r) = (; acggi=r.path in acggi, acggc=r.path in acggc)
    xname(v) = begin
        t = ""
        if v.acggi
            t *= " ACgG(I)"
        end
        if v.acggc
            t *= " ACgG(C)"
        end
        t = lstrip(t)
        isempty(t) && return "-"
        t
    end
    barplot_with_chisq!(ax2, filter(r -> r.typ == "ITc", df); rowx, xname, skipfirst=true)
    rotatexlabel!(ax2)
end

function plot_bundle_demo_ofc(grid, env; trunks, pathways, df, dfarbor, _...)
    dfx = filter(r -> r.w >= 5 && contains("OFC")(r.tgt_hier) && !select_intrinsic(r), pathways)
    sel1 = filter(r -> r.ic == :I, dfx).trunkcls
    @warn "sel1 $sel1"
    sel1 = [126, 595, 772, 776, 964, 982]
    sel1 = [595]
    # 772 prev
    # 964->xxx
    # 982/595
    # 126
    sel2 = filter(r -> r.ic == :C, dfx).trunkcls
    sel2 = [982]
    #@warn "sel2 $sel2"

    ax1 = LScene(grid[1, 1], width=pt(100), height=pt(100), show_axis=false)

    # ax = Makie.Axis(grid[1, 1], width=100, height=100)
    # plot_flatmap_base!(ax, highlights, objflat, objphy, label=false)
    # draw_pathways_flatmap!(ax, dfx)
    # ax.title[] = "to OFC"

    #obj = load("interm/mesh-nmt-lat_PFC-l.obj")
    cam = (fov=45.0, cent=(20581.25, 60086.03, 19007.01), deye=(-0.2965623931614364, 0.04839748259977958, 0.9537863653042885) .* 31051.408446745212, up=(0.88, -0.38, 0.29))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        cls2w = Dict(zip(pathways.trunkcls, pathways.w))
        cls2col = Dict()
        trunkcolor(rr) = begin
            rr === nothing && return :black
            get(cls2w, rr.trunkcls, 2) >= 5 || return :black
            rr.trunkcls in sel1 && return :red
            rr.trunkcls in sel2 && return :green
            #rr.trunkcls in sel3 && return :green
            :black
        end
        xxx = Set(filter(r -> r.trunkcls in union(sel1, sel2), trunks).path)
        draw_subtrees(filter(r -> r.path in xxx, trunks), mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor)..., inspect=:nmt)
    end
end

function plot_trunks(grid, env, sel; obj, trunks, pathways, _...)
    ax1 = LScene(grid[1, 1], width=pt(300), height=pt(300), show_axis=false)

    setup_camera(ax1 => :h)
    with_mixed_scene(ax1, GLMakie) do mix1
        cls2w = Dict(zip(pathways.trunkcls, pathways.w))
        cls2col = Dict()
        trunkcolor(rr) = begin
            rr === nothing && return :black
            get(cls2w, rr.trunkcls, 2) >= 5 || return :black
            if rr.trunkcls in sel
                return get!(() -> rand(to_colormap(:tab10)), cls2col, rr.trunkcls)
            end
            :black
        end
        draw_subtrees(filter(r -> r.trunkcls in sel, trunks), mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor)..., inspect=:nmt)
    end
    draw_mesh_overlay([obj], ax1)
end

function draw_pathway_graph!(ax, pathways)
    pathways = filter(r -> r.w >= 5 && r.ic == :I && contains("Frontal")(r.tgt_hier) && !isempty(r.src_hier), pathways)
    hiers = sort(unique(filter(!isempty, vcat(pathways.src_hier, pathways.tgt_hier))))
    h2i = Dict(reverse.(enumerate(hiers)))
    ww = zeros(length(hiers), length(hiers))
    for r in eachrow(pathways)
        r.src_hier in hiers || continue
        r.tgt_hier in hiers || continue
        r.src_hier != r.tgt_hier || continue
        ww[h2i[r.src_hier], h2i[r.tgt_hier]] += r.w
    end
    g = Graphs.SimpleDiGraph(ww)
    #Main.xxxg = deepcopy(g)
    abbr(h) = replace(h, r"^.*\|" => "")
    # for (i,h) in enumerate(hiers)
    #     MetaGraphs.set_prop!(g, i, :name, abbr(h))
    # end
    GraphMakie.graphplot!(ax, g, nlabels=abbr.(hiers), layout=NetworkLayout.Spring(), arrow_show=true, nlabels_align=(:right, :bottom))
end

function plot_pt_demo(grid, env; dfxpt, dfxct, obj, objflat, objphy, highlights, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(vcat(dfxct, dfxpt))

    ax1 = LScene(grid[1, 1][1, 1], width=pt(120), height=pt(82), show_axis=false)
    ax1ct = LScene(grid[1, 1][2, 1], width=pt(120), height=pt(82), show_axis=false)
    rowgap!(content(grid[1, 1]), 1, 0)
    ax2 = LScene(grid[2, 1][1, 1], width=pt(60), height=pt(60), show_axis=false)
    ax3 = LScene(grid[2, 1][1, 2], width=pt(60), height=pt(60), show_axis=false)
    rowgap!(grid, 1, 0)
    colgap!(content(grid[2, 1]), 1, 0)

    setup_camera(ax1 => :s)
    setup_camera(ax2 => :zoominhpt)
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            draw_neurons(dfxpt, mix1 => :hr, mix2 => :s; rowlabel, rowcolor, mirrorby=NmtDat)
            axislegend(ax1, mix1.scene2, unique=true) |> fix_line_legs
        end
    end
    draw_mesh_overlay([obj], ax1, ax2)
    draw_contours(["interm/mesh-nmt-$ba-lr.obj" => c for (ba, c) in [:Thal => :red, :Mid => :blue, :Pd => :orange, :Pons => :magenta]], ax1, ax2)
    #draw_contours(["interm/mesh-nmt-$ba-lr.obj" for ba in [:Thal, :Mid]], ax1, ax2)

    #plot_pt_demo_zoomin(GridLayout(grid[2, 1][1, 2]), env; env...)
    colgap!(content(grid[2, 1]), 1, 0)

    plot_ct_demo(ax1ct, ax3; env...)
end

function plot_pt_heatmap(grid, env; dfxpt, dfxct, obj, objflat, objphy, highlights, _...)
    ax0 = Makie.Axis(grid[1, 1])
    ax1 = Makie.Axis(grid[2, 1])
    ax2 = Makie.Axis(grid[3, 1])
    ax1x = Axis(grid[2, 2])
    ax2x = Axis(grid[3, 2])
    colgap!(grid, 1, -8)
    Makie.rowgap!(grid, 1, 2)
    Makie.rowgap!(grid, 2, 2)
    dfxpt = vcat(dfxct, dfxpt,)

    cols = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-pt.jl")
    # dm=1 .-StatsBase.corspearman(Matrix(dfxpt[:, cols])')
    # o1=Clustering.hclust(dm, linkage=:ward, branchorder=:barjoseph).order
    # oxx=zeros(Int, size(dfxpt, 1))
    # for cls in unique(dfxpt.cls)
    #     a,b=extrema(findall(==(cls), dfxpt.cls))
    #     oxx[a:b].=o1.+(a-1)
    # end
    #dfxpt=dfxpt[oxx, :]
    #rename!(dfxpt, :cls=>:prevcls, :subcls=>:cls)
    plot_subtyp_spans!(ax0, dfxpt; linkto=ax1, gridto=[ax1, ax2], rowgrp=r -> r.cls, grplabel=(i, g) -> "#$g")
    draw_neuron_proj_hm!(ax1, ax2, dfxpt, cols; ax1x, ax2x, skipic=true, longic=false)
    ax1.width[] = nothing
end

function plot_pt_flatmap(grid, env; dfxpt, dfxct, obj, objflat, objphy, highlights, _...)
    rowlabel(r) = string("#", r.cls)
    dfxpt = vcat(dfxpt, dfxct)
    rowcolor = cls_temp_color(dfxpt)
    ax1 = Makie.Axis(grid[1, 1], height=85, width=100)
    plot_flatmap_base!(ax1, highlights, objflat, objphy)
    plot_flatmap_soma!(ax1, dfxpt; rowcolor)
    limits!(ax1, -0.40, 0.40, -0.40, 0.30)
end

function plot_pt_demo_zoomin(grid, env; dfxpt, obj, objflat, objphy, highlights, _...)
    dfxx = pick_neurons_with_limit(dfxpt, r -> r.cls, 5)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfxx)

    ax1 = LScene(grid[1, 1], height=pt(60), width=pt(60), show_axis=false)
    setup_camera(ax1 => :ptthalpons)
    with_mixed_scene(ax1, GLMakie) do mix1
        draw_neurons(dfxx, mix1; rowcolor, rowlabel, mirrorby=NmtDat, inspect=:nmt)
    end
    #draw_mesh_overlay(["interm/mesh-nmt-$ba-lr.obj" => :red for ba in [:Thal, :Mid]], ax1,)
    #draw_contours(["interm/mesh-nmt-$ba-lr.obj" => c for (ba, c) in [(:Thal, :red), (:Mid, :blue)]], ax1,)
    draw_mesh_overlay(["interm/mesh-nmt-$ba-lr.obj" => c for (ba, c) in [:Thal => :red, :Mid => :blue]], ax1)
end


function select_one_arbor_per_neuron!(dfx; orderby=nothing)
    if orderby === nothing
        orderby = r -> rand()
    end
    Random.seed!(0)
    nrn2arbs = Dict()
    for r in eachrow(dfx)
        aa = get!(() -> [], nrn2arbs, r.path)
        push!(aa, r.arborpath => orderby(r))
    end
    arbs = Set(map(p -> first(sort!(p.second, by=last)).first, collect(nrn2arbs)))
    filter!(r -> r.arborpath in arbs, dfx)
end


function plot_pt_topo_term(grid, env, finegrain=false; trunks, dfxpt, dfxct, topo_pt_sc, obj, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(vcat(dfxct, dfxpt))
    ax2 = Axis(grid[1, 1][1, 1], height=pt(60))
    content(grid[1, 1]).alignmode = Outside()

    let dfxx = leftjoin(dfxpt, topo_pt_sc, on=:path) |> dropmissing!
        finegrain && filter!(r -> zz1data.matchhier("dlPFC")(r.somahier) && !r.qroot, dfxx)
        xpars, ypars = zz1data.find_best_coor_axis(dfxx.srcpos, dfxx.tgtpos, aligned_x=-1, aligned_y=-1)
        #.437/.391/.429/.391/.255
        #     # finegrain? .594/.507/.547/.489
        @warn "$xpars $ypars"
        insertcols!(dfxx, :xx => dot.(dfxx.srcpos, Ref(xpars)), :yy => dot.(dfxx.tgtpos, Ref(ypars)))
        linear_reg_scatter!(ax2, dfxx; rowx=r -> r.xx / 1000, rowy=r -> r.yy / 1000, rowcolor)
        ax2.xlabel[] = "Soma position"
        ax2.ylabel[] = "Arbor center"
        setup_axis_by_orient!(ax2, :x, dfxx.xx ./ 1000, xpars)
        setup_axis_by_orient!(ax2, :y, dfxx.yy ./ 1000, ypars)
    end
    ax2
end


function plot_ct_md_arbors(grid, env; dfxct, dfarbor, _...)
    #rowcolor = cls_temp_color(dfxct)
    rowcolor(r) = r.soma_pos[1]
    dfx = leftjoin(dfarbor, dfxct, on=:path) |> dropmissing!
    filter!(r -> zz1data.select_neurons(:ct_strict)(r) && contains("MD")(r.arborhier), dfx)
    colorrange = extrema(rowcolor.(eachrow(dfxct)))

    ax1 = LScene(grid[1, 1], width=pt(120), height=pt(120), show_axis=false)
    # ax2 = LScene(grid[1, 2], width=pt(130), height=pt(130), show_axis=false)
    # colgap!(grid, 1, 2)
    cam = (fov=9.0, cent=(35473.92, 37561.63, 26345.83), deye=(-0.9962770235009725, -0.08543700653055368, 0.011515657130222985) .* 62181.91884602918, up=(0.01, 0.0, 1.0))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        trunkcolor(rr) = begin
            rr === nothing && return NaN
            rowcolor(rr)
        end
        arborcolor(rr) = begin
            rr === nothing && return NaN
            rowcolor(rr)
        end
        draw_subtrees(dfx, mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)..., colorrange, colormap=[:red, :blue])
    end
    draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:MD => :white]], ax1)
    draw_scale_bar!(ax1, 1000, load("interm/mesh-nmt-MD-lr.obj"))
    Colorbar(grid[1, 1, Bottom()], tellheight=false, valign=:bottom; colorrange, colormap=[:red, :blue], vertical=false, width=pt(60), tellwidth=false, ticklabelsvisible=false)

    # cam = (fov=6.0, cent=(26925.1, 38570.66, 20581.41), deye=(-1.0, -0.0, -0.0) .* 270233.79, up=(0.0, 0.0, 1.0))
    # setup_camera(ax2 => cam)
    # with_mixed_scene(ax2, GLMakie) do mix1
    #     trunkcolor(rr) = begin
    #         rowcolor(rr)
    #     end
    #     arborcolor(rr) = begin
    #         return NaN
    #     end
    #     draw_subtrees(dfx, mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)..., colorrange, colormap=[:red, :blue])
    # end
    # draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:MD => :white]], ax2)
end

function plot_pt_sc_arbors(grid, env; dfxpt, dfxct, dfarbor, _...)
    rowcolor = cls_temp_color(vcat(dfxct, dfxpt))
    dfx = leftjoin(dfarbor, dfxpt, on=:path) |> dropmissing!
    filter!(r -> zz1data.select_neurons(:pt_strict)(r) && contains("SC")(r.arborhier), dfx)

    ax1 = LScene(grid[1, 1][1, 1], width=pt(80), height=pt(80), show_axis=false)
    # ax2 = LScene(grid[1, 2], height=pt(100), show_axis=false)
    for (j, n) in [1 => "Horizontal view", 2 => "Coronal sections"]
        Label(grid[1, j, Bottom()], n, tellwidth=false, tellheight=false, halign=:center, valign=:bottom)
    end
    cam = (fov=9.0, cent=(31827.6 - 1500 * 1, 31389.51, 24716.38), deye=(-0.0, -0.0, 1.0) .* 92342.68, up=(1.0, 0.0, 0.0))
    setup_camera(ax1 => cam)
    with_mixed_scene(ax1, GLMakie) do mix1
        trunkcolor(rr) = begin
            rr === nothing && return :transparent
            rowcolor(rr)
        end
        arborcolor(rr) = begin
            rr === nothing && return :transparent
            rowcolor(rr)
        end
        uvs, uv2pt = let (a, b, c, d, e, f) = Vec3f[
                (29288.621, 29801.918, 21323.984),
                (29275.391, 30310.301, 21534.143),
                (29331.496, 28050.166, 25113.037),
                (29550.107, 30475.43, 21155.42),
                (29547.594, 30798.568, 21329.217),
                (29493.84, 28385.988, 24987.662),
            ]
            b = b - a
            b2 = e - d
            b = b / norm(b) + b2 / norm(b2)
            c = c - a
            c2 = f - d
            c = c / norm(c) + c2 / norm(c2)
            a = (a + d) / 2
            u, v = normalize.((c, cross(c, b)))
            [Vec2d(c...) for c in eachcol(rand(2, 100) .* 2 .- 1)], (uv) -> begin
                (uv[1] .* u + uv[2] .* v) .+ a
            end
        end
        draw_cross_section_impl(Makie.Observable((-2800, 2800)), Makie.Observable((-2800, 2800)), uv2pt, (; mix1, gl=mix1.scene2); label=nothing, addrat=0.09, addfix=999, linewidth=100, arrowsize=300, color=:black)
        draw_subtrees(dfx, mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...)
        #scatter!(mix1.scene2, cspts, color=:red)
    end
    draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:SC => :ivory]], ax1)
    draw_scale_bar!(ax1, 2000, [cameracontrols(ax1).lookat[]], offset=(33, 12))

    # cam = (fov=6.0, cent=(26925.1, 38570.66, 20581.41), deye=(-1.0, -0.0, -0.0) .* 270233.79, up=(0.0, 0.0, 1.0))
    # setup_camera(ax2 => cam)
    # with_mixed_scene(ax2, GLMakie) do mix1
    #     trunkcolor(rr) = begin
    #         rowcolor(rr)
    #     end
    #     arborcolor(rr) = begin
    #         return :transparent
    #     end
    #     draw_subtrees(dfx, mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...)
    # end
    # draw_mesh_overlay([load("interm/mesh-nmt-$x-lr.obj") => c for (x, c) in [:SC => :white]], ax2)
    # draw_scale_bar!(ax2, 5000, load("interm/mesh-nmt-SC-lr.obj"), offset=(3, 12))

    # ax21 = Makie.Axis(grid[1, 2][1, 1])
    # ax22 = Makie.Axis(grid[1, 2][1, 2])
    GridLayout(grid[1, 2])
    LTeX(grid[1, 3][1, 1], PDFDocument(read("monkeytemp/wyz-upload/sc-layers-D17.pdf", String)), scale=0.95)
    LTeX(grid[1, 3][1, 2], PDFDocument(read("monkeytemp/wyz-upload/sc-layers-G97.pdf", String)), scale=0.95)
    colgap!(content(grid[1, 3]), 1, pt(1))
    colgap!(grid, 1, 2)
    colgap!(grid, 2, 2)
    # for (ax2, n) in (ax21 => "from dlPFC", ax22 => "from vlPFC")
    #     hidespines!(ax2)
    #     hidedecorations!(ax2)
    #     ax2.title = n
    #     ax2.titlegap = -pt(5)
    # end
end


function plot_ct_demo(ax1, ax2; dfxct, dfxpt, obj, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(vcat(dfxct, dfxpt))

    # ax1 = Makie.LScene(grid[1, 1], width=130, height=80, show_axis=false)
    # ax2 = Makie.LScene(grid[2, 1][1, 1], width=60, height=60, show_axis=false)
    # ax3 = Makie.LScene(grid[2, 1][1, 2], width=60, height=60, show_axis=false)
    # rowgap!(grid, 1, 0)
    setup_camera(ax1 => :s)
    setup_camera(ax2 => :zoominhpt)
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax2, GLMakie) do mix2
            draw_neurons(dfxct, mix1, mix2; rowlabel, rowcolor, mirrorby=NmtDat)
        end
        axislegend(ax1, mix1.scene2, unique=true) |> fix_line_legs
    end
    draw_contours(["interm/mesh-nmt-$ba-lr.obj" for ba in [:Thal, :MD]], ax1, ax2)
    draw_mesh_overlay([obj], ax1, ax2)
end

function plot_ct_flatmap(grid, env; objflat, objphy, highlights, dfxct, _...)
    rowlabel(r) = string("#", r.cls)
    rowcolor = cls_temp_color(dfxct)

    ax1 = Makie.Axis(grid[1, 1])
    plot_flatmap_base!(ax1, highlights, objflat, objphy)
    plot_flatmap_soma!(ax1, dfxct; rowcolor)
    limits!(ax1, -0.40, 0.40, -0.40, 0.30)
    ax1.width[] = 100
    ax1.height[] = 80
end

function plot_ct_topo(grid, env; dfxct, topo_ct_md, _...)
    grid.alignmode[] = Outside()
    rowlabel(r) = string("#", r.cls)
    #rowcolor = cls_temp_color(dfxct)
    rowcolor(r) = contains("vlPFC")(r.somahier) ? :red : :blue

    ax1 = Makie.Axis(grid[1, 1], height=pt(48))
    let dfxx = leftjoin(dfxct, topo_ct_md, on=:path) |> dropmissing!
        xpars, ypars = zz1data.find_best_coor_axis(dfxx.srcpos, dfxx.tgtpos, aligned_x=1, aligned_y=3)
        # 0.797/0.737/0.746/0.736
        @warn "$xpars $ypars"
        insertcols!(dfxx, :xx => dot.(dfxx.srcpos, Ref(xpars)), :yy => dot.(dfxx.tgtpos, Ref(ypars)))
        linear_reg_scatter!(ax1, dfxx; rowx=r -> r.xx / 1000, rowy=r -> r.yy / 1000, rowcolor)
        ax1.xlabel[] = "Soma position"
        ax1.ylabel[] = "Arbor center"
        setup_axis_by_orient!(ax1, :x, dfxx.xx ./ 1000, xpars)
        setup_axis_by_orient!(ax1, :y, dfxx.yy ./ 1000, ypars)
    end
    ax1
end

function draw_ptct_thal!(f::Function, ax3, dfx; typ2c)
    alltypes = sort(unique(dfx.typ))
    for t in alltypes
        Makie.density!(ax3, f.(filter(r -> r.typ == t, dfx).tgtpos), color=(typ2c(t), 0.5), label=t, direction=:x)
    end
    let t = HypothesisTests.MannWhitneyUTest([f.(filter(r -> r.typ == t, dfx).tgtpos) for t in ("CT", "PT")]...)
        write_stat(:CtVsPtPval, HypothesisTests.pvalue(t))
        write_stat(:CtVsPtMethod, :MannWhitneyUTestTwoSided)
    end
end

function plot_ctpt_diff(grid, env; dfxdlpfc, topo_ct_thal, topo_pt_thal, obj, highlights, objflat, objphy, _...)
    dfx = leftjoin(dfxdlpfc, vcat(topo_ct_thal, topo_pt_thal), on=:path) |> dropmissing!
    typ2c(r) = r == "CT" ? :red : :blue
    ax = Makie.Axis(grid[1, 1], alignmode=Outside())
    draw_ptct_thal!(v -> v[3], ax, dfx; typ2c)
    axislegend(ax, ax, margin=(2, 2, 2, 2))
    setup_axis_by_orient!(ax, :x, [r.tgtpos[3] for r in eachrow(dfx)], (0, 0, 1))
    #ax.width[] = 80
    ylims!(ax, 0, nothing)
    hideydecorations!(ax, label=false)
    ax.ylabel[] = "Density"
end

function plot_ctpt_diff_demo(grid, env; dfarbor, dfxdlpfc, topo_ct_thal, topo_pt_thal, obj, highlights, objflat, objphy, _...)
    dfx = leftjoin(dfxdlpfc, vcat(topo_ct_thal, topo_pt_thal), on=:path)
    dfx = leftjoin(dfarbor, dfxdlpfc, on=:path) |> dropmissing!
    filter!(r -> zz1data.matchhier("sarm", "Thal")(r.arborhier), dfx)
    typ2c(r) = r.typ == "CT" ? :red : :blue
    rowcolor(r) = typ2c(r)
    rowlabel(r) = string(r.typ)

    ax1 = LScene(grid[1, 1], width=pt(96), height=pt(70), show_axis=false)
    setup_camera(ax1 => :zoominctptthal)
    with_mixed_scene(ax1, GLMakie) do mix1
        nrn2color = Dict()
        trunkcolor(rr) = begin
            rr === nothing && return :transparent => 0.0
            return 0.5 * to_color(rowcolor(rr)) + 0.5 * to_color(:white) => 0.3
            #rowcolor(rr)
        end
        arborcolor(rr) = begin
            rr == nothing && return :transparent
            rowcolor(rr) => 0.5
        end
        draw_subtrees(dfx, mix1; autotrunk=false, mirrorby=NmtDat, node_color_trunks(; trunkcolor, arborcolor)...)
    end
    draw_contours(["interm/mesh-nmt-MD-lr.obj" => (; color=:cyan)], ax1)
    draw_mesh_overlay(["interm/mesh-nmt-Thal-lr.obj" => "Thal"], ax1)
    draw_scale_bar!(ax1, 2000, load("interm/mesh-nmt-MD-lr.obj"))
end

function draw_arbors_sidebyside!(ax, dfx; shift=0.3, rowcolor, dx0=0)
    swcs = map(eachrow(dfx)) do r
        swc = Gapr.load_swc(r.arborpath)
        p0 = swc[1].pos
        p1 = SVector(median(reduce(hcat, [n.pos for n in swc.nodes]), dims=2)...)
        u = p0 - p1
        u = u / norm(u)
        p2 = argmax(p -> norm((p - p1) - dot(p - p1, u) * u), [n.pos for n in swc.nodes])
        v = (p2 - p1) - dot(p2 - p1, u) * u
        v = v / norm(v)
        w = cross(u, v)
        w = w / norm(w)
        @warn "$u $v $w"
        for n in swc.nodes
            p = n.pos - p0
            n.pos = SVector(dot(p, v), dot(p, u), dot(p, w))
        end
        @assert swc[1].pos[1] == 0
        swc
    end
    dx = dx0
    somas = Pair{Point3f,Any}[]
    foreach(enumerate(swcs), eachrow(dfx)) do (i, swc), r
        c = rowcolor(r)
        z0, z1 = extrema([n.pos[3] for n in swc.nodes])
        mix(a, b, t) = t * a + (1 - t) * b
        pts, col = load_swc_as_linesx(n -> mix(to_color(c), 0.5 * to_color(c), (n.pos[3] - z0) / (z1 - z0)), swc)
        if i > 1
            dx += -minimum(p -> (v = first(p); isnan(v) ? Inf : v * shift), pts)
        end
        @warn "$(length(pts)) $dx"
        pts = map(p -> p .+ (dx, 0, 0), pts)
        push!(somas, swc[1].pos .+ (dx, 0, 0) => c)
        lines!(ax, pts, color=col, linewidth=0.5, joinstyle=:round, linecap=:round, transparency=true, fxaa=true)
        dx += maximum(p -> (v = first(p); isnan(v) ? -Inf : (v - dx) * shift), pts)
    end
    scatter!(ax, first.(somas), color=last.(somas), markersize=3)
end

# TAGxxx
#

function plot_ctpt_cmp(fig, env; dfxdlpfc, topo_ct_thal, topo_pt_thal, obj, highlights, objflat, objphy, _...)
    with_caption(L"Compare CT/PT.")
    grid = GridLayout(fig[1, 1])
    dfx = leftjoin(dfxdlpfc, vcat(topo_ct_thal, topo_pt_thal), on=:path) |> dropmissing!
    typ2c = typ2color_func(dfx)
    rowcolor(r) = typ2c(r.typ)
    rowlabel(r) = string(r.typ)

    tgtarea = include("monkeytemp/wyz-upload/julia/target-selection-ctpt-compare.jl")
    zz1data.calc_twoside_projection!(dfx)
    for (col, tgt) in enumerate(tgtarea)
        ax30 = Makie.Axis(grid[2, col])
        ax31 = Makie.Axis(grid[1, col])
        if col > 1
            colgap!(grid, col - 1, 2)
        end
        boxplot_and_compare!(ax30, stack(dfx, [tgt]), ax31; rowx=r -> r.variable, xname=x -> replace(x, r"^.*\|" => ""), rowy=r -> r.value / 1000, rowdodge=r -> r.typ, rowcolor, statif=(a, b) -> a[1] == b[1], violin_limits=(0, Inf))
    end
    rowgap!(grid, 1, 0)
end

soma_ml_mus(r) = begin
    p = r.soma_pos
    if !Allen.isleft(p)
        p = Allen.mirror(p)
    end
    p[3]
end
soma_ml_mac(r) = begin
    p = r.soma_pos
    if !NmtDat.isleft(p)
        p = NmtDat.mirror(p)
    end
    p[1]
end

function plot_mus_pt_bundle(grid, env; mac, mus, mustopo_ptsc, macarbor, _...)
    ax1 = Axis(grid[1, 1][1, 1], width=pt(70))
    content(grid[1, 1]).alignmode[] = Outside()
    dfx = leftjoin(mus, mustopo_ptsc, on=:path) |> dropmissing!
    filter!(r -> select_soma_mus_pfc_ctpt(r) > 0 && r.typ == "PT", dfx)
    xpars, ypars = zz1data.find_best_coor_axis(dfx.srcpos, dfx.tgtpos, aligned_x=-1, aligned_y=-1)
    insertcols!(dfx, :xx => [dot(xpars, r.srcpos) for r in eachrow(dfx)])
    insertcols!(dfx, :yy => [dot(ypars, r.tgtpos) for r in eachrow(dfx)])
    size(dfx, 1) == 0 && return
    @warn "$xpars, $ypars"
    linear_reg_scatter!(ax1, dfx; rowx=r -> r.xx / 1000, rowy=r -> r.yy / 1000, rowcolor=r -> :red)
    setup_axis_by_orient!(ax1, :y, dfx.yy / 1000, ypars, spe=:mus)
    setup_axis_by_orient!(ax1, :x, dfx.xx / 1000, xpars, spe=:mus)
    ax1.xlabel[] = "Soma position"
    ax1.ylabel[] = "Arbor center"
    text!(ax1, 1, 0, text="Mouse PT"; space=:relative, align=(:right, :bottom), set_font_h2x()...)

    ax2 = LScene(grid[1, 2][1,1], height=pt(110), show_axis=false)
    content(grid[1,2]).height=pt(90)
    colgap!(grid, 1, 4)
    setup_camera(ax2 => :allenhr)
    with_mixed_scene(ax2, GLMakie) do mix
        per_neuron(r) = begin
            return soma_ml_mus(r)
        end
        per_arbor(pn, rr, cond) = begin
            if cond
                return pn => pt(0.5)
            else
                return pn => 0.3
            end
        end
        per_node(pn, pa, n) = pa
        arbornodes(rr) = [rr.root] # [rr.tgtnid]
        colorrange = (extrema(per_neuron.(eachrow(dfx)))...,)
        draw_subtrees(pick_neurons_with_limit(dfx, r -> r.somahier, 20), mix; arbornodes, autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=Allen, colorrange, colormap=[:blue, :red], nan_color=:gray)

        cb = Colorbar(grid[1, 3], limits=colorrange, colormap=[:blue, :red])
        colgap!(grid, 2, 0)
        cb.ticks = ([mean(colorrange)], [L"\fontfamily{Arial}Lateral $⟷$ Medial"])
        cb.ticklabelrotation = π / 2
        cb.label = "Soma position"
        cb.height = pt(60)
    end
    draw_contours([load("interm/allen-$i.obj") => c for (i, c) in [294 => :black]], ax2)
    draw_mesh_overlay([load("interm/allen-$i.obj") => c for (i, c) in [997 => :lightgray]], ax2)
    global xxxsavedax2 = ax2
    return

    ax3 = LScene(grid[1, 3], height=pt(90), width=pt(120), show_axis=false)
    setup_camera(ax3 => :hr)
    with_mixed_scene(ax3, GLMakie) do mix
        per_neuron(r) = begin
            return soma_ml_mac(r)
        end
        per_arbor(pn, rr, cond) = begin
            if cond
                return pn => pt(0.5)
            else
                return pn => 0.3
            end
        end
        per_node(pn, pa, n) = pa
        arbornodes(rr) = [rr.root] # [rr.tgtnid]
        dfx = filter(r -> zz1data.matchhier("sarm", "SC")(r.arborhier), macarbor)
        leftjoin!(dfx, mac, on=:path) |> dropmissing!
        filter!(r -> r.typ == "PT", dfx)
        draw_subtrees(dfx, mix; arbornodes, autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=NmtDat, colorrange=(extrema(per_neuron.(eachrow(dfx)))...,), colormap=[:blue, :red], nan_color=:gray)
    end
    draw_contours([load("interm/mesh-nmt-$i-lr.obj") => c for (i, c) in [:SC => :black]], ax3)
    draw_mesh_overlay([load("interm/mesh-nmt-$i-lr.obj") => c for (i, c) in [:brain => :white]], ax3)
    colgap!(grid, 1, 2)
    colgap!(grid, 2, 2)
    global xxxsavedax3 = ax3

    #leftjoin!(dfx, mustopo_ptsc, on=:path) |> dropmissing!
    dfxmac = filter(r -> r.typ == "PT", mac)



    return
    # map(eachrow(dfx)) do rr
    #     swc = zz1data.load_swc_mirror(rr.path; mod=Allen)
    #     func = Main.trunk_from_arbor(swc; rr.soma_pos)
    #     func([rr.tgtnid])
    # end

    dfx = filter(r -> r.typ == "IT" && !zz1data.matchhier("ACA")(r.somahier) && !zz1data.matchhier("MO")(r.somahier) && !zz1data.matchhier("ORB")(r.somahier), mus) #= && r["allenI|root|grey|BS|MB|MBmot|SCm"] > 100  =#
    ax = LScene(grid[1, 1], width=pt(300), height=pt(300), show_axis=false)
    setup_camera(ax => :allenh)
    with_mixed_scene(ax, GLMakie) do mix
        per_neuron(r) = begin
            p = r.soma_pos
            if !Allen.isleft(p)
                p = Allen.mirror(p)
            end
            a = conda(r)
            b = condb(r)
            if a && b
                :orange
            elseif a
                :red
            elseif b
                :blue
            else
                :transparent
            end
        end
        scatter!(mix.scene2, dfx.soma_pos, color=per_neuron.(eachrow(dfx)))
        return
        per_arbor(pn, rr, cond) = begin
            if cond
                return pn => pt(0.5)
            else
                return pn => 0.3
            end
        end
        per_node(pn, pa, n) = pa
        arbornodes(rr) = [rr.root] # [rr.tgtnid]
        draw_subtrees(dfx, mix; arbornodes, autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=Allen, colorrange=(extrema(per_neuron.(eachrow(dfx)))...,), colormap=[:blue, :red], nan_color=:gray, inspect=:allen)
    end
    draw_mesh_overlay([load("interm/allen-$i.obj") => c for (i, c) in [997 => :white, 294 => :black]], ax)
end

function plot_mus_ct_bundle(grid, env; mac, mus, mustopo_ctmd, macarbor, _...)
    ax1 = Axis(grid[1, 1][1, 1], width=pt(70))
    content(grid[1, 1]).alignmode[] = Outside()
    dfx = leftjoin(mus, mustopo_ctmd, on=:path) |> dropmissing!
    filter!(r -> select_soma_mus_pfc_ctpt(r) > 0 && r.typ == "CT", dfx)
    xpars, ypars = zz1data.find_best_coor_axis(dfx.srcpos, dfx.tgtpos, aligned_x=3, aligned_y=2)
    insertcols!(dfx, :xx => [dot(xpars, r.srcpos) for r in eachrow(dfx)])
    insertcols!(dfx, :yy => [dot(ypars, r.tgtpos) for r in eachrow(dfx)])
    size(dfx, 1) == 0 && return
    @warn "$xpars, $ypars"
    linear_reg_scatter!(ax1, dfx; rowx=r -> r.xx / 1000, rowy=r -> r.yy / 1000, rowcolor=r -> :red)
    setup_axis_by_orient!(ax1, :y, dfx.yy / 1000, ypars, spe=:mus)
    setup_axis_by_orient!(ax1, :x, dfx.xx / 1000, xpars, spe=:mus)
    ax1.xlabel[] = "Soma position"
    ax1.ylabel[] = "Arbor center"
    text!(ax1, 1, 0, text="Mouse CT"; space=:relative, align=(:right, :bottom), set_font_h2x()...)

    ax2 = LScene(grid[1, 2][1,1], height=pt(100), show_axis=false)
    content(grid[1,2]).height=pt(90)
    colgap!(grid, 1, 4)
    setup_camera(ax2 => :allens)
    linkfov!(xxxsavedax2, ax2)
    with_mixed_scene(ax2, GLMakie) do mix
        per_neuron(r) = begin
            return soma_ml_mus(r)
        end
        per_arbor(pn, rr, cond) = begin
            if cond
                return pn => pt(0.5)
            else
                return pn => 0.3
            end
        end
        per_node(pn, pa, n) = pa
        arbornodes(rr) = [rr.root] # [rr.tgtnid]
        colorrange = (extrema(per_neuron.(eachrow(dfx)))...,)
        draw_subtrees(pick_neurons_with_limit(dfx, r -> r.somahier, 20), mix; arbornodes, autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=Allen, colorrange, colormap=[:blue, :red], nan_color=:gray)

        cb = Colorbar(grid[1, 3], limits=colorrange, colormap=[:blue, :red])
        colgap!(grid, 2, 0)
        cb.ticks = ([mean(colorrange)], [L"\fontfamily{Arial}Lateral $⟷$ Medial"])
        cb.ticklabelrotation = π / 2
        cb.label = "Soma position"
        cb.height = pt(60)
    end
    draw_contours([load("interm/allen-$i.obj") => c for (i, c) in [362 => :black]], ax2)
    draw_mesh_overlay([load("interm/allen-$i.obj") => c for (i, c) in [997 => :lightgray]], ax2)
    return

    ax3 = LScene(grid[1, 3], height=pt(90), width=pt(120), show_axis=false)
    setup_camera(ax3 => :s)
    linkfov!(xxxsavedax3, ax3)
    with_mixed_scene(ax3, GLMakie) do mix
        per_neuron(r) = begin
            return soma_ml_mac(r)
        end
        per_arbor(pn, rr, cond) = begin
            if cond
                return pn => pt(0.5)
            else
                return pn => 0.3
            end
        end
        per_node(pn, pa, n) = pa
        arbornodes(rr) = [rr.root] # [rr.tgtnid]
        dfx = filter(r -> zz1data.matchhier("sarm", "MD")(r.arborhier), macarbor)
        leftjoin!(dfx, mac, on=:path) |> dropmissing!
        filter!(r -> r.typ == "CT", dfx)
        draw_subtrees(dfx, mix; arbornodes, autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=NmtDat, colorrange=(extrema(per_neuron.(eachrow(dfx)))...,), colormap=[:blue, :red], nan_color=:gray)
    end
    draw_contours([load("interm/mesh-nmt-$i-lr.obj") => c for (i, c) in [:MD => :black]], ax3)
    draw_mesh_overlay([load("interm/mesh-nmt-$i-lr.obj") => c for (i, c) in [:brain => :white]], ax3)
    colgap!(grid, 1, 2)
    colgap!(grid, 2, 2)
end

function plot_compare_popu_comp(grid, env; mac, mus, _...)
    regions = get_regions(:; ff=r -> true)
    sel = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-mac-mus-it.jl")
    #sel = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-ct.jl")
    sel = map(v -> (v in vec(sel)), [regions(:mac, :I); regions(:mac, :C)])
    cols = filter(startswith("allenI"), [regions(:mus, :I); regions(:mus, :C)][sel])
    cols_mac = filter(startswith("nmtI"), [regions(:mac, :I); regions(:mac, :C)][sel])
    @warn "$(size(cols))"
    @warn "$(size(cols_mac))"
    getprojcol(df, col) = begin
        col in names(df) && return df[:, col]
        zeros(size(df, 1))
    end
    calc_proj(somahier) = begin
        musx = filter(r -> r.typ == "IT" && zz1data.matchhier(somahier)(r.somahier), mus)
        vec(mean(reduce(hcat, [getprojcol(musx, col) for col in cols]), dims=1))
    end
    macrgn(r) = begin
        if contains("dlPFC")(r.somahier)
            "dlPFC"
        elseif contains("vlPFC")(r.somahier)
            "vlPFC"
        else
            "other"
        end
    end
    calc_proj_mac(inj) = begin
        macx = filter(r -> r.typ == "IT" && macrgn(r) == inj, mac)
        vec(mean(reduce(hcat, [getprojcol(macx, col) for col in cols_mac]), dims=1))
    end
    mus_rgns = ["ORBm", "ORBvl", "ORBl", "PL", "ILA", "AId", "AIv"]
    musmat = reduce(hcat, calc_proj.(mus_rgns))
    mac_rgns = sort(unique(macrgn.(eachrow(mac))))
    macmat = reduce(hcat, calc_proj_mac.(mac_rgns))
    ax = Makie.Axis(grid[1, 1])
    heatmap!(ax, log.(musmat' .+ 1))
    ax2 = Makie.Axis(grid[1, 2])
    heatmap!(ax2, log.(macmat' .+ 1))
    ax.yticks[] = (axes(cols, 1), string.(cols))
    ax.xticks[] = (axes(mus_rgns, 1), string.(mus_rgns))
    ax2.xticks[] = (axes(mac_rgns, 1), string.(mac_rgns))
    @warn "$(size(macmat))"
    @warn "$(size(musmat))"

    calc_dist(i, j) = begin
        a = log.(musmat[:, i] .+ 1)
        b = log.(macmat[:, j] .+ 1)
        #a=musmat[:, i]
        #b=macmat[:, j]
        #dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        StatsBase.corspearman(a, b)
    end

    dist = [calc_dist(i, j) for i in axes(musmat, 2), j in axes(macmat, 2)]
    global xxxdist = deepcopy(dist)


end

function plot_compare_heatmap_temp(grid, env; mac, mus, _...)
    regions = get_regions(:; ff=r -> true)
    sel = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-mac-mus-it.jl")
    @warn "sel $sel $([regions(:mac, :I); regions(:mac, :C)])"
    sel = map(v -> v in vec(sel), [regions(:mac, :I); regions(:mac, :C)])
    @warn "sel $sel"
    macx = filter(r -> (r.typ == "ITc" || zz1data.select_neurons(:long_it_ipsi)(r)) && istypit(r), mac)
    filter!(r -> zz1data.matchhier("dlPFC")(r.somahier), macx)
    musx = filter(r -> r.typ == "IT" && zz1data.matchhier("AIv")(r.somahier), mus)
    insertcols!(musx, :cls => musx.subtyp)
    sort!(macx, :hclustord)
    # cols=filter(startswith("allenI"), [regions(:mus, :I); regions(:mus, :C)][sel])
    # mat=Matrix(musx[:,cols])'
    # hh=Clustering.hclust(1 .-cor(mat.+rand(size(mat))*0.01), linkage=:ward, branchorder=:barjoseph)
    # o1=hh.order
    # @warn "$(size(o1)) $(size(musx))"
    # musx.hclustord.=invperm(o1)
    # musx.cls.=Clustering.cutree(hh, k=8)
    #sort!(musx, :hclustord)
    sort!(musx, :cls)
    ax1 = Makie.Axis(grid[1, 1])
    ax2 = Makie.Axis(grid[1, 2])
    ax3 = Makie.Axis(grid[2, 1])
    ax4 = Makie.Axis(grid[2, 2])
    ax5 = Makie.Axis(grid[3, 1])
    ax6 = Makie.Axis(grid[3, 2])
    draw_neuron_proj_hm!(ax3, ax5, macx, [regions(:mac, :I); regions(:mac, :C)][sel]; ax3=ax1)
    draw_neuron_proj_hm!(ax4, ax6, musx, [regions(:mus, :I); regions(:mus, :C)][sel]; ax3=ax2)
    Makie.linkyaxes!(ax1, ax2)
    Makie.linkyaxes!(ax3, ax4)
    Makie.linkyaxes!(ax5, ax6)
end
function plot_mus_it_test(grid, env; mac, mus, _...)
    #return plot_compare_popu_comp(grid, env; env...)
    #conda(r) = zz1data.matchhier("ORBm")(r.somahier)
    #condb(r) = zz1data.matchhier("ORBvl")(r.somahier)
    wa(r) = r["allenI|root|grey|CH|CTX|CTXpl|Isocortex|RSP"]
    wb(r) = r["allenI|root|grey|CH|CTX|CTXpl|Isocortex|PTLp"]
    conda(r) = r["allenI|root|grey|CH|CTX|CTXpl|Isocortex|RSP"] > 10


    condb(r) = r["allenI|root|grey|CH|CTX|CTXpl|Isocortex|PTLp"] > 10
    condc(r) = false #zz1data.matchhier("ORBl")(r.somahier)
    dfx = filter(r -> r.typ == "IT" && (conda(r) || condb(r) || condc(r)), mus)
    filter!(r -> !zz1data.matchhier("ACA")(r.somahier) && !zz1data.matchhier("MO")(r.somahier), dfx)
    filter!(r -> zz1data.matchhier("ORB")(r.somahier), dfx)
    #&& "allenI|root|grey|CH|CTX|CTXpl|Isocortex|ORB|ORBvl"

    ax = LScene(grid[1, 1], width=pt(200), height=pt(300), show_axis=false)
    ax2 = Axis(grid[2, 1], height=pt(80))

    sts = []
    for ((cond, ww), col) in ((conda, wa) => :red, (condb, wb) => :blue)
        dfxx = filter(cond, dfx)
        @warn "cnt $(size(dfxx, 1))"
        vals = [mirror_by(Allen, r.soma_pos)[3] for r in eachrow(dfxx)]
        density!(ax2, color=(col, 0.5), vals, weights=ww.(eachrow(dfxx)))
        push!(sts, vals)
    end
    t = HypothesisTests.MannWhitneyUTest(sts...)
    per_neuron(r) = begin
        p = r.soma_pos
        if !Allen.isleft(p)
            p = Allen.mirror(p)
        end
        :red
        a = conda(r)
        b = condb(r)
        c = condc(r)
        if a && b
            :orange
        elseif a
            :red
        elseif b
            :blue
        elseif c
            :green
        else
            :transparent
        end
    end
    for c in unique(per_neuron.(eachrow(dfx)))
        n = size(filter(r -> per_neuron(r) == c, dfx), 1)
        @warn "$c $n"
    end
    @warn "$t"

    setup_camera(ax => :allenh)
    with_mixed_scene(ax, GLMakie) do mix

        scatter!(mix.scene2, [Vec3f(mirror_by(Allen, p)) for p in dfx.soma_pos], color=per_neuron.(eachrow(dfx)))
        return
        per_arbor(pn, rr, cond) = begin
            if cond
                return pn => pt(0.5)
            else
                return pn => 0.3
            end
        end
        per_node(pn, pa, n) = pa
        arbornodes(rr) = [rr.root] # [rr.tgtnid]
        draw_subtrees(dfx, mix; arbornodes, autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=Allen, colorrange=(extrema(per_neuron.(eachrow(dfx)))...,), colormap=[:blue, :red], nan_color=:gray, inspect=:allen)
    end
    draw_mesh_overlay([load("interm/allen-$i.obj") => c for (i, c) in [997 => :white, 294 => :black]], ax)
    return

    ax2 = LScene(grid[1, 2], width=pt(200), height=pt(300), show_axis=false)
    setup_camera(ax2 => :h)
    with_mixed_scene(ax2, GLMakie) do mix

        per_neuron(r) = begin
            p = r.soma_pos
            :red
            # a=conda(r)
            # b=condb(r)
            #       if a && b
            #           :orange
            #       elseif a
            #           :red
            #       elseif b
            #           :blue
            #       else
            #           :transparent
            #       end
        end
        per_arbor(pn, rr, cond) = begin
            if cond
                return pn => pt(0.5)
            else
                return pn => 0.3
            end
        end
        per_node(pn, pa, n) = pa
        arbornodes(rr) = [rr.root] # [rr.tgtnid]
        draw_subtrees(filter(r -> zz1data.matchhier("dlPFC")(r.somahier), mac), mix; arbornodes, autotrunk=false, per_neuron, per_arbor, per_node, mirrorby=NmtDat, colorrange=(extrema(per_neuron.(eachrow(dfx)))...,), colormap=[:blue, :red], nan_color=:gray, inspect=:nmt)
    end
    #draw_mesh_overlay([load("interm/allen-$i.obj") => c for (i, c) in [997 => :white, 294 => :black]], ax)
end

function plot_compare_lenx(gridx, env; mac, mus, spe2color, _...)
    mac = deepcopy(mac)
    mus = deepcopy(mus)
    mac6 = deepcopy(mac)
    mac6.spe .= :mac6
    scale = 5.86
    mac6.len .= mac6.len ./ scale
    mac6.axonlen .= mac6.axonlen ./ scale
    mac6.arborlen .= mac6.arborlen ./ scale
    mac6.maxdepth .= mac6.maxdepth ./ scale
    # "perarborlen"
    # "axonlenC"
    # "hclustord"
    # "majorlens"
    # "majortips"
    # "nonmajorlen"
    # "spe"

    # calc_arbor_stats!(mac; thr=3000)
    # calc_arbor_stats!(mus; thr=1000)
    calc_arbor_stats!(mac; thr=1900, scale=5.7)
    calc_arbor_stats!(mus; thr=700, scale=1)
    dd = extrema(mus.len1 .+ mus.len2 .- mus.axonlen)
    dd2 = extrema(mac.len1 .+ mac.len2 .- mac.axonlen)
    xx = sort(mac, :arboravg)[end, :path]
    yy = sort(mus, :arboravg)[end, :path]
    @warn "dddd $dd $dd2 $xx $yy"
    zz = NamedTuple.(eachrow(filter(r -> r.path == "/fugax/monkeytemp/mouse-pfc/195830_024.swc.gz", mus)[:, end-10:end] |> dropmissing!))
    println(zz[1])
    yy = NamedTuple.(eachrow(filter(r -> r.path == "monkeytemp/split-sample232717-ch2/root1058779201-qroot401-good.swc.gz", mac)[:, end-10:end] |> dropmissing!))
    println(yy[1])

    cols = intersect(names(mac), names(mus))
    df = vcat(mac[:, cols], mus[:, cols]) #= mac6[:, cols],  =#

    rowx(r) = r.typ[1:2]
    xname(x) = "$x"
    rowdodge(r) = r.spe
    statif(a, b) = begin
        a.x == b.x && (a.d === :mus) != (b.d === :mus)
    end

    filter!(r -> r.spe !== :mac6, df)

    ax1x = Axis(gridx[1, 1][1, 1])
    ax1 = Axis(gridx[1, 1][2, 1])
    boxplot_and_compare!(ax1, df, ax1x; rowx, xname, rowy=r -> log10(r.axonlen / 1000), rowdodge, rowcolor=spe2color, statif, legendpos=:rb)
    ax1.ylabel[] = "Axon length (mm)"
    ax1.yticks[] = ([0:3...], [string(10^y) for y in 0:3])

    ax2x = Axis(gridx[2, 1][1, 1])
    ax2 = Axis(gridx[2, 1][2, 1])
    boxplot_and_compare!(ax2, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> log10(r.trunklen / 1000), rowdodge, rowcolor=spe2color, statif)
    ax2.ylabel[] = "Trunk length (mm)"
    ax2.yticks[] = ([0:3...], [string(10^y) for y in 0:3])

    ax2x = Axis(gridx[3, 1][1, 1])
    ax3 = Axis(gridx[3, 1][2, 1])
    boxplot_and_compare!(ax3, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> log10(r.arbortot / 1000), rowdodge, rowcolor=spe2color, statif)
    ax3.ylabel[] = "Arbor length (mm)"
    ax3.yticks[] = ([0:3...], [string(10^y) for y in 0:3])

    #plot_compare_decomp(GridLayout(grid[3, 1]), env; env...)

    ax2x = Axis(gridx[4, 1][1, 1])
    ax4 = Axis(gridx[4, 1][2, 1])
    boxplot_and_compare!(ax4, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> r.collcnt, rowdodge, rowcolor=spe2color, statif)
    ax4.ylabel[] = "Collateral N.o."
    #ax2.yticks[] = ([0:3...], [string(10^y) for y in 0:3])

    ax2x = Axis(gridx[5, 1][1, 1])
    ax5 = Axis(gridx[5, 1][2, 1])
    #boxplot_and_compare!(ax5, filter(r -> r.arboravg > 0 && r.arboravg < 100000000, df), ax2x; rowx, xname, rowy=r -> r.arboravg, rowdodge, rowcolor=spe2color, statif)
    boxplot_and_compare!(ax5, filter(r -> r.arboravg > 0 && r.arboravg < 100000000, df), ax2x; rowx, xname, rowy=r -> r.arboravg, rowdodge, rowcolor=spe2color, statif)
    ax5.ylabel[] = "Normalized Arbor len."
    #ax2.yticks[] = ([0:3...], [string(10^y) for y in 0:3])

    ax2x = Axis(gridx[6, 1][1, 1])
    ax6 = Axis(gridx[6, 1][2, 1])
    boxplot_and_compare!(ax6, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> r.nbranches, rowdodge, rowcolor=spe2color, statif)
    ax6.ylabel[] = "Branch N.o."

    #ax2x = Axis(gridx[7, 1][1, 1])
    ax7 = Axis(gridx[7, 1][2, 1])
    #boxplot_and_compare!(ax7, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> r.nbranches, rowdodge, rowcolor=spe2color, statif)
    ax7.ylabel[] = "Arbor size"
    #ax2.yticks[] = ([0:3...], [string(10^y) for y in 0:3])
    #
    # ax2x = Axis(gridx[4, 1][1, 1])
    # ax4 = Axis(gridx[4, 1][2, 1])
    # boxplot_and_compare!(ax4, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> r.maxdepth / r.axonlen, rowdodge, rowcolor=spe2color, ingroupstat=true)
    # ax4.ylabel[] = "Max depth"
    # #ax4.yticks[] = ([0:3...], [string(10^y) for y in 0:3])
    #
    # ax2x = Axis(gridx[5, 1][1, 1])
    # ax5 = Axis(gridx[5, 1][2, 1])
    # #  boxplot_and_compare!(ax5, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> r.perarborlen, rowdodge, rowcolor=spe2color, ingroupstat=true)
    # boxplot_and_compare!(ax5, filter(r -> r.arborlen > 0, df), ax2x; rowx, xname, rowy=r -> r.axonlenC / r.axonlen, rowdodge, rowcolor=spe2color, ingroupstat=true)
    # ax5.ylabel[] = "Collateral len"
    # #ax5.yticks[] = ([0:3...], [string(10^y) for y in 0:3])
    Makie.linkxaxes!(ax1, ax2, ax3, ax4, ax5, ax6, ax7)
    for r in 1:7
        if r > 1
            rowgap!(gridx, r - 1, 0)
        end
        rowgap!(content(gridx[r, 1]), 1, 0)
        rowsize!(gridx, r, 45)
    end
end

function load_zf_data()
    files = readdir("mapZebrain__neurons/Aligned/Original", join=true)
    df = @showprogress asyncmap(files; ntasks=12) do path
        swc = try
            mktempdir() do d
                f1 = joinpath(d, "1.swc")
                e = expanduser("~/Desktop/gapr/nodist/nrn-model/build/sort-swc")
                run(`$e $path $f1`)
                Gapr.load_swc(f1; sort=true)
            end
        catch e
            @warn "$path"
            return missing
        end
        swc = Gapr.nodedegree(swc)
        nbranches = 0
        for n in swc.nodes
            nbranches += Gapr.isbranch(n, n.degree)
        end
        axonlen = 0.0
        for n in swc.nodes
            axonlen += n.length
        end
        (; path, nbranches, axonlen, swc)
    end
    DataFrame(collect(skipmissing(df)))
end

function load_mac_dend()
    paths = []
    for d1 in readdir("/tmp")
        match(r"xxx\d+", d1) !== nothing || continue
        for d2 in readdir(joinpath("/tmp", d1))
            match(r"\bgooddend\b", d2) !== nothing || continue
            push!(paths, joinpath("/tmp", d1, d2))
        end
    end
    df = @showprogress map(paths) do path
        swc = Gapr.load_swc(path)
        swc = Gapr.nodedegree(swc)
        swc = filter(n -> n.typename !== :axon, swc; toroot=true, toterm=false)
        nbranches = 0
        for n in swc.nodes
            nbranches += Gapr.isbranch(n, n.degree)
        end
        axonlen = 0.0
        for n in swc.nodes
            axonlen += n.length
        end
        (; path, nbranches, axonlen)
    end
    DataFrame(df)
end
function load_mus_dend(mus)
    mus = filter(r -> r.axonlen / r.len < 0.97, mus)
    mus = shuffle(mus)[1:500, :]
    df = @showprogress map(eachrow(mus)) do r
        swc = Gapr.load_swc(r.path)
        swc = Gapr.nodedegree(swc)
        swc = filter(n -> n.typename !== :axon, swc; toroot=true, toterm=false)
        nbranches = 0
        for n in swc.nodes
            nbranches += Gapr.isbranch(n, n.degree)
        end
        axonlen = 0.0
        for n in swc.nodes
            axonlen += n.length
        end
        (; r.path, nbranches, axonlen)
    end
    DataFrame(df)
end


function plot_compare_loglog(grid, env; mac, mus, spe2color, _...)
    mac = deepcopy(mac)
    mus = deepcopy(mus)
    calc_arbor_stats!(mac; thr=1900)
    calc_arbor_stats!(mus; thr=700)
    zf = zz1data.cached_result(:zf) do
        load_zf_data()
    end
    insertcols!(zf, :spe => :zf)
    zf = vcat([insertcols(zf, :typ => "$t") for t in (:IT, :PT, :CT)]...)
    musdend = zz1data.cached_result(:musdend, mus) do mus
        load_mus_dend(mus)
    end
    macdend = zz1data.cached_result(:macdend) do
        load_mac_dend()
    end
    insertcols!(musdend, :spe => :mus, :typ => "Dend")
    insertcols!(macdend, :spe => :mac, :typ => "Dend")
    t1 = HypothesisTests.MannWhitneyUTest(musdend.nbranches, macdend.nbranches)
    @info "t1 $t1"
    t2 = HypothesisTests.MannWhitneyUTest(musdend.axonlen, macdend.axonlen)
    @info "t1 $t2"
    for r in eachrow(mus)
        if r["allenC|root"] > 10 && r.typ == "IT"
            r.typ = "ITc"
        end
        if r["allenC|root"] < 0.01 && r.typ == "IT"
            r.typ = "ITi"
        end
    end

    if false
        dd = extrema(mus.len1 .+ mus.len2 .- mus.axonlen)
        dd2 = extrema(mac.len1 .+ mac.len2 .- mac.axonlen)
        xx = sort(mac, :arboravg)[end, :path]
        yy = sort(mus, :arboravg)[end, :path]
        @warn "dddd $dd $dd2 $xx $yy"
        zz = NamedTuple.(eachrow(filter(r -> r.path == "/fugax/monkeytemp/mouse-pfc/195830_024.swc.gz", mus)[:, end-10:end] |> dropmissing!))
        println(zz[1])
        yy = NamedTuple.(eachrow(filter(r -> r.path == "monkeytemp/split-sample232717-ch2/root1058779201-qroot401-good.swc.gz", mac)[:, end-10:end] |> dropmissing!))
        println(yy[1])
    end

    cols = intersect(names(mac), names(mus), names(zf), names(musdend), names(macdend))
    df = vcat(mac[:, cols], mus[:, cols], zf[:, cols], musdend[:, cols], macdend[:, cols])

    shuffle!(df)
    for (ccc, typ) in enumerate([:ITi, :ITc, :PT, :CT, :Dend])
        ax = Axis(grid[ccc, 1])
        ccc > 1 && rowgap!(grid, ccc - 1, 2)
        if ccc > 1
            linkaxes!(ax, content(grid[ccc-1, 1]))
        end
        for spe in [:mus, :mac, :zf]
            col = (; mac=:red, mus=:blue, zf=:orange)[spe]
            marker = (; IT=:circle, ITc=:circle, ITi=:circle, PT=:cross, CT=:triangle, Dend=:square)[typ]
            #dfx = filter(r -> r.spe === spe && r.typ[1:2] == "$typ"[1:2], df)
            dfx = filter(r -> r.spe === spe && r.typ == "$typ", df)
            size(dfx, 1) == 0 && continue
            linear_reg_scatter!(ax, dfx; rowx=r -> log10(r.axonlen + 1), rowy=r -> log10(r.nbranches + 1), rowcolor=r -> col, legpos=nothing, force=false, marker, linecolor=col, noleg=true, legoff=(2, spe === :mus ? -9 : -2), legcol=col)
        end
        ax.xlabel[] = "Axon length (log10)"
        ax.ylabel[] = "No. branches (log10)"
        text!(ax, 1, 0, text="$typ", align=(:right, :bottom), space=:relative; set_font_h2()...)
        hideydecorations!(ax, label=ccc != 2, ticklabels=false, ticks=false)
        hidexdecorations!(ax, label=false, ticklabels=false, ticks=false)
        if ccc != 3
            hidexdecorations!(ax, ticks=false)
        end
    end
    ax, _ = density(grid[5, 1][1, 1], musdend.nbranches)
    density!(ax, macdend.nbranches, color=(:red, 0.5))
    ax, _ = density(grid[5, 1][1, 2], musdend.axonlen)
    density!(ax, macdend.axonlen, color=(:red, 0.5))
    ax, _ = density(grid[6, 1][1, 1], filter(r -> r.spe == :mus && r.typ == "ITc", df).nbranches)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "ITc", df).nbranches, color=(:red, 0.5))
    ax, _ = density(grid[6, 1][1, 2], filter(r -> r.spe == :mus && r.typ == "ITc", df).axonlen)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "ITc", df).axonlen, color=(:red, 0.5))
    ax, _ = density(grid[7, 1][1, 1], filter(r -> r.spe == :mus && r.typ == "PT", df).nbranches)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "PT", df).nbranches, color=(:red, 0.5))
    ax, _ = density(grid[7, 1][1, 2], filter(r -> r.spe == :mus && r.typ == "PT", df).axonlen)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "PT", df).axonlen, color=(:red, 0.5))
    ax, _ = density(grid[8, 1][1, 1], filter(r -> r.spe == :mus && r.typ == "CT", df).nbranches)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "CT", df).nbranches, color=(:red, 0.5))
    ax, _ = density(grid[8, 1][1, 2], filter(r -> r.spe == :mus && r.typ == "CT", df).axonlen)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "CT", df).axonlen, color=(:red, 0.5))
    ax, _ = density(grid[9, 1][1, 1], filter(r -> r.spe == :mus && r.typ == "ITi", df).nbranches)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "ITi", df).nbranches, color=(:red, 0.5))
    ax, _ = density(grid[9, 1][1, 2], filter(r -> r.spe == :mus && r.typ == "ITi", df).axonlen)
    density!(ax, filter(r -> r.spe == :mac && r.typ == "ITi", df).axonlen, color=(:red, 0.5))
end

onearbor_per_nrnr!(df, dfarbor; noop=false) = begin
    noop && return
    per_dsarb = Dict()
    df2 = map(eachrow(df)) do r
        dfx = get!(per_dsarb, r.sample) do
            paths = Set(filter(rr -> rr.sample == r.sample, df).path)
            filter(rr -> in(paths)(rr.path), dfarbor)
        end
        dfx = filter(rr -> rr.path == r.path, dfx)
        isempty(dfx) && return (; r.path, f_totlen=NaN, f_volume=NaN, f_numbranches=NaN)
        rr = argmax(rr -> rr.f_totlen, eachrow(dfx))
        (; r.path, rr.f_totlen, rr.f_volume, rr.f_numbranches)
    end
    leftjoin!(df, DataFrame(df2), on=:path) |> dropmissing!
end


function find_all_temppari_neurons(df)
    outdir = "monkeytemp/wyz-upload/temporal-parietal-neurons"
    tgts = [
        "allen|root|grey|CH|CTX|CTXpl|Isocortex|TEa",
        #"allen|root|grey|CH|CTX|CTXpl|Isocortex|SS",
        "allen|root|grey|CH|CTX|CTXpl|Isocortex|RSP",
        "allen|root|grey|CH|CTX|CTXpl|Isocortex|PTLp",
    ]
    allen = Allen.load_images()
    ann2match = Dict()
    @showprogress for r in eachrow(df)
        swc = Gapr.load_swc(r.path)
        cnt = highlight_swc!(swc, allen, tgts, Allen; ann2match)
        if cnt > 0
            outfn = joinpath(outdir, basename(r.path))
            swc[1].typ = 1
            Gapr.save_swc(outfn, swc)
        end
    end
end

function highlight_swc!(swc, atlas, hiers, mod; ann2match)
    cnt = 0
    for n in swc
        idx = mod.pos2idx(n.pos, atlas.res)
        ann = mod.idx2ann(atlas, idx)
        mat = get!(ann2match, ann) do
            h = mod.ann2hier(atlas, ann)
            if hiers isa AbstractString
                hiers = (hiers,)
            end
            for hier in hiers
                if h == hier || startswith("$hier|")(h)
                    return true
                end
            end
            false
        end
        n.typ = mat ? 2 : 0
        if mat
            cnt += 1
        end
    end
    cnt
end

function plot_compare_temppari_arbor(grid, env; mac, mus, macarbor, musarbor, _...)
    obj1 = load("interm/mesh-nmt-brain-lr.obj")
    obj2 = load("interm/allen-997.obj")
    for (row, ff) in enumerate([
        #(r -> r.typ == "ITi", r -> r.typ == "IT" && r.axonlenC < 1 && r["allenI|root|grey|CH|CNU|STR"] < 1),
        #("nmt|charm|Parietal|IPL|lat_IPS|LIP", "allen|root|grey|CH|CTX|CTXpl|Isocortex|PTLp"),
        #("nmt|charm|Parietal|PMC|PCgG|RSC", "allen|root|grey|CH|CTX|CTXpl|Isocortex|RSP"),
        ("nmt|charm|Temporal", "allen|root|grey|CH|CTX|CTXpl|Isocortex|TEa"),
        #(r -> r.cls in (31, 30), r -> r.subtyp == 21),
    ])
        ax1 = LScene(grid[row, 1][1, 1])#, width=380, height=120, show_axis=false)
        ax2 = LScene(grid[row, 1][1, 2])#, width=380, height=120, show_axis=false)

        for (ax, df, mod, hhh) in [(ax1, macarbor, NmtDat, ff[1]), (ax2, musarbor, Allen, ff[2])]
            atlas = mod.load_images()
            #with_mixed_scene(ax2, GLMakie) do mix2
            ann2match = Dict()
            @showprogress for r in eachrow(shuffle(df)[1:5000, :])
                swc = Gapr.load_swc(r.arborpath)
                n = highlight_swc!(swc, atlas, hhh, mod; ann2match)
                n == 0 && continue
                c = Makie.rand(Makie.wong_colors())
                pts, col = load_swc_as_linesx(n -> n.typ == 0 ? (:gray, 0.03) : c, swc)
                lines!(ax, pts, color=c, transparency=true)
                #draw_neurons(filter(ff[2], mus), ax2 => :allenh; obj=obj2, inspect=:allen, mirrorby=Allen)
            end
            #end
        end
    end
end

function mark_significant!(ax, xvals, yvals)
    allgrps = sort(unique(xvals))
    allgrps = map(allgrps) do g
        g => yvals[xvals.==g]
    end
    o = 0.0
    for (g1, v1) in allgrps
        for (g2, v2) in allgrps
            if g1 < g2
                t = HypothesisTests.MannWhitneyUTest(v1, v2)
                p = HypothesisTests.pvalue(t)
                s = p < 0.0005 ? "***" : (p < 0.005 ? "**" : (p < 0.05 ? "*" : ""))
                if s != ""
                    Makie.bracket!(ax, g1, 0, g2, 0, text="$s (p=$(round(p, sigdigits=3)))", style=:square, offset=o, width=3)
                    o += 8
                    @warn "$g1 $g2 $p"
                end
            end
        end
    end
end

calc_proj_of_neuron(tgts...) = begin
    cols = Ref{Vector{String}}()
    ismatch(c) = zz1data.is_projcol(c) && zz1data.matchhier(tgts...)(c)
    r -> begin
        if !isassigned(cols)
            cols[] = filter(ismatch, names(r))
        end
        p = (10000, 0.0)
        for c in cols[]
            p = min(p, (length(c), r[c]))
        end
        p[2] > 10.0
    end
end

select_soma_mus_pfc(r) = begin
    zz1data.matchhier("ACA")(r.somahier) && return 0
    zz1data.matchhier("MO")(r.somahier) && return 0
    zz1data.matchhier("AI")(r.somahier) && return 0
    #!zz1data.matchhier("ORB")(r.somahier) && return 0
    !(zz1data.matchhier("ORBvl")(r.somahier) || zz1data.matchhier("ORBl")(r.somahier)) && return 0
    1
end
select_soma_mus_pfcit(r) = begin
    !zz1data.istypit(r) && return 0
    select_soma_mus_pfc(r)
end
select_soma_mus_pfc_ctpt(r) = begin
    zz1data.istypit(r) && return 0
    select_soma_mus_pfc(r)
end

function plot_compare_pari(grid, env; mac, mus, macarbor, musarbor, _...)
    objs = (; mac=load("interm/mesh-nmt-brain-lr.obj"), mus=load("interm/allen-997.obj"))

    mac = insertcols(mac, :projpat => selectrows_mac_temp_pari().(eachrow(mac)))
    filter!(r -> zz1data.istypit(r) && sum(r.projpat) == 1, mac)

    mus = insertcols(mus, :projpat => selectrows_mus_temp_pari().(eachrow(mus)))
    filter!(r -> select_soma_mus_pfcit(r) > 0 && sum(r.projpat) == 1, mus)

    ax1 = LScene(grid[1, 1][1, 1], width=pt(108), height=pt(85), show_axis=false)
    ax11 = LScene(grid[1, 1][2, 1], width=pt(108), height=pt(70), show_axis=false)
    ax2 = LScene(grid[1, 2][1, 1], width=pt(104), height=pt(85), show_axis=false)
    ax3 = LScene(grid[1, 2][2, 1], width=pt(104), height=pt(70), show_axis=false)
    # ax7 = LScene(grid[1, 3][1, 1], width=pt(110), height=pt(85), show_axis=false)
    # ax8 = LScene(grid[1, 3][2, 1], width=pt(110), height=pt(70), show_axis=false)
    ax60 = Axis(grid[2, 1])
    ax6 = Axis(grid[3, 1][1, 1], height=pt(50))
    ax40 = Axis(grid[2, 2][1, 1])
    ax4 = Axis(grid[3, 2][1, 1])
    # ax90 = Axis(grid[1, 3][3, 1][1, 1])
    # ax9 = Axis(grid[1, 3][3, 1][2, 1])
    rowgap!(grid, 1, 4)
    rowgap!(grid, 2, 0)
    for j in 1:2
        if j > 1
            colgap!(grid, j - 1, 4)
        end
        rowgap!(content(grid[1, j]), 1, 0)
        content(grid[3, j]).alignmode[] = Outside()
    end

    rowcolor(r) = (; RSC=:Red, LIP=:Blue, Temporal=:Green)[argmax(r.projpat)]
    setup_camera(ax1 => :hr, ax11 => :s)
    with_mixed_scene(ax1, GLMakie) do mix1
        with_mixed_scene(ax11, GLMakie) do mix11
            rowlabel(r) = string(argmax(r.projpat))
            dfx = pick_neurons_with_limit(mac, r -> r.projpat, 20)
            sort!(dfx, :projpat)
            draw_neurons(dfx, mix1, mix11; rowlabel, rowcolor, mirrorby=NmtDat)
            axislegend(ax1, mix1.scene2, unique=true) |> fix_line_legs
        end
    end
    draw_mesh_overlay([objs[:mac]], ax1, ax11)
    draw_contours(["interm/mesh-nmt-$ba-l.obj" for ba in [:LIP, :RSC, :Temporal]], ax1, ax11)

    setup_camera(ax2 => :allenhr, ax3 => :allens)
    with_mixed_scene(ax2, GLMakie) do mix2
        with_mixed_scene(ax3, GLMakie) do mix3
            rowlabel(r) = (; RSC="RSP", LIP="PTLp", Temporal="TEa")[argmax(r.projpat)]
            dfx = pick_neurons_with_limit(mus, r -> r.projpat, 20)
            sort!(dfx, :projpat)
            draw_neurons(dfx, mix2, mix3; rowlabel, rowcolor, mirrorby=Allen)
            #draw_neurons(filter(r -> r.sel === 3, dfxx), mix3; rowlabel, rowcolor, mirrorby=Allen)
            axislegend(ax2, mix2.scene2, unique=true) |> fix_line_legs
            #axislegend(ax3, mix3.scene2, unique=true) |> fix_line_legs
        end
    end
    draw_mesh_overlay([objs[:mus]], ax2, ax3)
    draw_contours(["interm/allen-$ba.obj" for (n, ba) in [:RSP => 254, :PTLp => 22, :TEa => 541]], ax2, ax3)

    for (ax, tag) in (ax6 => :Monkey, ax4 => :Mouse)
        dfx = deepcopy((; Mouse=mus, Monkey=mac)[tag])
        Mod = (; Mouse=Allen, Monkey=NmtDat)[tag]
        ddd = (; Mouse=(0, 0, 1), Monkey=(1, 0, 0))[tag]
        insertcols!(dfx, :yy => [dot(zz1data.mirrorby(Mod)(r.soma_pos), ddd) ./ 1000 for r in eachrow(dfx)])
        ax00 = (; Monkey=ax60, Mouse=ax40)[tag]
        violin_and_compare!(ax, dfx, ax00; compact=true, rowx=r -> (; RSC=1, LIP=2, Temporal=3)[argmax(r.projpat)], xname=x -> "", rowy=r -> r.yy, rowcolor, statif=(a, b) -> a[1] != b[1], scale=:width)
        setup_axis_by_orient!(ax, :y, dfx.yy, ddd, spe=(; Monkey=:mac, Mouse=:mus)[tag])
        ax.ylabel[] = "Soma position"
    end
    linkyaxes!(ax60, ax40)
    ax60.height = max(ax40.height[], ax60.height[])
    ax40.height = max(ax40.height[], ax60.height[])
    ylims!(ax40, 0, 2 + 1.0)
end

function plot_compare_venn(grid, env; mac, mus, macarbor, musarbor, _...)
    mac = insertcols(mac, :projpat => selectrows_mac_temp_pari().(eachrow(mac)))
    filter!(r -> zz1data.istypit(r), mac)
    mus = insertcols(mus, :projpat => selectrows_mus_temp_pari().(eachrow(mus)))
    filter!(r -> select_soma_mus_pfcit(r) > 0, mus)
    patcolor(pat) = begin
        :Black => (sum(pat) == 1 ? first : :Gray)
    end

    dfx = DataFrame(mac.projpat)
    ax1 = Axis(grid[1, 1])
    mac2cols = [:RSC => :Red, :LIP => :Blue, :Temporal => :Green]
    test_venn!(ax1, dfx, mac2cols..., scale=0.8; patcolor)
    dfx = DataFrame(mus.projpat)
    ax2 = Axis(grid[1, 2])
    for (j, mm) in enumerate((:Monkey, :Mouse))
        Label(grid[1, j, Left()], "$mm", rotation=π / 2; set_font_h2x()...)
    end
    mac2mus = (; RSC="RSP", LIP="PTLp", Temporal="TEa")
    test_venn!(ax2, dfx, mac2cols..., tagname=t -> mac2mus[t], scale=1.1; patcolor)
    Legend(grid[2, :], [PolyElement(color=c) for c in push!(last.(mac2cols), :Gray)], push!(["$n-/$(mac2mus[n])-only" for n in first.(mac2cols)], "Multi-targeted"), orientation=:horizontal)
    rowgap!(grid, 1, 0)
    for ax in (ax1, ax2)
        hidedecorations!(ax)
        hidespines!(ax)
    end
end

function plot_compare_pari_hm(grid, env; mac, mus, macarbor, musarbor, _...)
    mac = insertcols(mac, :projpat => selectrows_mac_temp_pari().(eachrow(mac)))
    filter!(r -> zz1data.istypit(r) && sum(r.projpat) > 0, mac)
    sort!(mac, :projpat)

    mus = insertcols(mus, :projpat => selectrows_mus_temp_pari().(eachrow(mus)))
    filter!(r -> select_soma_mus_pfcit(r) > 0 && sum(r.projpat) > 0, mus)
    sort!(mus, :projpat)

    regions = get_regions(:; ff=r -> true)
    sel = include("monkeytemp/wyz-upload/julia/heatmap-target-selection-mac-mus-it.jl")
    sel = map(v -> v in vec(sel), regions(:mac, :I))
    ax998x = Axis(grid[1, 1])
    ax998 = Axis(grid[2, 1])
    ax999x = Axis(grid[1, 2])
    ax999 = Axis(grid[2, 2])
    rowgap!(grid, 1)
    colgap!(grid, 4)
    Label(grid[2, 1, Left()], "Monkey"; rotation=π / 2, set_font_h2x()...)
    Label(grid[2, 2, Left()], "Mouse"; rotation=π / 2, set_font_h2x()...)
    let
        typ2tmpcls = Dict()
        for df in (mac, mus)
            for r in eachrow(df)
                r.cls = 1000 - get!(() -> length(typ2tmpcls) + 1, typ2tmpcls, r.projpat)
            end
        end
        #mac.cls .= 999
        #mus.cls .= 999
        #heatmap!(ax999, log.(Matrix(dfx[:, rgns]) .+ 1))
        grpcolor(i, g) = begin
            sum(g) != 1 && return :gray
            g.RSC && return :red
            g.LIP && return :blue
            g.Temporal && return :green
        end
        plot_subtyp_spans!(ax998x, mac; linkto=ax998, gridto=[ax998], rowgrp=r -> r.projpat, grplabel=(i, g) -> "", grpcolor, label="")
        draw_neuron_proj_hm!(ax998, nothing, mac, regions(:mac, :I)[sel]; ax3=nothing, skipic=true, yticks=:ignore)
        ax998.width[] = nothing
        plot_subtyp_spans!(ax999x, mus; linkto=ax999, gridto=[ax999], rowgrp=r -> r.projpat, grplabel=(i, g) -> "", grpcolor, label="Pattern")
        draw_neuron_proj_hm!(ax999, nothing, mus, regions(:mus, :I)[sel]; ax3=nothing, skipic=true, yticks=regions(:mac, :I)[sel])
        ax999.width[] = nothing
        #ax999.yticks[] = (axes(rgns, 1), [zz1data.areaname(rgn) for rgn in rgns])
    end
end

selectrows_mac_temp_pari() = begin
    fRSC = calc_proj_of_neuron("nmtI", "RSC")
    fLIP = calc_proj_of_neuron("nmtI", "LIP")
    fTemporal = calc_proj_of_neuron("nmtI", "Temporal")
    r -> begin
        (; RSC=fRSC(r), LIP=fLIP(r), Temporal=fTemporal(r))
    end
end
selectrows_mus_temp_pari() = begin
    fRSC = calc_proj_of_neuron("allenI", "RSP")
    fLIP = calc_proj_of_neuron("allenI", "PTLp")
    fTemporal = calc_proj_of_neuron("allenI", "TEa")
    r -> begin
        (; RSC=fRSC(r), LIP=fLIP(r), Temporal=fTemporal(r))
    end
end

function plot_compare_pari_bp(grid, env; mac, mus, macarbor, musarbor, _...)
    mus = insertcols(mus, :projpat => selectrows_mus_temp_pari().(eachrow(mus)))
    filter!(r -> zz1data.istypit(r) && sum(r.projpat) == 1 && !r.projpat.Temporal, mus)

    ax = Axis(grid[1, 1][1, 1], width=58,)
    content(grid[1, 1]).alignmode = Outside()
    musrgn = [:ORBvl, :ORBl, :ORBm, :MO, :PL, :ILA, :AId, :AIv] #= :ACA,  =#
    dat = Dict()
    for (iii, sss) in enumerate(musrgn)
        dfx = filter(r -> zz1data.matchhier("$sss")(r.somahier), mus)
        vvv = zz1data.vote(dfx, :projpat, :path)
        projpat = argmax.(first.(vvv))
        length(projpat) == 2 && push!(dat, iii => NamedTuple(zip(projpat, last.(vvv))))
        barplot!(ax, fill(iii, length(vvv)), n_dodge=2, dodge=[(; RSC=1, LIP=2)[p] for p in projpat], last.(vvv), color=[(; RSC=:red, LIP=:blue)[p] for p in projpat], bar_labels=:y, direction=:x, flip_labels_at=40, label_formatter=v -> string(Int(v)), dodge_gap=0)
    end
    ax.yticks[] = (axes(musrgn, 1), string.(musrgn))
    ax.yticklabelrotation[] = π / 2
    ax.xlabel = "No. of neurons"
    ax.ylabel = "Brain areas"
    ax.xminorticksvisible = false
    xlims!(ax, 0, nothing)
    ax.yreversed[] = true
    #hidexdecorations!(ax)
    #hidespines!(ax, :b)
    # ax.topspinevisible[] = true
    # ax.bottomspinevisible[] = true
    axislegend(ax, [PolyElement(color=c) for c in [:red, :blue]], ["to RSP", "to PTLp"], position=:rb)
    hideydecorations!(ax, ticklabels=false, label=false)
    mmmm = 0
    for (i, j) in [1 => 2]
        d1, d2 = dat[i], dat[j]
        vals = (d1.RSC, d2.RSC, d1.LIP, d2.LIP)
        mval = maximum(vals)
        t = HypothesisTests.FisherExactTest(vals...)
        p = HypothesisTests.pvalue(t)
        if p < 0.05
            write_stat(Symbol(:MousePairedRegion, musrgn[i], musrgn[j], :Pvalue), p)
            @warn "stasssss $i $j $p"
            bracket!(ax, mval, j, mval, i, offset=1, text="$(compact_pval(p))", textoffset=1, width=1.5, orientation=:down)
            mmmm = max(mmmm, mval)
        end
    end
    mmmm != 0 && xlims!(ax, 0, mmmm + 6)

    ax2 = LScene(grid[2, 1])
    rowgap!(grid, 1, 2)
    setup_camera(ax2 => :allenhr)
    musobj = load("interm/allen-997.obj")
    draw_mesh_overlay([musobj], ax2)
    areas = [:ORBvl => (i=746, c=:black), :ORBl => (i=723, c=:gray),]
    draw_contours(["interm/allen-$(i.i).obj" => i.c for (ba, i) in areas], ax2)
    ax2.height = pt(60)
    axislegend(ax2, [PolyElement(color=i.c) for (ba, i) in areas], [string(ba) for (ba, i) in areas], position=:rb)
end

function plot_compare_pari_proj(grid, env; mac, mus, macarbor, musarbor, _...)
    selectrows_mus(r) = begin
        !zz1data.istypit(r) && return 0
        a = calc_proj_of_neuron(r, "allenI", "RSP")
        b = calc_proj_of_neuron(r, "allenI", "PTLp")
        c = calc_proj_of_neuron(r, "allenI", "TEa")
        a + b + c != 1 && return 0
        a && return 1
        b && return 2
        #c && return 3
        return 0
    end

    ax2 = Axis(grid[1, 1], width=300)
    ax = Axis(grid[2, 1], width=300, height=60)
    musrgn = [:ORBvl, :ORBl, :ORBm, :MO, :PL, :ILA, :AId, :AIv] #= :ACA,  =#
    dfx = vcat([insertcols!(filter(r -> zz1data.matchhier("$sss")(r.somahier), mus), :somargn => sss) for sss in musrgn]...)
    boxplot_and_compare!(ax, dfx, ax2; rowx=r -> something(findfirst(==(r.somargn), musrgn)), xname=x -> string(musrgn[x]), rowy=r -> r["allenI|root|grey|CH|CTX|CTXpl|Isocortex|RSP"], rowdodge=nothing, rowcolor=nothing, statif=(a, b) -> a[1] != b[1], legendpos=nothing,)
    ax.xticks[] = (axes(musrgn, 1), string.(musrgn))
end

function plot_compare_tempparix_pari_sccc(grid, env; mac, mus, macarbor, musarbor, _...)
    objs = (; mac=load("interm/mesh-nmt-brain-lr.obj"), mus=load("interm/allen-997.obj"))

    selectrows(r, tag, aaa, bbb) = begin
        zz1data.istypit(r) || return 0
        zz1data.matchhier("ACA")(r.somahier) && return 0
        zz1data.matchhier("MO")(r.somahier) && return 0
        zz1data.matchhier("AI")(r.somahier) && return 0
        (zz1data.matchhier("ORBvl")(r.somahier) || zz1data.matchhier("ORBl")(r.somahier)) || return 0
        x = filter(rr -> rr.path == r.path, (; mac=macarbor, mus=musarbor)[tag])
        if tag === :mus
            conda(r) = r["allenI|root|grey|CH|CTX|CTXpl|Isocortex|RSP"] > 10
            condb(r) = r["allenI|root|grey|CH|CTX|CTXpl|Isocortex|PTLp"] > 10
            if conda(r) && condb(r)
                return 0
            elseif conda(r)
                return 0
            elseif condb(r)
                return 1
            else
                return 0
            end
        end
        aaa = Set(filter(r -> zz1data.matchhier(aaa)(r.arborhier) && r.ipsi, x).path)
        bbb = Set(filter(r -> zz1data.matchhier(bbb)(r.arborhier) && r.ipsi, x).path)
        if !isempty(aaa) && !isempty(bbb)
            return 0
        elseif !isempty(aaa)
            return 1
        elseif !isempty(bbb)
            return 2
        end
        0
    end

    #musdl, musvl = "ORBm", "ORBvl"
    musdl, musvl = "ORBvl", "ORBl"
    selectrows(r, tag, tgts...) = begin
        #zz1data.matchhier((; mac="dlPFC", mus=musdl)[tag])(r.somahier) && return 1
        #zz1data.matchhier((; mac="vlPFC", mus=musvl)[tag])(r.somahier) && return 2
        0
    end
    for (row, ff) in enumerate([
        (; mac=["RSC", "LIP"], mus=["RSP", "PTLp"]),
        #(; mac=["Temporal"], mus=["TEa"]),
    ])
        ax3 = Axis(grid[row, 1][1, 3], width=80)
        for (col, tag) in enumerate([:mac, :mus])
            ax1 = LScene(grid[row, 1][1, col], width=pt(120), height=pt(120), show_axis=false)
            setup_camera(ax1 => (; mac=:s, mus=:allens)[tag])
            with_mixed_scene(ax1, GLMakie) do mix1
                dfxxx = (; mac, mus)[tag]
                dfx = insertcols(dfxxx, :sel => [selectrows(r, tag, ff[tag]...) for r in eachrow(dfxxx)])
                filter!(r -> r.sel > 0, dfx)
                rowlabel(r) = (; mac=("dlPFC", "vlPFC"), mus=(musdl, musvl))[tag][r.sel]
                rowcolor(r) = (:red, :blue)[r.sel]
                if tag === :mus
                end
                #dfx = pick_neurons_with_limit(dfx, r -> r.sel, 20)
                sort!(dfx, :sel)

                inspect = (; mac=:nmt, mus=:allen)[tag]
                inspect = nothing
                sts = []
                for sel in sort(unique(dfx.sel))
                    dfxx = filter(r -> r.sel == sel, dfx)
                    n = size(dfxx, 1)
                    if tag === :mus
                        vals = [mirror_by(Allen, p)[3] for p in dfxx.soma_pos]
                        push!(sts, vals)
                        density!(ax3, vals, color=([:red, :blue][sel], 0.5))
                    end
                    @warn "$sel $n"
                end
                if length(sts) == 2
                    t = HypothesisTests.MannWhitneyUTest(sts...)
                    @warn "stat $t"
                end
                draw_neurons(dfx, mix1; rowlabel, rowcolor, mirrorby=(; mac=NmtDat, mus=Allen)[tag], inspect)
                #axislegend(ax1, mix1.scene2, unique=true) |> fix_line_legs
            end
            draw_mesh_overlay([objs[tag]], ax1)
            if tag === :mac
                draw_contours(["interm/mesh-nmt-$ba-l.obj" for ba in [:LIP, :RSC]], ax1)
            else
                draw_contours(["interm/allen-$ba.obj" for ba in [254, 22]], ax1)
            end
        end
    end
end

function plot_compare_tempparix_prev(grid, env; mac, mus, macarbor, musarbor, _...)
    objs = (; mac=load("interm/mesh-nmt-brain-lr.obj"), mus=load("interm/allen-997.obj"))
    #musdl, musvl = "ORBm", "ORBvl"
    musdl, musvl = "ORBvl", "ORBl"
    selectrows(r, tag, tgts...) = begin
        zz1data.istypit(r) || return 0
        x = filter(rr -> rr.path == r.path, (; mac=macarbor, mus=musarbor)[tag])
        sum([count(rr -> zz1data.matchhier(tgt)(rr.arborhier), eachrow(x)) for tgt in tgts]) > 0 || return 0
        zz1data.matchhier((; mac="dlPFC", mus=musdl)[tag])(r.somahier) && return 1
        zz1data.matchhier((; mac="vlPFC", mus=musvl)[tag])(r.somahier) && return 2
        0
    end
    for (row, ff) in enumerate([
        (; mac=["Parietal"], mus=["RSP", "PTLp"]),
        (; mac=["Temporal"], mus=["TEa"]),
    ])
        for (col, tag) in enumerate([:mac, :mus])
            ax1 = LScene(grid[row, 1][1, col], width=pt(120), height=pt(120), show_axis=false)
            setup_camera(ax1 => (; mac=:h, mus=:allenh)[tag])
            with_mixed_scene(ax1, GLMakie) do mix1
                dfxxx = (; mac, mus)[tag]
                dfx = insertcols(dfxxx, :sel => [selectrows(r, tag, ff[tag]...) for r in eachrow(dfxxx)])
                filter!(r -> r.sel > 0, dfx)
                rowlabel(r) = (; mac=("dlPFC", "vlPFC"), mus=(musdl, musvl))[tag][r.sel]
                rowcolor(r) = (:red, :blue)[r.sel]
                dfx = pick_neurons_with_limit(dfx, r -> r.sel, 10)
                sort!(dfx, :sel)
                draw_neurons(dfx, mix1; rowlabel, rowcolor, inspect=(; mac=:nmt, mus=:allen)[tag], mirrorby=(; mac=NmtDat, mus=Allen)[tag])
                axislegend(ax1, mix1.scene2, unique=true) |> fix_line_legs
            end
            draw_mesh_overlay([objs[tag]], ax1)
        end
    end
end


function plot_compare_arborsx(grid, env; mac, macarbor, mus, musarboregs, musarbor, _...)
    rgns = common_regions(:)
    rgns = unique([(; r.mus, r.mac) for r in rgns])
    filter!(r -> r.mac isa AbstractString && match(r"\|(TE|RSC|LIP)$", r.mac) !== nothing, rgns)
    @warn "$rgns"
    for (row, rgn) in enumerate(rgns)
        ax = Axis(grid[1, row])
        ax.height[] = pt(40)
        dats = Dict()
        for (spe, df) in [(:mus, musarbor), (:mac, macarbor)]
            dfx = filter(r -> zz1data.matchhier(rgn[spe])(r.arborhier) && r.ipsi, df)
            leftjoin!(dfx, (; mac, mus)[spe], on=:path) |> dropmissing!
            filter!(r -> zz1data.istypit(r), dfx)
            if spe === :mus
                filter!(r -> select_soma_mus_pfcit(r) > 0, dfx)
            end
            @warn "$(typeof(dfx.fv_whist))"
            dats[spe] = dfx
            xx, yy = calc_fvhist(dfx.fv_whist)
            spelong = (; mac=:Monkey, mus=:Mouse)[spe]
            specol = (; mac=:red, mus=:blue)[spe]
            Makie.lines!(ax, yy, xx, label="$spelong", color=specol)
            ax.title[] = replace(rgn.mac, r"^.*\|" => "")
            yaxis_as_layer(ax, 0, 1)
        end
        test_depth_dist(Symbol(:MouseVsMonkeyIn, rgn), dats[:mus].fv_whist, dats[:mac].fv_whist)
        if row == length(rgns)
            Legend(grid[1, row+1], ax, position=:rb) |> fix_line_legs
            colgap!(grid, row, 4)
        end
        #colsize!(grid, row, 50)
        if row > 1
            hideydecorations!(ax)
            colgap!(grid, row - 1, 2)
        end
        hidexdecorations!(ax)
        #hidespines!(ax)
    end
end

function plot_compare_arborsx_contra(grid, env; mac, macarbor, mus, musarbor, _...)
    ax = Axis(grid[1, 1])
    #ax.height[] = pt(40)
    for (row, src) in enumerate([:lat_PFC])
        dats = Dict()
        for (spe, df) in [(:mus, musarbor), (:mac, macarbor)]
            src2 = spe === :mac ? "lPFC" : "ORBvl/l"
            dfx = filter(r -> zz1data.matchhier((; mac="charm", mus="Isocortex")[spe])(r.arborhier) && !r.ipsi, df)
            leftjoin!(dfx, (; mac, mus)[spe], on=:path) |> dropmissing!
            filter!(r -> zz1data.istypit(r), dfx)
            if spe === :mac
                filter!(r -> zz1data.matchhier("$src")(r.somahier), dfx)
            end
            if spe === :mus
                filter!(r -> select_soma_mus_pfcit(r) > 0, dfx)
            end
            @warn "$(typeof(dfx.fv_whist))"
            dats[spe] = dfx
            xx, yy = calc_fvhist(dfx.fv_whist)
            spelong = (; mac=:Monkey, mus=:Mouse)[spe]
            specol = (; mac=:red, mus=:blue)[spe]
            linestyle = (; lat_PFC=:solid, vlPFC=:dot)[src]
            Makie.lines!(ax, xx, yy; label="$src2 → Contra. ($spelong)", color=specol, linestyle)
            #ax.title[] = "$src2 → $src2 (contra.)"
            xaxis_as_layer(ax, 0, 1)
        end
        test_depth_dist(Symbol(:MouseVsMonkeyIn, src), dats[:mus].fv_whist, dats[:mac].fv_whist)
        if row == 1
            Legend(grid[1, 2], ax, position=:rb, tellheight=false, tellwidth=false)
            colgap!(grid, 1, 4)
        end
        #row > 1 && colgap!(grid, row - 1, 2)
        hideydecorations!(ax, label=false)
        ax.ylabel = "Density"
        # ax.width = pt(80)
        ax.width = pt(100)
        #hidespines!(ax)
    end
end

function draw_density!(ax, df; rowx, rowy, rowcolor, kw...)
    xx = [rowx(r) for r in df]
    yy = [rowy(r) for r in df]
    cc = [rowcolor(r) for r in df]
    dl = get(kw, :datalimits, nothing)
    show_median = true
    mxx = median(yy)
    for a in @something(dl, (-Inf, Inf))
        @warn "nomedian? $mxx"
        if abs(mxx - a) < 1e-3
            show_median = false
        end
    end
    violin!(ax, xx, yy, color=[(c, 0.5) for c in cc], medianlinewidth=pt(1.0), mediancolor=cc, orientation=:horizontal; show_median, side=:right, kw...)
    ylims!(ax, -0.001, nothing)
    if dl !== nothing
        ##@xlims!(ax, dl[1] - 0.001, dl[2] + 0.001)
    end
end

function plot_compare_symm(grid, env; mus, mac, _...)
    mac1 = filter(r -> zz1data.istypit(r), mac)
    for ic in (:I, :C)
        insertcols!(mac1, Symbol(:ctx, ic) => mac1[:, "nmt$ic|charm"])
    end
    mac2 = filter(r -> r.ctxC > 10, mac1)
    mus1 = filter(r -> zz1data.istypit(r) && select_soma_mus_pfcit(r) > 0, mus)
    for ic in (:I, :C)
        insertcols!(mus1, Symbol(:ctx, ic) => mus1[:, "allen$ic|root|grey|CH|CTX"])
    end
    mus2 = filter(r -> r.ctxC > 10, mus1)

    spedat = (;
        Monkey=(; df1=mac1, df2=mac2, col=:red,),
        Mouse=(; df1=mus1, df2=mus2, col=:blue,),
        Null=(; col=:gray),
    )

    for (i, tag) in enumerate([:Monkey, :Mouse])
        dat = spedat[tag]
        ax = Axis(grid[1, 1][1, i], aspect=DataAspect(),)
        i > 1 && colgap!(content(grid[1, 1]), i - 1, 2)
        ncontra = size(dat.df2, 1)
        nipsi = size(dat.df1, 1) - size(dat.df2, 1)
        write_stat(Symbol(:CortexContraProportion, tag), ncontra / (ncontra + nipsi) * 100)
        pie!(ax, [ncontra, nipsi], color=[dat.col, :gray])
        hidedecorations!(ax)
        hidespines!(ax)
        #limits!(ax, -1, 1, -1, 1)
        Label(grid[1, 1][1, i, Top()], text="$tag")
    end
    leginfo = map([:Monkey, :Mouse, :Null]) do tag
        dat = spedat[tag]
        p = tag === :Null ? "without " : "with "
        s = tag === :Null ? "" : " ($tag)"
        PolyElement(color=dat.col) => "$(p)Contralateral$s"
    end
    Legend(grid[1, 1][2, :], first.(leginfo), last.(leginfo), valign=:top, halign=:left)
    rowgap!(content(grid[1, 1]), 1, 2)

    ax3 = Axis(grid[1, 2][1, 1])
    content(grid[1, 2]).alignmode = Outside()
    for (i, tag) in enumerate([:Monkey, :Mouse])
        dat = spedat[tag]
        symm = (dat.df2.ctxC .- dat.df2.ctxI) ./ (dat.df2.ctxC .+ dat.df2.ctxI)
        eee = extrema(symm)
        write_stat(Symbol(:CortexBiasIndexMedian, tag), median(symm))
        draw_density!(ax3, symm; rowx=r -> 0, rowy=r -> r, rowcolor=r -> dat.col, datalimits=(-1, 1),)
    end
    #ax3.yreversed[] = true
    ax3.xlabel[] = "Hemispheric bias"
    ax3.ylabel[] = "Desity"
    hidexdecorations!(ax3, label=false, ticklabels=false, ticks=false)
    hideydecorations!(ax3, label=false, ticklabels=false, ticks=false)
    Label(grid[0, :], "Neurons with bi-lateral cortical projection"; set_font_h2x()...)
    rowgap!(grid, 1, 1)
end

function plot_compare_symm_str(grid, env; mus, mac, _...)
    mac1 = filter(r -> zz1data.istypit(r), mac)
    for ic in (:I, :C)
        insertcols!(mac1, Symbol(:str, ic) => mac1[:, "nmt$ic|sarm|tel|BG|Str"])
    end
    filter!(r -> r.strI + r.strC > 20, mac1)
    mac2 = filter(r -> r.strC > 10, mac1)
    mus1 = filter(r -> zz1data.istypit(r) && select_soma_mus_pfcit(r) > 0, mus)
    for ic in (:I, :C)
        insertcols!(mus1, Symbol(:str, ic) => mus1[:, "allen$ic|root|grey|CH|CNU|STR"])
    end
    filter!(r -> r.strI + r.strC > 20, mus1)
    mus2 = filter(r -> r.strC > 10, mus1)

    spedat = (;
        Monkey=(; df1=mac1, df2=mac2, col=:red,),
        Mouse=(; df1=mus1, df2=mus2, col=:blue,),
        Null=(; col=:gray),
    )

    for (i, tag) in enumerate([:Monkey, :Mouse])
        dat = spedat[tag]
        ax = Axis(grid[1, 1][1, i], aspect=DataAspect(),)
        i > 1 && colgap!(content(grid[1, 1]), i - 1, 2)
        ncontra = size(dat.df2, 1)
        nipsi = size(dat.df1, 1) - size(dat.df2, 1)
        write_stat(Symbol(:StriatumContraProportion, tag), ncontra / (ncontra + nipsi) * 100)
        pie!(ax, [ncontra, nipsi], color=[dat.col, :gray])
        hidedecorations!(ax)
        hidespines!(ax)
        #limits!(ax, -1, 1, -1, 1)
        Label(grid[1, 1][1, i, Top()], text="$tag")
    end
    leginfo = map([:Monkey, :Mouse, :Null]) do tag
        dat = spedat[tag]
        p = tag === :Null ? "without " : "with "
        s = tag === :Null ? "" : " ($tag)"
        PolyElement(color=dat.col) => "$(p)Contralateral$s"
    end
    Legend(grid[1, 1][2, :], first.(leginfo), last.(leginfo), valign=:top, halign=:left)
    rowgap!(content(grid[1, 1]), 1, 2)

    ax3 = Axis(grid[1, 2][1, 1])
    colgap!(grid, 1, 4)
    content(grid[1, 2]).alignmode = Outside()
    for (i, tag) in enumerate([:Monkey, :Mouse])
        dat = spedat[tag]
        symm = (dat.df2.strC .- dat.df2.strI) ./ (dat.df2.strC .+ dat.df2.strI)
        eee = extrema(symm)
        @warn "$eee"
        write_stat(Symbol(:StriatumBiasIndexMedian, tag), median(symm))
        draw_density!(ax3, symm, rowx=r -> 0, rowy=r -> r, rowcolor=r -> dat.col, datalimits=(-1, 1))
        #ecdfplot!(ax3, symm, color=dat.col)
    end
    ax3.xlabel[] = "Hemispheric bias"
    ax3.ylabel[] = "Desity"
    hidexdecorations!(ax3, label=false, ticklabels=false, ticks=false)
    hideydecorations!(ax3, label=false, ticklabels=false, ticks=false)
    Label(grid[0, :], "Neurons with bi-lateral striatum projection"; set_font_h2x()...)
    rowgap!(grid, 1, 1)
end

function plot_compare_temppari_hm(grid, env; mac, mus, _...)
    ax1 = Makie.Axis(grid[1, 1])
    ax2 = Makie.Axis(grid[1, 2])
    ax3 = Makie.Axis(grid[2, 1])
    ax4 = Makie.Axis(grid[2, 2])
    ax5 = Makie.Axis(grid[3, 1])
    ax6 = Makie.Axis(grid[3, 2])
    regions = get_regions(:; ff=r -> r.mac isa AbstractString && r.mus isa AbstractString)
    macx = filter(r -> zz1data.select_neurons(:long_it_ipsi)(r), mac)
    musx = filter(r -> r.cls in [6, 21], mus)
    draw_neuron_proj_hm!(ax3, ax5, macx, [regions(:mac, :I); regions(:mac, :C)]; ax3=ax1)
    draw_neuron_proj_hm!(ax4, ax6, musx, [regions(:mus, :I); regions(:mus, :C)]; ax3=ax2)
    Makie.linkyaxes!(ax1, ax2)
    Makie.linkyaxes!(ax3, ax4)
    Makie.linkyaxes!(ax5, ax6)
end

function plot_compare_ctpt(grid, env; mac, mus, _...)
    regions = get_regions(:; ff=r -> contains("sarm")(r.mac))
    sel = include("monkeytemp/wyz-upload/julia/target-selection-ctpt-compare.jl")
    sel = map(v -> v in vec(sel), [regions(:mac, :I); regions(:mac, :C)])
    macx = filter(r -> (zz1data.select_neurons(:ct_strict)(r) || zz1data.select_neurons(:pt_strict)(r)) && !istypit(r), mac)
    musx = filter(r -> !istypit(r) && select_soma_mus_pfc_ctpt(r) > 0, mus)
    typ2tmpcls = Dict()
    for df in (macx, musx)
        for r in eachrow(df)
            r.cls = 1000 - get!(() -> length(typ2tmpcls) + 1, typ2tmpcls, r.typ)
        end
    end

    ax1 = Makie.Axis(grid[1, 1])
    ax2 = Makie.Axis(grid[1, 2])
    Label(grid[1, 1, Top()], "Monkey"; set_font_h2x()...)
    Label(grid[1, 2, Top()], "Mouse"; set_font_h2x()...)
    ax3 = Makie.Axis(grid[2, 1])
    ax4 = Makie.Axis(grid[2, 2])
    ax5 = Makie.Axis(grid[3, 1])
    ax6 = Makie.Axis(grid[3, 2])
    ax4x = Axis(grid[2, 3])
    ax6x = Axis(grid[3, 3])
    colgap!(grid, 2, 1)
    typ2c = typ2color_func(macx)
    plot_subtyp_spans!(ax1, macx; linkto=ax3, gridto=[ax3, ax5], rowgrp=r -> r.typ, grplabel=(i, g) -> "$g", grpcolor=(i, g) -> typ2c(g), label="")
    draw_neuron_proj_hm!(ax3, ax5, macx, [regions(:mac, :I); regions(:mac, :C)][sel]; skipic=true, yticks=:ignore)
    plot_subtyp_spans!(ax2, musx; linkto=ax4, gridto=[ax4, ax6], rowgrp=r -> r.typ, grplabel=(i, g) -> "$g", grpcolor=(i, g) -> typ2c(g), label="Type")
    #ax3.width[] = ax3.width[] / 2
    ax3.width[] = nothing
    draw_neuron_proj_hm!(ax4, ax6, musx, [regions(:mus, :I); regions(:mus, :C)][sel]; skipic=true, ax1x=ax4x, ax2x=ax6x, yticks=[regions(:mac, :I); regions(:mac, :C)][sel])
    #ax4.width[] = ax4.width[] / 2
    ax4.width[] = nothing
    Makie.linkyaxes!(ax1, ax2)
    Makie.linkyaxes!(ax3, ax4)
    Makie.linkyaxes!(ax5, ax6)
    colgap!(grid, 1, 2)
    rowgap!(grid, 1, 0)
    rowgap!(grid, 2, 0)
end

function plot_compare_len_model(df)
    func(thr) = begin
        #dfx=filter(r->zz1data.istypit(r), df)
        dfx = filter(r -> r.typ[1:2] == "CT", df)
        calc_arbor_stats!(dfx; thr)
        x1 = dfx.nbranches
        x2 = dfx.collcnt
        lr = LinearRegression.linregress(hcat(x1, x2), dfx.axonlen)
        pred = x1 .* lr.coeffs[1] .+ x2 .* lr.coeffs[2] .+ lr.coeffs[3]
        ll = sum((pred .- dfx.axonlen) .^ 2)
        @info "lr $thr $ll $lr"
        ll
    end
    argmin(func, 100:100:6000)
    # CT [ Info: lr 1400 3.495072676477679e9 LinearRegression.LinearRegressor{Vector{Float64}}(true, [248.22286823006027, 3745.3184901742047, 45817.2213864416])
    # PT [ Info: lr 2200 7.500974797799017e10 LinearRegression.LinearRegressor{Vector{Float64}}(true, [236.38324407592395, 8432.467561437012, 46883.296153115996])
    # IT [ Info: lr 1900 1.1117016345392346e12 LinearRegression.LinearRegressor{Vector{Float64}}(true, [298.9296967852815, 6333.104393970342, 19624.03879046412])
    #
    # CT [ Info: lr 600 2.4777993859974503e10 LinearRegression.LinearRegressor{Vector{Float64}}(true, [6.888489777329651, 1548.1912164027683, 11845.945989607551])
    # PT [ Info: lr 500 1.7203587530452325e11 LinearRegression.LinearRegressor{Vector{Float64}}(true, [6.117760290776652, 2007.3867641367833, 19074.57686720223])
    # IT [ Info: lr 700 6.204645073805602e11 LinearRegression.LinearRegressor{Vector{Float64}}(true, [45.45405457237133, 2016.7447888374058, 12899.295907012955])


end

function plot_compare_ctpt2(grid, env; mac, mus, _...)
    typ2c = typ2color_func(mac)
    regions = get_regions(:; ff=r -> contains("sarm")(r.mac))
    sel = include("monkeytemp/wyz-upload/julia/target-selection-ctpt-compare.jl")
    sel = map(v -> v in vec(sel), [regions(:mac, :I); regions(:mac, :C)])
    macx = filter(r -> (zz1data.select_neurons(:ct_strict)(r) || zz1data.select_neurons(:pt_strict)(r)) && !istypit(r), mac)
    musx = filter(r -> !istypit(r) && select_soma_mus_pfc_ctpt(r) > 0, mus)
    for dfx in (macx, musx)
        for r in eachrow(dfx)
            r.cls = r.typ == "CT" ? 999 : 998
        end
    end
    gridx = GridLayout(grid[1, 1])

    rgns = collect(zip(regions(:mac, :I) .=> regions(:mac, :C), regions(:mus, :I) .=> regions(:mus, :C)))[sel[1:end÷2]]
    for (j, (macrgnp, musrgnp)) in enumerate(rgns)
        ax1 = Axis(gridx[j, 1])
        ax2 = Axis(gridx[j, 2])
        j > 1 && rowgap!(gridx, j - 1, 0)
        getprojcol(df, col) = begin
            col in names(df) && return df[:, col]
            zeros(size(df, 1))
        end
        for (ax, tag) in [ax1 => :Monkey, ax2 => :Mouse]
            dfx = (; Mouse=musx, Monkey=macx)[tag]
            rgnp = (; Mouse=musrgnp, Monkey=macrgnp)[tag]
            subtyps = sort(unique(collect(zip(dfx.typ, dfx.cls))))
            for (dod, rgn) in enumerate(rgnp)
                proj = getprojcol(dfx, rgn)
                q95 = quantile(proj, 0.95)
                datalimits = (0, Inf)
                @info "$datalimits"
                cls2typ = Dict()
                for (jj, (typ, cls)) in enumerate(subtyps)
                    #proj = min.(filter(r -> r.cls == cls, dfx)[:, rgn], q95)
                    dfxxx = filter(r -> r.cls == cls, dfx)
                    clsproj = getprojcol(dfxxx, rgn)
                    otherproj = getprojcol(filter(r -> r.cls != cls, dfx), rgn)
                    t = HypothesisTests.MannWhitneyUTest(proj, clsproj)
                    strokecolor = HypothesisTests.pvalue(t) < 0.05 && median(clsproj) > median(proj) ? :red : :black
                    violin!(ax, fill(jj, length(clsproj)), log10.(clsproj .+ 1); datalimits, color=:gray, strokewidth=pt(0.5), strokecolor, dodge=fill(dod, length(clsproj)), n_dodge=2)
                end
            end
            ylims!(ax, 0, nothing)
            ax.height[] = pt(12)
            hideydecorations!(ax)
            hidespines!(ax, :l)
            if j == length(rgns)
                ax.xticks[] = (axes(subtyps, 1), [rich("$t", color=typ2c(t)) for (t, c) in subtyps])
                hidexdecorations!(ax, ticklabels=false)
            else
                hidexdecorations!(ax)
            end
            #rotatexlabel!(ax,)
        end
        n1 = areaname(macrgnp.first)
        n2 = areaname(musrgnp.first)
        if lowercase(n2) != lowercase(n1)
            n1 = "$n1 ($n2)"
        end
        Label(gridx[j, 3], text=n1, halign=:left)
    end
    colgap!(gridx, 1, 2)
    colgap!(gridx, 2, 2)
    Label(gridx[end+1, 1], text="Monkey", tellwidth=false)
    Label(gridx[end, 2], text="Mouse", tellwidth=false)
    rowgap!(gridx, length(rgns), 2)
    return
    ax1 = Makie.Axis(grid[1, 1])
    ax2 = Makie.Axis(grid[1, 2])
    ax3 = Makie.Axis(grid[2, 1])
    ax4 = Makie.Axis(grid[2, 2])
    ax5 = Makie.Axis(grid[3, 1])
    ax6 = Makie.Axis(grid[3, 2])
    draw_neuron_proj_hm!(ax3, ax5, macx, [sel]; ax3=ax1)
    ax3.width[] = ax3.width[] / 2
    draw_neuron_proj_hm!(ax4, ax6, musx, [sel]; ax3=ax2)
    ax4.width[] = ax4.width[] / 2
    Makie.linkyaxes!(ax1, ax2)
    Makie.linkyaxes!(ax3, ax4)
    Makie.linkyaxes!(ax5, ax6)
end

function plot_compare_ptct_numtgt(grid, env, tgttyp="PT"; mac, mus, macarbor, musarbor, _...)
    typ2c = typ2color_func(mac)
    regions = get_regions(:; ff=r -> contains("sarm")(r.mac))
    sel = include("monkeytemp/wyz-upload/julia/target-selection-ctpt-compare.jl")
    sel = map(v -> v in vec(sel), [regions(:mac, :I); regions(:mac, :C)])
    macx = filter(r -> (zz1data.select_neurons(:ct_strict)(r) || zz1data.select_neurons(:pt_strict)(r)) && !istypit(r), mac)
    musx = filter(r -> !istypit(r) && select_soma_mus_pfc_ctpt(r) > 0, mus)
    rgns = collect(zip([regions(:mac, :I); regions(:mac, :C)], [regions(:mus, :I); regions(:mus, :C)]))[sel]
    rgns = vcat(first.(rgns), last.(rgns))

    path2tgts = Dict{String,Set{String}}()
    tag2col = (; Mouse=:blue, Monkey=:red)
    for dfarbor in [macarbor, musarbor]
        for r in eachrow(dfarbor)
            ss = get!(() -> Set(), path2tgts, r.path)
            push!(ss, r.arborhier)
            votes = get(r, :hiervotes, nothing)
            @assert votes !== nothing
            if votes !== nothing
                for (h, l) in votes
                    l >= 10 && push!(ss, h)
                end
            end
        end
    end

    remove_ic(rgn) = replace(rgn, r"^(\w+)[IC]\|" => s"\1|")

    dats = []
    for typ in ["PT", "CT"]
        for tag in [:Monkey, :Mouse]
            dfx = (; Mouse=musx, Monkey=macx)[tag]
            dfx = filter(r -> r.typ == typ, dfx)
            for r in eachrow(dfx)
                tgts = get(path2tgts, r.path, [])
                numtgt = count(unique(remove_ic.(rgns))) do rgn
                    n = count(tgts) do t
                        zz1data.matchhier(rgn)(t)
                    end
                    n > 0
                end
                # for rgn in rgns
                #     if get(r, rgn, 0) > 10
                #         push!(ttt, replace(rgn, r"^(\w+)[IC]\|" => s"\1|"))
                #     end
                # end
                push!(dats, (; r.path, numtgt, tag, typ))
            end
        end
    end
    dats = DataFrame(dats)

    for (i, typ) in enumerate(["PT", "CT"])
        for (j, tag) in enumerate([:Monkey, :Mouse])
            ax = Axis(grid[j, i])
            ax.height = pt(22)
            cnts = zz1data.vote(filter(r -> r.typ == typ && r.tag == tag, dats), :numtgt, :path)
            barplot!(ax, first.(cnts), last.(cnts), color=tag2col[tag], gap=0)
            xlims!(ax, -0.5, nothing)
            ylims!(ax, 0, nothing)
            if j == 2
                hidexdecorations!(ax, ticks=false, ticklabels=false, label=false)
            else
                hidexdecorations!(ax, ticks=false)
            end
            if i == 1
                hideydecorations!(ax, ticks=false, ticklabels=false, label=false)
            else
                hideydecorations!(ax, ticks=false)
            end
            ax.xlabel[] = "No. of target areas"
            if j == 1
                Label(grid[j, i, Top()], "$typ"; set_font_h2x()...)
            end
            ax.ylabel[] = "Count"
        end
    end
    for c in 1:2
        linkxaxes!(content(grid[1, c]), content(grid[2, c]))
        if c == 1
            align_axis_label(:y, content(grid[1, c]), content(grid[2, c]))
        end
    end
    for c in 1:2
        linkyaxes!(content(grid[c, 1]), content(grid[c, 2]))
    end
    colgap!(grid, 1, 2)
    rowgap!(grid, 1, 2)
    Legend(grid[3, :], [PolyElement(color=tag2col[tag]) for tag in [:Monkey, :Mouse]], string.([:Monkey, :Mouse]), orientation=:horizontal)
    rowgap!(grid, 2, 2)
end

function plot_compare_ptct_symm(grid, env; mac, mus, macarbor, musarbor, _...)
    typ2c = typ2color_func(mac)
    regions = get_regions(:; ff=r -> contains("sarm")(r.mac))

    sel = include("monkeytemp/wyz-upload/julia/target-selection-ctpt-compare.jl")
    sel = map(v -> v in vec(sel), [regions(:mac, :I); regions(:mac, :C)])
    macx = filter(r -> (zz1data.select_neurons(:ct_strict)(r) || zz1data.select_neurons(:pt_strict)(r)) && !istypit(r), mac)
    musx = filter(r -> !istypit(r) && select_soma_mus_pfc_ctpt(r) > 0, mus)
    rgns = collect(zip([regions(:mac, :I); regions(:mac, :C)], [regions(:mus, :I); regions(:mus, :C)]))[sel]
    rgns = vcat(first.(rgns), last.(rgns))
    rgnsipsi = filter(v -> match(r"^\w+I\|", v) !== nothing, rgns)
    rgnscontra = filter(v -> match(r"^\w+C\|", v) !== nothing, rgns)
    getprojcol(df, col) = begin
        col in names(df) && return df[:, col]
        zeros(size(df, 1))
    end
    for df in (macx, musx)
        for (fff, rrr) in [:sarmipsi => rgnsipsi, :sarmcontra => rgnscontra]
            insertcols!(df, fff => reduce(+, [getprojcol(df, rr) for rr in rrr]))
        end
    end
    tag2col = (; Mouse=:blue, Monkey=:red)
    for (j, typ) in enumerate(["PT", "CT"])
        ax = Axis(grid[1, j])
        j > 1 && colgap!(grid, j - 1, 6)
        for (i, tag) in enumerate([:Monkey, :Mouse])
            dfx = filter(r -> r.typ == typ, (; Mouse=musx, Monkey=macx)[tag])
            symm = (dfx.sarmcontra .- dfx.sarmipsi) ./ (dfx.sarmcontra .+ dfx.sarmipsi)
            @warn "symm $(length(symm))"
            global xxxsymm = deepcopy(symm)
            symm = push!(vcat(symm, symm, symm, symm), -1, 1)
            plt = ecdfplot!(ax, symm, color=tag2col[tag])
            translate!(plt, 0, 0, 10000000)
            #draw_density!(ax, symm, rowx=r -> 0, rowy=r -> r, rowcolor=r -> tag2col[tag], datalimits=(-1, 1))
            ax.xlabel[] = "Hemispheric bias"
            text!(ax, 1, 0, space=:relative, text=typ, align=(:right, :bottom); set_font_h2x()...)
            ax.ylabel[] = "ECDF (%)"
            ax.ytickformat = (vv) -> [string(Int(v * 100)) for v in vv]
            xlims!(ax, -1.001, 1.001)
            ax.xticks = -1:1
            ylims!(ax, -0.001, nothing)
            hidexdecorations!(ax, label=false, ticklabels=false, ticks=false)
            hideydecorations!(ax, label=j != 1, ticklabels=j != 1, ticks=false)
        end
    end
    Legend(grid[2, :], [PolyElement(color=tag2col[tag]) for tag in [:Monkey, :Mouse]], string.([:Monkey, :Mouse]), orientation=:horizontal)
    rowgap!(grid, 1, 2)
end

get_regions(sel; ff=nothing) = begin
    pp = common_regions(sel)
    pp = filter(r -> r.mus isa AbstractString && r.mac isa AbstractString, pp)
    if ff !== nothing
        pp = filter(ff, pp)
    end
    pp = sort(unique(map(r -> (; r.mac, r.mus), pp)))
    sort(pp, by=v -> v[:mus])
    (spe, ic) -> begin
        [replace(t[spe], r"^(\w+)" => SubstitutionString("\\1$ic")) for t in pp]
    end
end

function calc_arbor_stats!(df; thr=6000, scale=1)
    df2 = map(eachrow(df)) do r
        cnt = 0
        len1, len2 = 0.0, 0.0
        for l in r.majorlens
            if l > thr || cnt == 0
                len1 += l
                cnt += 1
            else
                len2 += l
            end
        end
        len2 += r.nonmajorlen
        dec = min(len1, (cnt - 1) * thr)
        len1 -= dec
        len2 += dec
        colltot = (len1 - r.majorlens[1])
        (; r.path, colltot, collavg=colltot / (cnt - 1), collcnt=cnt - 1, arbortot=len2, arboravg=len2 / cnt / scale, arborrat=len2 / r.axonlen, collrat=colltot / r.axonlen, len1, len2, trunklen=r.majorlens[1])
    end
    leftjoin!(df, DataFrame(df2), on=:path)
end

function plot_compare_decomp(grid, env; mac, mus, _...)
    kk = 0
    for typ in sort(unique(mus.typ))
        kk += 1
        ax = Makie.Axis(grid[kk, 1])
        for (spe, df, s) in [(:Monkey, mac, 6), (:Mouse, mus, 1)]
            dfx = filter(r -> r.typ[1:2] == typ, df)
            get_data(i) = begin
                map(eachrow(dfx)) do r
                    (length(r.majorlens) >= i ? r.majorlens[i] : 0.0) / r.majorlens[1]
                end
            end
            xs = 1:40
            ys = get_data.(xs)
            s = median(ys[1])
            #ys = [d ./ s for d in ys]
            ys = [quantile(d, (0.25, 0.5, 0.75)) for d in ys]
            errorbars!(ax, xs, [d[2] for d in ys], [d[1] for d in ys], [d[3] for d in ys], whiskerwidth=10, label="$spe")
            lines!(ax, xs, [d[2] for d in ys])
        end
        Makie.axislegend(ax, ax)
    end
end

function plot_scatter_with_jitter!(ax, cats, vals; orientation=:vertical, jitter_width=0.5, gap=0.1, kwargs...)
    xvals = indexin(cats, sort(unique(cats)))
    xvals = xvals .- 1 .+ rand(length(xvals)) .* jitter_width
    if orientation === :horizontal
        xvals, vals = vals, xvals
    end
    Makie.scatter!(ax, xvals, vals; kwargs...)
end

using Dates
function plot_rec_progress(grid, env; stats, _...)
    tohour(p) = Dates.toms(p) / (1000 * 3600)
    tohour(p::DateTime) = tohour(p - DateTime(0))
    function accum_human_hour(catvals)
        annotators = unique(map(first, catvals))
        deltas = []
        for annotator in annotators
            peruser = sort(last.(filter(v -> first(v) == annotator, catvals)), by=v -> v.time)
            prevt = DateTime(0)
            usage = Second(0)
            for t in peruser
                dt = min(t.time - prevt, Second(isempty(t.usertag) ? 30 : 180))
                usage += dt
                push!(deltas, (t.time, dt))
                prevt = t.time
            end
            usage = tohour(usage)
            ###@info "user $user $usage H"
        end
        sort!(deltas)
        partialsums = []
        cursum = Second(0)
        for (t, dt) in deltas
            cursum += dt
            push!(partialsums, (t, tohour(cursum)))
        end
        first.(partialsums), last.(partialsums)
    end

    allcatvals = map(stats) do v
        if v.type == :tiff
            ("0conv", "0")
        elseif v.type == :webm
            ("0conv", "1")
        elseif v.type == :edit
            if v.edit == "add_patch_" || startswith(v.user, "bot")
                ("1auto", "all")
            else
                ("2manual", v.user)
            end
        else
            @assert false
        end => v
    end
    sort!(allcatvals, by=v -> (v.first, v.second.time))
    usedhours = accum_human_hour(filter(v -> v[1][1] == "2manual", allcatvals))
    all_cats = unique(first.(allcatvals))
    filter!(all_cats) do cat
        count(v -> first(v) == cat, allcatvals) > 8000
    end
    sel = indexin(first.(allcatvals), all_cats) .!== nothing
    catvals = allcatvals[sel]
    allcolors = Makie.wong_colors()
    roundtime(t) = round(t - DateTime(0), Minute(60)) + DateTime(0)
    t0, t1 = extrema(map(p -> p.second.time, allcatvals))
    t00 = DateTime(v -> dayofweek(v) == 1, year(t0), month(t0), day(t0)) - Hour(8)
    t11 = t1 - Day(2)
    xticks = range(t00, t11, step=Week(1))
    legends = []
    axvec = map(enumerate(unique(first.(all_cats)))) do (j, cat)
        catvals2 = filter(p -> p[1][1] == cat, catvals)
        vals = [p.second.time for p in catvals2]
        cats = first.(catvals2)
        colors = map(indexin(cats, all_cats)) do i
            i > 3 && return :black
            allcolors[i]
        end
        cats = last.(cats)
        ax = Makie.Axis(grid[j, 1], yreversed=j <= 2, yautolimitmargin=(0.00, 0.00))
        if j <= 2
            plot_scatter_with_jitter!(ax, cats, tohour.(vals); orientation=:horizontal, rasterize=6, color=colors, markersize=0.2, jitter_width=0.9, gap=0.01)
            hidexdecorations!(ax)
            ax.xgridvisible = true
            hidespines!(ax)
        else
            accum = Dict()
            foreach(cats, vals) do c, v
                k = (c, roundtime(v))
                accum[k] = get(accum, k, 0.0) + 1
            end
            alltimes = sort(unique(map(roundtime, range(t0, t1, step=Minute(2)))))
            for (i, c) in enumerate(unique(cats))
                plt = Makie.lines!(ax, tohour.(alltimes), [log(get(accum, (c, t), 0.0) + 1) for t in alltimes] ./ log(600) .+ (i - 1) .+ 0.01, color=:black, linewidth=pt(0.5))
                if i == 1
                    push!(legends, plt)
                end
                i >= 30 && break
            end
            Makie.ylims!(ax, 0, nothing)
            ax.xlabel[] = "Time (week)"
            hidexdecorations!(ax, ticks=false, ticklabels=false, label=false)
            hidespines!(ax)
            ax.xgridvisible = true

            ax2 = Makie.Axis(grid[j, 1], yticklabelcolor=(:red, 0.4), yaxisposition=:right, xautolimitmargin=(0.00, 0.00))
            plt = Makie.band!(ax2, tohour.(usedhours[1]), map(v -> 0, usedhours[2]), usedhours[2], color=(:red, 0.4))
            push!(legends, plt)
            Makie.translate!(plt, 0, 0, -1000)
            hidespines!(ax2, :l)
            ax2.rightspinevisible = true
            Makie.hidexdecorations!(ax2)
            Makie.hideydecorations!(ax2, ticks=false, ticklabels=false)
            Makie.linkxaxes!(ax, ax2)
            Makie.ylims!(ax2, 0, 7500)
            Makie.xlims!(ax2, tohour(t0), tohour(t1 - Day(3)))
            ax2.yticklabelrotation[] = -pi / 2
            ax2.yticksvisible[] = true
            ax2.yticks[] = 0:1500:7500
        end
        ticklabel(v) = string(round(v - t00, Week(1)).value)
        ax.xticks[] = (tohour.(xticks), ticklabel.(xticks))
        Makie.hideydecorations!(ax)
        Makie.rowsize!(grid, j, Makie.Auto(length(unique(cats)) * (j <= 2 ? 2.5 : 1)))
        if j > 1
            Makie.rowgap!(grid, j - 1, 2)
        end
        ax
    end
    Makie.linkxaxes!(axvec...)

    floor_n_user = (length(all_cats) - 1) ÷ 10 * 10
    dotmarker(i) = Makie.MarkerElement(marker=:circle, color=allcolors[i], strokecolor=:transparent)
    Makie.Legend(grid[0, :], [dotmarker.(1:2)..., dotmarker(3), legends...],
        ["Full scan of TIFF images", "On-demand WEBM cube generation", "Automatic reconstruction",
            "Manual proofreading by >$floor_n_user annotators", "Time usage (h)"],
        orientation=:horizontal, groupgap=20,
    )
    Makie.rowgap!(grid, 1, 3)
end

function plot_fig_result(fig, env; _...)
    with_caption(L"Projectome-based classification of macaque PFC single-neuron projectomes.")

    grid = fig[1, 1]
    with_subfig(grid, env) do grid, env
        with_caption(L"")
        plot_result_clust(grid[1, 1], env; env...)
    end

    grid = fig[2, 1]
    with_subfig(grid[1, 1][1, 1], env) do grid, env
        plot_neuron_func_alg(grid, env; env...)
    end
    gridy = with_subfig(grid[1, 1][2, 1], env) do grid, env
        grid
    end

    ax2 = with_subfig(grid[1, 2], env) do grid, env
        with_caption(L"")
        grid.alignmode = Mixed(top=0, bottom=0)
        ax2 = plot_func_table_per_cls(grid, env; env...)
        plot_neurons_func(gridy, env; env...)
        ax2
    end
end


function plot_fig_compare_itc_arbor(fig, env; mac, mus, macarbor, musarbor, _...)
    with_caption(L"Comparison to mouse PFC projectome.")
    ax = Axis(fig[1, 1])
    macarbor = leftjoin(macarbor, mac, on=:path) |> dropmissing!
    musarbor = leftjoin(musarbor, mus, on=:path) |> dropmissing!

    draw_arbor_depth(dfx, n) = begin
        xx, yy = calc_fvhist(dfx.fv_whist)
        @warn "$n $(size(dfx, 1))"
        lines!(ax, yy, xx, label="$n")
    end
    #filter!(r->r.sample=="234197", macarbor)
    dfx11 = filter(r -> r.typ[1:2] == "IT" && zz1data.matchhier("dlPFC")(r.somahier) && !r.ipsi && zz1data.matchhier("dlPFC")(r.arborhier), macarbor)
    draw_arbor_depth(dfx11, "Monkey dlPFC")
    dfx11 = filter(r -> r.typ[1:2] == "IT" && zz1data.matchhier("vlPFC")(r.somahier) && !r.ipsi && zz1data.matchhier("vlPFC")(r.arborhier), macarbor)
    draw_arbor_depth(dfx11, "Monkey vlPFC")

    dfx11 = filter(r -> r.typ[1:2] == "IT" && zz1data.matchhier("ORBl")(r.somahier) && !r.ipsi && zz1data.matchhier("ORBl")(r.arborhier), musarbor)
    draw_arbor_depth(dfx11, "Mouse ORBl")
    dfx11 = filter(r -> r.typ[1:2] == "IT" && zz1data.matchhier("ORBvl")(r.somahier) && !r.ipsi && zz1data.matchhier("ORBvl")(r.arborhier), musarbor)
    draw_arbor_depth(dfx11, "Mouse ORBvl")
    yaxis_as_layer(ax, 0, 1)

    Legend(fig[2, 1], ax) |> fix_line_legs


end

