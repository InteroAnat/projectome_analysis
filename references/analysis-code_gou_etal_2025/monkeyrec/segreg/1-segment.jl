import Images
import Interpolations

@rule :data "interm/segment-unet" let
    samples = [
        "data/seg-training/fmost234197.ch1@ch00-avg.tif" => [
            "/data/dat/dsimgs/fmost234197.ch1@ch00-avg.nrrd"
            "/data/dat/dsimgs/fmost234197.ch1@ch00-mip.nrrd"
        ]
    ]
    unet_siz = 512

    function prepare_samples(samples)
        map(samples) do (label, inputs)
            label = permutedims(FileIO.load(label), (2, 1, 3))
            @assert length(inputs) == 2
            @info "load inputs"
            outinputs = zeros(Float16, size(label)..., length(inputs))
            for (i, input) in enumerate(inputs)
                input = FileIO.load(input)
                @assert size(input) == size(label) "$(size(label)) != $(size(input)) "
                outinputs[:, :, :, i] .= input[:, :, :] ./ 256.0f0
            end

            label = [map(p -> getproperty(p, c), label) for c in (:r, :g, :b)]
            label = stack(label, dims=4)
            avg = Statistics.mean(label, dims=4)
            outlabel = BitArray(@. label > avg + 0.5f0)
            outlabel => outinputs
        end
    end
    samples = try
        @something Main.cache_unet_samples
    catch
        Main.cache_unet_samples = prepare_samples(samples)
    end
    GC.gc()

    function prepare_regions(samples)
        regions = []
        for (sampid, (label, input)) in enumerate(samples)
            label = sum(label, dims=4)[:, :, :, 1]
            for zi in 1:3
                xi, yi = setdiff(1:3, zi)
                #pooled=Flux.lpnormpool(label, 1, (d==zi ? 1 : 128 for d in 1:3))
                pooled = permutedims(label, (xi, yi, zi))
                idxvec = collect(Any, 1:3)
                @showprogress "sample regions" for z in axes(label, zi)
                    idxvec[zi] = z
                    for y in 1:(unet_siz÷3):(size(label, yi)-unet_siz÷2), x in 1:(unet_siz÷3):(size(label, xi)-unet_siz÷2)
                        idxvec[yi] = y:min(size(label, yi), y + unet_siz - 1)
                        idxvec[xi] = x:min(size(label, xi), x + unet_siz - 1)
                        hits = sum(view(pooled, idxvec[xi], idxvec[yi], idxvec[zi]))
                        if hits > 256
                            for dup in 1:(hits > 4096 ? 3 : 1)
                                push!(regions, (; sampid, index=(idxvec...,)))
                            end
                        end
                    end
                end
            end
        end
        Random.shuffle(regions)
    end

    all_regions = prepare_regions(samples)

    function get_slice_idx(index, off=0)
        map(v -> (v isa Integer ? v + off : v), index)
    end
    function flipswap(x, manipxy, rot)
        xrng = axes(x, 1)
        if manipxy[1]
            xrng = reverse(xrng)
        end
        yrng = axes(x, 2)
        if manipxy[2]
            yrng = reverse(yrng)
        end
        x = view(x, xrng, yrng, :)
        if manipxy[3]
            x = permutedims(x, (2, 1, 3))
        end
        siz = size(x)
        @assert length(siz) == 3
        x = stack((Images.imrotate(view(x, :, :, k), rot, method=Interpolations.Constant()).parent for k in axes(x, 3)), dims=3)
        x[isnan.(x)] .= 1.0f-3
        siz2 = size(x)
        a = max.((siz2 .- siz) .÷ 2 .+ 1, 1)
        b = min.(a .+ siz .- 1, siz2)
        view(x, a[1]:b[1], a[2]:b[2], 1:siz[3])
    end
    function get_training_sample(samples)
        nbat = 4
        samps = rand(all_regions, nbat)
        xvals = zeros(Float16, unet_siz, unet_siz, 2, 3, nbat)
        yvals = zeros(Float16, unet_siz, unet_siz, 4, nbat)
        for (sampi, samp) in enumerate(samps)
            label, input = samples[samp.sampid]
            scale = reshape(rand(Float16, 2) .* 0.5f0 .+ 0.75f0, 1, 1, :)
            flipseq = rand(Bool)
            manipxy = rand(Bool, 3)
            rot = rand(Float32) * pi
            for (oi, o) in enumerate(-1:1)
                s = flipswap(view(input, get_slice_idx(samp.index,)..., :) .* scale, manipxy, rot)
                ###@info "shape : $(size(s))"
                xvals[axes(s, 1), axes(s, 2), :, (flipseq ? oi : 4 - oi), sampi] .= s
            end
            for c in 1:3
                s = flipswap(view(label, samp.index..., c:c), manipxy, rot)
                yvals[axes(s, 1), axes(s, 2), c:c, sampi] .= s
            end
            yvals[:, :, 4, sampi] .= map(view(yvals, :, :, 1, sampi), view(yvals, :, :, 3, sampi)) do r, b
                r > 0.3f0 ? 0.0f0 : (b > 0.3f0 ? 1.0f0 : 0.5f0)
            end
        end
        reshape(xvals, unet_siz, unet_siz, :, nbat), yvals
    end

    function my_label_smoothing(y, α::Number; dims::Int=1)
        y .* (1 - α) .+ α * 1 // size(y, dims)
    end

    loss(m, x, y0, K, obs) = begin
        yy = m(x)
        dd = Flux.sigmoid_fast(view(yy, :, :, 4:4, :))
        yy = view(yy, :, :, 1:3, :)
        yyy = Flux.softmax(yy, dims=3)
        y, d, mskd, nd, yred, nyred = Flux.ignore_derivatives() do
            y000 = sum(x, dims=3) .== 0.0f0
            d0 = map(yyy[:, :, 1:1, :], yyy[:, :, 3:3, :]) do r, b
                r > 0.5f0 ? 0.0f0 : (b > 0.5f0 ? 1.0f0 : 0.5f0)
            end
            mskd0 = d0 .!= 0.5f0
            my_label_smoothing(y0[:, :, 1:3, :], 0.2, dims=3), d0, mskd0, sum(mskd0) + 1, y000, sum(y000) + 1
        end
        msk, nmsk, mskrb, nmskrb, mskr, nmskr = Flux.ignore_derivatives() do
            msk0 = sum(y, dims=3) .> 0.3f0
            msk1 = msk0 .- (view(y, :, :, 2:2, :) .> 0.3f0)
            msk2 = view(y, :, :, 1:1, :) .> 0.3f0
            msk0, sum(msk0) + 1, msk1, sum(msk1) + 1, msk2, sum(msk2) + 1
        end
        ###println("nssk $nmsk")
        l1 = 5 * sum(Flux.logitcrossentropy(yy, y; dims=3, agg=identity) .* msk) / nmsk
        l2 = 0 ####K < 100 ? 0 : 2 * sum(Flux.Losses.mse(dd, d; agg=identity) .* mskd) / nd
        probr = Flux.maxpool(yyy[:, :, 1:1, :], (7, 7); pad=3, stride=(1, 1))
        l3 = 13 * sum(Flux.Losses.mse(probr, view(y, :, :, 1:1, :), agg=identity) .* mskr) / nmskr
        l4 = 1 * sum(Flux.Losses.mse(probr, yred, agg=identity) .* yred) / nyred
        l61 = Flux.Losses.crossentropy(@view(yyy[2:end, :, :, :]), @view(yyy[1:end-1, :, :, :]), dims=3)
        l62 = Flux.Losses.crossentropy(@view(yyy[:, 2:end, :, :]), @view(yyy[:, 1:end-1, :, :]), dims=3)
        l6 = (l61 + l62)
        l5 = 0#  1000 * (Flux.mse(@view(dd[2:end, :, :, :]), @view(dd[1:end-1, :, :, :])) + Flux.mse(@view(dd[:, 2:end, :, :]), @view(dd[:, 1:end-1, :, :])))
        #############################################################################
        # w h ccc BBB
        #function crossentropy(ŷ, y; dims = 1, agg = mean, eps::Real = epseltype(ŷ), ϵ = nothing)     ■ Missing reference: mean
        ##
        # r
        # g
        # b
        # d
        # rgbrgbrgb
        Flux.ignore_derivatives() do
            if K % 30 == 0
                obs[] = (;
                    input=view(x, :, :, 1, 1) |> Flux.cpu,
                    output=Flux.softmax(view(yy, :, :, :, 1), dims=3) |> Flux.cpu,
                    depth=view(dd, :, :, 1, 1) |> Flux.cpu,
                    label=view(y, :, :, :, 1) |> Flux.cpu,
                )
                @info "iter $K $l1 $l2 $l3 $l4 $l5 $l6"
            end
        end
        l = l1 + l2 + l3 + l4 + l5 + l6
    end

    unet = create_unet(2 * 3, 4) |> Flux.gpu

    obs = let
        obs = Ref{Any}()
        loss(unet, Flux.gpu(get_training_sample(samples))..., 0, obs)
        obs = GLMakie.Observable(obs[])
        inp = map(obs) do tup
            tup.input
        end
        out = map(obs) do tup
            map((view(tup.output, :, :, c) for c in 1:3)...) do r, g, b
                GLMakie.RGBf(r, g, b)
            end
        end
        eg = map(obs) do tup
            map((view(tup.label, :, :, c) for c in 1:3)...) do r, g, b
                GLMakie.RGBf(r, g, b)
            end
        end
        dep = map(obs) do tup
            tup.depth
        end
        fig = GLMakie.Figure()
        display(fig)
        ax1, _ = GLMakie.image(fig[1, 1], inp, axis=(; aspect=GLMakie.DataAspect()))
        ax2, _ = GLMakie.image(fig[1, 2], out, axis=(; aspect=GLMakie.DataAspect()))
        ax3, _ = GLMakie.image(fig[2, 1], eg, axis=(; aspect=GLMakie.DataAspect()))
        ax4, _ = GLMakie.image(fig[2, 2], dep, axis=(; aspect=GLMakie.DataAspect()))
        GLMakie.linkaxes!(ax1, ax2, ax3, ax4)
        GLMakie.DataInspector(fig)
        obs
    end

    st = Flux.setup(Flux.AMSGrad(), unet)
    @showprogress for K in 1:10000
        x, y = get_training_sample(samples) |> Flux.gpu
        l, g = Flux.withgradient(m -> loss(m, x, y, K, obs), unet)
        Flux.update!(st, unet, g[1])
    end




    return unet |> Flux.cpu




    # @info "initial xform $name...... ..........."
    # prob = SegReg.Problem(ref_img, moving_imgs, ref_pts)
    # show(prob)
    # xform = SegReg.Network(prob)
    # show(xform)
    # open("/tmp/segreg1.log", "w") do log_io
    #     xform = prob(xform;
    #         landmark_weights=(10.0, 1000.0, 100000.0),
    #         n_iter=30000,
    #         log_io
    #     )
    #     xform = prob(xform;
    #         landmark_weights=(1000.0, 100.0, 100.0),
    #         n_iter=20000,
    #         log_io
    #     )
    #     xform, prob.group_pos
    # end
end

@rule tif = "data/seg-output/fmost$partch.tif" let unet = "interm/segment-unet"
    @deps img1 = "/data/dat/dsimgs/fmost$partch@ch00-avg.nrrd"
    @deps img2 = "/data/dat/dsimgs/fmost$partch@ch00-mip.nrrd"
    inputs = Float16.(stack([FileIO.load(img1), FileIO.load(img2)], dims=4) ./ 256.0f0)
    unet_siz = 512

    function sweep_with_unet(unet, img)
        out = zeros(Float16, size(img)[1:3]..., 3)
        batch = []
        function flush_batch()
            isempty(batch) && return
            bat = map(batch) do (xrng, yrng, z)
                za = max(1, z - 1)
                zb = min(size(img, 3), z + 1)
                inp = permutedims(view(img, xrng, yrng, [za, z, zb], :), (1, 2, 4, 3))
                if size(inp)[1:2] != (unet_siz, unet_siz)
                    inp = Flux.pad_zeros(inp, (0, unet_siz - size(inp, 1), 0, unet_siz - size(inp, 2), 0, 0, 0, 0))
                end
                reshape(inp, unet_siz, unet_siz, :)
            end
            bat = stack(bat, dims=4) |> Flux.gpu
            bato = unet(bat)
            bato = Flux.softmax(bato[:, :, 1:3, :], dims=3) |> Flux.cpu
            for (idx, (xrng, yrng, z)) in enumerate(batch)
                out[xrng, yrng, z, :] .= bato[axes(xrng, 1), axes(yrng, 1), :, idx]
            end
            empty!(batch)
        end
        function add_to_batch(xrng, yrng, z)
            push!(batch, (xrng, yrng, z))
            if length(batch) >= 8
                flush_batch()
            end
        end
        @showprogress for z in axes(img, 3)
            for x in 1:(unet_siz÷2):(size(img, 1)-unet_siz÷3//2), y in 1:(unet_siz÷2):(size(img, 2)-unet_siz÷3//2)
                x1 = min(size(img, 1), x + unet_siz - 1)
                y1 = min(size(img, 2), y + unet_siz - 1)
                add_to_batch(x:x1, y:y1, z)
            end
        end
        flush_batch()
        out
    end
    unet = unet |> Flux.gpu
    function encode_img(img)
        map((view(img, :, :, :, c) for c in axes(img, 4))...) do r, g, b
            #ColorTypes.RGB24(r, g, b)
            ColorTypes.RGB{ColorTypes.N0f8}(r, g, b)
        end
    end
    out = sweep_with_unet(unet, inputs)
    FileIO.save("/tmp/test1.nrrd", Float32.(out[:, :, :, 1]))
    FileIO.save("/tmp/test2.nrrd", Float32.(out[:, :, :, 2]))
    FileIO.save("/tmp/test3.nrrd", Float32.(out[:, :, :, 3]))
    #display(GLMakie.image())
    FileIO.save(tif, encode_img(out))
end
