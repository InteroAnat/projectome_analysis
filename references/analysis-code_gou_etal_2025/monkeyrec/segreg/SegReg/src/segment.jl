function create_seg_unet(unet_siz::Integer, (chin, chout)::Pair; chmid=4)
    backb = Metalhead.resnet(Metalhead.basicblock, [2, 3, 5, 7, 3]; inchannels=chmid, activation=Flux.leakyrelu)
    unet = Metalhead.UNet((unet_siz, unet_siz), chmid, chmid, backb[1])
    unet = Flux.Chain(
        Flux.Conv((1, 1), chin => chmid, bias=true),
        unet.layers[1],
        Flux.Chain(
            Flux.Conv((3, 3), chmid => chmid, pad=1, bias=false),
            Flux.BatchNorm(3),
            Flux.NNlib.relu,
            Flux.Conv((3, 3), chmid => chmid, pad=1, bias=false),
            Flux.BatchNorm(3),
        ),
        Flux.Conv((1, 1), chmid => chout, bias=true),
    )
    pretrain = "pretrain/unet-$unet_siz-$chin-$chmid-$chout.jld2"
    if isfile(pretrain)
        m = JLD2.load(pretrain, "model_state")
        Flux.loadmodel!(unet, m)
    end
    unet
end

#
# ** moving *U16 *1 *4ch
# ** ref F16 *1 1+3
# unet F32 F16 U16 1 /2 *2
# moving_lbl F16 1 (2)
# moving_seg F32 F16 U16 1 /2 *2
#
# moving_lbl: moving ref warp example
# unet: moving moving_lbl
# moving_seg: moving unet example

function label_moving_imgs(xform, moving_imgs, ref_img)
    prepare_xform!(xform)
    map(enumerate(moving_imgs)) do (B, img)
        temp=map(CartesianIndices(view(ref_img.data, :, :, :, 1))) do ind
            _,v2,v3,v4=view(ref_img.data, ind, :)
            coord=(Float32(ind[d]) for d in 1:3)
            SVector(coord...) => SVector(v2,v3,v4)
        end
        xyz=reshape(stack((map(v->v[1][d], temp) for d in 1:3), dims=1), 3, :) |>Flux.gpu
        xyz=(xyz.-0.5f0).*reshape(ref_img.resolution, 3, :)
        xyz= execute_xform(xform, B, xyz)
        xyz=round.(xyz./reshape(img.res, 3, :).+ 0.5f0)
        seg=map(v->v[2], temp)
        tree=NearestNeighbors.KDTree(xyz)
        outimg=zeros(Float16, (size(img)[1:end-1]..., 3))
        @showprogress for z in axes(img, 3)
            inds=map(CartesianIndices(view(img, :, :, z:z, 1))) do ind
                coord=(Float32(ind[d]) for d in 1:3)
                SVector(coord...)
            end
            tgtxyz=reshape(stack((map(v->v[1][d], inds) for d in 1:3), dims=1), 3, :)
            inds=reduce(vcat, NearestNeighbors.knn(tree, tgtxyz, 1)[1])
            outimg[:, :, z:z, :]=permutedims(reduce(hcat, map(i->seg[i], inds)), (2,1))
        end
        Main.xxxoutimg=outimg
    end
end

