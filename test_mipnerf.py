"""
test_mipnerf
=============
Mip-NeRF testing, use `python test_mipnerf.py` to run the module
"""
import os

import numpy as np
import torch
import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from torch import nn

import data_loader
import nerf
import render

###################################################################
# mip-NeRF Testing Hyperparameter
# (Use the same hyperparameters as the official implementation)
###################################################################
# OS parameters
DATA_BASE_DIR = "./data/nerf_synthetic/lego/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
RESIZE_COEF = 1
BACKGROUND_W = True
MULTISCALE = (1, 2, 4, 8)

# Model parameters
POS_ENCODE_DIM = (16, 0)
VIEW_ENCODE_DIM = 4
DENSE_FEATURES = 256
DENSE_DEPTH = 8

# Render parameters
TNEAR = 2.0
TFAR = 6.0
NUM_SAMPLES = 128
NUM_ISAMPLES = 128
RAY_CHUNK = 32768
SAMPLE5D_CHUNK = 65536
#############################################################################


def calcPSNR(img1, img2) -> float:
    return PSNR(img1, img2)


def calcSSIM(img1, img2) -> float:
    return SSIM(img1, img2, channel_axis=2)


def test_mip_nerf(
    datasets,
    net: nn.Module,
    loss_func,
    split="test"
):
    datasets = datasets[split]
    images, poses, focal = datasets["images"], datasets["poses"], datasets["focal"]
    samples, height, width, channel = images.shape
    # Pass relevant scene parameters, camera parameters,
    # geometric model (NeRF) into Volume Renderer
    renderer = render.MipVolumeRenderer(
        mip_nerf=net,
        width=width,
        height=height,
        focal=focal,
        tnear=TNEAR,
        tfar=TFAR,
        num_samples=NUM_SAMPLES,
        num_isamples=NUM_ISAMPLES,
        background_w=BACKGROUND_W,
        ray_chunk=RAY_CHUNK,
        sample5d_chunk=SAMPLE5D_CHUNK,
        is_train=False,
        device=DEVICE
    )
    if not os.path.exists("./out/other_imgs"):
        os.mkdir("./out/other_imgs")
    if MULTISCALE is not None and len(MULTISCALE) != 0:
        SCALE_LEN = len(MULTISCALE)
        loss_sum = np.zeros(SCALE_LEN)
        psnr_sum = np.zeros(SCALE_LEN)
        ssim_sum = np.zeros(SCALE_LEN)
    else:
        SCALE_LEN = 1
        loss_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
    print(f"[Log] number of {split}sets' images: {len(images) * SCALE_LEN}")
    for epoch in tqdm.trange(0, len(images) * SCALE_LEN):
        data_index = epoch // SCALE_LEN
        scale_index = epoch % SCALE_LEN
        image = images[data_index]
        pose = poses[data_index, :3, :4]

        
        if MULTISCALE is not None and len(MULTISCALE) != 0:
            scale = MULTISCALE[scale_index]
            H, W, F = height // scale, width // scale, focal / scale
            image = render.resize_img(image, H, W)
            renderer.change_res(H, W, F)
        else:
            scale = 1
            H, W, F = height, width, focal

        # Volume renderer render to obtain predictive rgb (image)
        # Including ray generation, sample coordinates, positional encoding,
        # Hierarchical volume sampling, NeRF(x,d)=(rgb,density),
        # computation of volume rendering equation
        image_hat = renderer.render_image(pose, use_tqdm=False)

        render.save_img(
            image_hat, f"./out/other_imgs/{split}_hat_r_Res{scale}_{data_index}.png"
        )

        # Calculate the loss and psnr
        loss = loss_func(
            torch.tensor(image),
            torch.tensor(image_hat)
        ).detach().item()
        
        if MULTISCALE is not None and len(MULTISCALE) != 0:
            loss_sum[scale_index] += loss
            psnr_sum[scale_index] += calcPSNR(image, image_hat)
            ssim_sum[scale_index] += calcSSIM(image, image_hat)
        else:
            loss_sum += loss
            psnr_sum += calcPSNR(image, image_hat)
            ssim_sum += calcSSIM(image, image_hat)

    loss_sum /= len(images)
    psnr_sum /= len(images)
    ssim_sum /= len(images)
    print(f"[Test] mip-NeRF avg Loss in {split}set: {loss_sum}")
    print(f"[Test] mip-NeRF avg PSNR in {split}set: {psnr_sum}")
    print(f"[Test] mip-NeRF avg SSIM in {split}set: {ssim_sum}")


if __name__ == "__main__":
    # get datasets from data_loader module
    datasets = data_loader.load_blender(
        base_dir=DATA_BASE_DIR,
        resize_coef=RESIZE_COEF,
        background_w=BACKGROUND_W
    )
    # get nerf from nerf module
    net = nerf.MipNeRF(
        pos_dim=POS_ENCODE_DIM,
        view_dim=VIEW_ENCODE_DIM,
        dense_features=DENSE_FEATURES,
        dense_depth=DENSE_DEPTH
    )
    # get loss function from torch.nn module
    loss_func = nn.MSELoss(reduction="mean")
    # Read the model of the largest epoch ever trained from the logs folder
    if os.path.exists("./out") and os.path.exists("./out/model"):
        for root, dirs, files in os.walk("./out/model"):
            files = list(filter(lambda file: file.startswith(
                "mip_nerf_train_") and file.endswith(".pt"), files))
            if files is not None and len(files) >= 1:
                result = max(files, key=lambda name: int(
                    name[len("mip_nerf_train_"):-len(".pt")]))
                path = os.path.join(root, result)
                print(f"[Model] mip-NeRF model Loader from {path}")
                net.load_state_dict(torch.load(path))
    # Start testing
    test_mip_nerf(
        datasets=datasets,
        net=net,
        loss_func=loss_func,
        split="test"
    )
