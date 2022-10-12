import os
import time
import torch
import cv2
import math
import numpy as np
import requests
import torchvision.transforms.functional as TF
import animation.pytorch3d_lite.py3d_tools as p3d
import random
import animation.generator.prompt_weighting as prompt_weighting
from skimage.exposure import match_histograms
from einops import rearrange
from PIL import Image


def load_img(path, shape):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    image = image.convert("RGB")
    image = image.resize(shape, resample=Image.LANCZOS)

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.0 * image - 1.0

    return image


def anim_frame_warp_3d(prev_img_cv2, depth, anim_args, keys, frame_idx, device):
    TRANSLATION_SCALE = 1.0 / 200.0  # matches Disco
    translate_xyz = [
        -keys.translation_x_series[frame_idx] * TRANSLATION_SCALE,
        keys.translation_y_series[frame_idx] * TRANSLATION_SCALE,
        -keys.translation_z_series[frame_idx] * TRANSLATION_SCALE,
    ]
    rotate_xyz = [
        math.radians(keys.rotation_3d_x_series[frame_idx]),
        math.radians(keys.rotation_3d_y_series[frame_idx]),
        math.radians(keys.rotation_3d_z_series[frame_idx]),
    ]
    rot_mat = p3d.euler_angles_to_matrix(
        torch.tensor(rotate_xyz, device=device), "XYZ"
    ).unsqueeze(0)
    result = transform_image_3d(
        prev_img_cv2, depth, rot_mat, translate_xyz, anim_args, device
    )
    torch.cuda.empty_cache()
    return result


def transform_image_3d(
    prev_img_cv2, depth_tensor, rot_mat, translate, anim_args, device
):
    # adapted and optimized version of transform_image_3d from Disco Diffusion https://github.com/alembics/disco-diffusion
    w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

    aspect_ratio = float(w) / float(h)
    near, far, fov_deg = (
        anim_args["near_plane"],
        anim_args["far_plane"],
        anim_args["fov"],
    )
    persp_cam_old = p3d.FoVPerspectiveCameras(
        near, far, aspect_ratio, fov=fov_deg, degrees=True, device=device
    )
    persp_cam_new = p3d.FoVPerspectiveCameras(
        near,
        far,
        aspect_ratio,
        fov=fov_deg,
        degrees=True,
        R=rot_mat,
        T=torch.tensor([translate]),
        device=device,
    )

    # range of [-1,1] is important to torch grid_sample's padding handling
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, dtype=torch.float32, device=device),
        torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=device),
    )
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(
        xyz_old_world
    )[:, 0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(
        xyz_old_world
    )[:, 0:2]

    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
    ).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(
        identity_2d_batch, [1, 1, h, w], align_corners=False
    )
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

    image_tensor = rearrange(
        torch.from_numpy(prev_img_cv2.astype(np.float32)), "h w c -> c h w"
    ).to(device)
    new_image = torch.nn.functional.grid_sample(
        image_tensor.add(1 / 512 - 0.0001).unsqueeze(0),
        offset_coords_2d,
        mode=anim_args["sampling_mode"],
        padding_mode=anim_args["padding_mode"],
        align_corners=False,
    )

    # convert back to cv2 style numpy array
    result = (
        rearrange(new_image.squeeze().clamp(0, 255), "c h w -> h w c")
        .cpu()
        .numpy()
        .astype(prev_img_cv2.dtype)
    )
    return result


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample


def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(
        np.float32
    )
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = sample_f32 * 255
    return sample_int8.astype(type)


def maintain_colors(prev_img, color_match_sample, mode):
    if mode == "Match Frame 0 RGB":
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == "Match Frame 0 HSV":
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else:  # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt


def next_seed(args):
    if args.seed_behavior == "iter":
        args.seed += 1
    elif args.seed_behavior == "fixed":
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed


def get_uc_and_c(prompts, model, args, frame=0):
    prompt = prompts[0]  # they are the same in a batch anyway

    # get weighted sub-prompts
    (
        negative_subprompts,
        positive_subprompts,
    ) = prompt_weighting.split_weighted_subprompts(
        prompt, frame, not args.normalize_prompt_weights
    )

    uc = prompt_weighting.get_learned_conditioning(
        model, negative_subprompts, "", args, -1
    )
    c = prompt_weighting.get_learned_conditioning(
        model, positive_subprompts, prompt, args, 1
    )

    return (uc, c)
