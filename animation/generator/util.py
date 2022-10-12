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


#
# Callback functions
#
class SamplerCallback(object):
    # Creates the callback function to be passed into the samplers for each step
    def __init__(
        self,
        args,
        run_id,
        mask=None,
        init_latent=None,
        sigmas=None,
        sampler=None,
        verbose=False,
        device="cuda",
    ):
        self.sampler_name = args.sampler
        self.dynamic_threshold = args.dynamic_threshold
        self.static_threshold = args.static_threshold
        self.mask = mask
        self.init_latent = init_latent
        self.sigmas = sigmas
        self.sampler = sampler
        self.verbose = verbose

        self.batch_size = args.n_samples
        self.save_sample_per_step = False
        self.show_sample_per_step = False
        self.paths_to_image_steps = [
            os.path.join(args.out_dir, f"{run_id}_{index:02}_{args.seed}")
            for index in range(args.n_samples)
        ]

        if self.save_sample_per_step:
            for path in self.paths_to_image_steps:
                os.makedirs(path, exist_ok=True)

        self.step_index = 0

        self.noise = None
        if init_latent is not None:
            self.noise = torch.randn_like(init_latent, device=device)

        self.mask_schedule = None
        if sigmas is not None and len(sigmas) > 0:
            self.mask_schedule, _ = torch.sort(sigmas / torch.max(sigmas))
        elif len(sigmas) == 0:
            self.mask = None  # no mask needed if no steps (usually happens because strength==1.0)

        if self.sampler_name in ["plms", "ddim"]:
            if mask is not None:
                assert (
                    sampler is not None
                ), "Callback function for stable-diffusion samplers requires sampler variable"

        if self.sampler_name in ["plms", "ddim"]:
            # Callback function formated for compvis latent diffusion samplers
            self.callback = self.img_callback_
        else:
            # Default callback function uses k-diffusion sampler variables
            self.callback = self.k_callback_

        self.verbose_print = print if verbose else lambda *args, **kwargs: None

    def view_sample_step(self, latents, path_name_modifier=""):
        if self.save_sample_per_step or self.show_sample_per_step:
            samples = model.decode_first_stage(latents)
            if self.save_sample_per_step:
                fname = f"{path_name_modifier}_{self.step_index:05}.png"
                for i, sample in enumerate(samples):
                    sample = sample.double().cpu().add(1).div(2).clamp(0, 1)
                    sample = torch.tensor(np.array(sample))
                    grid = make_grid(sample, 4).cpu()
                    TF.to_pil_image(grid).save(
                        os.path.join(self.paths_to_image_steps[i], fname)
                    )
            if self.show_sample_per_step:
                print(path_name_modifier)
                self.display_images(samples)
        return

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(np.array(images))
        grid = make_grid(images, 4).cpu()
        display.display(TF.to_pil_image(grid))
        return

    # The callback function is applied to the image at each step
    def dynamic_thresholding_(self, img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(self, args_dict):
        self.step_index = args_dict["i"]
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(args_dict["x"], self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(
                args_dict["x"], -1 * self.static_threshold, self.static_threshold
            )
        if self.mask is not None:
            init_noise = self.init_latent + self.noise * args_dict["sigma"]
            is_masked = torch.logical_and(
                self.mask >= self.mask_schedule[args_dict["i"]], self.mask != 0
            )
            new_img = init_noise * torch.where(is_masked, 1, 0) + args_dict[
                "x"
            ] * torch.where(is_masked, 0, 1)
            args_dict["x"].copy_(new_img)

        self.view_sample_step(args_dict["denoised"], "x0_pred")

    # Callback for Compvis samplers
    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(self, img, i):
        self.step_index = i
        # Thresholding functions
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(img, self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(img, -1 * self.static_threshold, self.static_threshold)
        if self.mask is not None:
            i_inv = len(self.sigmas) - i - 1
            init_noise = self.sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([i_inv] * self.batch_size).to(device),
                noise=self.noise,
            )
            is_masked = torch.logical_and(
                self.mask >= self.mask_schedule[i], self.mask != 0
            )
            new_img = init_noise * torch.where(is_masked, 1, 0) + img * torch.where(
                is_masked, 0, 1
            )
            img.copy_(new_img)

        self.view_sample_step(img, "x")
