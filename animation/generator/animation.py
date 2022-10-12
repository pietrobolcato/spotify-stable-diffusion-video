# -*- coding: utf-8 -*-
"""This module handles the animation computation"""

import os
import numpy as np
import pandas as pd
import json
import torch
import cv2
import time
import logging
import animation.generator.util as util
import animation.generator.post_process as anim_pp
from PIL import Image
from torch import autocast
from pytorch_lightning import seed_everything
from einops import repeat, rearrange
from contextlib import nullcontext
from animation.params.params import Params
from animation.model_loader import ModelLoader
from animation.stable_diffusion.helpers import sampler_fn
from animation.util import init_logging
from animation.generator.sampler_callback import SamplerCallback
from animation.stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from animation.k_diffusion.k_diffusion.external import CompVisDenoiser


class Animation:
    """Initializes the animation class with run-specific parameters, as well as the logging

    Args:
      diffusion_model (LatentModel): the loaded Stable Diffusion model from model_loader.py
      depth_model (DepthModel): the loaded Depth model from model_loader.py
      out_dir (str): the output directory of the generation
      init_image (str): the path or link to the init image to morph
      prompts (dict of int: str): dictionary mapping prompts to specific frames
      song (str): the song id,
      motion_type (str, optional): specifies the animation motion type - can be "default", "custom", or "random"
      half_precision (bool, optional): if true, loads the diffusion model in half_precision mode
      device (str, optional): can be "cuda" or "cpu", following pytorch standards
      logging_level (logging, optional): set level of logging
      **kwargs: arbitrary keyword arguments to override animation params
                eg: "max_frames=100" passed as kwarg, would set the animation params "max_frames" to 100,
                overriding the default value
    """

    def __init__(
        self,
        diffusion_model,
        depth_model,
        out_dir,
        init_image,
        prompts,
        song,
        motion_type="random",
        half_precision=True,
        device="cuda",
        logging_level=logging.INFO,
        **kwargs,
    ):
        self.diffusion_model = diffusion_model
        self.depth_model = depth_model
        self.song = song
        self.params = Params(
            out_dir=out_dir,
            init_image=init_image,
            prompts=prompts,
            motion_type=motion_type,
            **kwargs,
        )
        self.half_precision = half_precision
        self.device = device
        self.run_id = time.strftime("%Y%m%d_-_%H_%M_%S")

        init_logging(logging_level)

    def run(self):
        """Runs the animation generation, including video post processing

        Returns:
          str: the path to the generated video
        """

        self._generate_frames()
        out_video_path = anim_pp.generate_video_from_frames(self.params, self.run_id)

        logging.info(f"Video generated at: {out_video_path}")
        return out_video_path

    def _generate_frames(self):
        """Generate the frames of the animation"""

        anim_args = self.params.animation_params.anim_args

        # expand key frame strings to values
        keys = self.params.animation_params

        # create output folder for the batch
        os.makedirs(self.params.out_dir, exist_ok=True)
        logging.debug(f"Saving animation frames to {self.params.out_dir}")

        # save settings for the batch
        self._save_batch_settings()

        # expand prompts out to per-frame
        prompt_series = pd.Series([np.nan for a in range(self.params.max_frames)])
        for i, prompt in self.params.prompts.items():
            prompt_series[i] = prompt
        prompt_series = prompt_series.ffill().bfill()

        # state for interpolating between diffusion steps
        turbo_steps = int(anim_args["diffusion_cadence"])
        turbo_prev_image, turbo_prev_frame_idx = None, 0
        turbo_next_image, turbo_next_frame_idx = None, 0

        n_samples = 1
        frame_idx = 0
        prev_sample = None
        color_match_sample = None

        while frame_idx < self.params.max_frames:
            logging.debug(
                f"Rendering animation frame {frame_idx} of {self.params.max_frames}"
            )
            noise = keys.noise_schedule_series[frame_idx]
            strength = keys.strength_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            depth = None

            # emit in-between frames
            if turbo_steps > 1:
                tween_frame_start_idx = max(0, frame_idx - turbo_steps)
                for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                    tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(
                        frame_idx - tween_frame_start_idx
                    )
                    logging.debug(
                        f"Creating in between frame {tween_frame_idx} tween:{tween:0.2f}"
                    )

                    advance_prev = (
                        turbo_prev_image is not None
                        and tween_frame_idx > turbo_prev_frame_idx
                    )
                    advance_next = tween_frame_idx > turbo_next_frame_idx

                    assert turbo_next_image is not None
                    depth = self.depth_model.predict(turbo_next_image, anim_args)

                    if advance_prev:
                        turbo_prev_image = util.anim_frame_warp_3d(
                            turbo_prev_image,
                            depth,
                            anim_args,
                            keys,
                            tween_frame_idx,
                            device=self.device,
                        )

                    if advance_next:
                        turbo_next_image = util.anim_frame_warp_3d(
                            turbo_next_image,
                            depth,
                            anim_args,
                            keys,
                            tween_frame_idx,
                            device=self.device,
                        )

                    turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                    if turbo_prev_image is not None and tween < 1.0:
                        img = (
                            turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
                        )
                    else:
                        img = turbo_next_image

                    filename = f"{self.run_id}_{tween_frame_idx:05}.png"
                    cv2.imwrite(
                        os.path.join(self.params.out_dir, filename),
                        cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR),
                    )
                if turbo_next_image is not None:
                    prev_sample = util.sample_from_cv2(turbo_next_image)

            # apply transforms to previous frame
            if prev_sample is not None:
                prev_img_cv2 = util.sample_to_cv2(prev_sample)
                depth = (
                    self.depth_model.predict(prev_img_cv2, anim_args)
                    if self.depth_model
                    else None
                )

                prev_img = util.anim_frame_warp_3d(
                    prev_img_cv2,
                    depth,
                    anim_args,
                    keys,
                    frame_idx,
                    device=self.device,
                )

                # apply color matching
                if anim_args["color_coherence"] != "None":
                    if color_match_sample is None:
                        color_match_sample = prev_img.copy()
                    else:
                        prev_img = util.maintain_colors(
                            prev_img, color_match_sample, anim_args["color_coherence"]
                        )

                # apply scaling
                contrast_sample = prev_img * contrast
                # apply frame noising
                noised_sample = util.add_noise(
                    util.sample_from_cv2(contrast_sample), noise
                )

                # use transformed previous frame as init for current
                self.params.use_init = True
                if self.half_precision:
                    self.params.init_sample = noised_sample.half().to(self.device)
                else:
                    self.params.init_sample = noised_sample.to(self.device)
                self.params.strength = max(0.0, min(1.0, strength))

            # grab prompt for current frame
            current_frame_prompt = prompt_series[frame_idx]
            logging.debug(f"Prompt: {current_frame_prompt} | Seed: {self.params.seed}")

            # sample the diffusion model
            sample, image = self._generate_single_frame(
                prompt=current_frame_prompt, return_latent=False, return_sample=True
            )
            prev_sample = sample

            if turbo_steps > 1:
                turbo_prev_image, turbo_prev_frame_idx = (
                    turbo_next_image,
                    turbo_next_frame_idx,
                )
                turbo_next_image, turbo_next_frame_idx = (
                    util.sample_to_cv2(sample, type=np.float32),
                    frame_idx,
                )
                frame_idx += turbo_steps
            else:
                filename = f"{self.run_id}_{frame_idx:05}.png"
                image.save(os.path.join(self.params.out_dir, filename))
                frame_idx += 1

            self.params.seed = util.next_seed(self.params)

    def _generate_single_frame(
        self, prompt, frame=0, return_latent=False, return_sample=False, return_c=False
    ):
        """Sample diffusion model to get image output"""

        # set init settings
        seed_everything(self.params.seed)

        sampler = DDIMSampler(self.diffusion_model)

        model_wrap = CompVisDenoiser(self.diffusion_model)
        batch_size = self.params.n_samples
        assert prompt is not None
        data = [batch_size * [prompt]]
        precision_scope = (
            autocast if self.params.precision == "autocast" else nullcontext
        )

        init_latent = None
        init_image = None
        if self.params.init_latent is not None:
            init_latent = self.params.init_latent
        elif self.params.init_sample is not None:
            with precision_scope("cuda"):
                init_latent = self.diffusion_model.get_first_stage_encoding(
                    self.diffusion_model.encode_first_stage(self.params.init_sample)
                )
        elif (
            self.params.use_init
            and self.params.init_image != None
            and self.params.init_image != ""
        ):
            init_image = util.load_img(
                self.params.init_image, shape=(self.params.W, self.params.H)
            )
            init_image = init_image.to(self.device)
            init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
            with precision_scope("cuda"):
                init_latent = self.diffusion_model.get_first_stage_encoding(
                    self.diffusion_model.encode_first_stage(init_image)
                )  # move to latent space

        mask = None

        t_enc = int((1.0 - self.params.strength) * self.params.steps)

        # Noise schedule for the k-diffusion samplers (used for masking)
        k_sigmas = model_wrap.get_sigmas(self.params.steps)
        k_sigmas = k_sigmas[len(k_sigmas) - t_enc - 1 :]

        callback = SamplerCallback(
            args=self.params,
            run_id=self.run_id,
            mask=mask,
            init_latent=init_latent,
            sigmas=k_sigmas,
            sampler=sampler,
            verbose=False,
            device=self.device,
        ).callback

        results = []
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.diffusion_model.ema_scope():
                    for prompts in data:
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        if self.params.prompt_weighting:
                            uc, c = util.get_uc_and_c(
                                prompts, self.diffusion_model, self.params, frame
                            )
                        else:
                            uc = self.diffusion_model.get_learned_conditioning(
                                batch_size * [""]
                            )
                            c = self.diffusion_model.get_learned_conditioning(prompts)

                        if self.params.scale == 1.0:
                            uc = None
                        if self.params.init_c != None:
                            c = self.params.init_c

                        assert self.params.sampler in [
                            "klms",
                            "dpm2",
                            "dpm2_ancestral",
                            "heun",
                            "euler",
                            "euler_ancestral",
                        ], "Sampler: {self.params.sampler} not supported"

                        samples = sampler_fn(
                            c=c,
                            uc=uc,
                            args=self.params,
                            model_wrap=model_wrap,
                            init_latent=init_latent,
                            t_enc=t_enc,
                            device=self.device,
                            cb=callback,
                        )

                        if return_latent:
                            results.append(samples.clone())

                        x_samples = self.diffusion_model.decode_first_stage(samples)

                        if return_sample:
                            results.append(x_samples.clone())

                        x_samples = torch.clamp(
                            (x_samples + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        if return_c:
                            results.append(c.clone())

                        for x_sample in x_samples:
                            x_sample = 255.0 * rearrange(
                                x_sample.cpu().numpy(), "c h w -> h w c"
                            )
                            image = Image.fromarray(x_sample.astype(np.uint8))
                            results.append(image)
        return results

    def _save_batch_settings(self):
        """Dumps run settings to file for future inspection"""

        settings_filename = os.path.join(
            self.params.out_dir, f"{self.run_id}_settings.txt"
        )
        with open(settings_filename, "w+", encoding="utf-8") as settings_file:
            settings_dict = {
                "song": self.song,
                "run_id": self.run_id,
                "half_precision": self.half_precision,
                **self.params.dump_attributes(),
            }
            json.dump(settings_dict, settings_file, ensure_ascii=False, indent=4)
