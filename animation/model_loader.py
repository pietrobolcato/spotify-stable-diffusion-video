# -*- coding: utf-8 -*-
"""This module loads the required models such as Stable Diffusion, Adabins and MiDaS for the 3D animation to work"""

import os
import torch
import logging
import util
from omegaconf import OmegaConf
from animation.stable_diffusion.helpers import DepthModel
from animation.stable_diffusion.ldm.util import instantiate_from_config


class ModelLoader:
    """Loads diffusion and depth models

    Args:
      models_path (str): path where the models are located
      diffusion_model_config (str, optional): the name of the diffusion model config file
      diffusion_model_checkpoint (str, optional): the name of the diffusion model checkpoint file
      half_precision (bool, optional): if true, loads the diffusion model in half_precision mode
      device (str, optional): can be "cuda" or "cpu", following pytorch standards
      logging_level (logging, optional): set level of logging
    """

    def __init__(
        self,
        models_path,
        diffusion_model_config="v1-inference.yaml",
        diffusion_model_checkpoint="sd-v1-4.ckpt",
        half_precision=True,
        device="cuda",
        logging_level=logging.INFO,
    ):
        self.models_path = models_path
        self.diffusion_model_config = diffusion_model_config
        self.diffusion_model_checkpoint = diffusion_model_checkpoint
        self.half_precision = half_precision
        self.device = device

        self.diffusion_config_path = os.path.join(
            self.models_path, self.diffusion_model_config
        )
        assert os.path.exists(
            self.diffusion_config_path
        ), f"Diffusion config file doesn't exist at: {self.diffusion_config_path}"

        self.diffusion_ckpt_path = os.path.join(
            self.models_path, self.diffusion_model_checkpoint
        )
        assert os.path.exists(
            self.diffusion_ckpt_path
        ), f"Diffusion checkpoint file doesn't exist at: {self.diffusion_ckpt_path}"

        util.init_logging(logging_level)

    def load_diffusion_model(self):
        """Loads diffusion model from the parameters initialized in the class

        Returns:
          LatentModel: the loaded stable diffusion model
        """
        logging.info(f"Loading diffusion model from: {self.diffusion_config_path}")

        config = OmegaConf.load(self.diffusion_config_path)
        ckpt_load = torch.load(self.diffusion_ckpt_path, map_location=self.device)
        state_dict = ckpt_load["state_dict"]

        model = instantiate_from_config(config.model)
        model.load_state_dict(state_dict, strict=False)

        if self.half_precision:
            model = model.half().to(self.device)
        else:
            model = model.to(self.device)

        return model

    def load_depth_model(self):
        """Loads depth model from the parameters initialized in the class

        Returns:
          DepthModel: the loaded depth model
        """
        depth_model = DepthModel(self.device)
        depth_model.load_midas(self.models_path)
        depth_model.load_adabins()

        return depth_model
