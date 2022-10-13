# -*- coding: utf-8 -*-
"""This module exposes utility functions for the flask server"""

import base64
from animation.model_loader import ModelLoader
from util import get_global_settings


def video_to_base64(video):
    """Convert generated video to base64"""

    with open(video, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_models():
    """Wrapper to load diffusion and depth models"""

    # load settings
    global_settings = get_global_settings()

    # load models
    model_loader = ModelLoader(
        models_path=global_settings["models_path"],
        half_precision=global_settings["half_precision"],
        device=global_settings["device"],
    )
    diffusion_model = model_loader.load_diffusion_model()
    depth_model = model_loader.load_depth_model()

    return diffusion_model, depth_model
