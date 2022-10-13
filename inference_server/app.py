# -*- coding: utf-8 -*-
"""This module creates a REST API interface to get inference from the model"""

import torch
import time
import sys
import os
from torch.cuda import stream
import uvicorn
import inference_server.schema as schema
import animation.presets.presets as presets

from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from inference_server.exception_handler import validation_exception_handler
from inference_server.exception_handler import python_exception_handler

from animation.generator.animation import Animation

from inference_server.server_util import load_models, video_to_base64
from util import get_global_settings

# initialize API Server
app = FastAPI(
    title="Custom Video Stable Diffusion",
    description="Creates a 30s music video from a custom picture using Stable Diffusion",
    version="0.9",
    terms_of_service=None,
    contact=None,
    license_info=None,
)

# allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and model
    """

    # load models
    diffusion_model, depth_model = load_models()

    # load settings
    global_settings = get_global_settings()

    # add models and settings to app state
    app.package = {
        "diffusion_model": diffusion_model,
        "depth_model": depth_model,
        "global_settings": global_settings,
    }


@app.post(
    "/api/v1/predict",
    response_model=schema.InferenceResponse,
    responses={
        422: {"model": schema.ErrorResponse},
        500: {"model": schema.ErrorResponse},
    },
)
def do_predict(request: Request, body: schema.InferenceInput):
    """Runs inference on the model to compute the animation based on input preset and image

    Request params:
      preset (str): the preset for the animation
      init_image (str): the image in link format

    Returns:
      200: a json dictionary with the video encoded in base64, and the elapsed time
    """

    logger.info(f"Received inference request with data: {body}")

    # get request data
    preset = body.preset
    init_image = body.init_image

    # get prompts and song from preset
    prompts, song = presets.get_prompts_and_song_from_preset(preset)

    # generate animation
    start_stopwatch = time.time()

    generation = Animation(
        diffusion_model=app.package["diffusion_model"],
        depth_model=app.package["depth_model"],
        out_dir=app.package["global_settings"]["output_directory"],
        init_image=init_image,
        prompts=prompts,
        song=song,
        motion_type="random",
        device=app.package["global_settings"]["device"],
        max_frames=5,
    )

    out_path = generation.run()
    end_stopwatch = time.time()

    # create response json
    elapsed_time = end_stopwatch - start_stopwatch
    results = {"video": video_to_base64(out_path), "elapsed_time": elapsed_time}

    logger.info(f"Request handled in: {elapsed_time} seconds")

    return {"error": False, "results": results}


@app.get("/about")
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash("nvidia-smi"),
    }


if __name__ == "__main__":
    # server api
    log_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "log.ini"
    )

    uvicorn.run(
        "inference_server.app:app",
        host="0.0.0.0",
        port=5000,
        workers=4,
        reload=False,
        debug=True,
        log_config=log_config_path,
    )
