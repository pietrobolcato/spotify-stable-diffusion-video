# -*- coding: utf-8 -*-
"""This module creates a REST API interface to get inference from the model"""

import animation.presets.presets as presets
import logging
import time
from flask_server.server_util import load_models, video_to_base64
from flask import Flask, request, jsonify
from animation.generator.animation import Animation
from util import init_logging, get_global_settings

# load models
diffusion_model, depth_model = load_models()

# load settings
global_settings = get_global_settings()

# init flask and logging
app = Flask(__name__)
init_logging()


@app.route("/api", methods=["POST"])
def predict():
    """Runs inference on the model to compute the animation based on input preset and image

    Request params:
      preset (str): the preset for the animation
      init_image (str): the image in link format

    Returns:
      400: if request params aren't corrent
      200: a json dictionary with the video encoded in base64, and the elapsed time
           eg:
           {
              video: base_64
              elapsed_time: time
            }
    """

    data = request.get_json(force=True)

    logging.info(f"Received request with data: {data}")

    # check if request is valid
    if "preset" not in data or "init_image" not in data:
        logging.info(f"Bad request - preset or init_image not found")

        return "bad request - preset or init_image not found", 400
    else:
        logging.info("Handling request")

        # get request data
        preset = data["preset"]
        init_image = data["init_image"]

        # get prompts and song from preset
        prompts, song = presets.get_prompts_and_song_from_preset(preset)

        # generate animation
        start_stopwatch = time.time()

        generation = Animation(
            diffusion_model=diffusion_model,
            depth_model=depth_model,
            out_dir=global_settings["output_directory"],
            init_image=init_image,
            prompts=prompts,
            song=song,
            motion_type="random",
        )

        out_path = generation.run()
        end_stopwatch = time.time()

        # create response json
        elapsed_time = end_stopwatch - start_stopwatch

        response = {"video": video_to_base64(out_path), "elapsed_time": elapsed_time}

        logging.info(f"Request handled in: {elapsed_time} seconds")

        return jsonify(response)
