# -*- coding: utf-8 -*-
"""Computes one animation run

Example usage as script:

python start_animation.py --preset as_it_was --init_image /content/image.png
"""

import argparse
import animation.presets.presets as presets
from animation.model_loader import ModelLoader
from animation.generator.animation import Animation
from util import get_global_settings


def parse_args():
    """Parse arguments from command line"""

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        help="Preset name for the song / animation",
        nargs="?",
        default="as_it_was",
    )
    parser.add_argument(
        "-i",
        "--init_image",
        type=str,
        help="Inital image to morph. Can be a local path, a URL, a URI from AWS/GCP/Azure storage, or an HDFS path.",
        nargs="?",
        default="https://i.ibb.co/7zm8Bw2/spotify-img-test.jpg",
    )
    args = parser.parse_args()

    return args


def compute_animation(args):
    """Compute animation based on command line arguments"""

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

    # get prompts and song from preset
    prompts, song = presets.get_prompts_and_song_from_preset(args.preset)

    # generate animation
    generation = Animation(
        diffusion_model=diffusion_model,
        depth_model=depth_model,
        out_dir=global_settings["output_directory"],
        init_image=args.init_image,
        prompts=prompts,
        song=song,
        motion_type="default",
        device=global_settings["device"],
    )

    generation.run()


def main():
    """Runs the script"""

    args = parse_args()
    compute_animation(args)


if __name__ == "__main__":
    main()
