"""Downloads Stable Diffusion 1.4 config and checkpoint weights"""

import argparse
import gdown
import os


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory where to save pretrained models",
        default="./pretrained",
    )

    arguments = parser.parse_args()
    return arguments


def download_models(arguments):
    """Download models and save them in the output folder specified as argument"""

    if not os.path.exists(arguments.output):
        os.makedirs(arguments.output)

    gdown.download(
        id="1RWFZ9ImRQgY7jbl9UCA6yKeqnrbgCK4b",
        output=os.path.join(arguments.output, "sd-v1-4.ckpt"),
    )
    gdown.download(
        id="10psUyqJv2c4I66qxpEYVkeWuBO9Dt-va",
        output=os.path.join(arguments.output, "dpt_large-midas-2f21e586.pt"),
    )
    gdown.download(
        id="1Qn905-dqK0ExVINO3zZkHr5cSDTdtNn-",
        output=os.path.join(arguments.output, "v1-inference.yaml"),
    )


if __name__ == "__main__":
    arguments = parse_args()
    download_models(arguments)
