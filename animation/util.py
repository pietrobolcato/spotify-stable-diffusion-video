# -*- coding: utf-8 -*-
"""This module provides shared common functions"""

import os
import logging
import yaml
from yaml.loader import SafeLoader


def init_logging(logging_level=logging.INFO):
    """Define logger settings"""

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging_level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_global_settings(settings_path="global_settings.yaml"):
    """Get global settings from specified .yaml file"""

    assert os.path.exists(
        settings_path
    ), f"Global settings file doesn't exist at: {settings_path}"

    with open(settings_path) as f:
        data = yaml.load(f, Loader=SafeLoader)

    return data
