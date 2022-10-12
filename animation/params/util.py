# -*- coding: utf-8 -*-
"""Contains the helper functions for the parameters classes"""

import os
import yaml
from yaml.loader import SafeLoader


def get_default(custom_file=None):
    """Read default values from a specified .yaml file"""

    if custom_file != None:
        default_file_path = custom_file
    else:
        default_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "default.yaml"
        )

    assert os.path.exists(
        default_file_path
    ), f"Default settings file doesn't exist at: {default_file_path}"

    with open(default_file_path) as f:
        data = yaml.load(f, Loader=SafeLoader)

    return data
