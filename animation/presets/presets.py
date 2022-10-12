# -*- coding: utf-8 -*-
"""This module handles prompts and song presets"""

import os
import yaml
from yaml.loader import SafeLoader


def get_prompts_and_song_from_preset(preset_name, custom_presets_file=None):
    """Get prompts and song from the preset list
    It is possible to use the symbol $CURRENT_FOLDER$ in the preset file, to access file relatively to the preset file position

    Args:
      preset_name (str): the name of the preset
      custom_preset_file (:obj:`str`, optional): loads the presets values from a specified yaml file

    Returns:
      (dict of int: str, str): a tuple containing the prompts in dictionary form of frame-prompt pairs, as well as the path of the song audio file
    """
    presets = _get_presets(custom_presets_file)
    assert preset_name in presets, f"Preset: {preset_name} not found in preset list"

    preset = presets[preset_name]
    prompts = preset["prompts"]
    song = preset["song"].replace(
        "$CURRENT_FOLDER$", os.path.dirname(os.path.realpath(__file__))
    )

    return prompts, song


def _get_presets(custom_presets_file=None):
    """Load presets from a specified .yaml file"""

    if custom_presets_file != None:
        presets_file = custom_presets_file
    else:
        presets_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "presets.yaml"
        )

    assert os.path.exists(
        presets_file
    ), f"Presets file doesn't exist at: {presets_file}"

    with open(presets_file) as f:
        data = yaml.load(f, Loader=SafeLoader)

    return data
