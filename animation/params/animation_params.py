import re
import random
import pandas as pd
import numpy as np
from animation.params.util import get_default


class AnimationParams:
    def __init__(
        self, max_frames, motion_type="default", custom_default_file=None, **kwargs
    ):
        # load default values from yaml
        self.def_values = get_default(custom_default_file)["animation_params"]

        if motion_type == "default":
            self.anim_args = self._get_animation_args()
        elif motion_type == "custom":
            self.anim_args = self._get_animation_args(**kwargs)
        elif motion_type == "random":
            random_motion_params = self._generate_random_motion_params()
            self.anim_args = self._get_animation_args(**random_motion_params)

        self.angle_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["angle"]), max_frames
        )
        self.zoom_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["zoom"]), max_frames
        )
        self.translation_x_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["translation_x"]), max_frames
        )
        self.translation_y_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["translation_y"]), max_frames
        )
        self.translation_z_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["translation_z"]), max_frames
        )
        self.rotation_3d_x_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["rotation_3d_x"]), max_frames
        )
        self.rotation_3d_y_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["rotation_3d_y"]), max_frames
        )
        self.rotation_3d_z_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["rotation_3d_z"]), max_frames
        )
        self.noise_schedule_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["noise_schedule"]), max_frames
        )
        self.strength_schedule_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["strength_schedule"]), max_frames
        )
        self.contrast_schedule_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["contrast_schedule"]), max_frames
        )

    def dump_attributes(self):
        attributes = {}

        for attribute, value in self.anim_args.items():
            if attribute != "self":
                attributes[attribute] = value

        return attributes

    def _get_animation_args(self, **kwargs):
        translation_z = kwargs.get("translation_z", self.def_values["translation_z"])
        rotation_3d_x = kwargs.get("rotation_3d_x", self.def_values["rotation_3d_x"])
        rotation_3d_y = kwargs.get("rotation_3d_y", self.def_values["rotation_3d_y"])
        rotation_3d_z = kwargs.get("rotation_3d_z", self.def_values["rotation_3d_z"])

        angle = kwargs.get("angle", self.def_values["angle"])
        zoom = kwargs.get("zoom", self.def_values["zoom"])
        translation_x = kwargs.get("translation_x", self.def_values["translation_x"])
        translation_y = kwargs.get("translation_y", self.def_values["translation_y"])

        noise_schedule = kwargs.get("noise_schedule", self.def_values["noise_schedule"])
        strength_schedule = kwargs.get(
            "strength_schedule", self.def_values["strength_schedule"]
        )
        contrast_schedule = kwargs.get(
            "contrast_schedule", self.def_values["contrast_schedule"]
        )

        color_coherence = kwargs.get(
            "color_coherence", self.def_values["color_coherence"]
        )
        diffusion_cadence = kwargs.get(
            "diffusion_cadence", self.def_values["diffusion_cadence"]
        )

        use_depth_warping = kwargs.get(
            "use_depth_warping", self.def_values["use_depth_warping"]
        )
        midas_weight = kwargs.get("midas_weight", self.def_values["midas_weight"])
        near_plane = kwargs.get("near_plane", self.def_values["near_plane"])
        far_plane = kwargs.get("far_plane", self.def_values["far_plane"])
        fov = kwargs.get("fov", self.def_values["fov"])
        padding_mode = kwargs.get("padding_mode", self.def_values["padding_mode"])
        sampling_mode = kwargs.get("sampling_mode", self.def_values["sampling_mode"])

        border = kwargs.get("border", self.def_values["border"])

        return locals()

    def _generate_random_motion_params(self):
        trans_z = "0:(4)"
        rot_3d_x = f"0:({random.uniform(0., 1.)})"
        rot_3d_y = f"0:({random.uniform(0., 1.)})"
        rot_3d_z = "0:(0)"

        mid_change_frame = random.randint(18, 35)

        should_z_change, z_val = self._random_value_on_chance(6, 6, 9)
        if should_z_change:
            trans_z += f" {mid_change_frame}: {z_val}"

            should_x_change, x_val = self._random_value_on_chance(7, 2, 4)
            rot_3d_x += f" {mid_change_frame}: {x_val}" if should_x_change else ""

            should_y_change, y_val = self._random_value_on_chance(7, 2, 4)
            rot_3d_y += f" {mid_change_frame}: {y_val}" if should_y_change else ""

            should_z_change, z_val = self._random_value_on_chance(7, 2, 4)
            rot_3d_z += f" {mid_change_frame}: {z_val}" if should_z_change else ""

        motion_params = {
            "translation_z": trans_z,
            "rotation_3d_x": rot_3d_x,
            "rotation_3d_y": rot_3d_y,
            "rotation_3d_z": rot_3d_z,
        }

        return motion_params

    def _random_value_on_chance(self, chance, min, max):
        roll = random.randint(0, 10)

        if roll <= chance:
            return True, random.uniform(min, max)
        else:
            return False, -1

    def _get_inbetweens(
        self, key_frames, max_frames, integer=False, interp_method="Linear"
    ):
        key_frame_series = pd.Series([np.nan for a in range(max_frames)])

        for i, value in key_frames.items():
            key_frame_series[i] = value
        key_frame_series = key_frame_series.astype(float)

        if interp_method == "Cubic" and len(key_frames.items()) <= 3:
            interp_method = "Quadratic"
        if interp_method == "Quadratic" and len(key_frames.items()) <= 2:
            interp_method = "Linear"

        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[max_frames - 1] = key_frame_series[
            key_frame_series.last_valid_index()
        ]
        key_frame_series = key_frame_series.interpolate(
            method=interp_method.lower(), limit_direction="both"
        )
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    def _parse_key_frames(self, string, prompt_parser=None):
        pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
        frames = dict()
        for match_object in re.finditer(pattern, string):
            frame = int(match_object.groupdict()["frame"])
            param = match_object.groupdict()["param"]
            if prompt_parser:
                frames[frame] = prompt_parser(param)
            else:
                frames[frame] = param
        if frames == {} and len(string) != 0:
            raise RuntimeError("Key Frame string not correctly formatted")
        return frames

    def __str__(self):
        summary = (
            f"translation_z: {self.anim_args['translation_z']}\n"
            + f"rotation_3d_x: {self.anim_args['rotation_3d_x']}\n"
            + f"rotation_3d_y: {self.anim_args['rotation_3d_y']}\n"
            + f"rotation_3d_z: {self.anim_args['rotation_3d_z']}"
        )

        return summary
