import re
import random
import pandas as pd
import numpy as np

MAX_FRAMES = 200
TRANS_Z = "0:(4) 19: (4) 26: (7)"
ROT_3D_X = "0: (0)"
ROT_3D_Y = "0: (0)"
ROT_3D_Z = "0:(0) 19: (0) 26: (3) 50: (3.) 60: (0.)"


class AnimationParams:
    def __init__(self, motion_type="default", **kwargs):
        if motion_type == "default":
            self.anim_args = self._get_animation_args()
        elif motion_type == "custom":
            self.anim_args = self._get_animation_args(**kwargs)
        elif motion_type == "random":
            random_motion_params = self._generate_random_motion_params()
            self.anim_args = self._get_animation_args(**random_motion_params)

        self.max_frames = kwargs.get("max_frames", MAX_FRAMES)

        self.angle_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["angle"]), self.max_frames
        )
        self.zoom_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["zoom"]), self.max_frames
        )
        self.translation_x_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["translation_x"]), self.max_frames
        )
        self.translation_y_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["translation_y"]), self.max_frames
        )
        self.translation_z_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["translation_z"]), self.max_frames
        )
        self.rotation_3d_x_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["rotation_3d_x"]), self.max_frames
        )
        self.rotation_3d_y_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["rotation_3d_y"]), self.max_frames
        )
        self.rotation_3d_z_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["rotation_3d_z"]), self.max_frames
        )
        self.noise_schedule_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["noise_schedule"]), self.max_frames
        )
        self.strength_schedule_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["strength_schedule"]), self.max_frames
        )
        self.contrast_schedule_series = self._get_inbetweens(
            self._parse_key_frames(self.anim_args["contrast_schedule"]), self.max_frames
        )

    def dump_attributes(self):
        attributes = {}

        for attribute, value in self.anim_args.items():
            if attribute != "self":
                attributes[attribute] = value

        return attributes

    def _get_animation_args(self, **kwargs):
        """kwargs can change default:
        translation_z
        rotation_3d_x
        rotation_3d_y
        rotation_3d z"""

        translation_z = kwargs.get("translation_z", TRANS_Z)
        rotation_3d_x = kwargs.get("rotation_3d_x", ROT_3D_X)
        rotation_3d_y = kwargs.get("rotation_3d_y", ROT_3D_Y)
        rotation_3d_z = kwargs.get("rotation_3d_z", ROT_3D_Z)

        max_frames = 50
        border = "replicate"

        angle = "0:(0)"
        zoom = "0:(1.04)"
        translation_x = "0:(0)"
        translation_y = "0:(0)"

        noise_schedule = "0: (0.02) "
        strength_schedule = "0: (0.9) 20: (0.8) 30: (0.5) 34: (0.4) 36: (0.3) 38: (0.75) 50: (0.05) 100: (0.3)"
        contrast_schedule = "0: (1.0)"

        color_coherence = "None"
        diffusion_cadence = "2"

        use_depth_warping = True
        midas_weight = 0.3
        near_plane = 200
        far_plane = 10000
        fov = 40
        padding_mode = "border"
        sampling_mode = "bicubic"

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
