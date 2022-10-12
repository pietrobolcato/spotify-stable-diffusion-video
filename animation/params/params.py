import random
import os
from animation.params.util import get_default
from animation.params.animation_params import AnimationParams


class Params:
    def __init__(
        self,
        out_dir,
        init_image,
        prompts,
        motion_type="default",
        custom_default_file=None,
        **kwargs
    ):
        # load default values from yaml
        def_values = get_default(custom_default_file)["params"]

        self.out_dir = out_dir
        self.init_image = init_image
        self.prompts = prompts

        self.W = kwargs.get("W", def_values["W"])
        self.H = kwargs.get("H", def_values["H"])
        self.W, self.H = map(lambda x: x - x % 64, (self.W, self.H))

        self.max_frames = kwargs.get("max_frames", def_values["max_frames"])
        self.fps = kwargs.get("fps", def_values["fps"])

        self.animation_params = AnimationParams(
            max_frames=self.max_frames, motion_type=motion_type
        )

        self.seed = kwargs.get("seed", def_values["seed"])
        self.sampler = kwargs.get("sampler", def_values["sampler"])
        self.steps = kwargs.get("steps", def_values["steps"])
        self.scale = kwargs.get("scale", def_values["scale"])
        self.ddim_eta = kwargs.get("ddim_eta", def_values["ddim_eta"])
        self.dynamic_threshold = kwargs.get(
            "dynamic_threshold", def_values["dynamic_threshold"]
        )
        self.static_threshold = kwargs.get(
            "static_threshold", def_values["static_threshold"]
        )

        self.save_samples = kwargs.get("save_samples", def_values["save_samples"])
        self.save_settings = kwargs.get("save_settings", def_values["save_settings"])
        self.display_samples = kwargs.get(
            "display_samples", def_values["display_samples"]
        )

        self.n_batch = kwargs.get("n_batch", def_values["n_batch"])
        self.filename_format = kwargs.get(
            "filename_format", def_values["filename_format"]
        )
        self.seed_behavior = kwargs.get("seed_behavior", def_values["seed_behavior"])

        self.use_init = kwargs.get("use_init", def_values["use_init"])
        self.strength = kwargs.get("strength", def_values["strength"])

        self.n_samples = kwargs.get("n_samples", def_values["n_samples"])
        self.precision = kwargs.get("precision", def_values["precision"])
        self.C = kwargs.get("C", def_values["C"])
        self.f = kwargs.get("f", def_values["f"])

        self.init_latent = kwargs.get("init_latent", def_values["init_latent"])
        self.init_sample = kwargs.get("init_sample", def_values["init_sample"])
        self.init_c = kwargs.get("init_c", def_values["init_c"])

        self.prompt_weighting = kwargs.get(
            "prompt_weighting", def_values["prompt_weighting"]
        )
        self.normalize_prompt_weights = kwargs.get(
            "normalize_prompt_weights", def_values["normalize_prompt_weights"]
        )
        self.log_weighted_subprompts = kwargs.get(
            "log_weighted_subprompts", def_values["log_weighted_subprompts"]
        )

        if self.seed == -1:
            self.seed = random.randint(0, 2**32 - 1)

    def dump_attributes(self):
        attributes = {}

        for attribute, value in self.__dict__.items():
            attributes[attribute] = value

        attributes["animation_params"] = self.animation_params.dump_attributes()

        return attributes
