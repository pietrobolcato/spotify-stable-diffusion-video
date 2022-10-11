import random
from animation.params.animation_params import AnimationParams

SEED = 3664512517
MAX_FRAMES = 60
FPS = 10


class Params:
    def __init__(self, out_dir, init_image, prompts, motion_type="default", **kwargs):
        self.out_dir = out_dir
        self.init_image = init_image
        self.prompts = prompts

        W = 512
        H = 512
        self.W, self.H = map(lambda x: x - x % 64, (W, H))
        self.max_frames = kwargs.get("max_frames", MAX_FRAMES)
        self.fps = kwargs.get("fps", FPS)

        self.animation_params = AnimationParams(
            max_frames=self.max_frames, motion_type=motion_type
        )

        self.seed = kwargs.get("seed", SEED)
        self.sampler = "euler_ancestral"
        self.steps = 15
        self.scale = 7
        self.ddim_eta = 0.0
        self.dynamic_threshold = None
        self.static_threshold = None

        self.save_samples = True
        self.save_settings = True
        self.display_samples = True

        self.n_batch = 1
        self.filename_format = "{timestring}_{index}.png"
        self.seed_behavior = "iter"

        self.use_init = True
        self.strength = 1

        self.n_samples = 1
        self.precision = "autocast"
        self.C = 4
        self.f = 8

        self.init_latent = None
        self.init_sample = None
        self.init_c = None

        self.prompt_weighting = False
        self.normalize_prompt_weights = True
        self.log_weighted_subprompts = False

        if self.seed == -1:
            self.seed = random.randint(0, 2**32 - 1)

    def dump_attributes(self):
        attributes = {}

        for attribute, value in self.__dict__.items():
            attributes[attribute] = value

        attributes["animation_params"] = self.animation_params.dump_attributes()

        return attributes
