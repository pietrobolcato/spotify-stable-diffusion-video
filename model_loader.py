# TODO: document
# TODO: add kwargs for parameters 
# TODO: refactor

import os
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

MODEL_CONFIG = "v1-inference.yaml"
MODEL_CHECKPOINT = "sd-v1-4.ckpt"

class ModelLoader():
  def __init__(self, models_path, **kwargs):
    self.models_path = models_path
    self.model_config = kwargs.get('model_config', MODEL_CONFIG)
    self.model_checkpoint =  kwargs.get('model_checkpoint', MODEL_CHECKPOINT)
    self.half_precision = True
    self.check_sha256 = False 

    model_map = {
        "sd-v1-4.ckpt": {'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'},
    }

    # config path
    self.ckpt_config_path = os.path.join(self.models_path, self.model_config)
    if os.path.exists(self.ckpt_config_path):
        print(f"{self.ckpt_config_path} exists")
    else:
        self.ckpt_config_path = "./stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
    print(f"Using config: {self.ckpt_config_path}")

    # checkpoint path or download
    self.ckpt_path = os.path.join(self.models_path, self.model_checkpoint)
    ckpt_valid = True
    if os.path.exists(self.ckpt_path):
        print(f"{self.ckpt_path} exists")
    else:
        print(f"Please download model checkpoint and place in {os.path.join(self.models_path, self.model_checkpoint)}")
        ckpt_valid = False

    if self.check_sha256 and ckpt_valid:
        import hashlib
        print("\n...checking sha256")
        with open(self.ckpt_path, "rb") as f:
            bytes = f.read() 
            hash = hashlib.sha256(bytes).hexdigest()
            del bytes
        if model_map[self.model_checkpoint]["sha256"] == hash:
            print("hash is correct\n")
        else:
            print("hash in not correct\n")
            ckpt_valid = False

    if ckpt_valid:
        print(f"Using ckpt: {self.ckpt_path}")

  def load_model_from_config(self, verbose=False, device='cuda', half_precision=True):
      config = OmegaConf.load(self.ckpt_config_path)
      map_location = "cuda" 
      print(f"Loading model from {self.ckpt_config_path}")
      pl_sd = torch.load(self.ckpt_path, map_location=map_location)
      if "global_step" in pl_sd:
          print(f"Global Step: {pl_sd['global_step']}")
      sd = pl_sd["state_dict"]
      model = instantiate_from_config(config.model)
      m, u = model.load_state_dict(sd, strict=False)
      if len(m) > 0 and verbose:
          print("missing keys:")
          print(m)
      if len(u) > 0 and verbose:
          print("unexpected keys:")
          print(u)

      if half_precision:
          model = model.half().to(device)
      else:
          model = model.to(device)

      model.eval()

      return model

    # if load_on_run_all and ckpt_valid:
    #     local_config = OmegaConf.load(f"{ckpt_config_path}")
    #     model = load_model_from_config(local_config, f"{ckpt_path}", half_precision=half_precision)
    #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #     model = model.to(device)