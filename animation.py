import sys

sys.path.extend([
    'src/taming-transformers',
    'src/clip',
    'stable-diffusion/',
    'k-diffusion',
    'pytorch3d-lite',
    'AdaBins',
    'MiDaS',
])

import os
import subprocess
import numpy as np
from animation_params import AnimationParams
from run_params import RunParams
from helpers import DepthModel, sampler_fn
from base64 import b64encode

# TODO: move config to a separate file or argparse
models_path = "/content/drive/MyDrive/AI/Stable_Diffusion/"

FPS = 10

class Animation():
  def __init__(self, out_path, batch_name, init_image, prompts, song, motion_type = "random", **kwargs):
    self.animation_params = AnimationParams(motion_type, **kwargs)
    self.run_params = RunParams(out_path, batch_name, init_image, prompts)
    self.song = song
    self.fps = kwargs.get('fps', FPS) # TODO: move in run_params
    self.device = "cuda"
    self.depth_model = self._load_depth_model()

  def run(self):
    self._generate_frames()
    out_video_path = self._generate_video_from_frames()

    print('Video generated at:', out_video_path)
    return
  
  def __str__(self):
    ret = f"-- General params:\n" + \
          f"song: {self.song}\n" + \
          f"fps: {self.fps}\n" + \
          f"\n" + \
          f"-- Animation params:\n" + \
          f"{self.animation_params}"

    return ret

  def _load_depth_model(self):
    predict_depths = (self.animation_params.anim_args["animation_mode"] == "3D" and self.animation_params.anim_args["use_depth_warping"]) or self.animation_params.anim_args["save_depth_maps"]
    if predict_depths:
        depth_model = DepthModel(self.device)
        depth_model.load_midas(models_path)
        if self.animation_params.anim_args["midas_weight"] < 1.0:
            depth_model.load_adabins()
    else:
        depth_model = None
        self.animation_params.anim_args["save_depth_maps"] = False
    
    return depth_model

  def _generate_video_from_frames(self):
    image_path = os.path.join(self.run_params.outdir, f"{self.run_params.timestring}_%05d.png")
    temp_mp4_path = f"/content/{self.run_params.timestring}_temp.mp4"

    command = f'ffmpeg -y -vcodec png -r {self.fps} -start_number "0" -i "{image_path}" -frames:v {self.animation_params.max_frames} -c:v libx264 -vf fps="{fps}" -pix_fmt yuv420p -crf 17 -preset veryfast {temp_mp4_path}'
    os.system(command)

    postprocessed_video_path = self._post_process_video(temp_mp4_path)

    return postprocessed_video_path

  def _post_process_video(self, temp_mp4_path):
    # add boomerang

    temp_rev_mp4_path = temp_mp4_path.replace('_temp.mp4', '_temp_rev.mp4')
    os.system(f'ffmpeg -i {temp_mp4_path} -vf reverse {temp_rev_mp4_path}')

    temp_rev_x2_mp4_path = temp_rev_mp4_path.replace('_temp_rev.mp4', '_temp_rev_x2.mp4')
    os.system(f'ffmpeg -i {temp_rev_mp4_path} -filter:v "setpts=PTS/2" {temp_rev_x2_mp4_path}')

    concat_list_path = f'/content/{self.run_params.timestring}_concat_list.txt'
    boomerang_video_path = f'/content/{self.run_params.timestring}_boomerang.mp4'

    open(concat_list_path, 'w').write(f'file {temp_mp4_path}\nfile {temp_rev_x2_mp4_path}')
    os.system(f'ffmpeg -f concat -safe 0 -i {concat_list_path} -c copy {boomerang_video_path}')

    # add audio

    audio_path = '/content/as_it_was_cut_boomerang.mp3'
    boomerang_video_path_with_audio = os.path.join(self.run_params.outdir, f"{self.run_params.timestring}_boomerang_audio.mp4")
    os.system(f'ffmpeg -y -i {boomerang_video_path} -i {audio_path} -c copy -map 0:v:0 -map 1:a:0 {boomerang_video_path_with_audio}')

    return boomerang_video_path_with_audio

  def _generate_frames(self):
    anim_args = self.animation_params.anim_args

    # expand key frame strings to values
    keys = self.animation_params

    # create output folder for the batch
    os.makedirs(self.run_params.outdir, exist_ok=True)
    print(f"Saving animation frames to {self.run_params.outdir}")

    # save settings for the batch
    settings_filename = os.path.join(self.run_params.outdir, f"{self.run_params.timestring}_settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as settings_file:
        settings_dict = {**self.run_params.dump_attributes(), **self.animation_params.dump_attributes()}
        json.dump(settings_dict, settings_file, ensure_ascii=False, indent=4)

    # expand prompts out to per-frame
    prompt_series = pd.Series([np.nan for a in range(self.animation_params.max_frames)])
    for i, prompt in animation_prompts.items():
        prompt_series[i] = prompt
    prompt_series = prompt_series.ffill().bfill()

    # state for interpolating between diffusion steps
    turbo_steps = int(anim_args["diffusion_cadence"])
    turbo_prev_image, turbo_prev_frame_idx = None, 0
    turbo_next_image, turbo_next_frame_idx = None, 0

    n_samples = 1
    frame_idx = 0
    prev_sample = None
    
    while frame_idx < self.animation_params.max_frames:
        print(f"Rendering animation frame {frame_idx} of {self.animation_params.max_frames}")
        noise = keys.noise_schedule_series[frame_idx]
        strength = keys.strength_schedule_series[frame_idx]
        contrast = keys.contrast_schedule_series[frame_idx]
        depth = None
        
        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(0, frame_idx-turbo_steps)
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                advance_next = tween_frame_idx > turbo_next_frame_idx

                if self.depth_model is not None:
                    assert(turbo_next_image is not None)
                    depth = self.depth_model.predict(turbo_next_image, anim_args)

                if anim_args["animation_mode"] == '2D':
                    if advance_prev:
                        turbo_prev_image = anim_frame_warp_2d(turbo_prev_image, self.run_params, anim_args, keys, tween_frame_idx)
                    if advance_next:
                        turbo_next_image = anim_frame_warp_2d(turbo_next_image, self.run_params, anim_args, keys, tween_frame_idx)
                else: # '3D'
                    if advance_prev:
                        turbo_prev_image = anim_frame_warp_3d(turbo_prev_image, depth, anim_args, keys, tween_frame_idx)
                    if advance_next:
                        turbo_next_image = anim_frame_warp_3d(turbo_next_image, depth, anim_args, keys, tween_frame_idx)
                turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                if turbo_prev_image is not None and tween < 1.0:
                    img = turbo_prev_image*(1.0-tween) + turbo_next_image*tween
                else:
                    img = turbo_next_image

                filename = f"{self.run_params.timestring}_{tween_frame_idx:05}.png"
                cv2.imwrite(os.path.join(self.run_params.outdir, filename), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                if anim_args["save_depth_maps"]:
                    self.depth_model.save(os.path.join(self.run_params.outdir, f"{self.run_params.timestring}_depth_{tween_frame_idx:05}.png"), depth)
            if turbo_next_image is not None:
                prev_sample = sample_from_cv2(turbo_next_image)

        # apply transforms to previous frame
        if prev_sample is not None:
            if anim_args["animation_mode"] == '2D':
                prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), args, anim_args, keys, frame_idx)
            else: # '3D'
                prev_img_cv2 = sample_to_cv2(prev_sample)
                depth = self.depth_model.predict(prev_img_cv2, anim_args) if self.depth_model else None
                prev_img = anim_frame_warp_3d(prev_img_cv2, depth, anim_args, keys, frame_idx)

            # apply color matching
            if anim_args["color_coherence"] != 'None':
                if color_match_sample is None:
                    color_match_sample = prev_img.copy()
                else:
                    prev_img = maintain_colors(prev_img, color_match_sample, anim_args["color_coherence"])

            # apply scaling
            contrast_sample = prev_img * contrast
            # apply frame noising
            noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

            # use transformed previous frame as init for current
            self.run_params.use_init = True
            if half_precision:
                self.run_params.init_sample = noised_sample.half().to(device)
            else:
                self.run_params.init_sample = noised_sample.to(device)
            self.run_params.strength = max(0.0, min(1.0, strength))

        # grab prompt for current frame
        self.run_params.prompt = prompt_series[frame_idx]
        print(f"{self.run_params.prompt} {self.run_params.seed}")

        # sample the diffusion model
        sample, image = generate(self.run_params, return_latent=False, return_sample=True)
        prev_sample = sample

        if turbo_steps > 1:
            turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
            frame_idx += turbo_steps
        else:    
            filename = f"{self.run_params.timestring}_{frame_idx:05}.png"
            image.save(os.path.join(self.run_params.outdir, filename))
            if anim_args["save_depth_maps"]:
                if depth is None:
                    depth = self.depth_model.predict(sample_to_cv2(sample), anim_args)
                self.depth_model.save(os.path.join(self.run_params.outdir, f"{self.run_params.timestring}_depth_{frame_idx:05}.png"), depth)
            frame_idx += 1

        display.clear_output(wait=True)
        display.display(image)

        self.run_params.seed = next_seed(self.run_params)

if __name__ == "__main__":
    prompts = {
      0: "LSD acid blotter art featuring a face, surreal psychedelic hallucination, screenprint by kawase hasui, moebius, colorful flat surreal design, artstation",
      30: "LSD acid blotter art featuring the planet earth in space, surreal psychedelic hallucination, screenprint by kawase hasui, moebius, colorful flat surreal design, artstation",
    }

    generation = Animation(batch_name="rlon_test_AIO",
                          out_path="/",
                          init_image="https://i.ibb.co/7zm8Bw2/spotify-img-test.jpg",
                          prompts=prompts,
                          song="as_it_was",
                          motion_type="random")
    generation.run()