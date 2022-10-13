import animation.presets.presets as presets
from animation.model_loader import ModelLoader
from animation.generator.animation import Animation
from util import get_global_settings

if __name__ == "__main__":
    # load settings
    global_settings = get_global_settings()

    # load models
    model_loader = ModelLoader(
        models_path=global_settings["models_path"],
        half_precision=global_settings["half_precision"],
        device=global_settings["device"],
    )
    diffusion_model = model_loader.load_diffusion_model()
    depth_model = model_loader.load_depth_model()

    # get prompts and song from preset
    prompts, song = presets.get_prompts_and_song_from_preset("as_it_was")

    # generate animation
    generation = Animation(
        diffusion_model=diffusion_model,
        depth_model=depth_model,
        out_dir=global_settings["output_directory"],
        init_image="https://i.ibb.co/7zm8Bw2/spotify-img-test.jpg",
        prompts=prompts,
        song=song,
        motion_type="default",
    )

    generation.run()
