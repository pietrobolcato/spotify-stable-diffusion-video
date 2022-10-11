from model_loader import ModelLoader
from animation import Animation
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

    # generate animation
    prompts = {
        0: "LSD acid blotter art featuring a face, surreal psychedelic hallucination, screenprint by kawase hasui, moebius, colorful flat surreal design, artstation",
        30: "LSD acid blotter art featuring the amazonian forest, surreal psychedelic hallucination, screenprint by kawase hasui, moebius, colorful flat surreal design, artstation",
        50: "LSD acid blotter art featuring smiling and sad faces, surreal psychedelic hallucination, screenprint by kawase hasui, moebius, colorful flat surreal design, artstation",
    }

    generation = Animation(
        diffusion_model=diffusion_model,
        depth_model=depth_model,
        out_dir="/content/test_outdir",
        init_image="https://i.ibb.co/7zm8Bw2/spotify-img-test.jpg",
        prompts=prompts,
        song="as_it_was",
        motion_type="default",
    )

    generation.run()
