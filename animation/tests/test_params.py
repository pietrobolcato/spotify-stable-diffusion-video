from animation.params.params import Params
from animation.params.util import get_default


def test_load_default():
    """Tests that default values of params are loaded correctly"""

    default_values = get_default()["params"]
    params_dump = Params("out_dir", "init_image", "prompts").dump_attributes()

    for key, value in default_values.items():
        if key != "seed":  # seed gets randomized, therefore ignore it
            assert default_values[key] == params_dump[key]


def test_custom_override():
    """Tests that default values get overrided by a correct kwarg input"""

    max_frames = -1

    default_values = get_default()["params"]
    params_dump = Params(
        "out_dir", "init_image", "prompts", max_frames=max_frames
    ).dump_attributes()

    assert params_dump["max_frames"] == max_frames
    assert default_values["max_frames"] != params_dump["max_frames"]
