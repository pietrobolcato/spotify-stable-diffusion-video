import os


def generate_video_from_frames(params, run_id):
    """Using ffmpeg, encodes the generated frames into a video"""

    image_path = os.path.join(params.out_dir, f"{run_id}_%05d.png")
    temp_mp4_path = os.path.join(params.out_dir, f"{run_id}_temp.mp4")

    command = f'ffmpeg -y -vcodec png -r {params.fps} -start_number "0" -i "{image_path}" -frames:v {params.max_frames} -c:v libx264 -vf fps="{params.fps}" -pix_fmt yuv420p -crf 17 -preset veryfast {temp_mp4_path}'
    os.system(command)

    postprocessed_video_path = post_process_video(temp_mp4_path, params, run_id)

    return postprocessed_video_path


def post_process_video(temp_mp4_path, params, run_id):
    """Applies the video post processing steps using ffmpeg.

    First, it creates a reversed version of the original video running at double speed, creating a boomerang effect.
    This is then concatenated to the original video, to create a looping motion
    Finally, the song audio is added to the video"""

    # add boomerang
    temp_rev_mp4_path = temp_mp4_path.replace("_temp.mp4", "_temp_rev.mp4")
    os.system(f"ffmpeg -i {temp_mp4_path} -vf reverse {temp_rev_mp4_path}")

    temp_rev_x2_mp4_path = temp_rev_mp4_path.replace(
        "_temp_rev.mp4", "_temp_rev_x2.mp4"
    )
    os.system(
        f'ffmpeg -i {temp_rev_mp4_path} -filter:v "setpts=PTS/2" {temp_rev_x2_mp4_path}'
    )

    concat_list_path = os.path.join(params.out_dir, f"{run_id}_concat_list.txt")
    boomerang_video_path = os.path.join(params.out_dir, f"{run_id}_boomerang.mp4")

    open(concat_list_path, "w").write(
        f"file {temp_mp4_path}\nfile {temp_rev_x2_mp4_path}"
    )
    os.system(
        f"ffmpeg -f concat -safe 0 -i {concat_list_path} -c copy {boomerang_video_path}"
    )

    # add audio - TODO
    audio_path = "/content/as_it_was_cut_boomerang.mp3"
    boomerang_video_path_with_audio = os.path.join(
        params.out_dir, f"{run_id}_boomerang_audio.mp4"
    )
    os.system(
        f"ffmpeg -y -i {boomerang_video_path} -i {audio_path} -c copy -map 0:v:0 -map 1:a:0 {boomerang_video_path_with_audio}"
    )

    return boomerang_video_path_with_audio
