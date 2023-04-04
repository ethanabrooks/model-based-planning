import os
import numpy as np
import skvideo.io
import wandb


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(filename, video_frames, fps=60, video_format="mp4"):
    assert fps == int(fps), fps

    _make_dir(filename)
    if wandb.run is not None:
        (time, height, width, channel) = [i for i, _ in enumerate(video_frames.shape)]
        data = np.transpose(video_frames, (time, channel, height, width))
        wandb.log({filename: wandb.Video(data, fps=fps, format=video_format)})
    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={
            "-f": video_format,
            "-pix_fmt": "yuv420p",  # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        },
    )


def save_videos(filename, *video_frames, **kwargs):
    ## video_frame : [ N x H x W x C ]
    video_frames = np.concatenate(video_frames, axis=2)
    save_video(filename, video_frames, **kwargs)
