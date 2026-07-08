import os

SUPPORTED_FORMATS = (".mp4", ".avi", ".mov", ".mkv")


def load_videos(folder):

    video_paths = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(SUPPORTED_FORMATS):

            video_paths.append(
                os.path.join(folder, filename)
            )

    return sorted(video_paths)