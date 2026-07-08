import os

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")


def load_images(folder):

    image_paths = []

    for filename in os.listdir(folder):

        if filename.lower().endswith(SUPPORTED_FORMATS):

            image_paths.append(
                os.path.join(folder, filename)
            )

    return sorted(image_paths)