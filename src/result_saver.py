import os


def get_output_path(output_folder, input_path):

    filename = os.path.basename(input_path)

    return os.path.join(output_folder, filename)