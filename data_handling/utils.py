import os


"""
Use this module to put any helper function you may need.
"""


def list_files_in_dir(path_to_dir, extension):
    """
    This function returns a list of files names
    in a specific directory with a specific extension
    Args:
        path_to_dir: path to directory
        extension: file extension to seach for
    """
    return [f for f in os.listdir(path_to_dir)
            if os.path.isfile(os.path.join(path_to_dir, f))
            and f.endswith(extension)]
