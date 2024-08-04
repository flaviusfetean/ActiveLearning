import os
import yaml
import shutil


def create_folder(path):
    os.makedirs(path, exist_ok=True)


def get_paths(path):
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]


def get_paths_recursive(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


def _get_paths_with_extension(path, extension):
    return [os.path.join(path, name) for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and name.endswith(extension)]


def get_paths_with_extensions(path, extensions: list):
    paths = []
    for extension in extensions:
        paths.extend(_get_paths_with_extension(path, extension))
    return paths


def get_paths_recursive_with_extension(path, extension):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))
    return paths

def get_names(path):
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]


def get_name_from_path(path):
    return path[path.rfind(os.path.sep) + 1:]


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def read_listing_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()


def copy_files(files, target_dir):
    for file in files:
        shutil.copy(file, target_dir)