import shutil
import numpy as np
from utils.files_utils import get_paths_with_extensions, get_names, get_paths
import os


class DataHandler:
    """
    Responsible with handling of datasets. It will abstract the manipulation
    of AL split datasets into indices
    """

    def __init__(self, dataset_path, labels_path):
        self.dataset = dataset_path
        self.labels = labels_path

        self.images_paths = np.array(get_paths_with_extensions(dataset_path, ["jpg", "png"]))
        self.labels_paths = np.array(get_paths(labels_path))

        self.selected_mask = np.full((len(self.images_paths),), False)

        self.selected_indices = np.array([])
        self.unselected_indices = np.arange(0, len(self.images_paths))

        self.latest_acquired = []

    def _update_indices_by_mask(self):
        self.selected_indices = np.where(self.selected_mask)[0]
        self.unselected_indices = np.where(np.logical_not(self.selected_mask))[0]
        self._update_images_labels_dict()

    def _update_images_labels_dict(self):
        images_names = np.array(get_names(self.dataset))[self.selected_indices]
        labels_names = np.array(get_names(self.dataset))[self.selected_indices]
        self.images_labels_dict = {image_name: label_name for image_name, label_name in zip(images_names, labels_names)}

    def _split_according_to_existing_labels(self):
        labels_names = [path[-1][:path.rfind('.')] for path in self.labels_paths]
        indices_of_labels = self.get_indices_of_names(labels_names, include_extension=False)

        self.selected_mask[indices_of_labels] = True
        self._update_indices_by_mask()

    def get_paths_by_indices(self, indices):
        return self.images_paths[indices]

    def get_labels_paths_by_indices(self, indices):
        return self.labels_paths[indices]

    def move_to_selected_by_indices(self, indices, target_dir: str = None):
        assert len(indices) <= len(self.unselected_indices), f'Too many indices selected'
        self.selected_mask[indices] = True
        self._update_indices_by_mask()
        self.latest_acquired = indices

        if target_dir is not None:
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(self.dataset, target_dir)

    def move_to_selected_by_paths(self, images_paths, target_dir: str = None):
        indices = self.get_indices_of_paths(images_paths)
        self.move_to_selected_by_indices(indices, target_dir)

    def move_to_selected_by_names(self, images_names, include_extension: bool = True, target_dir: str = None):
        indices = self.get_indices_of_names(images_names, include_extension)
        self.move_to_selected_by_indices(indices, target_dir)

    def get_selected_images(self):
        return self.get_paths_by_indices(self.selected_indices)

    def get_selected_labels(self):
        return self.get_labels_paths_by_indices(self.selected_indices)

    def get_unselected_images(self):
        return self.get_paths_by_indices(self.unselected_indices)

    def get_random_unselected_indices(self, size):
        assert 0 <= size <= len(self.unselected_indices), f'Selection size should be smaller than unselected pool. Requested size: {size}, Available size: {len(self.unselected_indices)}'
        np.random.seed(42)
        indices = np.random.permutation(self.unselected_indices)[:size]
        return indices

    def get_indices_of_paths(self, images_paths):
        indices = [i for i, img_path in enumerate(self.images_paths) if img_path in images_paths]
        return np.array(indices)

    def get_indices_of_names(self, images_names, include_extension: bool = True):
        names = np.array(get_names(self.dataset))
        if not include_extension:
            names = np.array([name[:name.rfind(".")] for name in names])
        indices = [i for i, name in enumerate(names) if name in images_names]
        return np.array(indices)
