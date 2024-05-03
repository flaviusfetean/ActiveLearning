import copy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

from learners import Learner
from data_handler import DataHandler
from typing import Callable

import tensorflow as tf
from data_tooling.utils.files_utils import create_folder, copy_files, get_paths
from data_selectors import SelectorWrapper
from segmentation.selection_strategies.strategies import StrategyClass

from multiprocessing import Process

DEBUG = os.environ.get("DEBUG", False)
DISABLE_VERBOSE = os.environ.get("DISABLE_VERBOSE", False)


class ActiveLearningProcess:
    """
    Class that models a whole Active learning process
    Defines main parts of the process. May be subclassed and with some parts overwritten for custom functionality
    """

    def __init__(self, learner: Learner, config: dict, input_preprocess: Callable[[np.ndarray], np.ndarray],
                 output_preprocess: Callable[[np.ndarray], np.ndarray]):
        """
        Args:
            config: specifies test configuration, including:
                        -selection_strategy
                        -num_labeled
                        -budget
                        -selection_size
                        -images_path
                        -labels_path
                        -test_path
                        -model_path
                        -color_to_label dict (dictionary with keys being tuples
                            representing a BGR color and value being label - id of a class)
                        -label_to_color dict (dictionary with keys being labels - id of a class -
                            and value being a list representing a BGR color)
            input_preprocess: method used to preprocess the data points
            output_preprocess: method used to preprocess the labels
        """
        self.learner = learner
        self.config = config
        self.input_preprocess = input_preprocess
        self.output_preprocess = output_preprocess
        self.selector = None

    def step(self, data_handler: DataHandler):
        """
        Defines a single active learning step that consists in:
            -building a model
            -training the model
            -evaluating and saving the model
            -acquire the next batch of images
            -at the end of the step, the latest_acquired field of the data_handler holds the newly acquired images

        Returns:
            pair consisting of 2 arrays representing(newly_acquired_images_paths, newly_acquired_labels_paths)

        """

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            self.learner.build_model()

        xs = data_handler.get_selected_images()
        ys = data_handler.get_selected_labels()

        if self.config['test_images_path'] is None and self.config['test_labels_path'] is None:
            xs, xs_test, ys, ys_test = train_test_split(xs, ys, test_size=0.2, shuffle=True)
        else:
            xs_test = get_paths(self.config['test_images_path'])
            ys_test = get_paths(self.config['test_labels_path'])

        print("[STATUS] Training the model")
        self.learner.train_model(xs, ys)
        print("[STATUS] Evaluating the model")
        self.learner.evaluate_model(xs_test, ys_test, self.config['test_path'])

        self.learner.save_model(self.config['model_path'])

        print("[STATUS] Selecting the next batch of images...")
        self.selector.run(model=self.learner.get_model(), preprocess_func=self.input_preprocess)

        new_xs_paths = data_handler.get_paths_by_indices(data_handler.latest_acquired)
        new_ys_paths = data_handler.get_labels_paths_by_indices(data_handler.latest_acquired)

        return new_xs_paths, new_ys_paths

    def loop(self):
        """
            runs a complete active learning loop to test different AL algorithms
            includes model training, evaluation and new sample acquisition
        """

        data_handler = DataHandler(self.config['images_path'], self.config['labels_path'])
        rand_sel = data_handler.get_random_unselected_indices(self.config['num_labeled'] if not DEBUG else 20)
        data_handler.move_to_selected_by_indices(rand_sel)

        self.selector = SelectorWrapper(selection_strategy=StrategyClass[self.config["selection_strategy"]].value,
                                        data_handler=data_handler,
                                        selection_size=self.config['selection_size'] if not DEBUG else 10)

        acquired_samples = 0
        i = 0

        while acquired_samples < self.config['budget']:
            print(f"Beginning {self.__class__.__name__} iteration {i + 1}")
            new_xs, new_ys = self.step(data_handler)

            selection_folder = os.path.sep.join([self.config['test_path'], f"selection{i}"])
            imgs_folder = os.path.sep.join([selection_folder, "imgs"])
            masks_folder = os.path.sep.join([selection_folder, "imgs"])

            create_folder(imgs_folder)
            create_folder(masks_folder)

            copy_files(new_xs, imgs_folder)
            copy_files(new_ys, masks_folder)

            acquired_samples += len(new_xs)
            i += 1


class ActiveLearningCrossValidationProcess(ActiveLearningProcess):
    """
    Class that models an Active Learning process with cross-validation
    Inherits everything for base process and overwrites the step function
    to perform the cross-validation train-evaluation loop
    """

    def __init__(self, learner: Learner, config: dict, input_preprocess: Callable[[np.ndarray], np.ndarray],
                 output_preprocess: Callable[[np.ndarray], np.ndarray], test_size=0.2):
        super().__init__(learner, config, input_preprocess, output_preprocess)
        self.test_size = test_size

    def step(self, data_handler: DataHandler):
        """
               Defines a single active learning step that consists in:
                   -building a model
                   -training the model
                   -evaluating and saving the model
                   -acquire the next batch of images
                   -at the end of the step, the latest_acquired field of the data_handler holds the newly acquired images
                Evaluation is cross-validation, so the train-evaluate step is run in a loop

               Returns:
                   pair consisting of 2 arrays representing(newly_acquired_images_paths, newly_acquired_labels_paths)

        """

        xs = data_handler.get_selected_images()
        ys = data_handler.get_selected_labels()

        xs, ys = shuffle(xs, ys)
        test_size_int = int(self.test_size * len(xs))

        for i in range(0, len(xs), test_size_int):
            # using mask
            mask = np.zeros(len(xs), dtype=bool)
            mask[i:i + test_size_int] = True
            xs_test, ys_test = xs[mask], ys[mask]
            xs_train, ys_train = xs[np.logical_not(mask)], ys[np.logical_not(mask)]

            # retrain model
            self.learner.build_model()
            print(f"[STATUS] Training the model, cross-validation iteration {i // test_size_int + 1}")
            self.learner.train_model(xs_train, ys_train)
            print(f"[STATUS] Evaluating the model, cross-validation iteration {i // test_size_int + 1}")
            self.learner.evaluate_model(xs_test, ys_test, self.config['test_path'])

        self.learner.save_model(self.config['model_path'])

        eval_folder = os.path.sep.join([self.config['test_path'], "eval"])
        eval_file = os.path.sep.join([eval_folder, "eval_data.txt"])
        with open(eval_file, "a+") as f:
            f.write("\n")

        print("[STATUS] Selecting the next batch of images...")
        self.selector.run(model=self.learner.get_model(), preprocess_func=self.input_preprocess)

        new_xs_paths = data_handler.get_paths_by_indices(data_handler.latest_acquired)
        new_ys_paths = data_handler.get_labels_paths_by_indices(data_handler.latest_acquired)

        return new_xs_paths, new_ys_paths


def cross_val_step(learner, xs_train, ys_train, xs_test, ys_test, config, gpu_id):
    learner.train_model(xs_train, ys_train, gpu_id=gpu_id)
    learner.evaluate_model(xs_test, ys_test, config['test_path'], gpu_id=gpu_id)


class ParallelCrossValidationProcess(ActiveLearningCrossValidationProcess):

    def step(self, data_handler: DataHandler):
        """
               Defines a single active learning step that consists in:
                   -building a model
                   -training the model
                   -evaluating and saving the model
                   -acquire the next batch of images
                   -at the end of the step, the latest_acquired field of the data_handler holds the newly acquired images
                Evaluation is cross-validation, so the train-evaluate step is run in a loop

               Returns:
                   pair consisting of 2 arrays representing(newly_acquired_images_paths, newly_acquired_labels_paths)

        """

        xs = data_handler.get_selected_images()
        ys = data_handler.get_selected_labels()

        xs, ys = shuffle(xs, ys)
        test_size_int = int(self.test_size * len(xs))

        procs = []

        for i in range(0, len(xs), test_size_int):
            # using mask
            mask = np.zeros(len(xs), dtype=bool)
            mask[i:i + test_size_int] = True
            xs_test, ys_test = xs[mask], ys[mask]
            xs_train, ys_train = xs[np.logical_not(mask)], ys[np.logical_not(mask)]
            new_learner = self.learner.clone()

            proc = Process(target=cross_val_step,
                           args=(new_learner, xs_train, ys_train, xs_test, ys_test, self.config, i,))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

        self.learner.save_model(self.config['model_path'])

        eval_folder = os.path.sep.join([self.config['test_path'], "eval"])
        eval_file = os.path.sep.join([eval_folder, "eval_data.txt"])
        with open(eval_file, "a+") as f:
            f.write("\n")

        print("[STATUS] Selecting the next batch of images...")
        self.selector.run(model=self.learner.get_model(), preprocess_func=self.input_preprocess)

        new_xs_paths = data_handler.get_paths_by_indices(data_handler.latest_acquired)
        new_ys_paths = data_handler.get_labels_paths_by_indices(data_handler.latest_acquired)

        return new_xs_paths, new_ys_paths
