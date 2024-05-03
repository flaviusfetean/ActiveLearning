from abc import ABC
import numpy as np
import keras


class Learner(ABC):

    def get_model(self) -> keras.Model:
        """

        Returns
        -------
         The underlying model of the selector
        """
        pass

    def build_model(self):
        """
               Method responsible with model creation.
        """
        pass

    def train_model(self, xs: np.ndarray, ys: np.ndarray):
        """
        Defines the training part of the AL process. May be overwritten to provide desired functionality.
        Different processes may require different training steps

        Args:
            xs: available labeled data for training
            ys: corresponding labels for the training data

        Returns:
            the fitted model

        """
        pass

    def evaluate_model(self, xs: np.ndarray, ys: np.ndarray, test_folder):
        """
        Method to evaluate a model
        Args:
            xs: paths to test images
            ys: paths to test_labels
            test_folder: folder where evaluation data is written
        """
        pass

    def save_model(self, output_folder):
        """

        Parameters
        ----------
        output_folder: folder in which to save the model

        Returns
        -------

        """

        pass

    def clone(self):
        """

        Returns
        -------
        An instance of a cloned learner, with the same parameters as this one

        """
        pass
