import os.path
from segmentation.selection_strategies.strategies import *
import tensorflow as tf
import cv2
from typing import Callable
import tqdm
from data_handler import DataHandler
from utils.files_utils import get_name_from_path
from utils.models_utils import create_model_with_dropout_heads

DEBUG = os.environ.get("DEBUG", False)
DISABLE_VERBOSE = os.environ.get("DISABLE_VERBOSE", False)


class Selector(ABC):

    def __init__(self, selection_strategy: SegmentationSelectionStrategy, data_handler: DataHandler,
                 selection_size: int = 100):
        self.data_handler = data_handler
        self.selection_strategy = selection_strategy
        self.selection_size = selection_size

    def build_score_dict(self, model: tf.keras.Model, images_paths, preprocess_func: Callable[[np.ndarray], np.ndarray]) -> dict:
        """
        Builds a dictionary where each image is attributed a score by the selection strategy implemented by the selector

        Parameters
        ----------
        model: tf.keras model used for predictions to compute the score
        images_paths: paths to unlabeled images
        preprocess_func:  Preprocessing function. This function will be executed before inference

        Returns
        -------
        dictionary of pairs where key is the image_path and the value is the image's score by model prediction and implemented strategy

        """
        pass

    def run(self, model: tf.keras.Model, preprocess_func: Callable[[np.ndarray], np.ndarray]):
        """
        Performs an active learning acquisition step.

        Parameters
        ----------
        model: tf.keras model used to make predictions that will be scored
        preprocess_func: Preprocessing function. This function will be executed before inference.

        """
        images_paths = self.data_handler.get_unselected_images()[:40 if DEBUG else -1]
        scores_dict = self.build_score_dict(model, images_paths, preprocess_func)
        best_samples = self.selection_strategy.get_best_samples(scores=scores_dict,
                                                                selection_size=self.selection_size)
        self.data_handler.move_to_selected_by_names(best_samples, include_extension=True)


class BaseSelector(Selector):
    """
    Implementation of metric-based active learning image selection for track segmentation feature.
    """

    def __init__(self, selection_strategy: SegmentationSelectionStrategy,
                 data_handler: DataHandler, selection_size=100):
        """

        Args:
            selection_strategy: Selection strategy used for active learning.
            data_handler: Data Handler objects which will handle dataset operations (abstracts images as indices in backend)
            selection_size: Number of images to be selected.
        """
        super(BaseSelector, self).__init__(selection_strategy, data_handler, selection_size)
        self.selection_strategy = selection_strategy
        self.selection_size = selection_size
        self.data_handler = data_handler

    def build_score_dict(self, model: tf.keras.Model, images_paths, preprocess_func: Callable[[np.ndarray], np.ndarray]):
        prediction_dict = {}
        for image_path in tqdm.tqdm(images_paths, disable=DISABLE_VERBOSE):
            model_input = cv2.imread(image_path)
            model_input = preprocess_func(model_input)
            model_input = np.expand_dims(model_input, axis=0)
            score = self.selection_strategy.score(model.predict(model_input, verbose=0))
            prediction_dict[get_name_from_path(image_path)] = score

        return prediction_dict


class BayesianSelector(Selector):
    """
    Implementation of bayesian with Monte-Carlo Dropout active learning image selection for track segmentation feature.
    """

    def __init__(self, selection_strategy: SegmentationSelectionStrategy, data_handler: DataHandler,
                 ensemble_size=10, selection_size=100):
        """

        Args:
            selection_strategy: Selection strategy used for active learning.
            ensemble_size: Number of instances of the model to apply MC-dropout to
            selection_size: Number of images to be selected.
        """
        super().__init__(selection_strategy, data_handler, selection_size)
        self.selection_strategy = selection_strategy
        self.selection_size = selection_size
        self.data_handler = data_handler
        self.ensemble_size = ensemble_size

    def montecarlo_dropout(self, model: tf.keras.Model, model_input):
        """
        Performs Monte-Carlo Dropout for approximating a bayesian inference of the model

        Parameters
        ----------
        model: tf.keras model that will be used for approximating a bayesian model
        model_input: input image for the model

        Return
        ------
        array of predictions computed by models with multiple parameter sets sampled from
        distribution (different instances of dropout before the last layer)

        """

        model_dropout = create_model_with_dropout_heads(model, dropout_rate=0.5, num_heads=self.ensemble_size)
        predictions = model_dropout.predict(model_input, verbose=0)

        return np.array(predictions)

    def build_score_dict(self, model: tf.keras.Model, images_paths, preprocess_func: Callable[[np.ndarray], np.ndarray]):

        prediction_dict = {}
        for image_path in tqdm.tqdm(images_paths, disable=DISABLE_VERBOSE):
            model_input = cv2.imread(image_path)
            model_input = preprocess_func(model_input)
            model_input = np.expand_dims(model_input, axis=0)
            score = self.selection_strategy.score(self.montecarlo_dropout(model, model_input))
            prediction_dict[get_name_from_path(image_path)] = score

        return prediction_dict


class SelectorWrapper(Selector):
    def __init__(self, selection_strategy: SegmentationSelectionStrategy, data_handler: DataHandler,
                 ensemble_size=10, selection_size=100):
        """
            Based on selection strategy, the Selector composition can slightly differ.
            There are strategies specific to BayesianLearners and to BaseLearners.

            params are the same as for class build, but will return a different selector based on selection strategy

        """
        super().__init__(selection_strategy, data_handler, selection_size)
        self.selector = None
        if isinstance(selection_strategy, (RandomSelection, EntropySelection, MarginSelection)):
            self.selector = BaseSelector(selection_strategy, data_handler, selection_size)
        else:
            self.selector = BayesianSelector(selection_strategy, data_handler, ensemble_size, selection_size)

    def build_score_dict(self, model: tf.keras.Model, images_paths, preprocess_func: Callable[[np.ndarray], np.ndarray]) -> dict:
        return self.selector.build_score_dict(model, images_paths, preprocess_func)

    def run(self, model: tf.keras.Model, preprocess_func: Callable[[np.ndarray], np.ndarray]):
        self.selector.run(model, preprocess_func)
