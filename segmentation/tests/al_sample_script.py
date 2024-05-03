import argparse
import os

from imutils.paths import list_images
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import Callable

import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping

from segmentation.augmentation import *
from unet import build_unet
import tqdm

from data_tooling.utils.files_utils import get_name_from_path, read_yaml_file
from learners import Learner
from segmentation.utils import categorical_to_mask, segmentation_preprocess, \
    segmentation_preprocess_output, tf_segmentation_preprocess, tf_segmentation_preprocess_output
from segmentation.active_learning_process import ActiveLearningCrossValidationProcess, \
    ParallelCrossValidationProcess, ActiveLearningProcess

TRAIN = os.environ.get("TRAIN", False)
DEBUG = os.environ.get("DEBUG", False)
DISABLE_VERBOSE = os.environ.get("DISABLE_VERBOSE", False)


def lr_schedule(lr_decay, lr_decay_freq):

    def schedule(epoch, lr):
        if epoch > 0 and epoch % lr_decay_freq == 0:
            return lr * lr_decay
        else:
            return lr
    return schedule


class UnetLearner(Learner):
    def __init__(self, input_shape: tuple, tf_loader, preprocess_input: Callable[[np.ndarray], np.ndarray] = (lambda x: x),
                 preprocess_output: Callable[[np.ndarray, dict], np.ndarray] = (lambda x: x), **kwargs):
        self.model = None
        self.model_input_shape = input_shape
        self.tf_loader = tf_loader
        self.preprocess_input = preprocess_input
        self.preprocess_output = preprocess_output
        self.train_config = kwargs
        self.build_model()

    def get_model(self):
        return self.model

    def build_model(self):
        self.model = build_unet(input_size=self.model_input_shape, config=self.train_config)

    def train_model(self, xs: np.ndarray, ys: np.ndarray):

        xs, xs_val, ys, ys_val = train_test_split(xs, ys, test_size=0.1, shuffle=True)

        train_dataset = tf.data.Dataset.from_tensor_slices((xs, ys), name="TrainDataset")
        train_dataset = train_dataset.repeat(self.train_config['num_epochs']).shuffle(128).map(self.tf_loader)
        train_dataset = train_dataset.map(tfCrop()).map(tfFlip()).map(tfBrightness())

        val_dataset = tf.data.Dataset.from_tensor_slices((xs_val, ys_val), name="ValDataset")
        val_dataset = val_dataset.repeat(self.train_config['num_epochs']).shuffle(32).map(self.tf_loader)

        callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_schedule(self.train_config['lr_decay'],
                                                                          self.train_config['lr_decay_freq'])),
                     tf.keras.callbacks.ModelCheckpoint(filepath=self.train_config['model_path'],
                                                        save_weights_only=True,
                                                        monitor='val_dice_coef', mode='max',
                                                        save_best_only=True)]

        self.model.fit(train_dataset.batch(self.train_config['bsize']), steps_per_epoch=len(xs) // self.train_config['bsize'],
                       epochs=1 if DEBUG else self.train_config['num_epochs'], verbose=2 if DISABLE_VERBOSE else 1,
                       validation_steps=len(xs_val) // self.train_config['bsize'], validation_data=val_dataset.batch(self.train_config['bsize']),
                       callbacks=callbacks)

        self.model.load_weights(self.train_config['model_path'])

    def evaluate_model(self, xs: np.ndarray, ys: np.ndarray, test_folder: str, gpu_id: int = 0):
        pr_folder = os.path.sep.join([test_folder, "predictions"])
        eval_folder = os.path.sep.join([test_folder, "eval"])
        eval_file = os.path.sep.join([eval_folder, "eval_data.txt"])

        os.makedirs(pr_folder, exist_ok=True)
        os.makedirs(eval_folder, exist_ok=True)

        predictions = {}
        print("[STATUS] Predicting on the test set..")

        test_dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
        test_dataset = test_dataset.shuffle(32).map(self.tf_loader).batch(self.train_config['bsize'])

        for x in tqdm.tqdm(xs[:20], disable=DISABLE_VERBOSE):
            img = self.preprocess_input(cv2.imread(x))
            name = get_name_from_path(x)
            img = np.expand_dims(img, axis=0)
            prediction = self.model.predict(img, verbose=0)[0]

            predictions[name] = prediction

        eval_data = self.model.evaluate(test_dataset, steps=len(xs) // self.train_config['bsize'],
                                        verbose=2 if DISABLE_VERBOSE else 1)

        print("[STATUS] saving predictions to file")
        for name, prediction in tqdm.tqdm(predictions.items(), disable=DISABLE_VERBOSE):
            mask_color = categorical_to_mask(prediction, self.train_config['label_to_color'])

            cv2.imwrite(os.path.sep.join([pr_folder, name]), mask_color.astype("uint8"))

        with open(eval_file, "a+") as f:
            f.write(f"{eval_data[0]} {eval_data[1]} {eval_data[2]};")

    def save_model(self, output_folder):
        self.model.save(output_folder)

    def clone(self):
        return UnetLearner(self.model_input_shape, self.preprocess_input, self.preprocess_output,
                           **self.train_config)

def norm(x: np.ndarray, m, o) -> np.ndarray:
    return (x - m) / o

def compute_norm_data(xs, image_size):
    print("[INFO] computing dataset stats...")
    print("[INFO] reading images...")
    imgs = np.array([cv2.resize(cv2.imread(path), image_size) for path in tqdm.tqdm(xs, disable=DISABLE_VERBOSE)], dtype="float32")

    m = np.mean(imgs)
    o = np.std(imgs)

    print(f"[INFO]Computed mean is: {m}, and stddev: {o}")

    return m, o

def data_generator(batch_size, xs, ys, preprocess_input, preprocess_otput,
                   color_to_label: dict, rgb_input: bool, aug: SemsegAugmentation=None):
    i = 0
    size = len(xs)

    while i < size:

        x = xs[i: i + batch_size]
        y = ys[i: i + batch_size]

        x_preproc, y_preproc = [], []

        for x_path, y_path in zip(x, y):
            x_i = cv2.imread(x_path)[..., ::-1]
            y_i = cv2.imread(y_path)[..., :: 1 if rgb_input else -1]

            if aug is not None:
                x_i, y_i = aug.aug(x_i, y_i)

            x_preproc.append(x_i)
            y_preproc.append(y_i)

        x_preproc = np.array([preprocess_input(x_p) for x_p in x_preproc])
        y_preproc = np.array([preprocess_otput(y_p, color_to_label) for y_p in y_preproc])

        i += batch_size
        if i >= size:
            xs, ys = shuffle(xs, ys)
            i = 0

        yield x_preproc, y_preproc

def unet_preprocess_input(h, w, mean, stddev):
    def base_func(x):
        return segmentation_preprocess(x, (w, h), mean, stddev)

    return base_func


def unet_preprocess_output(h, w):

    def base_func(x: np.ndarray, color_to_label: dict):
        return segmentation_preprocess_output(x, (w, h), color_to_label)

    return base_func

def tf_loader(h, w, mean, stddev, rgb_input, color_to_label: dict):
    def base_func(image_path, label_path):
        # image_path, label_path = input_pair

        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=3)

        mask = tf.io.read_file(label_path)
        mask = tf.io.decode_png(mask, channels=3)
        if rgb_input:
            mask = tf.reverse(mask, axis=[-1])

        image = tf_segmentation_preprocess(image, [h, w], mean, stddev)
        mask = tf_segmentation_preprocess_output(mask, [h, w], color_to_label)

        return image, mask

    return base_func

def tfCrop(p=0.9):
    def base_func(x, y):
        h, w = x.shape[:2]
        d = y.shape[-1]
        w_cut = int(p * w)
        h_cut = int(p * h)

        rand_seed = (int(random.uniform(0, 10000)),
                     int(random.uniform(0, 10000)))

        x_cut = tf.image.stateless_random_crop(x, (h_cut, w_cut, 3), rand_seed)
        y_cut = tf.image.stateless_random_crop(y, (h_cut, w_cut, d), rand_seed)

        x_new = tf.image.resize(x_cut, (h, w))
        y_new = tf.image.resize(y_cut, (h, w), tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return x_new, y_new

    return base_func

def tfFlip():
    def base_func(x, y):
        x_new, y_new = x, y

        if random.randint(0, 10) < 5:
            x_new = tf.image.flip_left_right(x_new)
            y_new = tf.image.flip_left_right(y_new)

        return x_new, y_new
    return base_func

def tfBrightness(min_b=-0.1, max_b=0.1):
    def base_func(x, y):
        rand_p = random.uniform(min_b, max_b)

        x_new = tf.image.adjust_brightness(x, rand_p)

        return x_new, y
    return base_func

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)

    return parser.parse_args()


def get_config(path):
    """
    An example of config path can be found in segmentation/tests/config/cityscapes_random.yaml
    """
    config = read_yaml_file(path)

    config['test_images_path'] = config.get('test_images_path', None)
    config['test_labels_path'] = config.get('test_labels_path', None)

    config['num_classes'] = len(config['colors'])
    config['label_to_color'] = {label: color for label, color in enumerate(config['colors'])}
    config['color_to_label'] = {tuple(color): label for label, color in enumerate(config['colors'])}
    config['img_h'] = config.get('img_h', 512)
    config['img_w'] = config.get('img_w', 960)
    config['rgb_input'] = config.get('rgb_input', False)
    config['loss'] = config.get('loss', 'crossentropy')
    config['bsize'] = config.get('bsize', 1)
    config['lr_init'] = config.get('lr_init', 0.001)
    config['lr_decay'] = config.get('lr_decay', 1)
    config['lr_decay_freq'] = config.get('lr_decay_freq', 0)
    config['dropout'] = config.get('dropout', 0)
    config['batch_norm'] = config.get('batch_norm', False)
    config['num_filters'] = config.get('num_filters', 32)
    config['num_epochs'] = config.get('num_epochs', 50)

    return config

def main_simple():
    args = get_args()
    configuration = get_config(args.config_file)
    h, w = configuration['img_h'], configuration['img_w']
    m, o = 76, 48 #compute_norm_data(list(list_images(configuration['images_path']))[:10 if DEBUG else -1], (h, w))
    loader = tf_loader(h, w, m, o, configuration['rgb_input'], configuration['color_to_label'])
    unet_learner = UnetLearner((h, w, 3), loader, unet_preprocess_input(h, w, m, o),
                               unet_preprocess_output(h, w), **configuration)
    al_process = ActiveLearningProcess(unet_learner, configuration,
                                       unet_preprocess_input(h, w, m, o),
                                       unet_preprocess_output(h, w))
    al_process.loop()

def main_crossval():
    args = get_args()
    configuration = get_config(args.config_file)
    h, w = configuration['img_h'], configuration['img_w']
    m, o = compute_norm_data(list(list_images(configuration['images_path']))[:10 if DEBUG else -1], (w, h))
    unet_learner = UnetLearner((h, w, 3), unet_preprocess_input(h, w, m, o),
                               unet_preprocess_output(h, w), **configuration)
    al_process = ActiveLearningCrossValidationProcess(unet_learner, configuration,
                                                      unet_preprocess_input(h, w, m, o),
                                                      unet_preprocess_output(h, w))
    al_process.loop()


def simple_training():
    args = get_args()
    configuration = get_config(args.config_file)
    configuration["num_labeled"] = 3400
    configuration["budget"] = 3440
    h, w = configuration['img_h'], configuration['img_w']
    m, o = compute_norm_data(list(list_images(configuration['images_path']))[:10 if DEBUG else -1], (w, h))
    tf_preprocess = tf_loader(h, w, m, o, configuration['rgb_input'], configuration['color_to_label'])
    unet_learner = UnetLearner((h, w, 3), tf_preprocess, unet_preprocess_input(h, w, m, o),
                               unet_preprocess_output(h, w), **configuration)
    al_process = ActiveLearningProcess(unet_learner, configuration,
                                       unet_preprocess_input(h, w, m, o),
                                       unet_preprocess_output(h, w))
    al_process.loop()


if __name__ == "__main__":
    if TRAIN:
        simple_training()
    else:
        main_simple()
