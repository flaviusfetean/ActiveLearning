from abc import ABC

import numpy as np
import cv2
import random

class SemsegAugmentation(ABC):

    def __init__(self, next_item=None):
        self.next_item = next_item

    def aug(self, x: np.ndarray, y: np.ndarray):
        """
        implements the specific augmentation on both the image and the mask
        """
        pass

class RandomCrop(SemsegAugmentation):

    def __init__(self, next_item, p=0.9):
        super().__init__(next_item)
        self.p = p

    def aug(self, x, y):
        h, w = x.shape[:2]

        w_cut = random.randint(0, int(w * (1. - self.p)))
        h_cut = random.randint(0, int(h * (1. - self.p)))

        h_lim = int(h_cut + h * self.p)
        w_lim = int(w_cut + w * self.p)

        x_new = cv2.resize(x[h_cut: h_lim, w_cut: w_lim], (w, h))
        y_new = cv2.resize(y[h_cut: h_lim, w_cut: w_lim], (w, h),
                           interpolation=cv2.INTER_NEAREST)

        if self.next_item is not None:
            x_new, y_new = self.next_item.aug(x_new, y_new)
        return x_new, y_new



class RandomFlip(SemsegAugmentation):

    def __init__(self, next_item):
        super().__init__(next_item)

    def aug(self, x, y):
        x_new, y_new = x, y

        if random.randint(0, 10) < 5:
            x_new, y_new = cv2.flip(x, 1), cv2.flip(y, 1)

        if self.next_item is not None:
            x_new, y_new = self.next_item.aug(x_new, y_new)
        return x_new, y_new


class RandomBrightness(SemsegAugmentation):

    def __init__(self, next_item, min_b=-0.1, max_b=0.1):
        super().__init__(next_item)
        self.min_b = min_b
        self.max_b = max_b


    def aug(self, x, y):
        rand_p = random.uniform(self.min_b, self.max_b)
        max_val = np.max(x)
        rand_modif = max_val * rand_p

        x_new = cv2.addWeighted(x, 1, np.zeros_like(x), 0, rand_modif)
        y_new = y

        if self.next_item is not None:
            x_new, y_new = self.next_item.aug(x_new, y)

        return x_new, y_new

