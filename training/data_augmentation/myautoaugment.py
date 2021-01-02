import random
import numpy as np
from data_augmentation.transformation import apply_policy
import random
import cv2


def randomize_policies(policies):
    my_policy = []
    for policy in policies:
        if random.random() < 0.5:
            my_policy += policy
    return my_policy


class MyAutoAugmentation():
    
    def __init__(self, standard_policies, blur_policies, noise_policies):
        self.standard_policies = standard_policies
        self.blur_policies = blur_policies
        self.noise_policies = noise_policies
        self.current_policy = ''

    def before_cut(self, img, _):
        return img

    def augment_roi(self, roi):
        return roi

    def after_cut(self, img):
        policy = randomize_policies([
            random.choice(self.standard_policies),
            random.choice(self.blur_policies),
            random.choice(self.noise_policies)
        ])
        self.current_policy = str(policy)

        img = apply_policy(policy, img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        img = np.clip(img.astype(np.uint8), 0, 255)
        return img
