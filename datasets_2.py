import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import random
import textwrap
from tensorflow.keras.utils import Sequence
import math
import random
import glob
import json
import imgaug as ia
from imgaug import augmenters as iaa

class DataSequence(Sequence):

    def __init__(self, image_folder, label_file, batch_size=8, input_size=(224, 224), shuffle=True, augment=False, random_flip=True, normalize=True):
        """
        Keras Sequence object to train a model on larger-than-memory data.
        """

        self.batch_size = batch_size
        self.input_size = input_size
        self.image_folder = image_folder
        self.random_flip = random_flip
        self.augment = augment
        self.normalize = normalize

        with open(label_file, "r") as fp:
            self.data = json.load(fp)["labels"]

        if shuffle:
            random.shuffle(self.data)

        self.img_augment = iaa.Sequential(
            [
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        iaa.CropAndPad(
                            percent=(-0.1, 0.1),
                            pad_mode=ia.ALL,
                            pad_cval=(0, 255)
                        ),
                        iaa.Crop(
                            percent=0.2,
                            keep_size=True
                        ),
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 0.1)),
                            iaa.AverageBlur(k=(2, 3)),
                            iaa.MedianBlur(k=(1, 3)),
                            iaa.Sharpen(alpha=(0, 0.2), lightness=(0.75, 1.5)), # sharpen images
                            iaa.Emboss(alpha=(0, 0.2), strength=(0, 0.25)), # emboss images
                            # search either for all edges or for directed edges,
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                            iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                            # either change the brightness of the whole image (sometimes
                            # per channel) or change the brightness of subareas
                        ]),
                        iaa.contrast.LinearContrast((0.5, 2.0), per_channel=0.1), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 0.1)),
                        iaa.AdditiveLaplaceNoise(scale=0.01*255),
                        iaa.AdditivePoissonNoise(lam=2),
                        iaa.Multiply(mul=(0.9, 1.1)),
                        iaa.Dropout(p=(0.1, 0.2)),
                        iaa.CoarseDropout(p=0.1, size_percent=0.05),
                        iaa.LinearContrast(),
                        iaa.AveragePooling(2),
                        iaa.MotionBlur(k=3),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
       
    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return int(math.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """

        batch_data = self.data[idx * self.batch_size: (1 + idx) * self.batch_size]

        batch_image = []
        batch_is_pushing_up = []

        for data in batch_data:
            
            # Load image
            # Flip 50% of images
            flip = False
            if self.random_flip and random.random() < 0.5:
                flip = True
        
            image, is_pushing_up = self.load_data(self.image_folder, data, augment=self.augment, flip=flip)

            batch_image.append(image)
            batch_is_pushing_up.append(is_pushing_up)

        batch_image = self.img_augment(images=batch_image)
        if self.normalize:
            batch_image = utils.normalize_img_batch(batch_image)
        batch_image = np.array(batch_image)
        batch_is_pushing_up = np.array(batch_is_pushing_up).astype(int)

        return batch_image, batch_is_pushing_up

    def set_normalization(self, normalize):
        self.normalize = normalize

    def load_data(self, img_folder, data, augment=False, flip=False):

        is_pushing_up = data["is_pushing_up"]
        path = os.path.join(img_folder, data["image"])
        img = cv2.imread(path)
        img = cv2.resize(img, (self.input_size))

        if flip:
            img = cv2.flip(img, 1)

        # cv2.imshow("draw", img)
        # cv2.waitKey(0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, is_pushing_up