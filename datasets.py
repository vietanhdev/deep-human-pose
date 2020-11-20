import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import random
import textwrap
from tensorflow.keras.utils import Sequence
import math
from augmentation import augment_img
import random
import glob
import json
from augmentation import augment_img

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
        batch_landmark = []
        batch_is_pushing_up = []
        batch_contains_person = []

        for data in batch_data:
            
            # Load image
            # Flip 50% of images
            flip = False
            if self.random_flip and random.random() < 0.5:
                flip = True
        
            image, landmark, is_pushing_up, contains_person = self.load_data(self.image_folder, data, augment=self.augment, flip=flip)

            batch_image.append(image)
            batch_landmark.append(landmark)
            batch_is_pushing_up.append(is_pushing_up)
            batch_contains_person.append(contains_person)

        print(batch_landmark)

        batch_image = np.array(batch_image)
        batch_landmark = np.array(batch_landmark)
        batch_landmark = batch_landmark.reshape(batch_landmark.shape[0], -1)
        batch_is_pushing_up = np.array(batch_is_pushing_up).astype(int)
        batch_contains_person = np.array(batch_contains_person).astype(int)

        return batch_image, [batch_landmark, batch_is_pushing_up, batch_contains_person]

    def set_normalization(self, normalize):
        self.normalize = normalize

    def load_data(self, img_folder, data, augment=False, flip=False):

        landmark = data["points"]
        is_pushing_up = data["is_pushing_up"]
        contains_person = data["contains_person"]
        path = os.path.join(img_folder, data["image"])
        # print(path)
        img = cv2.imread(path)
        landmark = utils.normalize_landmark(landmark, (img.shape[1], img.shape[0]))
        img = cv2.resize(img, (self.input_size))

        if flip:

            # Flip landmark
            landmark[:, 0] = 1 - landmark[:, 0]

            # Change the indices of landmark points and visibility
            l = landmark
            landmark = [l[6], l[5], l[4], l[3], l[2], l[1], l[0]]


        unnomarlized_landmark = utils.unnormalize_landmark(landmark, self.input_size)


        if flip:
            img = cv2.flip(img, 1)
        
        if augment:
            img, unnomarlized_landmark = augment_img(img, unnomarlized_landmark)

        landmark = utils.normalize_landmark(unnomarlized_landmark, self.input_size)

        # Uncomment following lines to write out augmented images for debuging
        # cv2.imwrite("aug_" + str(random.randint(0, 50)) + ".png", img)
        # cv2.waitKey(0)

        # draw = img.copy()
        # unnomarlized_landmark = utils.unnormalize_landmark(landmark, self.input_size)
        # for i in range(len(unnomarlized_landmark)):
        #     x = int(unnomarlized_landmark[i][0])
        #     y = int(unnomarlized_landmark[i][1])

        #     draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,  
        #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.circle(draw, (int(x), int(y)), 1, (0,0,255))

        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.normalize:
            img = img.astype(np.float, copy=False)
            img /= 255.
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img[..., :] -= mean
            img[..., :] /= std

        return img, landmark, is_pushing_up, contains_person