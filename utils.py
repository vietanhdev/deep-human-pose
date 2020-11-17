import os
import numpy as np
import cv2
import scipy.io as sio
from math import cos, sin
from imutils import face_utils

def get_list_from_filenames(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def normalize_landmark_point(original_point, image_size):
    '''
    original_point: (x, y)
    image_size: (W, H)
    '''
    x, y = original_point
    x -= image_size[0] // 2
    y -= image_size[1] // 2
    x /= image_size[0]
    y /= image_size[1]
    return [x, y]

def unnormalize_landmark_point(normalized_point, image_size, scale=[1,1]):
    '''
    normalized_point: (x, y)
    image_size: (W, H)
    '''
    x, y = normalized_point
    x *= image_size[0]
    y *= image_size[1]
    x += image_size[0] // 2
    y += image_size[1] // 2
    x *= scale[0]
    y *= scale[1]
    return [x, y]

def unnormalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    landmark = np.multiply(np.array(landmark), np.array(image_size)) 
    landmark = landmark + image_size / 2
    return landmark

def normalize_landmark(landmark, image_size):
    image_size = np.array(image_size)
    landmark = np.array(landmark) - image_size / 2
    landmark = np.divide(landmark, np.array(image_size))
    return landmark

def draw_landmark(img, landmark):
    im_width = img.shape[1]
    im_height = img.shape[0]
    img_size = (im_width, im_height)
    landmark = landmark.reshape((-1, 2))
    unnormalized_landmark = unnormalize_landmark(landmark, img_size)
    for i in range(unnormalized_landmark.shape[0]):
        img = cv2.circle(img, (int(unnormalized_landmark[i][0]), int(unnormalized_landmark[i][1])), 2, (0,255,0), 2)
    return img