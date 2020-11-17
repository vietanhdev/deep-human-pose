import argparse
from RetinaFace.retinaface import RetinaFace
import models
import utils
import datasets
from imutils import face_utils
import scipy.io as sio
import cv2
import numpy as np
import os
import json
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--model_file', default="./models/shuffle_net_dhp.h5",
    help='Output model file')
parser.add_argument(
    '-c',
    '--conf_file', default="config.json",
    help='Configuration file')
args = parser.parse_args()

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

# Build model
net = models.HeadPoseNet(config["model"]["im_width"], config["model"]
                         ["im_height"],
                         learning_rate=config["train"]["learning_rate"],
                         backbond=config["model"]["backbond"])
# Load model
net.load_weights(config["test"]["model_file"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Convert crop image to RGB color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_landmark, batch_visibility = net.predict_batch(np.array([frame]))

        draw = frame.copy()

        landmark = batch_landmark[0]
        print(landmark)

        cv2.imshow("Result", draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
