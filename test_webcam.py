import argparse
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
    ret, origin_frame = cap.read()
    if ret:
        # Convert crop image to RGB color space
        frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.imread("/mnt/DATA/PUSHUP_PROJECT/deep-human-pose/data/mpii/processed_images/0.png")
        frame = cv2.resize(frame, (224, 224))
        batch_landmark, batch_is_pushing_up, batch_contains_person = net.predict_batch(np.array([frame]))

        net_input_size = (config["model"]["im_width"], config["model"]["im_height"])
        scale = np.divide(np.array([origin_frame.shape[1], origin_frame.shape[0]]), np.array(net_input_size))

        draw = origin_frame.copy()

        landmark = batch_landmark[0]
        is_pushing_up = batch_is_pushing_up[0]
        contains_person = batch_contains_person[0]

        for j in range(7):
            x = landmark[2 * j]
            y = landmark[2 * j + 1]
            x, y = utils.unnormalize_landmark_point(
                (x, y), net_input_size, scale=scale)
            x = int(x)
            y = int(y)
            draw = cv2.circle(draw, (x, y), 2, (255, 0, 0), 2)
            cv2.putText(draw, 'Pushing:{}, Person:{}'.format(is_pushing_up, contains_person), (100, 100), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.5, (0, 0, 255), 1, cv2.LINE_AA) 


        cv2.imshow("Result", draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
