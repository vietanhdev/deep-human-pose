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
        frame = cv2.resize(frame, (224, 224))
        batch_landmark, batch_visibility = net.predict_batch(np.array([frame]))

        draw = origin_frame.copy()

        landmark = batch_landmark[0]
        landmark = landmark.reshape((7, 2))

        unnomarlized_landmark = utils.unnormalize_landmark(landmark, (origin_frame.shape[1], origin_frame.shape[0]))
        for i in range(len(unnomarlized_landmark)):
            x = int(unnomarlized_landmark[i][0])
            y = int(unnomarlized_landmark[i][1])

            draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,  
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(draw, (int(x), int(y)), 3, (0,0,255))
        

        cv2.imshow("Result", draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
