import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import math
import time
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import callbacks
from backbonds.shufflenetv2_backbond import *
import efficientnet.tfkeras as efn 
from tensorflow.keras import optimizers
import pathlib

class HeadPoseNet:
    def __init__(self, im_width, im_height, learning_rate=0.001, loss_weights=[1,1], backbond="SHUFFLE_NET_V2"):
        self.im_width = im_width
        self.im_height = im_height
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.backbond = backbond
        self.model = self.__create_model()

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.im_height, self.im_width, 3))

        if self.backbond == "SHUFFLE_NET_V2":
            feature = ShuffleNetv2(66)(inputs)
            feature = tf.keras.layers.Flatten()(feature)
        elif self.backbond == "EFFICIENT_NET_B0":
            efn_backbond = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(self.im_height, self.im_width, 3))
            efn_backbond.trainable = False
            feature = efn_backbond(inputs)
            feature = tf.keras.layers.Flatten()(feature)
            feature = tf.keras.layers.Dense(1024, activation='relu')(feature)
        else:
            raise ValueError('No such arch!... Please check the backend in config file')

        fc_1_landmarks = tf.keras.layers.Dense(512, activation='relu', name='fc_landmarks')(feature)
        fc_2_landmarks = tf.keras.layers.Dense(14, name='landmarks')(fc_1_landmarks)

        fc_1_visibility = tf.keras.layers.Dense(512, activation='relu', name='fc_visibility')(feature)
        fc_2_visibility = tf.keras.layers.Dense(7, name='visibility')(fc_1_visibility)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_2_landmarks, fc_2_visibility])
        
        losses = { 'landmarks':'mean_squared_error', 'visibility': 'binary_crossentropy'}

        model.compile(optimizer=optimizers.Adam(self.learning_rate),
                        loss=losses, loss_weights=self.loss_weights)
       
        return model

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def train(self, train_dataset, val_dataset, train_conf):

        # Load pretrained model
        if train_conf["load_weights"]:
            print("Loading model weights: " + train_conf["pretrained_weights_path"])
            self.model.load_weights(train_conf["pretrained_weights_path"])

        # Make model path
        pathlib.Path(train_conf["model_folder"]).mkdir(parents=True, exist_ok=True)

        # Define the callbacks for training
        tb = TensorBoard(log_dir=train_conf["logs_dir"], write_graph=True)
        mc = ModelCheckpoint(filepath=os.path.join(train_conf["model_folder"], train_conf["model_base_name"] + "_ep{epoch:03d}.h5"), save_weights_only=True, save_format="h5", verbose=2)
        
        self.model.fit_generator(generator=train_dataset,
                                epochs=train_conf["nb_epochs"],
                                steps_per_epoch=len(train_dataset),
                                validation_data=val_dataset,
                                validation_steps=len(val_dataset),
                                max_queue_size=64,
                                workers=1,
                                use_multiprocessing=False,
                                callbacks=[tb, mc],
                                verbose=1)
            
    def test(self, test_dataset, show_result=False):
        landmark_error = .0
        total_time = .0
        total_samples = 0

        test_dataset.set_normalization(False)
        for images, labels in test_dataset:

            batch_landmark = labels[0]

            start_time = time.time()
            batch_landmark_pred, _ = self.predict_batch(images, normalize=True)
            total_time += time.time() - start_time
            
            total_samples += np.array(images).shape[0]
    
            # Mean absolute error
            landmark_error += np.sum(np.abs(batch_landmark - batch_landmark_pred))

            # Show result
            if show_result:
                for i in range(images.shape[0]):
                    image = images[i]
                    landmark = batch_landmark_pred[i]
                    image = utils.draw_landmark(image, landmark)
                    cv2.imshow("Test result", image)
                    cv2.waitKey(0)
        
        avg_time = total_time / total_samples
        avg_fps = 1.0 / avg_time

        print("### MAE: ")
        print("- Landmark MAE: {}".format(landmark_error / total_samples / 14))
        print("- Avg. FPS: {}".format(avg_fps))
        

    def predict_batch(self, face_imgs, verbose=1, normalize=True):
        if normalize:
            img_batch = self.normalize_img_batch(face_imgs)
        else:
            img_batch = np.array(face_imgs)
        pred_landmark, pred_visibility = self.model.predict(img_batch, batch_size=1, verbose=verbose)
        return pred_landmark, pred_visibility

    def normalize_img_batch(self, face_imgs):
        image_batch = np.array(face_imgs, dtype=np.float32)
        image_batch /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_batch[..., 0] -= mean[0]
        image_batch[..., 1] -= mean[1]
        image_batch[..., 2] -= mean[2]
        image_batch[..., 0] /= std[0]
        image_batch[..., 1] /= std[1]
        image_batch[..., 2] /= std[2]
        return image_batch
