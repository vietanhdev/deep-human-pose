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
from sklearn.metrics import precision_recall_fscore_support

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
            feature = ShuffleNetv2()(inputs)
            feature = tf.keras.layers.Flatten()(feature)
        elif self.backbond == "EFFICIENT_NET_B0":
            efn_backbond = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(self.im_height, self.im_width, 3))
            efn_backbond.trainable = False
            feature = efn_backbond(inputs)
            feature = tf.keras.layers.Flatten()(feature)
            feature = tf.keras.layers.Dense(1024, activation='relu')(feature)
        elif self.backbond == "EFFICIENT_NET_B3":
            efn_backbond = efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(self.im_height, self.im_width, 3))
            efn_backbond.trainable = False
            feature = efn_backbond(inputs)
            feature = tf.keras.layers.Flatten()(feature)
            feature = tf.keras.layers.Dense(1024, activation='relu')(feature)
        elif self.backbond == "EFFICIENT_NET_B4":
            efn_backbond = efn.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(self.im_height, self.im_width, 3))
            efn_backbond.trainable = False
            feature = efn_backbond(inputs)
            feature = tf.keras.layers.Flatten()(feature)
            feature = tf.keras.layers.Dense(1024, activation='relu')(feature)
        else:
            raise ValueError('No such arch!... Please check the backend in config file')

        fc_1_landmarks = tf.keras.layers.Dense(256, activation='relu', name='fc_landmarks')(feature)
        fc_2_landmarks = tf.keras.layers.Dense(14, name='landmarks')(fc_1_landmarks)

        fc_1_is_pushing_up = tf.keras.layers.Dense(256, activation='relu', name='fc_is_pushing_up')(feature)
        fc_2_is_pushing_up = tf.keras.layers.Dense(1, name='is_pushing_up', activation="sigmoid")(fc_1_is_pushing_up)

        fc_1_contains_person = tf.keras.layers.Dense(256, activation='relu', name='fc_contains_person')(feature)
        fc_2_contains_person = tf.keras.layers.Dense(1, name='contains_person', activation="sigmoid")(fc_1_contains_person)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_2_landmarks, fc_2_is_pushing_up, fc_2_contains_person])
        
        def landmark_loss():
            def landmark_loss_func(y_true, y_pred):
                lm_loss = tf.keras.backend.switch(
                    tf.keras.backend.max(y_pred) == -1.0 and tf.keras.backend.min(y_pred) == -1.0,
                    0.0,
                    tf.keras.losses.MSE(y_true, y_pred)
                )
                return lm_loss
            return landmark_loss_func
        losses = { 'landmarks': landmark_loss(), 'is_pushing_up': 'binary_crossentropy', 'contains_person': 'binary_crossentropy'}

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
                                # max_queue_size=64,
                                # workers=6,
                                # use_multiprocessing=True,
                                callbacks=[tb, mc],
                                verbose=1)
            
    def test(self, test_dataset, show_result=False):
        landmark_error = .0
        total_time = .0
        total_samples = 0

        test_dataset.set_normalization(False)
        total_landmark = []
        total_is_pushing_up = []
        total_contains_person = []
        total_landmark_pred = []
        total_is_pushing_up_pred = []
        total_contains_person_pred = []
        for images, labels in test_dataset:

            batch_landmark, batch_is_pushing_up, batch_contains_person = labels
            total_landmark += batch_landmark.tolist()
            total_is_pushing_up += batch_is_pushing_up.tolist()
            total_contains_person += batch_contains_person.tolist()

            start_time = time.time()
            batch_landmark_pred, batch_is_pushing_up_pred, batch_contains_person_pred = self.predict_batch(images, normalize=True)
            total_time += time.time() - start_time

            total_landmark_pred += batch_landmark_pred.tolist()
            total_is_pushing_up_pred += batch_is_pushing_up_pred.tolist()
            total_contains_person_pred += batch_contains_person_pred.tolist()
            
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
        print("- Pushing up: ", precision_recall_fscore_support(total_is_pushing_up, total_is_pushing_up_pred, average='macro'))
        print("- Contains person: ", precision_recall_fscore_support(total_contains_person, total_contains_person_pred, average='macro'))
        print("- Avg. FPS: {}".format(avg_fps))
        

    def predict_batch(self, face_imgs, verbose=1, normalize=True):
        if normalize:
            img_batch = self.normalize_img_batch(face_imgs)
        else:
            img_batch = np.array(face_imgs)
        pred_landmark, pred_is_pushing_up, pred_contains_person = self.model.predict(img_batch, batch_size=1, verbose=verbose)
        return pred_landmark, pred_is_pushing_up, pred_contains_person

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
