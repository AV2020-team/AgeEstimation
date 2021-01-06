import h5py
import pickle
import csv
from tqdm import tqdm
import os
import sys
import numpy as np
import cv2


from hdf5_loader import save_dict_to_hdf5
from vgg2_utils import get_id_from_vgg2, PARTITION_TEST, PARTITION_VAL, PARTITION_TRAIN
from vgg2_dataset_age import EXT_ROOT
from vgg2_dataset_age import CACHE_DIR
from vgg2_dataset_age import DATA_DIR
from vgg2_dataset_age import HDF5_DIR
from vgg2_dataset_age import increase_roi, get_age_from_vgg2, _readcsv


sys.path.append("../training")
from dataset_tools import DataGenerator
from model_build import senet_model_build, vgg16_keras_build, vggface_custom_build, mobilenet_224_build,\
mobilenet_96_build, mobilenet_64_build, squeezenet_build, shufflenet_224_build, xception_build, densenet_121_build, efficientnetb3_224_build


hdf = h5py.File("/home/anlaki/av/AgeEstimation/AgeEstimation/dataset/train_val_complementary_to_360375.hdf5", "r", swmr=True)

cache_file_name = "/home/anlaki/av/AgeEstimation/AgeEstimation/dataset/train_val_complementary_to_360375.cache"
weights_path = "/home/anlaki/Downloads/checkpoint.45.hdf5"

with open(cache_file_name, 'rb') as f:
    data = pickle.load(f)
    print("Data loaded. %d samples, from cache" % (len(data)))


model, _ = mobilenet_96_build(num_classes=101)
model.load_weights(weights_path)


from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import keras
NUM_CLASSES=101
def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

def age_mse(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    mse = tf.keras.losses.mean_squared_error(true_age, pred_age)
    return mse

#model = load_model(weights_path, custom_objects={'age_mae': age_mae, 'age_mse':age_mse, 'loss': keras.losses.categorical_crossentropy}, compile=False)
loss = keras.losses.categorical_crossentropy
optimizer = keras.optimizers.SGD(momentum=0.9)
model.compile(loss=loss, optimizer=optimizer, metrics=[age_mae,age_mse])

gen = DataGenerator(
                        data=data, 
                        target_shape=tuple(model.layers[0].input_shape[0][1:]),
                        batch_size=128,
                        preprocessing="full_normalization",
                        fullinfo=False,
                        with_augmentation=False,
                        num_classes=NUM_CLASSES,
                        hdf=hdf,
                        shape_predictor_path='../training/resources/shape_predictor_68_face_landmarks.dat',
                    )

print(model.evaluate_generator(gen, steps=len(gen), workers=4, verbose=True))

hdf.close()
