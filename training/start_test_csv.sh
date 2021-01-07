#! /bin/bash

pip install imgaug==0.4.0
pip install imagecorruptions
pip install omegaconf
pip install dlib

python3 train.py \
--net vgg16 \
--pretraining imagenet \
--dataset vggface2_age \
--trained_weights checkpoint.18.hdf5 \
--preprocessing full_normalization \
--batch 128 \
--dir ../testing  \
--mode test
