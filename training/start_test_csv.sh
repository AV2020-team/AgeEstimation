#! /bin/bash

pip install imgaug==0.4.0
pip install imagecorruptions
pip install omegaconf
pip install dlib

python3 train.py \
--net mobilenet96 \
--pretraining imagenet \
--dataset vggface2_age \
--trained_weights checkpoint.45.hdf5 \
--preprocessing full_normalization \
--batch 64 \
--dir ../testing  \
--mode test