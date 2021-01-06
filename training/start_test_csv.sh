#! /bin/bash

pip install imgaug==0.4.0
pip install imagecorruptions
pip install omegaconf
pip install dlib

python3 train.py \
--net efficientnetb3 \
--dataset vggface2_age \
--pretraining efficent.hdf5 \
--preprocessing full_normalization \
--batch 64 \
--dir #add folder  \
--mode test