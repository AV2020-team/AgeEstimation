#! /bin/bash

pip install imgaug==0.4.0
pip install imagecorruptions
pip install omegaconf
pip install dlib

python3 train.py \
--net vgg16 \
--dataset vggface2_age \
--pretraining imagenet \
--preprocessing full_normalization \
--augmentation myautoaugment \
--batch 128 \
--lr 0.005:0.5:5 \
--sel_gpu 0 \
--training-epochs 70 \
--weight_decay 0.005 \
--momentum \
--resume True
