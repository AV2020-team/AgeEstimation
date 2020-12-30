#! /bin/bash

python3 train.py \
--net efficientnetb3 \
--dataset vggface2_age \
--pretraining EfficientNetB3_224_weights.11-3.44.hdf5 \
--preprocessing full_normalization \
--augmentation myautoaugment \
--batch 128 \
--lr 0.005:0.2:20 \
--sel_gpu 0 \
--training-epochs 70 \
--weight_decay 0.005 \
--momentum