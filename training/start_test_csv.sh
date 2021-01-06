#! /bin/bash


python3 train.py \
--net mobilenet96 \
--pretraining imagenet \
--dataset vggface2_age \
--trained_weights checkpoint.45.hdf5 \
--preprocessing full_normalization \
--batch 64 \
--dir /home/anlaki/Downloads  \
--mode test