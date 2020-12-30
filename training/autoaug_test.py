import sys
import numpy as np
import cv2

from data_augmentation.myautoaugment import MyAutoAugmentation
from data_augmentation.policies import standard_policies, blur_policies, noise_policies

sys.path.append('./dataset')
from vgg2_dataset_age import Vgg2DatasetAge as Dataset


def show_one_image():
    TARGET_SHAPE= (300, 300, 3)
    P = 'train'
    print('Partition: %s'%P)
    while True:
        NUM_ROWS = 2
        NUM_COLS = 2
        imout = np.zeros( (TARGET_SHAPE[0]*NUM_ROWS,TARGET_SHAPE[1]*NUM_COLS,3), dtype=np.uint8 )
        print(imout.shape)
        for ind1 in range(NUM_ROWS):
            for ind2 in range(NUM_COLS):
                a = MyAutoAugmentation(standard_policies, blur_policies, noise_policies)
                
                dataset_test = Dataset(partition=P, target_shape=TARGET_SHAPE,
                            debug_max_num_samples=1, augment=False, custom_augmentation=a)
                imex = np.squeeze(dataset_test.get_generator(1).__getitem__(0)[0],0)
                imex = ((imex*127)+127).clip(0,255).astype(np.uint8)
                #imex_corrupted = a.before_cut(imex)
                imex_corrupted = imex
                off1=ind1*TARGET_SHAPE[0]
                off2=ind2*TARGET_SHAPE[1]
                imout[off1:off1+TARGET_SHAPE[0],off2:off2+TARGET_SHAPE[1],:] = imex_corrupted

        #imout = cv2.resize(imout, (TARGET_SHAPE[0]*2, TARGET_SHAPE[1]*2))
        cv2.imshow('imout', imout)
        k = cv2.waitKey(0)
        if k==27:
            sys.exit(0)


if '__main__' == __name__:
    show_one_image()
