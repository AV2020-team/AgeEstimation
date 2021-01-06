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
from dataset_tools import enclosing_square

def complementary_ds(debug_max_num_samples=None,
                    hdf5_file_name='train_val_complementary_to_SIZE.hdf5'
                    ):
    csvmeta = 'vggface2_data/annotations/train.detected.csv'
    agecsv = 'vggface2_data/annotations/pics_with_age.csv'
    imagesdir = 'vggface2_data/train'
    csvmeta = os.path.join(DATA_DIR, csvmeta)
    agecsv = os.path.join(DATA_DIR, agecsv)
    meta = _readcsv(csvmeta, None)
    
    images_root = os.path.join(EXT_ROOT, DATA_DIR)
    imagesdir = os.path.join(images_root, imagesdir)

    existent_ds_val = 'vggface2_age_val.cache.medium'
    existent_ds_train = 'vggface2_age_train.cache.medium'

    cache_root = os.path.join(EXT_ROOT, CACHE_DIR)
    with open(os.path.join(cache_root, existent_ds_val), 'rb') as f:
        data_val = pickle.load(f)
        print("Data val loaded. %d samples, from cache" % (len(data_val)))

    with open(os.path.join(cache_root, existent_ds_train), 'rb') as f:
        data_train = pickle.load(f)
        print("Data train loaded. %d samples, from cache" % (len(data_train)))

    all_path = []
    for e in tqdm(data_val):
        all_path.append(e['img'])

    for e in tqdm(data_train):
        all_path.append(e['img'])

    index = 0
    data = []
    np.random.shuffle(meta)
    if debug_max_num_samples is not None:
        meta = meta[:debug_max_num_samples]
    hdf5_file = h5py.File(hdf5_file_name.replace('SIZE', str(len(data_val) + len(data_train))), "r", swmr=True)
    n_discarded = 0
    del data_val
    del data_train
    for d in tqdm(meta):
        path = os.path.join(imagesdir, '%s' % (d[2]))
        if path in all_path:
            all_path.remove(path)
            continue
        if not os.path.isfile(path):
            #print("WARNING! Unable to read %s" % path)
            n_discarded += 1
            continue
        img = cv2.imread(path)
        if img is None:
            #print("WARNING! Unable to read %s" % path)
            n_discarded += 1
            continue
        roi = [int(x) for x in d[4:8]]
        roi = enclosing_square(roi)
        roi = increase_roi(img.shape[1], img.shape[0], roi, 0.2)
        sub_category_label, ret2 = get_age_from_vgg2(d[2][1:-4], agecsv)
        if ret2 == -1:
            continue
        binary_data = None
        with open(path, 'rb') as img_f:
            binary_data = img_f.read()
        example = {
                str(index): { #path non funziona, a meno che non si fa un replace degli /
                    'index': str(index),
                    'img': path,
                    'label': np.float64(sub_category_label),
                    'roi': np.asarray(roi),
                    'part': np.int64(0),
                }
            }
        if np.max(img) == np.min(img):
            print('Warning, blank image: %s!' % path)
        else:
            data.append(dict(example[str(index)]))
            example[str(index)]['img_bin'] = np.asarray(binary_data)
            save_dict_to_hdf5(example, h5file=hdf5_file)
            index += 1

    print("Data loaded. %d samples (%d discarded)" % (len(data), n_discarded))
    hdf5_file.close()
    return data

complementary_ds(debug_max_num_samples=150000)
a = 42
print(a)
