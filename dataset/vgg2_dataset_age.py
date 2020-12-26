#!/usr/bin/python3
from cv2 import cv2
from tqdm import tqdm
import os
import pickle
import numpy as np
import csv
import sys

import h5py

from vgg2_utils import get_id_from_vgg2, PARTITION_TEST, PARTITION_VAL, PARTITION_TRAIN
from hdf5_loader import save_dict_to_hdf5

sys.path.append("../training")
from dataset_tools import enclosing_square, add_margin, DataGenerator, VGGFace2Augmentation

EXT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "cache"
DATA_DIR = "data"
HDF5_DIR = "hdf5"

NUM_CLASSES = 82

FEMALE_LABEL = 0
MALE_LABEL = 1

MIN_AGE = 1
MAX_AGE = 82

vgg2age = None
age2vgg = None

MAX_DATASET_DIMENSION = 1_000_000
MAX_SAMPLES_PER_AGE = MAX_DATASET_DIMENSION // (MAX_AGE - MIN_AGE + 1)

def _load_identities(agecsv):
    global vgg2age
    global age2vgg
    if vgg2age is None:
        vgg2age = {}
        age2vgg = []
        arr = _readcsv(agecsv)
        i = 0
        for line in arr:
            try:
                vggpic = line[0][1:-4]
                age_label = get_age_label(line[-1])
                if age_label is not None:
                    vgg2age[vggpic] = (age_label, i)
                    age2vgg.append((age_label, vggpic))
                    i += 1
            except ValueError:
                pass
        print(len(age2vgg), len(vgg2age), NUM_CLASSES)


def get_age_label(age_string):
    try:
        return float(age_string)
    except:
        print("Error age deserialize")
        return None


def get_age_from_vgg2(vggpic, agecsv='vggface2/pics_with_age.csv'):
    _load_identities(agecsv)
    try:
        return vgg2age[vggpic]
    except KeyError:
        print('ERROR: n%s unknown' % vggpic)
        return 'unknown', -1


def _readcsv(csvpath, debug_max_num_samples=None):
    data = []
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True, delimiter=',', quotechar='|')
        i = 0
        for row in reader:
            if debug_max_num_samples is not None and i >= debug_max_num_samples:
                break
            i = i + 1
            data.append(row)
    return np.array(data)


def increase_roi(w, h, roi, qty):
    xmin = max(0, roi[0] - qty/2 * roi[2])
    ymin = max(0, roi[1] - qty/2 * roi[3])
    xmax = min(w, (1 + qty) * roi[2])
    ymax = min(h, (1 + qty) * roi[3])
    return xmin, ymin, xmax, ymax


def _load_vgg2(csvmeta, imagesdir, partition, debug_max_num_samples=None, hdf5_file_name="file.hdf5"):
    imagesdir = imagesdir.replace('<part>', partition)
    csvmeta = csvmeta.replace('<part>', partition)
    meta = _readcsv(csvmeta, debug_max_num_samples)
    print('csv %s read complete: %d.' % (csvmeta, len(meta)))
    idmetacsv = os.path.join(os.path.dirname(csvmeta), 'identity_meta.csv')
    agecsv = os.path.join(os.path.dirname(csvmeta), 'pics_with_age.csv')
    hdf5_file = h5py.File(hdf5_file_name, "w", swmr=True)
    data = []
    n_discarded = 0
    ages_dict = {k: 0 for k in range(MIN_AGE, MAX_AGE + 1)}
    print("shuffling meta...")
    np.random.shuffle(meta)

    index = 0
    for d in tqdm(meta):
        path = os.path.join(imagesdir, '%s' % (d[2]))
        if not os.path.isfile(path):
            #print("WARNING! Unable to read %s" % path)
            n_discarded += 1
            continue
        _, category_label = get_id_from_vgg2(int(d[3]), idmetacsv)
        sub_category_label, ret2 = get_age_from_vgg2(d[2][1:-4], agecsv)
        if ret2 == -1 or ages_dict[round(float(sub_category_label))] > MAX_SAMPLES_PER_AGE:
            continue
        img = cv2.imread(path)
        if img is None or category_label == -1:
            #print("WARNING! Unable to read %s" % path)
            n_discarded += 1
            continue
        roi = [int(x) for x in d[4:8]]
        roi = enclosing_square(roi)
        roi = increase_roi(img.shape[1], img.shape[0], roi, 0.2)
        if partition.startswith("train") or partition.startswith('val'):
            sample_partition = get_partition(category_label, sub_category_label)
        else:
            sample_partition = PARTITION_TEST
            
        if img is not None and sample_partition is not None:
            # only string as keys, type as value in dic np.ndarray, np.int64, np.float64, str, bytes, dict(recursive)
            binary_data = None
            with open(path, 'rb') as img_f:
                binary_data = img_f.read()
            example = {
                str(index): { #path non funziona, a meno che non si fa un replace degli /
                    'index': str(index),
                    'img': path,
                    'label': np.float64(sub_category_label),
                    'roi': np.asarray(roi),
                    'part': np.int64(sample_partition),
                }
            }
            if np.max(img) == np.min(img):
                print('Warning, blank image: %s!' % path)
            else:
                data.append(dict(example[str(index)]))
                example[str(index)]['img_bin'] = np.asarray(binary_data)
                save_dict_to_hdf5(example, h5file=hdf5_file)
                ages_dict[round(sub_category_label)] += 1
                index += 1
        else:  # img is None
            print("WARNING! Unable to read %s" % path)
            n_discarded += 1
    print("Data loaded. %d samples (%d discarded)" % (len(data), n_discarded))
    hdf5_file.close()
    return data


people_by_age = {age: dict() for age in range(MIN_AGE, MAX_AGE + 1)}
people_by_identity = {}


def get_partition(identity_label, age_label):
    try:
        age = round(float(age_label))
    except ValueError:
        return None
    if MIN_AGE <= age <= MAX_AGE:
        return split_by_identity(age, identity_label)
    else:
        return None


def split_by_identity(age_label, identity_label):
    global people_by_age
    try:
        faces, partition = people_by_age[age_label][identity_label]
        people_by_age[age_label][identity_label] = (faces + 1, partition)
    except KeyError:
        l = len(people_by_age[age_label])
        # split 10/90 stratified by identity
        l = (l + 1) % 10
        if l == 0:
            partition = PARTITION_VAL
        else:
            partition = PARTITION_TRAIN
        people_by_age[age_label][identity_label] = (1, partition)
    return partition


class Vgg2DatasetAge:
    def __init__(self,
                partition='train',
                imagesdir='vggface2_data/<part>',
                csvmeta='vggface2_data/annotations/<part>.detected.csv',
                target_shape=(224, 224, 3),
                augment=True,
                custom_augmentation=None,
                preprocessing='full_normalization',
                debug_max_num_samples=None):
        if partition.startswith('train'):
            partition_label = PARTITION_TRAIN
        elif partition.startswith('val'):
            partition_label = PARTITION_VAL
        elif partition.startswith('test'):
            partition_label = PARTITION_TEST
        else:
            raise Exception("unknown partition")

        self.target_shape = target_shape
        self.custom_augmentation = custom_augmentation
        self.augment = augment
        self.gen = None
        self.preprocessing = preprocessing
        self.hdf = None
        self.data = []
        print('Loading %s data...' % partition)

        num_samples = '_' + str(debug_max_num_samples) if debug_max_num_samples is not None else ''
        cache_file_name = 'vggface2_gender_{partition}{num_samples}.cache'.format(partition=partition, num_samples=num_samples)
        hdf5_file_name = f"test_{num_samples}.hdf5" if partition_label == PARTITION_TEST else f"train_val{num_samples}.hdf5"
        cache_root = os.path.join(EXT_ROOT, CACHE_DIR)
        hdf5_root = os.path.join(EXT_ROOT, HDF5_DIR)
        if not os.path.isdir(cache_root): os.mkdir(cache_root)
        cache_file_name = os.path.join(cache_root, cache_file_name)
        hdf5_file_name = os.path.join(hdf5_root, hdf5_file_name)

        print("cache file name %s" % cache_file_name)
        print("hdf5 file name %s" % hdf5_file_name)
        try:
            self.hdf = h5py.File(hdf5_file_name, 'r', swmr=True)
            with open(cache_file_name, 'rb') as f:
                self.data = pickle.load(f)
                self.data = self.data[:debug_max_num_samples]
                print("Data loaded. %d samples, from cache" % (len(self.data)))
        except (FileNotFoundError, OSError):
            if self.hdf is not None:
                print(f"Trying to reconstruct {partition} data list from hdf5...")
                loaded_data = []
                for v in tqdm(self.hdf.values()):
                    loaded_data.append({
                        'index': v['index'].value,
                        'img':  v['img'].value,
                        'label':  v['label'].value,
                        'roi':  v['roi'].value,
                        'part':  v['part'].value,
                    })
            else:
                print("Loading %s data from scratch" % partition)
                images_root = os.path.join(EXT_ROOT, DATA_DIR)
                csvmeta = os.path.join(images_root, csvmeta)
                imagesdir = os.path.join(images_root, imagesdir)

                load_partition = "train" if partition_label == PARTITION_TRAIN or partition_label == PARTITION_VAL else "test"
                loaded_data = _load_vgg2(csvmeta, imagesdir, load_partition, debug_max_num_samples, hdf5_file_name=hdf5_file_name)
            if partition.startswith('test'):
                self.data = loaded_data
            else:
                self.data = [x for x in loaded_data if x['part'] == partition_label]
            self.hdf = h5py.File(hdf5_file_name, 'r', swmr=True) if self.hdf is not None else self.hdf
            with open(cache_file_name, 'wb') as f:
                print("Pickle dumping")
                pickle.dump(self.data, f)

    def get_generator(self, batch_size=64):
        if self.gen is None:
            self.gen = DataGenerator(self.data, self.target_shape, with_augmentation=self.augment,
                                     custom_augmentation=self.custom_augmentation, batch_size=batch_size,
                                     num_classes=self.get_num_classes(), preprocessing=self.preprocessing)
        return self.gen

    def get_num_classes(self):
        return NUM_CLASSES

    def get_num_samples(self):
        return len(self.data)



def test1(dataset="test", debug_samples=None):
    global people_by_age
    if dataset.startswith("train") or dataset.startswith("val"):
        print(dataset, debug_samples if debug_samples is not None else '')
        dt = Vgg2DatasetAge(dataset, target_shape=(224, 224, 3), preprocessing='vggface2',
                               custom_augmentation=VGGFace2Augmentation(), debug_max_num_samples=debug_samples)
        print("SAMPLES %d" % dt.get_num_samples())

        """
        if len(people_by_age[MALE_LABEL]):
            print("Males %d" % (len(people_by_age[MALE_LABEL])))
            samples = [v[0] for k, v in people_by_age[MALE_LABEL].items() if v[1] == PARTITION_TRAIN]
            print("Male samples in train %d (people %d)" % (sum(samples), len(samples)))
            samples = [v[0] for k, v in people_by_age[MALE_LABEL].items() if v[1] == PARTITION_VAL]
            print("Male samples in validation %d (people %d)" % (sum(samples), len(samples)))

        if len(people_by_age[FEMALE_LABEL]):
            print("Females %d" % (len(people_by_age[FEMALE_LABEL])))
            samples = [v[0] for k, v in people_by_age[FEMALE_LABEL].items() if v[1] == PARTITION_TRAIN]
            print("Female samples in train %d (people %d)" % (sum(samples), len(samples)))
            samples = [v[0] for k, v in people_by_age[FEMALE_LABEL].items() if v[1] == PARTITION_VAL]
            print("Female samples in validation %d (people %d)" % (sum(samples), len(samples)))
        """

        print('Now generating from %s set' % dataset)
        gen = dt.get_generator()
    else:
        dv = Vgg2DatasetAge('test', target_shape=(224, 224, 3), preprocessing='full_normalization',
                               debug_max_num_samples=debug_samples, augment=False)
        print("SAMPLES %d" % dv.get_num_samples())
        print('Now generating from test set')
        gen = dv.get_generator()

    i = 0
    while True:
        print(i)
        i += 1
        for batch in tqdm(gen):
            for im, age in zip(batch[0], batch[1]):
                facemax = np.max(im)
                facemin = np.min(im)
                im = (255 * ((im - facemin) / (facemax - facemin))).astype(np.uint8)
                cv2.putText(im, "%.2f" % age, (0, im.shape[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))
                cv2.imshow('vggface2 image', im)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    return


if '__main__' == __name__:
    test1("train")
    # test1("val")
    # test1("test", 2_000)
    
    # test1("train") # cache
    # test1("val") # cache
    # test1("test") # cache
