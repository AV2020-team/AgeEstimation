import numpy as np
import h5py
import os


def save_dict_to_hdf5(dic, filename="hdf5file.hdf5", h5file=None):
    if h5file is None:
        with h5py.File(filename, 'w') as h5file:
            recursively_save_dict_contents_to_group(h5file, '/', dic)
    else:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))


def load_dict_from_hdf5(filename):
    try:
        with h5py.File(filename, 'r') as h5file:
            return recursively_load_dict_contents_from_group(h5file, '/')
    except OSError:
        raise FileNotFoundError


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans
