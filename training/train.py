import warnings
warnings.filterwarnings("ignore")

import argparse

available_nets = ['senet50', 'vgg16', 'densenet121bc', 'xception', 'xception71', 'mobilenet96', 'mobilenet224',
                  'mobilenet64_bio', 'shufflenet224', 'squeezenet', 'efficientnetb3', 'resnet50']

available_normalizations = ['z_normalization', 'full_normalization', 'vggface2']
available_augmentations = ['default', 'vggface2', 'autoaugment-rafdb', 'no', 'myautoaugment']
available_modes = ['train', 'training', 'test', 'train_inference', 'test_inference']
available_lpf = [0, 1, 2, 3, 5, 7]

parser = argparse.ArgumentParser(description='Common training and evaluation.')
parser.add_argument('--lpf', dest='lpf_size', type=int, choices=available_lpf, default=1, help='size of the lpf filter (1 means no filtering)')
parser.add_argument('--cutout', action='store_true', help='use cutout augmentation')
parser.add_argument('--center_loss', action='store_true', help='use center loss')
parser.add_argument('--weight_decay', type=float)
parser.add_argument('--lr', default='0.002', help='Initial learning rate or init:factor:epochs', type=str)
parser.add_argument('--momentum', action='store_true')
parser.add_argument('--dataset', dest='dataset', type=str, default="vggface2_age", help='dataset to use for the training')
parser.add_argument('--mode', dest='mode', type=str,choices=available_modes, default='train', help='train or test')
parser.add_argument('--epoch', dest='test_epoch', type=int, default=None, help='epoch to be used for testing, mandatory if mode=test')
parser.add_argument('--training-epochs', dest='n_training_epochs', type=int, default=220, help='epoch to be used for training, default 220')
parser.add_argument('--dir', dest='dir', type=str, default=None, help='directory for reading/writing training data and logs')
parser.add_argument('--batch', dest='batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--ngpus', dest='ngpus', type=int, default=1, help='Number of gpus to use.')
parser.add_argument('--sel_gpu', dest='selected_gpu', type=str, default="0", help="one number or two numbers separated by a hyphen")
parser.add_argument('--net', type=str, choices=available_nets, help='Network architecture')
parser.add_argument('--resume', type=bool, default=False, help='resume training')
parser.add_argument('--pretraining', type=str, default=None, help='Pretraining weights, do not set for None, can be vggface or imagenet or a file')
parser.add_argument('--preprocessing', type=str, default='full_normalization', choices=available_normalizations)
parser.add_argument('--augmentation', type=str, default='default', choices=available_augmentations)
parser.add_argument('--validation_steps', type=int, default=None)
parser.add_argument('--steps_per_epoch', type=int, default=None)

args = parser.parse_args()


import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os
import numpy as np
from glob import glob
import re
import tensorflow as tf
import keras
import time
from center_loss import center_loss
from datetime import datetime
from model_build import senet_model_build, vgg16_keras_build, vggface_custom_build, mobilenet_224_build,\
mobilenet_96_build, mobilenet_64_build, squeezenet_build, shufflenet_224_build, xception_build, densenet_121_build, efficientnetb3_224_build


if args.dataset == 'vggface2_age':
    sys.path.append("../dataset")
    from vgg2_dataset_age import Vgg2DatasetAge as Dataset, NUM_CLASSES
else:
    print('unknown dataset %s' % args.dataset)
    exit(1)


# Learning Rate
lr = args.lr.split(':')
initial_learning_rate = float(lr[0])  # 0.002
learning_rate_decay_factor = float(lr[1]) if len(lr) > 1 else 0.5
learning_rate_patience = int(lr[2]) if len(lr) > 2 else 5

# Epochs to train
n_training_epochs = args.n_training_epochs

# Batch size
batch_size = args.batch_size


def reduce_lr_on_plateau(decay_factor=0.5, patience=10):
	# Learning Rate
	monitor = "val_loss"
	verbose = 1
	mode = "min"
	min_delta = 0.01

	return keras.callbacks.ReduceLROnPlateau(
		monitor=monitor,
		factor=decay_factor,
		patience=patience,
		verbose=verbose,
		mode=mode,
		min_delta=min_delta
	)


# Model building
INPUT_SHAPE = None

def get_model():
    global INPUT_SHAPE
    if args.net.startswith('senet') or args.net.startswith('resnet') or args.net.startswith('vgg'):
        INPUT_SHAPE = (224, 224, 3)
        if args.pretraining.startswith('imagenet'):
            if args.net.startswith('senet') or args.net.startswith('resnet'):
                return senet_model_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
            else:
                return vgg16_keras_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
        else:
            print("VGGFACE Network")
            return vggface_custom_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining, args.net, args.lpf_size)
    elif args.net == 'mobilenet96':
        INPUT_SHAPE = (96, 96, 3)
        return mobilenet_96_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == 'mobilenet224':
        INPUT_SHAPE = (224, 224, 3)
        return mobilenet_224_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == 'mobilenet64_bio':
        INPUT_SHAPE = (64, 64, 3)
        return mobilenet_64_build(INPUT_SHAPE, NUM_CLASSES)
    elif args.net == 'densenet121bc':
        INPUT_SHAPE = (224, 224, 3)
        return densenet_121_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining, args.lpf_size)
    elif args.net.startswith('xception'):
        INPUT_SHAPE = (71, 71, 3) if args.net == 'xception71' else (299, 299, 3)
        return xception_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining, args.lpf_size)
    elif args.net == "shufflenet224":
        INPUT_SHAPE = (224, 224, 3)
        return shufflenet_224_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == "squeezenet":
        INPUT_SHAPE = (224, 224, 3)
        return squeezenet_build(INPUT_SHAPE, NUM_CLASSES, args.pretraining)
    elif args.net == "efficientnetb3":
        INPUT_SHAPE = (224, 224, 3)
        return efficientnetb3_224_build(INPUT_SHAPE, NUM_CLASSES)


# Model creating
gpu_to_use = [str(s) for s in args.selected_gpu.split(',') if s.isdigit()]
if args.ngpus <= 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
    model, feature_layer = get_model()
else:
    if len(gpu_to_use) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_to_use)
    print("WARNING: Using GPU %s" % os.environ["CUDA_VISIBLE_DEVICES"])
    print("WARNING: Using %d gpus" % args.ngpus)
    with tf.device('/cpu:0'):
        model, feature_layer = get_model()
    model = keras.utils.multi_gpu_model(model, args.ngpus)
model.summary()

from keras import backend as K
def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

def age_mse(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, NUM_CLASSES, dtype="float32"), axis=-1)
    mse = tf.keras.losses.mean_squared_error(true_age, pred_age)
    return mse

# model compiling
if args.weight_decay:
    weight_decay = args.weight_decay  # 0.0005
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, keras.layers.DepthwiseConv2D) or isinstance(
                layer, keras.layers.Dense):
            layer.add_loss(lambda: keras.regularizers.l2(weight_decay)(layer.kernel))
        if hasattr(layer, 'bias_regularizer') and layer.use_bias:
            layer.add_loss(lambda: keras.regularizers.l2(weight_decay)(layer.bias))
optimizer = keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.9) if args.momentum else keras.optimizers.SGD(learning_rate=initial_learning_rate)
if args.center_loss:
    loss = center_loss(feature_layer, keras.losses.categorical_crossentropy, 0.9, NUM_CLASSES, 0.01, features_dim=2048)
else:
    loss = keras.losses.categorical_crossentropy if NUM_CLASSES > 1 else keras.losses.mean_squared_error
accuracy_metrics = [keras.metrics.categorical_accuracy] if NUM_CLASSES > 1 else [keras.metrics.mean_squared_error]
model.compile(loss=loss, optimizer=optimizer, metrics=[age_mae,age_mse])


# Directory creating to store model checkpoints
datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
dirnm = "inference_time_test" if args.mode.endswith('inference') else "trained"
dirnm = os.path.join("..", dirnm)
if not os.path.isdir(dirnm): os.mkdir(dirnm)
argstring = ''.join(sys.argv[1:]).replace('--', '_').replace('=', '').replace(':', '_')
dirnm += '/%s' % (argstring)
if args.cutout: dirnm += '_cutout'
if args.dir: dirnm = args.dir
if not os.path.isdir(dirnm): os.mkdir(dirnm)
filepath = os.path.join(dirnm, "checkpoint.{epoch:02d}.hdf5")
logdir = dirnm
ep_re = re.compile('checkpoint.([0-9]+).hdf5')


def _find_latest_checkpoint(d):
    all_checks = glob(os.path.join(d, '*'))
    max_ep = 0
    max_c = None
    for c in all_checks:
        epoch_num = re.search(ep_re, c)
        if epoch_num is not None:
            epoch_num = int(epoch_num.groups(1)[0])
            if epoch_num > max_ep:
                max_ep = epoch_num
                max_c = c
    return max_ep, max_c



# AUGMENTATION 
if args.cutout:
    from cropout_test import CropoutAugmentation
    custom_augmentation = CropoutAugmentation()
elif args.augmentation == 'autoaugment-rafdb':
    from autoaug_test import MyAutoAugmentation
    from autoaugment.rafdb_policies import rafdb_policies
    custom_augmentation = MyAutoAugmentation(rafdb_policies)
elif args.augmentation == 'default':
    from dataset_tools import DefaultAugmentation
    custom_augmentation = DefaultAugmentation()
elif args.augmentation == 'vggface2':
    from dataset_tools import VGGFace2Augmentation
    custom_augmentation = VGGFace2Augmentation()
elif args.augmentation == 'myautoaugment':
    sys.path.append("./data_augmentation")
    from myautoaugment import MyAutoAugmentation
    from policies import blur_policies, noise_policies, standard_policies
    custom_augmentation = MyAutoAugmentation(standard_policies, blur_policies, noise_policies)
else:
    custom_augmentation = None


if args.mode.startswith('train'):
    print("TRAINING %s" % dirnm)
    dataset_training = Dataset('train', target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing, custom_augmentation=custom_augmentation)
    dataset_validation = Dataset('val', target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing)

    lr_sched = reduce_lr_on_plateau(decay_factor=learning_rate_decay_factor, patience=learning_rate_patience)
    monitor = 'val_categorical_accuracy' if NUM_CLASSES > 1 else 'val_mean_squared_error'
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=False, monitor=monitor)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=True, write_images=True)
    callbacks_list = [lr_sched, checkpoint, tbCallBack]

    if args.mode == "train_inference":
        batch_size = 1

    initial_epoch = 0
    if args.resume:
        pattern = filepath.replace('{epoch:02d}', '*')
        epochs = glob(os.path.join(dirnm, "*"))
        print(dirnm)
        print(epochs)
        epochs = [e for e in epochs if "hdf5" in e]
        checkpoints = []
        for x in epochs:
            try:
                checkpoints.append(int(x[-8:-5].replace('.', '')))
            except ValueError:
                continue
        if len(checkpoints) != 0:
            initial_epoch = max(checkpoints)
            print('Resuming from epoch %d...' % initial_epoch)
            model.load_weights(filepath.format(epoch=initial_epoch))

    model.fit_generator(generator=dataset_training.get_generator(batch_size),
                        validation_data=dataset_validation.get_generator(batch_size),
                        verbose=1, callbacks=callbacks_list, epochs=n_training_epochs, workers=32,
                        initial_epoch=initial_epoch)
    if args.mode == "train_inference":
        print("Warning: TEST ON CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
        model.fit_generator(generator=dataset_training.get_generator(batch_size),
                            validation_data=dataset_validation.get_generator(batch_size),
                            verbose=1, callbacks=callbacks_list, epochs=n_training_epochs, workers=8,
                            initial_epoch=initial_epoch)
elif args.mode == 'test':
    if args.test_epoch is None:
        args.test_epoch, _ = _find_latest_checkpoint(dirnm)
        print("Using epoch %d" % args.test_epoch)
    #model.load_weights(filepath.format(epoch=int(args.test_epoch)))
    model.load_weights(os.path.join(dirnm, args.pretraining))

    # TODO : add test_inference mode 
    
    def evalds(part):
        from tqdm import tqdm
        import csv

        import cv2
        import matplotlib.pyplot as plt
        from matplotlib import interactive
        interactive(True)
        xs = [x for x in range(101)]

        print('Evaluating %s results...' % part)
        dataset = Dataset(part, target_shape=INPUT_SHAPE, augment=False, preprocessing=args.preprocessing)
        if part == 'test':
            gen = dataset.get_generator(batch_size=batch_size, fullinfo=True)
            results = {}
            fig = plt.figure()

            for batch in tqdm(gen):
                for img, abs_path, path in zip(batch[0], batch[1], batch[2]):
                    frame = np.reshape(img, [1] + list(INPUT_SHAPE))
                    results[path] = model.predict(frame)
                    """
                    human_img = cv2.imread(abs_path)
                    human_img = cv2.resize(human_img, INPUT_SHAPE[:2])
                    cv2.putText(human_img, "%.2f" % (np.argmax(results[path]) + 1), (1, human_img.shape[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 8)
                    cv2.imshow('vggface2 image', human_img)
                    fig = plt.figure()
                    plt.plot(xs, results[path][0])
                    fig.canvas.draw()
                    plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    plot  = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR)
                    cv2.imshow("plot", plot)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return
                    """
           
        else:
            results = model.evaluate_generator(dataset.get_generator(batch_size), verbose=1, workers=4)
            print('%s results: loss %.3f - accuracy %.3f' % (part, results[0], results[1]))

        import pickle
        with open(os.path.join(dirnm, "results.dat"), 'wb') as f:
            print("Pickle dumping")
            pickle.dump(results, f)

        lines = []
        with open(os.path.join(dirnm, "results.csv"), 'w') as f:
            writer = csv.writer(f, delimiter=',')
            for image, res in tqdm(results.items()):
                writer.writerow([image,(np.argmax(res) + 1)])

    evalds('test')
    #evalds('val')
    #evalds('train')
