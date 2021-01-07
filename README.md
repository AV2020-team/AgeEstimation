# Age Estimation - Group 10

Final project for Artificial Vision 2020/2021 - Group 10

## Team Members
* [Giovanni Ammendola](https://github.com/giorge1)
* [Edoardo Maffucci](https://github.com/emaff)
* [Vincenzo Petrone](https://github.com/v8p1197)
* [Salvatore Scala](https://github.com/knowsx2)

## Usage

### Training

1. Open the [Google Colab Notebook](https://colab.research.google.com/drive/1gQ9vi4v3GxIAruQbz6h5aE_1zmVrpaMX?authuser=1)
2. Run all cells in sections 1 and 2

### Testing

1. Open the [Google Colab Notebook](https://colab.research.google.com/drive/1gQ9vi4v3GxIAruQbz6h5aE_1zmVrpaMX?authuser=1)
2. Run all cells in sections 1 and 3

## Contents

This project derives from [MIVIA](https://github.com/MiviaLab)'s [Gender Recognition Framework](https://github.com/AV2020-team/GenderRecognitionFramework).

Our changes consist in:

* [training](https://github.com/AV2020-team/AgeEstimation/tree/master/training) package:
	* [data_augmentation](https://github.com/AV2020-team/AgeEstimation/tree/master/training/data_augmentation) package:
		* [myautoaugment.py](https://github.com/AV2020-team/AgeEstimation/blob/master/training/data_augmentation/myautoaugment.py): defines `MyAutoAugment` class, performing our custom data augmentation on an image
		* [policies.py](https://github.com/AV2020-team/AgeEstimation/blob/master/training/data_augmentation/policies.py): defines standard, blur and noise policies
		* [transformation.py](https://github.com/AV2020-team/AgeEstimation/blob/master/training/data_augmentation/transformation.py): defines the methods that actual perform data augmentation transformations, using library [imgaug](https://imgaug.readthedocs.io/en/latest/) ([GitHub repository](https://github.com/aleju/imgaug))
	* [dataset_tools.py](https://github.com/AV2020-team/AgeEstimation/blob/master/training/dataset_tools.py) script:
		* Face alignment method
		* `DataGenerator` class modified: it loads samples from an `hdf5` file, performing preprocessing and augmentation
		* `DataTestGenerator` class created: loads test samples with or without ROI available, performing only preprocessing
	* [model_build.py](https://github.com/AV2020-team/AgeEstimation/blob/master/training/model_build.py) script:
		* `efficientnetb3_224_build`  method added: uses EfficientNetB3_224x224 backbone
	* [start_test_csv.sh](https://github.com/AV2020-team/AgeEstimation/blob/master/training/start_test_csv.sh) script: performs testing
	* [start_train.py.sh](https://github.com/AV2020-team/AgeEstimation/blob/master/training/start_train.py.sh) script: performs training
	* [train.py](https://github.com/AV2020-team/AgeEstimation/blob/master/training/train.py) script:
		* `reduce_lr_on_plateau` method created: returns a [ReduceLROnPlateau](https://keras.io/api/callbacks/reduce_lr_on_plateau/) scheduling
		* `age_mae` and `age_mse` metrics methods created: compute MAE and MSE on predicted and true labels
		* `test` mode: generates a `csv` file containing the labelled test samples, using the output predicted by the loaded model
* [dataset](https://github.com/AV2020-team/AgeEstimation/tree/master/dataset) package:
	* [vgg2_dataset_age.py](https://github.com/AV2020-team/AgeEstimation/blob/master/dataset/vgg2_dataset_age.py) script: all methods have been modified to generate also an `hdf5` file containing the original compressed images and ages as labels
	* [complementary_ds.py](https://github.com/AV2020-team/AgeEstimation/blob/master/dataset/complementary_ds.py) script: creates a labelled test set containing samples not included in the original training set
	* [test_complementary_ds.py](https://github.com/AV2020-team/AgeEstimation/blob/master/dataset/test_complementary_ds.py) script: tests the loaded model on the generated complementary dataset
