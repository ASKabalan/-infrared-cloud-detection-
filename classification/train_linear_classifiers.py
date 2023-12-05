# -*- coding: utf-8 -*-
# pylint: disable=W0108
# pylint: disable=C0114
# pylint: disable=C0116
# pylint: disable=R0913

import numpy
from astropy.io import fits
from plots import plot_confusion_matrix, roc_plots
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils import get_folders, get_user_data_general, parallel_style_w_one_arg

# ---------------------------------------------------------------------------------------------------------------------


common_args = {
    "verbose": 0,
    "max_iter": 1000,
    "tol": 1e-4,
    "penalty": "l2",
    "random_state": 10,  # int or None
    "shuffle": True,
    "learning_rate": "optimal",
    "early_stopping": True,
    "n_iter_no_change": 100,
}
MODELS = [SGDClassifier(loss="hinge", **common_args), SGDClassifier(loss="log_loss", **common_args), SGDClassifier(loss="perceptron", **common_args)]


# ---------------------------------------------------------------------------------------------------------------------


# USER DATA
NAME_DB, PATH_FOLDERS, DIRECTORIES, PERCENTAGE_TRAIN, NORMA = get_user_data_general()
FOLDERS = get_folders(PATH_FOLDERS, NAME_DB, DIRECTORIES)
FOLDER_DATABASE, FOLDER_PLOTS = FOLDERS[0], FOLDERS[1]
path_image_files = sorted(FOLDER_DATABASE.glob("*.fits"))
path_labels_files = sorted(FOLDER_DATABASE.glob("*.npy"))
NB_IMGS = len(path_image_files)

# COLLECT DATASET
raw_images_data = parallel_style_w_one_arg(func=lambda arg: fits.getdata(arg), data=path_image_files)
labels_data = parallel_style_w_one_arg(func=lambda arg: numpy.load(arg), data=path_labels_files)
del path_image_files, path_labels_files

# SPLIT DATASET
raw_training_images_data, raw_test_images_data, training_labels_data, test_labels_data = train_test_split(
    raw_images_data, labels_data, train_size=PERCENTAGE_TRAIN, shuffle=True
)
del raw_images_data

# NORMALISE
if NORMA == "min_max":
    MIN_GLOBAL = numpy.amin(raw_training_images_data)
    MAX_GLOBAL = numpy.amax(raw_training_images_data)
    training_images_data = (raw_training_images_data - MIN_GLOBAL) / (MAX_GLOBAL - MIN_GLOBAL)
    test_images_data = (raw_test_images_data - MIN_GLOBAL) / (MAX_GLOBAL - MIN_GLOBAL)
elif NORMA == "mean_std":
    MEAN_GLOBAL = numpy.mean(raw_training_images_data)
    STD_GLOBAL = numpy.std(raw_training_images_data)
    training_images_data = (raw_training_images_data - MEAN_GLOBAL) / STD_GLOBAL
    test_images_data = (raw_test_images_data - MEAN_GLOBAL) / STD_GLOBAL
del raw_training_images_data, raw_test_images_data

# RESHAPE FOR FIT
reshaped_training_images_data = numpy.reshape(training_images_data, (numpy.shape(training_images_data)[0], -1))
reshaped_test_images_data = numpy.reshape(test_images_data, (numpy.shape(test_images_data)[0], -1))
del training_images_data, test_images_data

# TRAINING
for num_model, model in enumerate(MODELS):
    case = f"MODEL{num_model+1}_ratio{int(PERCENTAGE_TRAIN*100)}"
    model.fit(reshaped_training_images_data, training_labels_data)
    predictions = model.predict(reshaped_test_images_data)
    accuracy = numpy.round(accuracy_score(test_labels_data, predictions, normalize=True), decimals=2)
    plot_confusion_matrix(test_labels_data, predictions, FOLDER_PLOTS, f"{case}_acc{int(accuracy*100)}", "LINEAR CLASSIFIER")
    roc_plots(predictions, test_labels_data, FOLDER_PLOTS, f"{case}_acc{int(accuracy*100)}")
