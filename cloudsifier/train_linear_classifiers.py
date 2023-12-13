# -*- coding: utf-8 -*-
# pylint: disable=W0108

import numpy
from astropy.io import fits
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.model_selection import train_test_split

from plots import matrix_confusion, report, roc
from utils import get_folders, get_user_data_general, parallel_style_w_one_arg

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


common_args = {"max_iter": 5000, "tol": 1e-4, "learning_rate": "optimal", "early_stopping": True, "n_iter_no_change": 50}

# MODELS = [
#     SGDClassifier(loss="hinge", **common_args),
#     SGDClassifier(loss="log_loss", **common_args),
#     SGDClassifier(loss="perceptron", **common_args),
#     RidgeClassifier(max_iter=5000, solver="svd"),
# ]

# NAMES = ["SVM", "LOGISTIC_REGRESSION", "PERCEPTRON", "RIDGE_REGRESSION"]

MODELS = [
    RidgeClassifier(max_iter=5000, solver="svd"),
]

NAMES = ["RIDGE_REGRESSION"]

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------


# USER DATA
NAME_DB, PATH_FOLDERS, DIRECTORIES, PERCENTAGE_TRAIN, _ = get_user_data_general()
FOLDERS = get_folders(PATH_FOLDERS, NAME_DB, DIRECTORIES)
FOLDER_DATABASE, FOLDER_PLOTS = FOLDERS[0], FOLDERS[1]
path_image_files = sorted(FOLDER_DATABASE.glob("*.fits"))
path_labels_files = sorted(FOLDER_DATABASE.glob("*.npy"))

# COLLECT DATASET
raw_images_data = parallel_style_w_one_arg(func=lambda arg: fits.getdata(arg), data=path_image_files)
labels_data = parallel_style_w_one_arg(func=lambda arg: numpy.load(arg), data=path_labels_files)
del path_image_files, path_labels_files

# SPLIT DATASET
raw_train_images_data, raw_test_images_data, training_labels_data, test_labels_data = train_test_split(
    raw_images_data, labels_data, train_size=PERCENTAGE_TRAIN, shuffle=True
)
del raw_images_data

# NORMALISE
MIN_GLOBAL, MAX_GLOBAL = numpy.amin(raw_train_images_data), numpy.amax(raw_train_images_data)
train_images_data = (raw_train_images_data - MIN_GLOBAL) / (MAX_GLOBAL - MIN_GLOBAL)
test_images_data = (raw_test_images_data - MIN_GLOBAL) / (MAX_GLOBAL - MIN_GLOBAL)
del raw_train_images_data, raw_test_images_data

# RESHAPE FOR FIT
reshaped_train_images_data = numpy.reshape(train_images_data, (numpy.shape(train_images_data)[0], -1))
reshaped_test_images_data = numpy.reshape(test_images_data, (numpy.shape(test_images_data)[0], -1))
del train_images_data, test_images_data

# TRAINING
for num_model, model in enumerate(MODELS):
    case = f"Model{NAMES[num_model]}_Ratio{int(PERCENTAGE_TRAIN*100)}"
    print(case)
    model.fit(reshaped_train_images_data, training_labels_data)
    predictions = model.predict(reshaped_test_images_data)
    matrix_confusion(test_labels_data, predictions, FOLDER_PLOTS, case, title=NAMES[num_model])
    roc(predictions, test_labels_data, FOLDER_PLOTS, case)
    report(test_labels_data, predictions, FOLDER_PLOTS, case)
