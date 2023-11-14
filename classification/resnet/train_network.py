# -*- coding: utf-8 -*-
# pylint: disable=C0114
# pylint: disable=C0116
# pylint: disable=E1102

import jax
from tqdm import tqdm

from data_loader import DataLoader, chosen_datasets
from model import (
    create_train_state,
    eval_function,
    load_model,
    save_model,
    update_model,
)
from plots import plot_confusion_matrix, plot_loss_and_accuracy, roc_plots
from utils import (
    check_slurm_mode,
    get_folders,
    get_statistics,
    get_user_data_general,
    get_user_data_network,
    number_clear_cloud,
)

# ---------------------------------------------------------------------------------------------------------------------


# USER PARAMETERS
TYPE_RESNET, BATCH_SIZE, NB_EPOCHS, EARLY_STOPPING, TYPE_OPTIMIZER, DYNAMIC = get_user_data_network()
NAME_DB, PATH_FOLDERS, DIRECTORIES, PERCENTAGE, NORMA = get_user_data_general()

FOLDERS = get_folders(PATH_FOLDERS, NAME_DB, DIRECTORIES)
FOLDER_DATABASE, FOLDER_PLOTS, FOLDER_MODELS = FOLDERS[0], FOLDERS[1], FOLDERS[2]
case = f"ResNetType{TYPE_RESNET}_batch{str(BATCH_SIZE)}_epoch{str(NB_EPOCHS)}"
print("case", case)
path_image_files = sorted(FOLDER_DATABASE.glob("*.fits"))
path_labels_files = sorted(FOLDER_DATABASE.glob("*.npy"))

# CREATE DATASETS
training_images_files, training_labels_files, test_images_files, test_labels_files = chosen_datasets(
    PERCENTAGE, path_image_files, path_labels_files
)
MEAN_GLOBAL, STD_GLOBAL, MIN_GLOBAL, MAX_GLOBAL = get_statistics(training_images_files)

# CONFIG
NB_TRAIN_IMGS = len(training_images_files)
NB_TEST_IMGS = len(test_images_files)
NB_BATCH_TRAIN = NB_TRAIN_IMGS // BATCH_SIZE + 1
NB_BATCH_TEST = NB_TEST_IMGS // BATCH_SIZE + 1
print(f"TRAIN_IMGS {NB_TRAIN_IMGS} & NB CLEAR/CLOUD IMAGES", number_clear_cloud(training_labels_files))
print(f"TEST_IMGS {NB_TEST_IMGS} & NB CLEAR/CLOUD IMAGES", number_clear_cloud(test_labels_files))
print(f"PERCENTAGE train/test : {PERCENTAGE} & NB of BATCH_TRAIN {NB_BATCH_TRAIN} NB of BATCH_TEST {NB_BATCH_TEST}")

# SPECIFICS NN
state, schedule = create_train_state(TYPE_RESNET, TYPE_OPTIMIZER, NB_EPOCHS, NB_BATCH_TRAIN, DYNAMIC)
data_loader_training = DataLoader(
    image_files=training_images_files,
    labels_files=training_labels_files,
    batch_size=BATCH_SIZE,
    mean_global=MEAN_GLOBAL,
    std_global=STD_GLOBAL,
    min_global=MIN_GLOBAL,
    max_global=MAX_GLOBAL,
    shuffle=True,
    normalisation=NORMA,
)
data_loader_test = DataLoader(
    test_images_files,
    test_labels_files,
    BATCH_SIZE,
    mean_global=MEAN_GLOBAL,
    std_global=STD_GLOBAL,
    min_global=MIN_GLOBAL,
    max_global=MAX_GLOBAL,
    normalisation=NORMA,
)

# TRAINING
TQDM_DISABLE = check_slurm_mode()
BEST_LOSS = 100
list_avg_losses, list_avg_accuracies, list_avg_test_losses, list_avg_test_accuracies = [], [], [], []

for epoch in range(NB_EPOCHS):
    list_losses, list_accuracies, list_test_losses, list_test_accuracies = [], [], [], []

    # LOOP OVER ALL BATCHES
    for batch_images, batch_labels in tqdm(
        data_loader_training.generate_batches(), total=NB_BATCH_TRAIN, desc=f"epoch {epoch+1}", disable=TQDM_DISABLE
    ):
        state, loss, accuracy = update_model(state, batch_images, batch_labels)

        list_losses.append(loss)
        list_accuracies.append(accuracy)

    # SAVE RES FOR PLOTS
    avg_losses = jax.numpy.mean(jax.numpy.stack(list_losses))
    avg_accuracies = jax.numpy.mean(jax.numpy.stack(list_accuracies))
    list_avg_losses.append(avg_losses)
    list_avg_accuracies.append(avg_accuracies)
    step, lr = state.step.item(), schedule(state.step).item()
    print(f"loss : {avg_losses}  accuracy {avg_accuracies}  lr {step, lr}")

    # TEST LOSS
    for batch_images_test, batch_labels_test in tqdm(
        data_loader_test.generate_batches(), total=NB_BATCH_TEST, desc="test batches", disable=TQDM_DISABLE
    ):
        test_loss, test_accuracies = eval_function(state, batch_images_test, batch_labels_test)
        list_test_losses.append(test_loss)
        list_test_accuracies.append(test_accuracies)

    # SAVE RES FOR PLOTS
    avg_test_losses = jax.numpy.mean(jax.numpy.stack(list_test_losses))
    avg_test_accuracies = jax.numpy.mean(jax.numpy.stack(list_test_accuracies))
    list_avg_test_losses.append(avg_test_losses)
    list_avg_test_accuracies.append(avg_test_accuracies)
    print(f"loss : {avg_test_losses}  accuracy {avg_test_accuracies}")

    # Early-Stopping
    if avg_losses < BEST_LOSS:
        BEST_LOSS = avg_losses
        save_model(state, FOLDER_MODELS / case)
        PATIENCE_COUNTER = 0
    else:
        PATIENCE_COUNTER += 1
        if PATIENCE_COUNTER == EARLY_STOPPING:
            print("early stopping")
            break

# PLOTS
list_predictions = []
list_probs = []
truth = []
best_state_model = load_model(FOLDER_MODELS / case)
for batch_images_test, batch_labels_test in tqdm(data_loader_test.generate_batches(), total=NB_BATCH_TEST):
    logits = best_state_model.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, batch_images_test, train=False
    )
    predictions = jax.numpy.round(jax.nn.sigmoid(logits)).astype(int)
    list_predictions.append(predictions)
    list_probs.append(jax.nn.sigmoid(logits))
    truth.append(batch_labels_test)

concatenated_preds = jax.numpy.concatenate(list_predictions, axis=0)
concatenated_probs = jax.numpy.concatenate(list_probs, axis=0)
concatenated_truth = jax.numpy.concatenate(truth, axis=0)

plot_confusion_matrix(concatenated_truth, concatenated_preds, FOLDER_PLOTS, case)
roc_plots(concatenated_preds, concatenated_truth, FOLDER_PLOTS, case=f"{case}_preds")
roc_plots(concatenated_probs, concatenated_truth, FOLDER_PLOTS, case=f"{case}_probs")
plot_loss_and_accuracy(
    list_avg_losses, list_avg_test_losses, list_avg_accuracies, list_avg_test_accuracies, FOLDER_PLOTS, case
)
