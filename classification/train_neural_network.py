# -*- coding: utf-8 -*-
# pylint: disable=E1102

import jax
from dataloader import DataLoader, chosen_datasets
from flax.training.early_stopping import EarlyStopping
from model import create_train_state, eval_function, load_model, pred_function, save_model, update_model
from plots import plot_confusion_matrix, plot_loss_and_accuracy, roc_plots
from tqdm import tqdm
from utils import check_slurm_mode, get_folders, get_statistics, get_user_data_general, get_user_data_network, number_clear_cloud

# ---------------------------------------------------------------------------------------------------------------------


# USER PARAMETERS
TYPE_RESNET, BATCH_SIZE, NB_EPOCHS, EARLY_STOP, TYPE_OPTIMIZER, MOMENTUM = get_user_data_network()
NAME_DB, PATH_FOLDERS, DIRECTORIES, PERCENTAGE, NORMA = get_user_data_general()
FOLDERS = get_folders(PATH_FOLDERS, NAME_DB, DIRECTORIES)
FOLDER_DATABASE, FOLDER_PLOTS, FOLDER_MODELS = FOLDERS[0], FOLDERS[1], FOLDERS[2]
case = f"ResNetType{TYPE_RESNET}_batch{str(BATCH_SIZE)}_epoch{str(NB_EPOCHS)}"

# CREATE DATASETS
path_image_files = sorted(FOLDER_DATABASE.glob("*.fits"))
path_labels_files = sorted(FOLDER_DATABASE.glob("*.npy"))
training_images_files, training_labels_files, test_images_files, test_labels_files = chosen_datasets(PERCENTAGE, path_image_files, path_labels_files)
MEAN_GLOBAL, STD_GLOBAL, MIN_GLOBAL, MAX_GLOBAL = get_statistics(training_images_files)

# CONFIG
NB_BATCH_TRAIN = len(training_images_files) // BATCH_SIZE + 1
NB_BATCH_TEST = len(test_images_files) // BATCH_SIZE + 1
print(f"TRAIN_IMGS {len(training_images_files)} & NB CLEAR/CLOUD IMAGES", number_clear_cloud(training_labels_files))
print(f"TEST_IMGS {len(test_images_files)} & NB CLEAR/CLOUD IMAGES", number_clear_cloud(test_labels_files))
print(f"PERCENTAGE train/test : {PERCENTAGE} & NB of BATCH_TRAIN {NB_BATCH_TRAIN} NB of BATCH_TEST {NB_BATCH_TEST}")

# SPECIFICS NN
early_stopping = EarlyStopping(min_delta=1e-9, patience=EARLY_STOP)
state, schedule = create_train_state(TYPE_RESNET, TYPE_OPTIMIZER, NB_EPOCHS, NB_BATCH_TRAIN, MOMENTUM)
dataloader_train = DataLoader(training_images_files, training_labels_files, BATCH_SIZE, MEAN_GLOBAL, STD_GLOBAL, MIN_GLOBAL, MAX_GLOBAL, normalisation=NORMA)
dataloader_test = DataLoader(test_images_files, test_labels_files, BATCH_SIZE, MEAN_GLOBAL, STD_GLOBAL, MIN_GLOBAL, MAX_GLOBAL, shuffle=False, normalisation=NORMA)

# TRAINING
TQDM_DISABLE = check_slurm_mode()
list_avg_losses, list_avg_accuracies, list_avg_test_losses, list_avg_test_accuracies = [], [], [], []

for epoch in range(NB_EPOCHS):
    list_losses, list_accuracies, list_test_losses, list_test_accuracies = [], [], [], []

    # LOOP OVER ALL BATCHES
    for batch_images, batch_labels in tqdm(dataloader_train.generate_batches(), total=NB_BATCH_TRAIN, desc=f"epoch {epoch+1}", disable=TQDM_DISABLE):
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
    for batch_images_test, batch_labels_test in tqdm(dataloader_test.generate_batches(), total=NB_BATCH_TEST, desc="test batches", disable=TQDM_DISABLE):
        test_loss, test_accuracies = eval_function(state, batch_images_test, batch_labels_test)
        list_test_losses.append(test_loss)
        list_test_accuracies.append(test_accuracies)

    # SAVE RES FOR PLOTS
    list_avg_test_losses.append(jax.numpy.mean(jax.numpy.stack(list_test_losses)))
    list_avg_test_accuracies.append(jax.numpy.mean(jax.numpy.stack(list_test_accuracies)))

    # Early-Stopping
    has_improved, early_stopping = early_stopping.update(metric=avg_losses)
    if has_improved:
        save_model(state, FOLDER_MODELS / case)
    if early_stopping.should_stop:
        print(f"Met early stopping criteria, breaking at epoch {epoch}")
        break

# PLOTS
list_preds = []
list_truths = []
best_state_model = load_model(FOLDER_MODELS / case)
for batch_images_test, batch_labels_test in tqdm(dataloader_test.generate_batches(), total=NB_BATCH_TEST):
    batch_preds = pred_function(best_state_model, batch_images_test)
    list_preds.append(batch_preds)
    list_truths.append(batch_labels_test)

concatenated_preds, concatenated_truth = jax.numpy.concatenate(list_preds, axis=0), jax.numpy.concatenate(list_truths, axis=0)
plot_confusion_matrix(concatenated_truth, concatenated_preds, FOLDER_PLOTS, case, "NON-LINEAR CLASSIFIER")
roc_plots(concatenated_preds, concatenated_truth, FOLDER_PLOTS, case=f"{case}_preds")
plot_loss_and_accuracy(list_avg_losses, list_avg_test_losses, list_avg_accuracies, list_avg_test_accuracies, FOLDER_PLOTS, case)
