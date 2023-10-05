# predict.py

import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from model import CIRRUS_Net, load_model, predict
from CloudDataSetGen import load_dataset
from ..utilities import utilities as util
import random

def visualize_predictions(X_test, y_test,y_pred, num_visualize=3):
    for _ in range(num_visualize):
        random_index = random.randint(0, len(X_test) - 1)
        util.plot_image_pred(X_test[random_index], y_test[random_index], y_pred[random_index],predmask_cmap='jet')

def load_model_predict(model_path,input_folder,batch_size=64,num_visualize=3,output_folder=None):

    # Load model
    state,apply_fn = load_model(model_path)

    # Load dataset (assuming you have a function to load only images without masks)
    ds_inference , _ = load_dataset(input_folder,batch_size)

    # Predict
    # Evaluate the model
    # Evaluate the model
    all_images = []
    all_predictions = []
    all_groudtruth = []
    for batch in ds_inference:
        images, masks = batch
        batch_predictions = predict(state["params"],apply_fn, images)
        all_images.append(batch_predictions)
        all_predictions.append(images)
        all_groudtruth.append(masks)

    all_images = jnp.concatenate(all_images, axis=0)
    x_img = all_images.squeeze(axis=-1)
    all_predictions = jnp.concatenate(all_predictions, axis=0)
    y_pred = all_predictions.squeeze(axis=-1)
    all_groudtruth = jnp.concatenate(all_groudtruth, axis=0)
    y_test = all_groudtruth.squeeze(axis=-1)
    util.evaluate_model(y_test, y_pred)
    visualize_predictions(x_img,y_test,y_pred,num_visualize)
    # Visualize
