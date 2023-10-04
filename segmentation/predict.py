# predict.py

import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
from model import CIRRUS_Net, load_model, predict
from CloudDataSetGen import load_inference_ds
from ..utilities import utilities as util
import random

def visualize_predictions(images, predictions, num_visualize=3):
    for _ in range(3):
        random_index = random.randint(0, len(images) - 1)
        util.plot_image_pred(X_test[random_index], y_test[random_index], y_pred[random_index],predmask_cmap='jet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using the CIRRUS CloudSeg model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing images for prediction.')
    parser.add_argument('--num_visualize', type=int, default=3, help='Number of predictions to visualize.')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to save the prediction images. If not provided, images will not be saved.')
    args = parser.parse_args()

    # Load model
    state,apply_fn = load_model(args.model_path,CIRRUS_Net)

    # Load dataset (assuming you have a function to load only images without masks)
    ds_inference = load_inference_ds(args.input_folder)

    # Predict
    # Evaluate the model
    all_images = []
    all_predictions = []
    for images in ds_inference:
        batch_predictions = predict(state, images)
        all_predictions.append(batch_predictions)
    all_predictions = jnp.concatenate(all_predictions, axis=0)
    y_pred = all_predictions.squeeze(axis=-1)

    # Visualize
    
    if 
    visualize_predictions(images, predictions, args.num_visualize)

    # Save predictions if output_folder is provided
    if args.output_folder:
        for idx, pred in enumerate(predictions):
            plt.imsave(f"{args.output_folder}/prediction_{idx}.png", pred.squeeze(), cmap='jet')
