import time
import jax
from jax import numpy as jnp
import argparse
from model import train_step, eval_step,predict, save_model
from CloudDataSetGen import load_dataset
from ..utilities import utilities as util
from ..utilities.training_utilities import TrainingMetrics, TrainingTimer
from model import create_train_state, train_step, eval_step
from CloudDataSetGen import load_dataset


def train_and_evaluate(input_folder_path, output_model_path, unet_conv=[64, 128, 256, 512],bottle_neck_conv = 1024, \
                learning_rate=0.001, train_batch_size=64, aug_batch_size=64, val_batch_size=64,\
                num_epochs=50, patience=10, verbose=1):


    ds_train_gen , ds_val_gen = load_dataset(input_folder_path,train_batch_size,aug_batch_size,val_batch_size)
    # Train the model
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    input_shape = (24, 160, 128, 1)
    n_batches = len(ds_train_gen)

    state = create_train_state(init_rng, input_shape, unet_conv, bottle_neck_conv, learning_rate, n_batches)
    best_val_loss = float('inf')
    counter = 0

    metrics = TrainingMetrics()
    timer = TrainingTimer()

    for epoch in range(args.num_epochs):
        metrics.reset()
        timer.start()

        for cnt, batch in enumerate(ds_train_gen):
            images, masks = batch
            state, loss, accuracy = train_step(state, images, masks)
            metrics.train_loss.append(loss)
            metrics.train_accuracy.append(accuracy)
            if args.verbose == 2 and cnt % (n_batches // 10) == 0:
                avg_loss = metrics.avg_loss
                avg_accuracy = metrics.avg_accuracy
                print("\r", end="")
                print(f"Batch Number {cnt + 1}/{n_batches} - Loss: {avg_loss}, Accuracy: {avg_accuracy}", end="")

        for cnt, batch in enumerate(ds_val_gen):
            images, masks = batch
            loss, accuracy = eval_step(state, images, masks)
            metrics.test_loss.append(loss)
            metrics.test_accuracy.append(accuracy)

        epoch_time = timer.stop()
        if verbose >= 1:
            print("\r", end="")
            print(f"Epoch {epoch + 1}/{num_epochs} done - Loss: {metrics.avg_loss}, Accuracy: {metrics.avg_accuracy}, - val_Loss: {metrics.avg_val_loss}, val_Accuracy: {metrics.avg_val_acc},  Time taken: {epoch_time:.2f} seconds")

        if metrics.avg_val_loss < best_val_loss:
            best_val_loss = metrics.avg_val_loss
            counter = 0
            # Save the model checkpoint
            save_model(state, output_model_path,epoch)
        else:
            counter += 1
            if verbose >= 1:
                print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                if verbose >= 1:
                    print("EarlyStopping: Stop training")
                break

    # Evaluate the model
    all_predictions = []
    all_groudtruth = []
    for batch in ds_val_gen:
        images, masks = batch
        batch_predictions = predict(state.params,state.apply_fn, images)
        all_predictions.append(batch_predictions)
        all_groudtruth.append(masks)
    all_predictions = jnp.concatenate(all_predictions, axis=0)
    y_pred = all_predictions.squeeze(axis=-1)
    all_groudtruth = jnp.concatenate(all_groudtruth, axis=0)
    y_test = all_groudtruth.squeeze(axis=-1)
    util.evaluate_model(y_test, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the CIRRUS CloudSeg model.')
    parser.add_argument('--input_folder_path', type=str, required=True, help='Path to the input folder containing FITS files.')
    parser.add_argument('--output_model_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--unet_conv', type=int, nargs='+', default=[64, 128, 256, 512], help='List defining the UNET convolution model.')
    parser.add_argument('--bottle_neck_conv', type=int, default=1024, help='Bottle Neck convolution size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--train_batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--aug_batch_size', type=int, default=64, help='Augmentation batch size.')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Validation batch size.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level. 0: No logs, 1: Epoch logs, 2: Batch logs.')
    args = parser.parse_args()

    train_and_evaluate(
        input_folder_path=args.input_folder_path,
        output_model_path=args.output_model_path,
        unet_conv=args.unet_conv,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        aug_batch_size=args.aug_batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        verbose=args.verbose
    )