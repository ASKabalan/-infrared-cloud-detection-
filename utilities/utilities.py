import numpy as np
import pandas as pd
from keras.losses import BinaryCrossentropy
import cv2
import os
from pathlib import Path
from astropy.io import fits
from skimage.transform import resize
from sklearn.metrics import roc_curve, auc,f1_score
import jax.numpy as jnp


def rebin_fits(filename, bin=(128, 160)):
    """
    Reads a FITS file, rebins the image data, and writes the rebinned image to a new FITS file.
    
    Args:
    - filename (str): Path to the input FITS file.
    - bin (tuple): Desired shape for rebinning. Default is (128, 160).
    """
    target_file = f"BIN_SUBSET/{Path(filename).name.replace('.fits', '_binned.fits')}"
    try: 
        fits_file = fits.open(name=filename)
        image = fits_file[0].data
        image = rebin(image, bin)
        fits_file[0].data = image
        fits_file.writeto(target_file, overwrite=True)
        fits_file.close()
        del fits_file
    except:
        print(f"Fail : {target_file}")
        
def save_image_pred(cloud_image, binary_mask, y_pred, output_path):
    """
    Save the three images (cloud_image, binary_mask, y_pred) to a FITS file.
    
    Parameters:
    cloud_image (array-like): The first image data.
    binary_mask (array-like): The second image data.
    y_pred (array-like): The third image data.
    output_path (str): The path where the FITS file will be saved.
    """
    # Create a PrimaryHDU object for each image
    hdu1 = fits.PrimaryHDU(cloud_image)
    hdu2 = fits.ImageHDU(binary_mask)
    hdu3 = fits.ImageHDU(y_pred)

    # Create an HDUList to hold them
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])

    # Write to a new FITS file
    hdulist.writeto(f'{output_path}.fits', overwrite=True)


def rebin(arr, new_shape):
    """
    Rebins a 2D array to a new shape by averaging.
    
    Args:
    - arr (array): Input 2D array.
    - new_shape (tuple): Desired shape for the output array.
    
    Returns:
    - array: Rebinned array.
    """
    shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def rebin_samping(image, new_shape):
    """
    Resizes an image to a new shape using anti-aliasing.
    
    Args:
    - image (array): Input image.
    - new_shape (tuple): Desired shape for the output image.
    
    Returns:
    - array: Resized image.
    """
    return resize(image, new_shape, anti_aliasing=True)

def evaluate_model(y_true, y_pred_proba, threshold=0.5, output_file=None, roc_data_file=None):
    """
    Evaluates the performance of a model using various metrics and saves the ROC data to a CSV file.
    
    Args:
    - y_true (array): Ground truth labels.
    - y_pred_proba (array): Predicted probabilities.
    - threshold (float): Threshold for classifying probabilities. Default is 0.5.
    - output_file (str): File to save the evaluation results text. If None, the results are printed.
    - roc_data_file (str): File to save the ROC data (FPR, TPR). If None, the data is not saved.
    """
    
    y_pred = (y_pred_proba > threshold).astype(int)
    y_true = (y_true > threshold).astype(int)

    # Flatten the arrays for pixel-wise operations
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    # Compute pixel-wise accuracy, precision, recall, and F1 score
    accuracy = np.mean(y_true_flat == y_pred_flat)
    precision = np.sum(y_true_flat * y_pred_flat) / (np.sum(y_pred_flat) + 1e-10)
    recall = np.sum(y_true_flat * y_pred_flat) / (np.sum(y_true_flat) + 1e-10)
    f1 = f1_score(y_true_flat, y_pred_flat)
    
    # Compute Error Rate (ER)
    TP = np.sum(y_true_flat * y_pred_flat)
    TN = np.sum((1 - y_true_flat) * (1 - y_pred_flat))
    FP = np.sum((1 - y_true_flat) * y_pred_flat)
    FN = np.sum(y_true_flat * (1 - y_pred_flat))
    ER = (FP + FN) / (TP + TN + FP + FN)
    
    bce = BinaryCrossentropy()
    loss = bce(y_true, y_pred_proba).numpy()
    
    # Compute the IOU metric (Intersection Over Union)
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    iou = intersection / (union + 1e-10)
    
    y_true_np = np.array(y_true.ravel())
    y_pred_proba_np = np.array(y_pred_proba.ravel())
    # Prepare the evaluation results text
    fpr, tpr, thresholds = roc_curve(y_true_np, y_pred_proba_np)
    auc_value = auc(fpr, tpr)
    
    results_text = (
        f"Mean Accuracy: {accuracy:.4f}\n"
        f"Mean Precision: {precision:.4f}\n"
        f"Mean Recall: {recall:.4f}\n"
        f"Mean F1 Score: {f1:.4f}\n"
        f"Error Rate (ER): {ER:.4f}\n"
        f"BinaryCrossEntropy Loss: {loss:.4f}\n"
        f"IOU: {iou:.4f}\n"
        f"Mean AUC: {auc_value:.4f}\n"
    )

    # Print the evaluation results
    print(results_text)

    # Save the evaluation results to a file
    if output_file is not None:
        with open(output_file, 'w') as file:
            file.write(results_text)
    
    # Save the ROC data to a CSV file
    if roc_data_file is not None:
        roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_data.to_csv(roc_data_file, index=False)

def rgb_to_gray(image_path, output_path, save_to_fits=True):
    """
    Converts an RGB image to grayscale and optionally saves it as a FITS file.
    
    Args:
    - image_path (str): Path to the input RGB image.
    - output_path (str): Directory where the grayscale image/FITS file will be saved.
    - save_to_fits (bool): Whether to save the grayscale image as a FITS file. Default is True.
    
    Returns:
    - array: Grayscale image.
    """
    # Read the input image in RGB format
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Extract the filename from the existing file path.
    # file_name = os.path.basename(image_path)

    # Extract the filename and extension from the existing file path.
    file_name, file_extension = os.path.splitext(os.path.basename(image_path))

    # Construct the new filename with the suffix.
    suffix = '_gray'
    new_file_name = f"{file_name}{suffix}{file_extension}"

    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    new_file_path = os.path.join(output_path, new_file_name)

    # Save the grayscale image
    #cv2.imwrite(new_file_path, gray_image)
    if save_to_fits:
        hdu = fits.PrimaryHDU(gray_image)
        hdu.scale('uint8')
        hdul = fits.HDUList([hdu])
        print(new_file_path.replace(file_extension, '.fits'))
        hdul.writeto(new_file_path.replace(file_extension, '.fits'), overwrite=True)

    return gray_image


def process_data_and_create_histogram(data_tuple, image_index=2, bins=50):
    """
    Processes a tuple of JAX arrays to create a histogram of a specified image.

    Args:
    - data_tuple (tuple): A tuple of JAX arrays.
    - image_index (int, optional): Index of the image in the tuple to create a histogram for. Defaults to 2.
    - bins (int, optional): Number of bins for the histogram. Defaults to 50.

    Returns:
    Tuple (data_tuple, hist, bin_edges): The data tuple, histogram values, and bin edges.
    """
    # Ensure the image index is within the range of the tuple
    if image_index >= len(data_tuple):
        raise IndexError("Image index is out of range for the provided data tuple.")

    image_data = data_tuple[image_index]
    hist, bin_edges = jnp.histogram(image_data.flatten(), bins=bins)

    return (data_tuple, hist, bin_edges)


from scipy.stats import pearsonr

def compute_correlation_matrix(fits_histogram_tuples):
    """
    Computes the Pearson correlation matrix between histograms from a list of tuples.

    Args:
    - fits_histogram_tuples (list of tuples): Each tuple contains a FITS file, histogram, and bin edges.

    Returns:
    - np.ndarray: Correlation matrix.
    """
    num_files = len(fits_histogram_tuples)
    correlation_matrix = np.zeros((num_files, num_files))

    for i in range(num_files):
        for j in range(num_files):
            if i != j:
                # Calculate correlation only for different histograms
                correlation, _ = pearsonr(fits_histogram_tuples[i][1], fits_histogram_tuples[j][1])
                correlation_matrix[i, j] = correlation

    return correlation_matrix


def find_correlated_fits(correlation_matrix, index, fits_data, N=3):
    """
    Finds the N most and least correlated FITS files to a given FITS file, identified by index.

    Args:
    - correlation_matrix (np.ndarray): The correlation matrix.
    - index (int): The index of the FITS file in question.
    - fits_data (list): The list of FITS data corresponding to the indices.
    - N (int, optional): Number of most and least correlated files to find. Defaults to 3.

    Returns:
    - dict: Dictionary with keys 'most_correlated' and 'least_correlated', each containing a list of tuples (index, FITS file).
    """
    # Exclude the index itself and get correlations
    correlations = correlation_matrix[index, :]
    correlations[index] = np.nan  # Ignore self-correlation

    # Find the indices of the N most and least correlated
    most_correlated_indices = np.argsort(-correlations)[:N]  # Negative for descending order
    least_correlated_indices = np.argsort(correlations)[:N]

    # Retrieve the corresponding FITS files
    original_fits = fits_data[index]
    most_correlated = [fits_data[idx] for idx in most_correlated_indices if not np.isnan(correlations[idx])]
    least_correlated = [fits_data[idx] for idx in least_correlated_indices if not np.isnan(correlations[idx])]

    return {
        'original'  : original_fits,
        'most_correlated': most_correlated,
        'least_correlated': least_correlated
    }