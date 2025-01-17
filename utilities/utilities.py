import numpy as np
import pandas as pd
from keras.losses import BinaryCrossentropy
import cv2
import os
from pathlib import Path
from astropy.io import fits
from skimage.transform import resize
from sklearn.metrics import roc_curve, auc,f1_score
import multiprocessing
num_cores = multiprocessing.cpu_count()
from joblib import parallel_backend, Parallel, delayed


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

def open_fits_with_mask(filename,DR = 2**14):
    image = fits.open(filename)
    cloud = image[0].data
    mask = image[1].data
    del image

    # Normalize image
    cloud  = cloud / DR
    return cloud , mask

def open_fits_with_mask_and_pred(filename,DR = 2**14):
    image = fits.open(filename)
    cloud = image[0].data
    mask = image[1].data
    pred = image[2].data
    del image

    # Normalize image
    cloud  = cloud / DR
    return cloud , mask, pred

def p_open_fits_with_mask(filenames,DR = 2**14):
    with parallel_backend('threading', n_jobs=num_cores):
        l_fits = Parallel(verbose=5)(delayed(open_fits_with_mask)(filename = filename, DR=DR) for filename in filenames)
    return l_fits

def p_open_fits_with_mask_and_pred(filenames,DR = 2**14):
    with parallel_backend('threading', n_jobs=num_cores):
        l_fits = Parallel(verbose=5)(delayed(open_fits_with_mask_and_pred)(filename = filename, DR=DR) for filename in filenames)
    return l_fits

