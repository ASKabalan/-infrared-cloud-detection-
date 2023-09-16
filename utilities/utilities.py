import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import visualization as aviz
from astropy.nddata.blocks import block_reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
from sklearn.metrics import roc_curve, auc
import cv2
import os

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_image(data,figsize=(10, 4)):
    
    cloud_image, binary_mask = data
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    N = 2

    ax1 = axes[0]
    ax1.set_title('Cloud image')
    im1 = ax1.imshow(cloud_image, cmap='jet')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar1.ax.set_ylabel('ADU')

    ax2 = axes[1]
    ax2.set_title('Binary mask')
    im2 = ax2.imshow(binary_mask, cmap=discrete_cmap(N, 'gray'))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=range(N))
    cbar.ax.set_yticklabels(['', ''], rotation=90)  # vertically oriented colorbar
    cbar.ax.set_ylabel('0 = Sky         1 = Cloud')

    plt.tight_layout()
    plt.show()

def plot_image_pred(cloud_image, binary_mask, y_pred , figsize=(8,4),predmask_cmap='grayscale'):

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    N = 2

    ax1 = axes[0]
    ax1.set_title('Cloud image')
    im1 = ax1.imshow(cloud_image, cmap='jet')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar1.ax.set_ylabel('ADU')

    ax2 = axes[1]
    ax2.set_title('Binary mask')
    im2 = ax2.imshow(binary_mask, cmap=discrete_cmap(N, 'gray'))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=range(N))
    cbar2.ax.set_yticklabels(['', ''], rotation=90)  # vertically oriented colorbar
    cbar2.ax.set_ylabel('0 = Sky         1 = Cloud')

    ax3 = axes[2]
    ax3.set_title('Predicted Binary mask')
    im3 = ax3.imshow(y_pred, cmap=predmask_cmap)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax, orientation='vertical', ticks=range(N))
    cbar3.ax.set_ylabel('0 = Sky         1 = Cloud')

    plt.tight_layout()
    plt.show()

from pathlib import Path
from astropy.io import fits

def rebin_fits(filename , bin = (128, 160)):
    try: 
        fits_file = fits.open(name=filename)
        image = fits_file[0].data
        image = rebin(image,bin)
        fits_file[0].data = image

        fits_file.writeto('BIN_SUBSET/'+Path(filename).name.replace('.fits', '_binned.fits'), overwrite=True)
        fits_file.close()
        del fits_file
    except:
        pass

def rebin(arr, new_shape):

        shape = (new_shape[0], arr.shape[0] // new_shape[0],
        new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)


def show_image(image,
               percl=99, percu=None, is_mask=False,
               figsize=(10, 10),
               cmap='viridis', log=False, clip=True,
               show_colorbar=True, show_ticks=True,
               fig=None, ax=None, input_ratio=None):
    """
    Show an image in matplotlib with some basic astronomically-appropriat stretching.

    Parameters
    ----------
    image
        The image to show
    percl : number
        The percentile for the lower edge of the stretch (or both edges if ``percu`` is None)
    percu : number or None
        The percentile for the upper edge of the stretch (or None to use ``percl`` for both)
    figsize : 2-tuple
        The size of the matplotlib figure in inches
    """
    if percu is None:
        percu = percl
        percl = 100 - percl

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise ValueError('Must provide both "fig" and "ax" '
                         'if you provide one of them')
    elif fig is None and ax is None:
        if figsize is not None:
            # Rescale the fig size to match the image dimensions, roughly
            image_aspect_ratio = image.shape[0] / image.shape[1]
            figsize = (max(figsize) * image_aspect_ratio, max(figsize))

        fig, ax = plt.subplots(1, 1, figsize=figsize)


    # To preserve details we should *really* downsample correctly and
    # not rely on matplotlib to do it correctly for us (it won't).

    # So, calculate the size of the figure in pixels, block_reduce to
    # roughly that,and display the block reduced image.

    # Thanks, https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
    fig_size_pix = fig.get_size_inches() * fig.dpi

    ratio = (image.shape // fig_size_pix).max()

    if ratio < 1:
        ratio = 1

    ratio = input_ratio or ratio

    reduced_data = block_reduce(image, ratio)

    if not is_mask:
        # Divide by the square of the ratio to keep the flux the same in the
        # reduced image. We do *not* want to do this for images which are
        # masks, since their values should be zero or one.
         reduced_data = reduced_data / ratio**2

    # Of course, now that we have downsampled, the axis limits are changed to
    # match the smaller image size. Setting the extent will do the trick to
    # change the axis display back to showing the actual extent of the image.
    extent = [0, image.shape[1], 0, image.shape[0]]

    if log:
        stretch = aviz.LogStretch()
    else:
        stretch = aviz.LinearStretch()

    norm = aviz.ImageNormalize(reduced_data,
                               interval=aviz.AsymmetricPercentileInterval(percl, percu),
                               stretch=stretch, clip=clip)

    if is_mask:
        # The image is a mask in which pixels should be zero or one.
        # block_reduce may have changed some of the values, so reset here.
        reduced_data = reduced_data > 0
        # Set the image scale limits appropriately.
        scale_args = dict(vmin=0, vmax=1)
    else:
        scale_args = dict(norm=norm)

    im = ax.imshow(reduced_data, origin='lower',
                   cmap=cmap, extent=extent, aspect='equal', **scale_args)

    if show_colorbar:
        # I haven't a clue why the fraction and pad arguments below work to make
        # the colorbar the same height as the image, but they do....unless the image
        # is wider than it is tall. Sticking with this for now anyway...
        # Thanks: https://stackoverflow.com/a/26720422/3486425
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # In case someone in the future wants to improve this:
        # https://joseph-long.com/writing/colorbars/
        # https://stackoverflow.com/a/33505522/3486425
        # https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes

    if not show_ticks:
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)


# Save to PDF
"""
r_list = random.choices(images_list, k=100)

with parallel_backend('dask', n_jobs=num_cores):
    l_plots = Parallel(verbose=10)(delayed(generate_mask)(fits.getdata(r), a_log = 100000, contrast = 0.9, display = True) for r in r_list)
    
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('ground_truth_masks_comparisons.pdf')
for plot in l_plots:
    pp.savefig(plot)
pp.close()
"""

def evaluate_model(y_true,y_pred_proba, threshhold = 0.5):
    """
    Evaluates the multi-output labels and prints mean accuracy, precision, recall, and plots mean AUC curve.
    Also computes IOU metric.
    
    Args:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    
    """
    
    y_pred = (y_pred_proba > threshhold).astype(int)

    # Flatten the arrays for pixel-wise operations
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()

    # Compute pixel-wise accuracy, precision, and recall
    accuracy = np.mean(y_true_flat == y_pred_flat)
    precision = np.sum(y_true_flat * y_pred_flat) / (np.sum(y_pred_flat) + 1e-10)
    recall = np.sum(y_true_flat * y_pred_flat) / (np.sum(y_true_flat) + 1e-10)
    
    bce = BinaryCrossentropy()
    loss = bce(y_true, y_pred_proba).numpy()
        # Compute the IOU metric (Intersection Over Union)
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat) - intersection
    iou = intersection / (union + 1e-10)
    
    
    print(f"Mean Accuracy: {accuracy:.4f}")
    print(f"Mean Precision: {precision:.4f}")
    print(f"Mean Recall: {recall:.4f}")
    print(f"BinaryCrossEntropy Loss: {loss:.4f}")
    print(f"IOU: {iou:.4f}")
    
    # Compute pixel-wise FPR, TPR and thresholds
    fpr, tpr, thresholds = roc_curve(y_true_flat, y_pred_flat)
    auc_value = auc(fpr, tpr)
    
    # Plot the AUC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'Mean AUC: {auc_value:.4f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    plt.show()

def rgb_to_gray(image_path, output_path):
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
    print(new_file_path)


    # Save the grayscale image
    cv2.imwrite(new_file_path, gray_image)

    return gray_image