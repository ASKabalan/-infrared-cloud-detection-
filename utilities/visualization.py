import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import visualization as aviz
from astropy.nddata.blocks import block_reduce
from astropy.io import fits
from pathlib import Path

import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = ['tgheros']
matplotlib.rcParams['font.sans-serif'] = ['helvet']
# matplotlib.rcParams['font.serif'] = ['cm10']
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"""
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{tgheros}
\usepackage[helvet]{sfmath}

"""
import matplotlib.pyplot as plt


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map.

    Args:
    - N (int): Number of bins.
    - base_cmap (str, optional): Base colormap to use. Defaults to None.

    Returns:
    - Colormap: Discrete colormap.
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_image(data,figsize=(10, 4)):
    """
    Plots a cloud image alongside its binary mask.

    Args:
    - data (tuple): Tuple containing cloud image and binary mask.
    - figsize (tuple, optional): Size of the figure. Defaults to (10, 4).
    """
    cloud_image, binary_mask = data
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    N = 2

    ax1 = axes[0]
    #ax1.set_title('Cloud image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    im1 = ax1.imshow(cloud_image, cmap='jet')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar1.ax.set_ylabel('ADU')

    ax2 = axes[1]
    #ax2.set_title('Binary mask')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    im2 = ax2.imshow(binary_mask, cmap=discrete_cmap(N, 'gray'))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=range(N))
    cbar.ax.set_yticklabels(['', ''], rotation=90)  # vertically oriented colorbar
    cbar.ax.set_ylabel('0 = Sky         1 = Cloud')

    plt.tight_layout()
    plt.show()

def plot_image_pred(cloud_image, binary_mask, y_pred , figsize=(8,4),predmask_cmap='grayscale'):
    """
    Plots a cloud image, its binary mask, and the predicted binary mask.

    Args:
    - cloud_image (array): Cloud image.
    - binary_mask (array): Binary mask.
    - y_pred (array): Predicted binary mask.
    - figsize (tuple, optional): Size of the figure. Defaults to (8,4).
    - predmask_cmap (str, optional): Colormap for the predicted mask. Defaults to 'grayscale'.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    N = 2

    ax1 = axes[0]
    #ax1.set_title('Cloud image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    im1 = ax1.imshow(cloud_image, cmap='jet')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar1.ax.set_ylabel('ADU')

    ax2 = axes[1]
    #ax2.set_title('Binary mask')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    im2 = ax2.imshow(binary_mask, cmap=discrete_cmap(N, 'gray'))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=range(N))
    cbar2.ax.set_yticklabels(['', ''], rotation=90)  # vertically oriented colorbar
    cbar2.ax.set_ylabel('0 = Sky         1 = Cloud')

    ax3 = axes[2]
    #ax3.set_title('Predicted Binary mask')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    im3 = ax3.imshow(y_pred, cmap=predmask_cmap)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax, orientation='vertical', ticks=range(N))
    cbar3.ax.set_ylabel('0 = Sky         1 = Cloud')

    plt.tight_layout()
    plt.show()

def save_image_pred(cloud_image, binary_mask, y_pred, output_path, figsize=(8,4), predmask_cmap='grayscale'):
    """
    Saves a cloud image, its binary mask, and the predicted binary mask to the specified output path.

    Args:
    - cloud_image (array): Cloud image.
    - binary_mask (array): Binary mask.
    - y_pred (array): Predicted binary mask.
    - output_path (str): Path to save the output image.
    - figsize (tuple, optional): Size of the figure. Defaults to (8,4).
    - predmask_cmap (str, optional): Colormap for the predicted mask. Defaults to 'grayscale'.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    N = 2

    ax1 = axes[0]
    #ax1.set_title('Cloud image')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    im1 = ax1.imshow(cloud_image, cmap='jet')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar1.ax.set_ylabel('ADU')

    ax2 = axes[1]
    #ax2.set_title('Binary mask')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    im2 = ax2.imshow(binary_mask, cmap=discrete_cmap(N, 'gray'))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax, orientation='vertical', ticks=range(N))
    cbar2.ax.set_yticklabels(['', ''], rotation=90)  # vertically oriented colorbar
    cbar2.ax.set_ylabel('0 = Sky         1 = Cloud')

    ax3 = axes[2]
    #ax3.set_title('Probabilistic output map')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    im3 = ax3.imshow(y_pred, cmap=predmask_cmap)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar3 = fig.colorbar(im3, cax=cax, orientation='vertical', ticks=range(N))
    cbar3.ax.set_ylabel('0 = Sky         1 = Cloud')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def show_image(image,
               percl=99, percu=None, is_mask=False,
               figsize=(10, 10),
               cmap='viridis', log=False, clip=True,
               show_colorbar=True, show_ticks=True,
               fig=None, ax=None, input_ratio=None):
    """
    Display an image with astronomically-appropriate stretching.

    Args:
    - image (array): Image to display.
    - percl (int, optional): Lower percentile for stretch. Defaults to 99.
    - percu (int, optional): Upper percentile for stretch. If None, uses percl. Defaults to None.
    - is_mask (bool, optional): If True, the image is treated as a mask. Defaults to False.
    - figsize (tuple, optional): Size of the figure. Defaults to (10, 10).
    - cmap (str, optional): Colormap to use. Defaults to 'viridis'.
    - log (bool, optional): If True, uses logarithmic stretch. Defaults to False.
    - clip (bool, optional): If True, clips the image. Defaults to True.
    - show_colorbar (bool, optional): If True, displays a colorbar. Defaults to True.
    - show_ticks (bool, optional): If True, displays axis ticks. Defaults to True.
    - fig (Figure, optional): Matplotlib figure object. If provided, must also provide ax. Defaults to None.
    - ax (Axes, optional): Matplotlib axes object. If provided, must also provide fig. Defaults to None.
    - input_ratio (int, optional): Ratio for block reduction. Defaults to None.
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