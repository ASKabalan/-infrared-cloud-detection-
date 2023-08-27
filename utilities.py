import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_image(data):


    cloud_image, binary_mask = data
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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

def plot_image_pred(cloud_image, binary_mask, y_pred , predmask_cmap='grayscale'):

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
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


# Copy to another folder
"""
import os
import shutil

from pathlib import Path    

r_list = random.choices(images_list, k=1000)

destination_folder = os.getcwd()+'/SUBSET_5000/'

# fetch all files
for r in r_list:
    destination = destination_folder + Path(os.path.abspath(r)).name
    shutil.copy(r, destination)
"""


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