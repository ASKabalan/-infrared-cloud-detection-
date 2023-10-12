import numpy as np
from astropy.visualization import LogStretch,ZScaleInterval
import matplotlib.pyplot as plt
import skimage.filters
from utilities import discrete_cmap,rebin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from astropy.io import fits


def generate_mask(filename, bin_size=(128, 160), a_log=100000, contrast=0.9, display=False, return_mask=False, write_to_fits=False):
    """
    Generates a binary mask for an image using the Otsu thresholding method.
    
    Parameters:
    - filename (str): Path to the FITS file containing the image.
    - bin_size (tuple, optional): Size for rebinning the image. Defaults to (128, 160).
    - a_log (float, optional): Parameter for the LogStretch. Defaults to 100000.
    - contrast (float, optional): Contrast for the ZScaleInterval. Defaults to 0.9.
    - display (bool, optional): If True, displays the binary mask and linearized image. Defaults to False.
    - return_mask (bool, optional): If True, returns the generated binary mask. Defaults to False.
    - write_to_fits (bool, optional): If True, writes the mask to a FITS file. Defaults to False.
    
    Returns:
    - ndarray (optional): The binary mask if return_mask is set to True.
    """
    DR = 2**14
    fits_file = fits.open(name=filename)
    image = fits_file[0].data
    image = rebin(image, bin_size)
    fits_file[0].data = image
    # Normalize image by camera internal bit-depth of 14bits
    image_div = image/DR
    image_linearized = np.log((DR/image) - 1)
    stretch = LogStretch(a=a_log) + ZScaleInterval(contrast=contrast)
    
    image_stretch = stretch(image_div)
    
    # Compute threshold with Otsu method
    threshold = skimage.filters.threshold_otsu(image_stretch)
    binary_mask = image_stretch > threshold
    binary_mask = np.array(binary_mask, dtype=np.int8)
    
    target_file = f"MASKED_SUBSET/{Path(filename).name.replace('_flux.fits', '_masked.fits')}"
    if write_to_fits:
        try:
            mask_hdu = fits.ImageHDU(binary_mask)
            fits_file.append(mask_hdu)
            fits_file[1].header['IMGTYPE'] = 'GROUND-TRUTH'
            fits_file.writeto(target_file, overwrite=True)
            fits_file.close()
            del fits_file
        except:
            print(f"Fail : {target_file}")
        
    if display:
        print("Found automatic threshold t = {}.".format(threshold))
        fig = plt.figure(figsize=(10, 4))
        
        ax1 = fig.add_subplot(121)
        ax1.set_title('Binary mask')
        N = 2
        im1 = ax1.imshow(binary_mask, cmap=discrete_cmap(N, 'gray'))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im1, cax=cax, orientation='vertical', ticks=range(N))
        cbar.ax.set_yticklabels(['', ''], rotation=90)  # vertically oriented colorbar
        cbar.ax.set_ylabel('0 = Sky         1 = Cloud')
        
        ax2 = fig.add_subplot(122)
        ax2.set_title('Linearized image')
        im2 = ax2.imshow(image_linearized, cmap='jet_r')

        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        
        cbar2 = fig.colorbar(im2, cax=cax, orientation='vertical')
        cbar2.ax.set_ylabel('ln(DR/S - 1)')
        
        plt.tight_layout()
        plt.show()
    
    if return_mask:
        return binary_mask
