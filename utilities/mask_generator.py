import numpy as np
from astropy.visualization import *
import matplotlib.pyplot as plt
import skimage.filters
from utilities import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from astropy.io import fits


DR = 2**14
def generate_mask(filename, bin_size = (128, 160) , a_log = 100000, contrast = 0.9, display = False, return_mask = False, write_to_fits = False):
    
    fits_file = fits.open(name=filename)
    image = fits_file[0].data
    image = rebin(image, bin_size)
    fits_file[0].data = image
    # Normalize image by camera internal bit-depth of 14bits
    image_div = image/DR
    image_linearized = np.log((DR/image) - 1)
    stretch = LogStretch(a=a_log) + ZScaleInterval(contrast=contrast)
    
    image_stretch = stretch(image_div)
    
    # Compute threshold with Otsu method
    # https://datacarpentry.org/image-processing/07-thresholding.html
    threshold = skimage.filters.threshold_otsu(image_stretch)
    binary_mask = image_stretch > threshold
    binary_mask = np.array(binary_mask, dtype=np.int8)
    
    if write_to_fits == True:
        
        try:
            mask_hdu = fits.ImageHDU(binary_mask)
            fits_file.append(mask_hdu)
            fits_file[1].header['IMGTYPE'] = 'GROUND-TRUTH'
            fits_file.writeto('MASKED_SUBSET/'+Path(filename).name.replace('_flux.fits', '_masked.fits'), overwrite=True)
            fits_file.close()
            del fits_file
        except:
            pass
            #print(filename)
        
    if display == True:
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
    
    if return_mask == True:
        return binary_mask