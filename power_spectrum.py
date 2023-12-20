
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def compute_power_spectrum(filename):
    try:
        img = fits.getdata(filename)  # Read FITS image data
    except:
        print(f"Error opening {filename}")
        return 1

    img = img/np.max(img)
    F = fftpack.fftshift(fftpack.fft2(img))  # Compute 2D FFT
    psd2D = np.abs(F)**2  # Compute 2D Power Spectrum
    psd1D = azimuthalAverage(psd2D)  # Compute azimuthally averaged 1D Power Spectrum

    # Create a single figure with three subplots arranged in 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Image
    axs[0].imshow(np.log10(img), cmap=plt.cm.Greys)
    axs[0].set_title('Image')

    # Plot 2D Power Spectrum
    axs[1].imshow(np.log10(psd2D), cmap=plt.cm.jet)
    axs[1].set_title('2D Power Spectrum')

    # Plot azimuthally averaged 1D Power Spectrum
    axs[2].plot(psd1D)
    axs[2].set_title('1D Power Spectrum')
    axs[2].set_yscale('symlog')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()  # Show the combined plot

    return psd1D


train_images_list = glob.glob('../Results/LWIRISEG/Results/*Train*.fits')
val_images_list = glob.glob('../Results/LWIRISEG/Results/*Val*.fits')
print(len(train_images_list))
print(len(val_images_list))

# Create figure and axis
fig, ax = plt.subplots()

ax.set_title('1D Power Spectrum')
ax.set_yscale('log')

ax.plot(ps_cloud, color='blue', label='Cloud')
ax.plot(ps_cloud2, color='cyan', label='Cloud 2')
ax.plot(ps_clear, color='red', label='Clear')

# Legend and labels
ax.legend(loc='best', fontsize='large', frameon=True)
ax.grid(True, linestyle='--', color='gray', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize='large')
ax.set_xlabel('Spatial frequency', fontsize='large')
ax.set_ylabel('Spectral power', fontsize='large')


plt.tight_layout()  # Adjust spacing between subplots
plt.show()  # Show the combined plot