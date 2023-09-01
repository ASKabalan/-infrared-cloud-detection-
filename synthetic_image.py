import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import visualization as aviz
from astropy.nddata.blocks import block_reduce
from scipy.ndimage import gaussian_filter
import os
from augmentations import rotate, shear, zoom
import random
from pathlib import Path
from astropy.io import fits

# Set up the random number generator, allowing a seed to be set from the environment
seed = os.getenv('GUIDE_RANDOM_SEED', None)

if seed is not None:
    seed = int(seed)
    
# This is the generator to use for any image component which changes in each image, e.g. read noise
# or Poisson error
noise_rng = np.random.default_rng(seed)

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result

def read_noise(image, amount, gain=1):
    """
    Generate simulated read noise.
    
    Parameters
    ----------
    
    image: numpy array
        Image whose shape the noise array should match.
    amount : float
        Amount of read noise, in electrons.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    shape = image.shape
    noise = noise_rng.normal(scale=amount/gain, size=shape)
    return noise

def bias(image, value, realistic=False, number_of_colums = 5):
    """
    Generate simulated bias image.
    
    Parameters
    ----------
    
    image: numpy array
        Image whose shape the bias array should match.
    value: float
        Bias level to add.
    realistic : bool, optional
        If ``True``, add some columns with somewhat higher bias value (a not uncommon thing)
    """
    # This is the whole thing: the bias is really suppose to be a constant offset!
    bias_im = np.zeros_like(image) + value
    
    # If we want a more realistic bias we need to do a little more work. 
    if realistic:
        shape = image.shape
        number_of_colums = number_of_colums
        
        # We want a random-looking variation in the bias, but unlike the readnoise the bias should 
        # *not* change from image to image, so we make sure to always generate the same "random" numbers.
        rng = np.random.RandomState(seed=np.random.randint(0, 1000))  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])
        
        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            bias_im[:, c] = value + np.random.normal(np.zeros(col_pattern.shape), col_pattern) 

        rng = np.random.RandomState(seed=np.random.randint(0, 1000))  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])
        
        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            #bias_im[:, c] = value - col_pattern
            bias_im[:, c] = value + np.random.normal(np.zeros(col_pattern.shape), col_pattern)
            
    return bias_im

def sky_background(image, sky_counts, gain=1):
    """
    Generate sky background, optionally including a gradient across the image (because
    some times Moons happen).
    
    Parameters
    ----------
    
    image : numpy array
        Image whose shape the cosmic array should match.
    sky_counts : float
        The target value for the number of counts (as opposed to electrons or 
        photons) from the sky.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.
    """
    sky_im = noise_rng.poisson(sky_counts * gain, size=image.shape) / gain
    return sky_im


def simulate_clear_sky_image(start=1000, stop=400, width=640, height=512, is_horizontal=True, bias_level=300, read_noise_level = 5, fpn_level = 5, bad_pixel_columns = 50, sky_noise_level = 10, return_original = True, augment_synthetic = True, apply_narcissus_effect = True, radius=160, center_x=320, center_y=256, smoothness=80, nar_intensity=0.025, seed = None, write_to_fits = False, index = 0):
    """
    Simulate an infrared clear sky image with realistic noise

    Parameters
    ----------
    start: float
        Starting value of gradient
    stop: float
        End value of gradient
    width: int
        2D array width
    height: int
        2D array height
    is_horizontal: bool
        If True, the gradient is horizontal (vertical if False)
    bias_level: float
        Offset ground level of image
    read_noise_level: float
        Amount of read noise, in ADU (estimated to be ~ 5 with FFC)
    fpn_level: float
        Amount of Fixed Pattern Noise (FPN) to add
    bad_pixel_columns: int
        Number of bad behaving pixel columns in the image (~50)
    sky_noise_level: float
        The target value for the number of counts from the sky.
    return_original: bool
        If True, returns both synthetic and noisy images
    augment_synthetic: bool
        If True, apply one random augmentation to the synthetic image
    apply_narcissus_effect: bool
        If True, apply the Narcissus effect (e.g own camera reflection)
    radius (int):
        Radius of the circular mask.
    center_x (int):
        X-coordinate of the circle center.
    center_y (int):
        Y-coordinate of the circle center.
    smoothness (int):
        Smoothness factor for mask edges.
    nar_intensity (float):
        Intensity of the narcissus effect.

    seed: int
        Seed for random noise generation (optional).

    Returns
    -------
    np.array or tuple
    """

    start_stop = [(i, int(i+50)) for i in range(100, 1000)]
    start_stop = np.random.normal(start_stop, 20)
    start, stop = random.choice(start_stop)

    synthetic_image = get_gradient_2d(start=start, stop=stop, width=width, height=height, is_horizontal=is_horizontal)

    # Alterations : zoom, rotate, shear, flip
    if augment_synthetic == True:
        synthetic_image, _ = zoom(synthetic_image, synthetic_image, zoom_range=(1, 2))
        synthetic_image, _ = rotate(synthetic_image, synthetic_image, angle_range=(-45, 45))
        synthetic_image, _ = shear(synthetic_image, synthetic_image, shear_range=(-0.2, 0.2))
        # Random flip
        flip = True
        if flip and random.choice([True, False]):
            augmented_image = np.fliplr(synthetic_image)

    noise_im = synthetic_image + read_noise(synthetic_image, read_noise_level)

    bias_only = bias(synthetic_image, bias_level, realistic=True, number_of_colums = bad_pixel_columns)
    bias_noise_im = noise_im + bias_only

    sky_only = sky_background(synthetic_image, sky_noise_level)
    noisy_synthetic_image = bias_noise_im + sky_only

    if seed is not None:
        np.random.seed(seed)

    # Generate random FPN pattern with values between -intensity and +intensity
    fpn_pattern = np.random.uniform(low=-fpn_level, high=fpn_level, size=sky_only.shape)

    # Add the FPN pattern to the simulated image
    noisy_synthetic_image += fpn_pattern

    if apply_narcissus_effect == True:
        noisy_synthetic_image = narcissus_effect(noisy_synthetic_image, radius = radius, center_x = center_x, center_y = center_y, smoothness = smoothness, intensity = nar_intensity)

    if return_original == True:
        return synthetic_image, noisy_synthetic_image
    else:
        if write_to_fits == True:
            noisy_synthetic_image = np.array(noisy_synthetic_image, dtype=np.int16)

            d_header = {'START': start,
                        'STOP': stop,
                        'WIDTH': width,
                        'HEIGHT': height,
                        'HORIZ': is_horizontal,
                        'BIASLVL': bias_level,
                        'READLVL': read_noise_level,
                        'FPNLVL': fpn_level,
                        'BADPXCOL': bad_pixel_columns,
                        'SKYNOISE': sky_noise_level,
                        'AUGMENT': augment_synthetic,
                        'NARCIS': apply_narcissus_effect,
                        'RADIUS': radius,
                        'CENTER_X': center_x,
                        'CENTER_Y': center_y,
                        'SMOOTH': smoothness,
                        'NARINT': nar_intensity,
                        'INDEX': index}

            hdu = fits.PrimaryHDU(noisy_synthetic_image)
            hdul = fits.HDUList([hdu])
            hdul[0].header['IMGTYPE'] = 'SIMCLEAR'
            hdul[0].header.extend(d_header.items())
            hdul.writeto('SIM_CLEAR/{}_ADU_synthetic_sim.fits'.format(index), overwrite=True)
            hdul.close()
            del hdul
            return True
        else:
            return noisy_synthetic_image

def plot_synthetic_noisy_images(syn, noisy, cmap='gray'):
    fig, axes = plt.subplots(1, 2, figsize=(7, 4), dpi=150)
    ax1 = axes[0]
    ax1.set_title('Original synthetic image')
    im1 = ax1.imshow(syn, cmap=cmap)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax, orientation='vertical')
    cbar1.ax.set_ylabel('ADU')

    ax2 = axes[1]
    ax2.set_title('Simulated image (read noise + bias + sky noise)')
    im2 = ax2.imshow(noisy, cmap=cmap)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax, orientation='vertical')
    cbar2.ax.set_ylabel('ADU')

    plt.tight_layout()
    plt.show()

def narcissus_effect(image, radius=160, center_x=320, center_y=256, smoothness=80, intensity=0.025):
    """
    Applies the narcissus effect to an image.

    Parameters
    ----------
    image (numpy.ndarray):
        Input image to apply the effect on.
    radius (int):
        Radius of the circular mask.
    center_x (int):
        X-coordinate of the circle center.
    center_y (int):
        Y-coordinate of the circle center.
    smoothness (int):
        Smoothness factor for mask edges.
    intensity (float):
        Intensity of the narcissus effect.

    Returns
    -------
        numpy.ndarray: Image with applied narcissus effect.
    """

    canvas = np.zeros((512, 640))

    # Generate a grid of coordinates
    x, y = np.meshgrid(np.arange(640), np.arange(512))

    # Calculate distances from the center
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Create a circular mask
    circular_mask = distances <= radius

    # Apply the mask to the canvas
    canvas[circular_mask] = 1

    circular_mask_f = circular_mask.astype(float)

    gmask = gaussian_filter(circular_mask_f, sigma=smoothness)

    narcissus_effect = gmask * np.ones((512, 640)) / np.max(gmask * np.ones((512, 640)))
    image_nar = (-intensity * narcissus_effect + 1) * image

    return image_nar