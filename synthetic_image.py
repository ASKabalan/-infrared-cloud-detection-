import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import visualization as aviz
from astropy.nddata.blocks import block_reduce
from scipy.ndimage import gaussian_filter
import os

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
        rng = np.random.RandomState(seed=8392)  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        # This adds a little random-looking noise into the data.
        col_pattern = rng.randint(0, int(0.1 * value), size=shape[0])
        
        # Make the chosen columns a little brighter than the rest...
        for c in columns:
            bias_im[:, c] = value + np.random.normal(np.zeros(col_pattern.shape), col_pattern) 

        rng = np.random.RandomState(seed=8393)  # 20180520
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

def simulate_clear_sky_image(start=1000, stop=400, width=640, height=512, is_horizontal=True, bias_level=300, read_noise_level = 5, bad_pixel_columns = 50, sky_noise_level = 10, return_original = True):
    """
    Simulate an infrared clear sky image

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
    bad_pixel_columns: int
        Number of bad behaving pixel columns in the image (~50)
    sky_noise_level : float
        The target value for the number of counts from the sky.
    return_original:
        If True, returns both synthetic and noisy images

    Returns
    -------
    np.array or tuple
    """

    synthetic_image = get_gradient_2d(start=start, stop=stop, width=width, height=height, is_horizontal=is_horizontal)

    noise_im = synthetic_image + read_noise(synthetic_image, read_noise_level)

    bias_only = bias(synthetic_image, bias_level, realistic=True, number_of_colums = bad_pixel_columns)
    bias_noise_im = noise_im + bias_only

    sky_only = sky_background(synthetic_image, sky_noise_level)
    noisy_synthetic_image = bias_noise_im + sky_only

    if return_original == True:
        return synthetic_image, noisy_synthetic_image
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

def apply_narcissus_effect(image, radius=160, center_x=320, center_y=256, smoothness=80, intensity=0.025):
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