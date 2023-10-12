import numpy as np
from scipy import ndimage

def zoom(image, mask, zoom_range=(1, 3)):
    """
    Zooms the given image and mask by a random factor within the specified range.
    
    Parameters:
    - image (ndarray): The input image.
    - mask (ndarray): The input mask.
    - zoom_range (tuple, optional): The range of zoom factors. Defaults to (1, 3).
    
    Returns:
    - tuple: The zoomed image and mask.
    """
    zoom_factor = np.random.uniform(*zoom_range)
    new_size = np.round(np.array(image.shape) * zoom_factor).astype(int)
    zoomed_image = ndimage.zoom(image, zoom_factor, order=1)
    zoomed_mask = ndimage.zoom(mask, zoom_factor, order=0)

    # Crop or pad the zoomed image and mask to match the original size
    diff_h = image.shape[0] - new_size[0]
    diff_w = image.shape[1] - new_size[1]
    pad_h = max(-diff_h // 2, 0)
    pad_w = max(-diff_w // 2, 0)
    cropped_image = zoomed_image[pad_h:pad_h+image.shape[0], pad_w:pad_w+image.shape[1]]
    cropped_mask = zoomed_mask[pad_h:pad_h+image.shape[0], pad_w:pad_w+image.shape[1]]

    return cropped_image, cropped_mask

def rotate(image, mask, angle_range=(-45, 45)):
    """
    Rotates the given image and mask by a random angle within the specified range.
    
    Parameters:
    - image (ndarray): The input image.
    - mask (ndarray): The input mask.
    - angle_range (tuple, optional): The range of rotation angles in degrees. Defaults to (-45, 45).
    
    Returns:
    - tuple: The rotated image and mask.
    """
    angle = np.random.uniform(*angle_range)
    rotated_image = ndimage.rotate(image, angle, reshape=False, mode='reflect')
    rotated_mask = ndimage.rotate(mask, angle, reshape=False, mode='reflect')
    return rotated_image, rotated_mask

def shift(image, mask, shift_range=(-200, 200)):
    """
    Shifts the given image and mask by a random amount within the specified range.
    
    Parameters:
    - image (ndarray): The input image.
    - mask (ndarray): The input mask.
    - shift_range (tuple, optional): The range of shift values. Defaults to (-200, 200).
    
    Returns:
    - tuple: The shifted image and mask.
    """
    shift_values = np.random.randint(*shift_range, size=2)
    shifted_image = ndimage.shift(image, shift_values, mode='reflect')
    shifted_mask = ndimage.shift(mask, shift_values, mode='reflect')
    return shifted_image, shifted_mask

def shear(image, mask, shear_range=(-0.2, 0.2)):
    """
    Shears the given image and mask by a random amount within the specified range.
    
    Parameters:
    - image (ndarray): The input image.
    - mask (ndarray): The input mask.
    - shear_range (tuple, optional): The range of shear values. Defaults to (-0.2, 0.2).
    
    Returns:
    - tuple: The sheared image and mask.
    """
    shear_values = np.random.uniform(*shear_range, size=2)
    shear_matrix = np.array([[1, shear_values[0]], [shear_values[1], 1]])
    sheared_image = ndimage.affine_transform(image, shear_matrix, mode='reflect')
    sheared_mask = ndimage.affine_transform(mask, shear_matrix, mode='reflect')
    return sheared_image, sheared_mask

def flip(image, mask):
    """
    Flips the given image and mask horizontally.
    
    Parameters:
    - image (ndarray): The input image.
    - mask (ndarray): The input mask.
    
    Returns:
    - tuple: The flipped image and mask.
    """
    flipped_image = np.fliplr(image)
    flipped_mask = np.fliplr(mask)
    return flipped_image, flipped_mask

def random_augmentation(image, mask):
    """
    Applies a random augmentation (zoom, rotate, shift, shear, or flip) to the given image and mask.
    
    Parameters:
    - image (ndarray): The input image.
    - mask (ndarray): The input mask.
    
    Returns:
    - tuple: The augmented image and mask.
    """
    augmentation_functions = [zoom, rotate, shift, shear, flip]
    chosen_function = np.random.choice(augmentation_functions)
    augmented_image, augmented_mask = chosen_function(image, mask)
    return augmented_image, augmented_mask

def random_augment(img_mask):
    """
    Wrapper function for random_augmentation that takes a tuple of image and mask.
    
    Parameters:
    - img_mask (tuple): Tuple containing the image and mask.
    
    Returns:
    - tuple: The augmented image and mask.
    """
    img , mask = img_mask
    aug_image, aug_mask = random_augmentation(img,mask)
    aug_fit = (aug_image, aug_mask)

    return aug_fit
