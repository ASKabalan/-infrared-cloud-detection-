import imageio
import numpy as np
from astropy.io import fits
from utilities.utilities import rebin_samping
from pathlib import Path
import glob
import multiprocessing
num_cores = multiprocessing.cpu_count()
from joblib import parallel_backend, Parallel, delayed

def process_dataset(entry, output_path):
    """
    Processes a dataset by reading an image and its corresponding binary mask, 
    rebinning them, and writing the processed data to a FITS file.
    
    Parameters:
    - entry (tuple): A tuple containing the paths to the FITS file and its corresponding binary mask.
    - output_path (str): The path where the processed FITS file will be saved.
    """
    fits_file, GT_img = entry
    fits_data = fits.open(name=fits_file)
    image = fits_data[0].data
    binary_mask = imageio.imread(GT_img)
    binary_mask = np.array(binary_mask)

    image = rebin_samping(image, new_shape=(128, 160))
    binary_mask = rebin_samping(binary_mask, new_shape=(128, 160))

    target_file = f"{output_path}/{Path(fits_file).name.replace('.fits', '_processed.fits')}"
    try:
        mask_hdu = fits.ImageHDU(binary_mask)
        fits_data.append(mask_hdu)
        fits_data[0].data = image
        fits_data[1].header['IMGTYPE'] = 'GROUND-TRUTH'
        fits_data.writeto(target_file, overwrite=True)
        fits_data.close()
        del fits_data
    except:
        print(f"Fail : {target_file}")

def preprocess_ds(input_path, output_path, GT_path):
    """
    Preprocesses a dataset by processing each image and its corresponding binary mask in parallel.
    
    Parameters:
    - input_path (str): The path to the directory containing the FITS files.
    - output_path (str): The path where the processed FITS files will be saved.
    - GT_path (str): The path to the directory containing the binary masks.
    
    Example of usage:
    ```
    input_path = "/path/to/fits/files"
    output_path = "/path/to/save/processed/files"
    GT_path = "/path/to/binary/masks"
    preprocess_ds(input_path, output_path, GT_path)
    ```
    """
    fits_list = glob.glob(input_path+'/*.fits')
    GT_list = glob.glob(GT_path + '/*.[pj][np][g]*')
    assert(len(fits_list) == len(GT_list))

    fits_list.sort()
    GT_list.sort()
    public_ds = zip(fits_list, GT_list)

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    with parallel_backend('threading', n_jobs=num_cores):
        Parallel(verbose=5)(delayed(process_dataset)(entry=entry, output_path=output_path) for entry in public_ds)
