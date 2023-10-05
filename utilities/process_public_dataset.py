import imageio
import numpy as np
from astropy.io import fits
from utilities.utilities import rebin
from pathlib import Path
import glob
import multiprocessing
num_cores = multiprocessing.cpu_count()
from joblib import parallel_backend, Parallel, delayed

def process_dataset(entry,output_path):
    # Read the image using imageio
    fits_file,GT_img = entry
    fits_file = fits.open(name=fits_file)
    image = fits_file[0].data
    binary_mask = imageio.imread(GT_img)
    binary_mask = np.array(binary_mask)

    print(GT_img)
    image = rebin(image, new_shape= (128, 160))
    binary_mask = rebin(binary_mask, new_shape= (128, 160))

    
    try:
        mask_hdu = fits.ImageHDU(binary_mask)
        fits_file.append(mask_hdu)
        fits_file[0].data = image
        fits_file[1].header['IMGTYPE'] = 'GROUND-TRUTH'
        fits_file.writeto(f"{output_path}/{Path(fits_file).name.replace('.fits', '_processed.fits')}", overwrite=True)
        fits_file.close()
        del fits_file
    except:
        print(fits_file)
        

def preprocess_ds(input_path,output_path,GT_path):

    fits_list = glob.glob(input_path+'/*.fits')
    GT_list = glob.glob(GT_path + '/*.[pj][np][g]*')

    print(len(fits_list))
    print(len(GT_list))
    assert(len(fits_list) == len(GT_list))
    public_ds = zip(fits_list,GT_list)

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    with parallel_backend('threading', n_jobs=num_cores):
        Parallel(verbose=5)(delayed(process_dataset)(entry = entry,output_path=output_path) for entry in public_ds)


