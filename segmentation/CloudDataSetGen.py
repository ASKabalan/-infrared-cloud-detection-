# CloudDataSetGen.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import random
from joblib import parallel_backend, Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
from ..utilities import augmentations as aug
from ..utilities import utilities as util
from astropy.io import fits
from jax import numpy as jnp

def open_fits_with_mask(filename):
    image = fits.open(filename)
    cloud = image[0].data
    mask = image[1].data
    del image

    # Normalize image
    cloud  = cloud / 2**14
    return cloud , mask

# @title
class CloudImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, batch_size=8,aug_batch_size=2, shuffle=True):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.aug_batch_size = aug_batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.x_data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.x_data) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indices = self.indices[start_idx:end_idx]

        if self.y_data is None: # if only x is requested (inference case)
            batch_x = self.x_data[batch_indices]
            batch_x = jnp.array(batch_x[..., np.newaxis])

            return batch_x
        else:
            batch_x = self.x_data[batch_indices]
            batch_y = self.y_data[batch_indices]

            # Create a list of tuples using zip
            batch_list = list(zip(batch_x, batch_y))

            if self.aug_batch_size != 0:
                l_fits_aug = [aug.random_augment(img_mask=random.choice(batch_list)) for _ in range(self.aug_batch_size)]
                fits_images_aug = np.array(l_fits_aug)

                batch_x =  np.concatenate((batch_x, fits_images_aug[:,0]), axis=0)
                batch_y =  np.concatenate((batch_y, fits_images_aug[:,1]), axis=0)

            # Convert the TensorFlow tensors to JAX arrays
            batch_x = jnp.array(batch_x[..., np.newaxis])
            batch_y = jnp.array(batch_y[..., np.newaxis])


            return batch_x, batch_y


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_dataset(input_folder_path, train_batch_size, aug_batch_size, val_batch_size):
    images_list = glob.glob(f'{input_folder_path}/*.fits')
    images_list = random.choices(images_list, k=len(images_list))

    with parallel_backend('threading', n_jobs=num_cores):
        l_fits = Parallel(verbose=0)(delayed(open_fits_with_mask)(filename=r) for r in images_list)
    l_fits = np.array(l_fits)
    
    X_train, X_test, y_train, y_test = train_test_split(l_fits[:,0], l_fits[:,1], test_size=0.2, random_state=42)
    
    ds_train_gen = CloudImageDataGenerator(x_data=X_train, y_data=y_train, batch_size=train_batch_size, aug_batch_size=aug_batch_size, shuffle=True)
    ds_val_gen = CloudImageDataGenerator(x_data=X_test, y_data=y_test, batch_size=val_batch_size, aug_batch_size=0, shuffle=False)
    
    return ds_train_gen, ds_val_gen

def load_inference_ds(input_folder_path,batch_size):
    images_list = glob.glob(f'{input_folder_path}/*.fits')
    images_list = random.choices(images_list, k=len(images_list))

    with parallel_backend('threading', n_jobs=num_cores):
        l_fits = Parallel(verbose=0)(delayed(open_fits_with_mask)(filename=r) for r in images_list)
    l_fits = np.array(l_fits)

    ds_pred_gen = CloudImageDataGenerator(x_data=l_fits[:,0], y_data=None, batch_size=batch_size, aug_batch_size=0, shuffle=False)
    
    return ds_pred_gen
