## **IRIS-CloudDeep**: Infrared Radiometric Image Classification and Segmentation of Cloud Structure using a Deep-learning Framework for Ground-based Long-wave Infrared Thermal Camera Observation

Deep-learning architecture classifies and identifies cloud structure in sky infrared images. A standard Convolutional Neural Network (CNN) distinguishes clear sky images from cloud images. A UNet-based segmentation model detects cloud structure in pre-identified cloudy images and outputs a probabilistic map for each pixel.

**Example:**

```python
import multiprocessing
from joblib import parallel_backend, Parallel, delayed
num_cores = multiprocessing.cpu_count()

dataset_directory = ''
images_list = glob.glob(dataset_directory+'/*.fits')

# To bin (Classification)

with parallel_backend('threading', n_jobs=num_cores):
    l_plots = Parallel(verbose=5)(delayed(rebin_fits)(filename = filename, bin_size=(128, 160)) for filename in images_list)

# To generate the mask and bin at the same time (Segmentation)

with parallel_backend('threading', n_jobs=num_cores):
    l_plots = Parallel(verbose=5)(delayed(generate_mask)(filename = filename,
    bin_size=(128, 160), a_log = 100000, contrast = 0.9, display = False, return_mask = False, write_to_fits = True) for filename in images_list)
```

## Process

### Classification

1. Generation of the dataset
2. Create environment: `micromamba env create -f ENV_LINUX_CLASSIFIER.yml`
3. Change the `.jsonc` file to your convenience:

   ```json
   {
       "general":{
           "NAME_DB" : "TOTAL_2800",        // name of the database
           "path_folders" : "pisco/CLOUD",  // subfolder of the database
           "directories" : ["/home", "/net/GECO"],     // folder of the database
           "percentage" : 0.7,                         // train percentage
           "normalisation" : "min_max"                 // mean_std or min_max
       },
       "network": {
           "batch_size" : 64,
           "num_epochs" : 10,
           "early_stopping" : 10,
           "type_optimizer" : "piecewise",    	   // exponential or piecewise
           "momentum" : 0.5                           // ]0;1[
       }
   }
   ```
4. Train linear classifiers: `python train_linear_classifiers.py`
5. Train neural network: `python train_neural_network.py`

### Segmentation

1. Generate "ground-truth" masks using astropy stretch tools.
2. Data augmentation (rotate, flip, shear, zoom).
3. Normalize raw images (ADU) to accelerate calculations.
4. Select 1,000 random images for training. It is important not to take consecutive images to avoid resemblance.
5. Split the data into training, testing, and validation sets.
6. Train the UNet model.

Training for segmentation is done in the notebook `segmentation/CloudSegmentation_Jax.ipynb`, which uses a UNet model implemented in Flax.

## Usage Notebooks

### Cloud Classification and Segmentation

The notebook `notebooks/Cloud_classification_segmentation.ipynb` demonstrates how to use the classification and segmentation models together. The classifier first identifies if an image contains clouds, and if clouds are detected, the image is passed through the segmentation model. The final output includes the original sky images and the segmented images with ground truth.

### Blob Counter

The notebook `notebooks/blob_counter.ipynb` shows how to use the model for post-processing. It includes functions to binarize images, count blobs (connected regions), and quantify sky quality. This notebook demonstrates how the segmentation results can be utilized for further analysis and processing.

## References

[CloudSegNet code](https://github.com/Soumyabrata/CloudSegNet) `<br>`
[CloudSegNet arXiv paper](https://arxiv.org/pdf/1904.07979.pdf) `<br>`
[DeepL4Astro](https://github.com/ASKabalan/deeplearning4astro_tools/blob/master/dltools/batch.py) `<br>`
[Day and Night Clouds Detection Using a Thermal-Infrared All-Sky-View Camera](https://doi.org/10.3390/rs13091852) `<br>`

This GitHub repo accompanies the paper.
