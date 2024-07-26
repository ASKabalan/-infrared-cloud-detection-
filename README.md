## **IRIS-CloudDeep**: Infrared Radiometric Image classification and Segmentation of Cloud structure using Deep-learning framework for ground-based long-wave infrared thermal camera observation

Deep-learning architecture classify and identify cloud structure on sky infrared images. Standard Convolutional Neural Network (CNN) distinguishes clear sky images from cloud images. An UNet-based segmentation model detects cloud structure onto pre-identified cloudy images and outputs a probabilistic map for each pixel.

## Datasets

- raw image Seg : https://filesender.renater.fr/?s=download&token=2bd01943-b173-4bb3-b0fa-41e6528ecbb9
- raw image classification : https://filesender.renater.fr/?s=download&token=6277e8d4-3fde-4903-84fa-fa44db669ef4
- preprocess image by wass : https://filesender.renater.fr/?s=download&token=8035173f-b7a1-4549-a24d-b210527d20d6

__Note:__ Attention pour les Raw image il faut les binner

**Example:**

```python
import multiprocessing
from joblib import parallel_backend, Parallel, delayed
num_cores = multiprocessing.cpu_count()

dataset_directory = ''
images_list = glob.glob(dataset_directory+'/*.fits')

# Pou binner (Classification)

with parallel_backend('threading', n_jobs=num_cores):
    l_plots = Parallel(verbose=5)(delayed(rebin_fits)(filename = filename,bin_size=(128, 160)) for filename in images_list)

# Pour Generer les mask et binner en même temps (Segmentation)

with parallel_backend('threading', n_jobs=num_cores):
    l_plots = Parallel(verbose=5)(delayed(generate_mask)(filename = filename,
    bin_size=(128, 160), a_log = 100000, contrast = 0.9, display = False, return_mask = False, write_to_fits = True) for filename in images_list)

```

## Process

### Classification

1. Generation of the dataset 
2. micromamba env create -f ENV_LINUX_CLASSIFIER.yml
3. change the .jsonc file to your conveniance:

   ```
   {
       "general":{
           "NAME_DB" : "TOTAL_2800",		    // name of the database
           "path_folders" : "pisco/CLOUD",		    // subfolder of the database
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
4. python train_linear_classifiers.py
5. python train_neural_network.py

### Segmentation

1. Génération des masques "ground-truth" avec les outils de strech de astropy
2. Augmentation des données (rotate, flip, shear, zoom)
3. Normalisation des images brutes (ADU) pour accélérer les calculs
4. Sélectionner N = 1,000 images random pour l'entrainement. Étape importante, il ne faut pas prendre des images consécutives pour éviter la ressemblance entre elles.
5. Split les données en train + test + validation
6. Entraîner le modèle UNet

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

This github repo is accompaniying the paper
