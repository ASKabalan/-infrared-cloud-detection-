# **CIRRUS**: Cloud Infrared Recognition & Reconstruction via UNET System

Alternative title : **Single channel long-wave infrared sky images classification and probablistic segmentation of cloud structures with deep-learning / CNN and U-Net architectures**


Deep-learning architecture classify and identify cloud structure on sky infrared images. Standard Convolutional Neural Network (CNN) distinguishes clear sky images from cloud images. An UNet-based segmentation model detects cloud structure onto pre-identified cloudy images and outputs a probabilistic map for each pixel.

## Tasks

- ~~Mettre les données sur Renater~~
- Implémenter optimisation des hyperparamètres avec Optuna

### Wassim

- Mettre le modèle Flax en propre et finir le markdown du notebook
- Faire un script python pour creer un model et l'entrainer
  - args : (Flax ou Keras) Dossier image, channel list, nombre epoch, batch_size, val_batch_size, aug_batch_size, (bool)evaluate,  output save model, output_save_image_prediction
- Faire un script python inference 
  - args : (Flax ou keras) chemin model, dossier_image_cible, batch_size, output_save_image_prediction
- Faire un tikz de l'UNET et double CONV UNET pour le papier

### Kelian

- acces GPU 
- Papier

### Romain

- Classification des image en keras
- (peut-être) faire un script pour l'entrainement from scratch et la classification depuis un model.h5

## DataSet : 

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

### Segmentation

1. Génération des masques "ground-truth" avec les outils de strech de astropy
2. Augmentation des données (rotate, flip, shear, zoom)
3. Normalisation des images brutes (ADU) pour accélérer les calculs
4. Sélectionner N = 1,000 images random pour l'entrainement. Étape importante, il ne faut pas prendre des images consécutives pour éviter la ressemblance entre elles.
5. Split les données en train + test + validation
6. Entraîner le modèle UNet

## Notes

- UNet segmentation
- x = image, y = mask (ground truth)
- Optimizer : adaptive moment (ADAM)
- Loss: binary cross-entropy (binary car True ou False)
- sklearn metrics permet d'avoir le accuracy, recall, etc.
- union over intersection comme métrique pour le U-Net(cf. A. Boucaud) -> voir fonction wassim

## References

[CloudSegNet code](https://github.com/Soumyabrata/CloudSegNet) <br>
[arXiv paper](https://arxiv.org/pdf/1904.07979.pdf) <br>
[DeepL4Astro](https://github.com/ASKabalan/deeplearning4astro_tools/blob/master/dltools/batch.py) <br>
[Day and Night Clouds Detection Using a Thermal-Infrared All-Sky-View Camera](https://doi.org/10.3390/rs13091852) <br>
[Cloud Detection and Classification with the Use of Whole-Sky Ground-Based Images]( https://www.researchgate.net/publication/227860342) <br>
