# **CIRRUS**: Cloud Infrared Recognition & Reconstruction via UNET System

Alternative title : **Single channel long-wave infrared sky images classification and probablistic segmentation of cloud structures with deep-learning / CNN and U-Net architectures**


Deep-learning architecture classify and identify cloud structure on sky infrared images. Standard Convolutional Neural Network (CNN) distinguishes clear sky images from cloud images. An UNet-based segmentation model detects cloud structure onto pre-identified cloudy images and outputs a probabilistic map for each pixel.

## Tasks

- Mettre les données sur Renater
- Implémenter optimisation des hyperparamètres avec Optuna

## Process

### Classification

### Segmentation
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

## References

[CloudSegNet code](https://github.com/Soumyabrata/CloudSegNet) <br>
[arXiv paper](https://arxiv.org/pdf/1904.07979.pdf) <br>
[DeepL4Astro](https://github.com/ASKabalan/deeplearning4astro_tools/blob/master/dltools/batch.py) <br>
[Day and Night Clouds Detection Using a Thermal-Infrared All-Sky-View Camera](https://doi.org/10.3390/rs13091852) <br>
[Cloud Detection and Classification with the Use of Whole-Sky Ground-Based Images]( https://www.researchgate.net/publication/227860342) <br>
