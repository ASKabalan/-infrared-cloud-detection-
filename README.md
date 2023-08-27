# Infrared cloud detection

Design ML model to detect cloud images and structure on sky infrared images for the StarDICE experiment.
Two models need to be implemented :
1. **Classification** : detect if the image contains cloud or not
2. **Segmentation** : if cloud(s), detect the structure onto the image

## Process

### Classification

### Segmentation
1. Génération des masques "ground-truth" avec les outils de strech de astropy
2. Augmentation des données (rotate, flip, shear, zoom)
3. Normalisation des images brutes (ADU) pour accélérer les calculs
4. Sélectionner N = 1,000 images random pour l'entrainement. Étape importante, il ne faut pas prendre des images consécutives pour éviter la ressemblance entre elles.
5. Split les données en train + test + validation
6. Entraîner le modèle UNet

## Tasks

- Mettre les données sur Renater
- Appliquer les augmentations de façon random sur les images
- Implémenter optimisation des hyperparamètres avec Optuna
- Enlever les images contenant le "toit"


## Notes

- Réseau UNet
- x = image, y = mask (ground truth)
- optimizer : adaptive moment (ADAM)
- loss: binary cross-entropy (binary car True ou False)
- csklearn metrics permet d'avoir le accuracy, recall, etc.
- union over intersection comme métrique (cf. A. Boucaud) -> voir fonction wassim


## References

[CloudSegNet code](https://github.com/Soumyabrata/CloudSegNet) <br>
[arXiv paper](https://arxiv.org/pdf/1904.07979.pdf) <br>
[DeepL4Astro](https://github.com/ASKabalan/deeplearning4astro_tools/blob/master/dltools/batch.py) <br>
[Day and Night Clouds Detection Using a Thermal-Infrared All-Sky-View Camera](https://doi.org/10.3390/rs13091852) <br>
[Cloud Detection and Classification with the Use of Whole-Sky Ground-Based Images]( https://www.researchgate.net/publication/227860342) <br>
