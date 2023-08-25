# Infrared cloud detection

Design ML model to detect cloud images and structure on sky infrared images for the StarDICE experiment.

## Notes

- Faire les masques avec le ZScaleInterval de astropy avec un contraste de 1

- Sélectionner 2,000 à 3000 images contenant des nuages
ne pas mettre d'images sans nuages

- UNet
-> prendre la fonction que tu m'envoies sur le notebook
-> x = image, y = mask
-> optimizer : adaptive moment (ADAM)
-> loss: binary cross-entropy (binary car True ou False)

- split training test avec keras ou sklearn (train_test_split)

- méthode self-supervised

- optuna : framework qui permet d'optimiser les hyper parameters

- métrique : val loss

-sklearn metrics permet d'avoir le accuracy, recall, etc.


- union over intersection comme métrique (cf. A. Boucaud)
-> voir fonction wassim

https://github.com/ASKabalan/deeplearning4astro_tools/blob/master/dltools/batch.py


- renater filesender pour envoyer les images avec et sans nuages

-
