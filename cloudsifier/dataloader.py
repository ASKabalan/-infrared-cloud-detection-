# -*- coding: utf-8 -*-
# pylint: disable=R0902

import jax
import numpy
from astropy.io import fits


def chosen_datasets(percentage, path_image_files, path_labels_files, reproductible=False):
    nb_imgs = len(path_image_files)
    if reproductible:
        key = jax.random.PRNGKey(10)
        vec = jax.numpy.arange(nb_imgs)
        shuffled_indices = jax.random.permutation(key, len(vec))
        shuffled_vec = vec[shuffled_indices]
    else:
        shuffled_vec = numpy.arange(nb_imgs)
        numpy.random.shuffle(shuffled_vec)

    split_index = int(len(shuffled_vec) * percentage)
    training_vector = shuffled_vec[:split_index]
    test_vector = shuffled_vec[split_index:]

    training_images_files = [path_image_files[i] for i in training_vector]
    training_labels_files = [path_labels_files[i] for i in training_vector]
    test_images_files = [path_image_files[i] for i in test_vector]
    test_labels_files = [path_labels_files[i] for i in test_vector]

    return training_images_files, training_labels_files, test_images_files, test_labels_files


class DataLoader:
    def __init__(
        self, image_files, labels_files, batch_size, mean_global=None, std_global=None, min_global=None, max_global=None, shuffle=True, normalisation="mean_std"
    ):
        self.image_files = image_files
        self.labels_files = labels_files
        self.batch_size = batch_size
        self.min_global = min_global
        self.max_global = max_global
        self.mean_global = mean_global
        self.std_global = std_global
        self.shuffle = shuffle
        self.normalisation = normalisation

        self.indices = numpy.arange(len(self.image_files))
        assert len(self.image_files) == len(self.labels_files)

    @staticmethod
    def load_batch_labels(batch_indices, data_files):
        return numpy.array([numpy.load(data_files[idx]) for idx in batch_indices])

    @staticmethod
    def load_batch_imgs(batch_indices, data_files):
        return numpy.array([fits.getdata(data_files[idx]) for idx in batch_indices])

    @staticmethod
    def replace_bad_values(array):
        return jax.numpy.nan_to_num(array)

    def min_max_normalise(self, images):
        return (images - self.min_global) / (self.max_global - self.min_global)

    def mean_std_normalise(self, images):
        return (images - self.mean_global) / self.std_global

    def generate_batches(self):
        if self.shuffle:
            numpy.random.shuffle(self.indices)

        for i in range(0, len(self.image_files), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]  # noqa

            batch_images = self.load_batch_imgs(batch_indices, self.image_files)
            batch_labels = self.load_batch_labels(batch_indices, self.labels_files)

            if self.normalisation == "min_max":
                batch_images = self.min_max_normalise(batch_images)
            elif self.normalisation == "mean_std":
                batch_images = self.mean_std_normalise(batch_images)

            batch_images = self.replace_bad_values(batch_images)
            batch_labels = self.replace_bad_values(batch_labels)

            batch_labels = numpy.expand_dims(batch_labels, axis=-1)
            batch_images = numpy.expand_dims(batch_images, axis=-1)

            yield jax.numpy.array(batch_images), jax.numpy.array(batch_labels)
