from jax import jit
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr

def process_data_and_create_histogram(data_tuple, image_index=2, bins=50):
    """
    Processes a tuple of JAX arrays to create a histogram of a specified image.

    Args:
    - data_tuple (tuple): A tuple of JAX arrays.
    - image_index (int, optional): Index of the image in the tuple to create a histogram for. Defaults to 2.
    - bins (int, optional): Number of bins for the histogram. Defaults to 50.

    Returns:
    Tuple (data_tuple, hist, bin_edges): The data tuple, histogram values, and bin edges.
    """
    # Ensure the image index is within the range of the tuple
    if image_index >= len(data_tuple):
        raise IndexError("Image index is out of range for the provided data tuple.")

    image_data = data_tuple[image_index]
    hist, bin_edges = np.histogram(image_data.flatten(), bins=bins)

    return (data_tuple, hist, bin_edges)


from scipy.stats import pearsonr

def compute_correlation_matrix(fits_histogram_tuples):
    """
    Computes the Pearson correlation matrix between histograms from a list of tuples.

    Args:
    - fits_histogram_tuples (list of tuples): Each tuple contains a FITS file, histogram, and bin edges.

    Returns:
    - np.ndarray: Correlation matrix.
    """
    num_files = len(fits_histogram_tuples)
    correlation_matrix = np.zeros((num_files, num_files))

    for i in range(num_files):
        for j in range(num_files):
            if i != j:
                # Calculate correlation only for different histograms
                correlation, _ = pearsonr(fits_histogram_tuples[i][1], fits_histogram_tuples[j][1])
                correlation_matrix[i, j] = correlation

    return correlation_matrix


def find_correlated_fits(correlation_matrix, index, fits_data, N=3):
    """
    Finds the N most and least correlated FITS files to a given FITS file, identified by index.

    Args:
    - correlation_matrix (np.ndarray): The correlation matrix.
    - index (int): The index of the FITS file in question.
    - fits_data (list): The list of FITS data corresponding to the indices.
    - N (int, optional): Number of most and least correlated files to find. Defaults to 3.

    Returns:
    - dict: Dictionary with keys 'most_correlated' and 'least_correlated', each containing a list of tuples (index, FITS file).
    """
    # Exclude the index itself and get correlations
    correlations = correlation_matrix[index, :]
    correlations[index] = np.nan  # Ignore self-correlation

    # Find the indices of the N most and least correlated
    most_correlated_indices = np.argsort(-correlations)[:N]  # Negative for descending order
    least_correlated_indices = np.argsort(correlations)[:N]

    # Retrieve the corresponding FITS files
    original_fits = fits_data[index]
    most_correlated = [fits_data[idx] for idx in most_correlated_indices if not np.isnan(correlations[idx])]
    least_correlated = [fits_data[idx] for idx in least_correlated_indices if not np.isnan(correlations[idx])]

    return {
        'original'  : original_fits,
        'most_correlated': most_correlated,
        'least_correlated': least_correlated
    }