import matplotlib.pyplot as plt
import numpy as np

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

# Copy to another folder
"""
import os
import shutil

from pathlib import Path    

r_list = random.choices(images_list, k=1000)

destination_folder = os.getcwd()+'/SUBSET_5000/'

# fetch all files
for r in r_list:
    destination = destination_folder + Path(os.path.abspath(r)).name
    shutil.copy(r, destination)
"""


# Save to PDF
"""
r_list = random.choices(images_list, k=100)

with parallel_backend('dask', n_jobs=num_cores):
    l_plots = Parallel(verbose=10)(delayed(generate_mask)(fits.getdata(r), a_log = 100000, contrast = 0.9, display = True) for r in r_list)
    
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('ground_truth_masks_comparisons.pdf')
for plot in l_plots:
    pp.savefig(plot)
pp.close()
"""