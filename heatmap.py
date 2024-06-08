import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
# Plot heatmap
def plot_heatmap( grid: NDArray, S: NDArray, L: NDArray, name_s, name_l ):
    """
    grid S x L
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='viridis', interpolation = 'nearest')
    plt.colorbar(label='Mean Avg Error')
    plt.xlabel(name_s)
    plt.ylabel(name_l)
    plt.xticks( np.arange(len(S)), np.round(S, 2) )
    plt.yticks( np.arange(len(L)), np.round(L, 2) )
    plt.title('Diff. in MAE: Filter and Smoother F.B.')
    plt.show()

