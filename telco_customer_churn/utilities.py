"""Provides a few common utilities for the notebooks
"""

import numpy as np
import matplotlib.pyplot as plt


def compile_correlation_matrix(
    binary_categories :dict,
    diagonal_values=1.0,
    correlation_func=None,
    symmetric=True):
    """Compiles the correlation between the categories.
    Args:
        binary_categories: A dictionary of category names to lists or 
            numpy arrays of bools.
        diagonal_values: The value to use for diagonal elements. Default
            is 1.0.
        correlation_func: A callable used to compute correlations. 
            Default reduction is xnor.
        symmetric: If False, will allow for asymmetric corellation/dependecy
            computation. Useful in conjunction with `correlation_func`.
        
    Returns:
        Tuple of (n, n) correlations array and list of categories.
    """
    correlation_matrix = np.eye(
        len(binary_categories), dtype=np.single)
    
    for row_idx, row_feature_name in enumerate(binary_categories.keys()):
        correlation_matrix[row_idx, row_idx] = diagonal_values
        for col_idx, col_feature_name in enumerate(binary_categories.keys()):
            if symmetric and col_feature_name == row_feature_name:
                break
            #
            if correlation_func is not None:
                correlation_matrix[row_idx, col_idx] = correlation_func(
                    binary_categories[row_feature_name],
                    binary_categories[col_feature_name])
            else:
                correlation_matrix[row_idx, col_idx] = np.sum(
                    np.logical_not(np.logical_xor(
                        binary_categories[row_feature_name],
                        binary_categories[col_feature_name])
                )).astype(np.single) / len(binary_categories[row_feature_name])
            # The correlation matrix is symmetric:
            if symmetric:
                correlation_matrix[col_idx, row_idx] = correlation_matrix[row_idx, col_idx]
                
    return (correlation_matrix, 
            [x.replace("_", " ").lower() for x in binary_categories.keys()])

                
def plot_correlation_matrix(
    correlation_matrix: np.ndarray, 
    category_names: list, 
    show_lower_half_only=False,
    **kwargs):
    """Plots the correlation matrix.
  
    Args:
        correlation_matrix: Array of correlations of the shape (n, n).
        category_names: A list or numpy array of strings/objects of length n.
        show_lower_half_only: If `True`, will mask the upper half of the matrix.
        **kwargs
        
    Returns:
        None
    """
     
    fig = plt.figure(figsize=kwargs.get("figsize", (8., 8.)))
    
    ax = plt.gca()
    if "title" in kwargs.keys():
        ax.set_title(kwargs["title"], fontsize=kwargs.get("title_fontsize", 16))
    
    # Mask the upper half of the matrix, if specified:
    if show_lower_half_only:
        matrix_shape = correlation_matrix.shape
        rr, cc = np.meshgrid(np.arange(matrix_shape[0]), np.arange(matrix_shape[1]))
        masked_correlation_matrix = np.ma.masked_where(cc < rr, correlation_matrix)
        ax.imshow(masked_correlation_matrix, cmap=kwargs.get("cmap", "Blues"))
    else:
        ax.imshow(correlation_matrix, cmap=kwargs.get("cmap", "Blues"))
        
    ax.set_xticks(range(len(category_names)))
    ax.set_xticklabels(category_names, 
                       rotation=45, ha="right", va="top", 
                       fontsize=12, fontweight="normal", color="gray")
    ax.set_yticks(range(len(category_names)))
    ax.set_yticklabels(category_names, 
                       rotation=45, ha="right", va="top", 
                       fontsize=12, fontweight="normal", color="gray")
    
    for row_idx in range(correlation_matrix.shape[0]):
        for col_idx in range(correlation_matrix.shape[1]):
            if show_lower_half_only and col_idx > row_idx:
                break
            ax.text(col_idx, row_idx, str(round(float(correlation_matrix[row_idx, col_idx]), 3)), 
                    ha="center", va="center", color=kwargs.get("text_color", "royalblue"), 
                    fontsize=12, fontweight="normal", transform=ax.transData)
            
    for name in ax.spines.keys():
        ax.spines[name].set_visible(False)
        
    ax.tick_params(axis="both", which="both",length=0)
    
    plt.show()