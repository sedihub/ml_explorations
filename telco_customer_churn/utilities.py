"""Provides a few common utilities for the notebooks
"""

import tensorflow as tf

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
    

class SparseCategoricalTP(tf.keras.metrics.TruePositives):
    """True Positive metric.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(tf.math.argmax(y_pred, axis=-1))
        super(SparseCategoricalTP, self).update_state(y_true, y_pred, sample_weight)

        
class SparseCategoricalFN(tf.keras.metrics.FalseNegatives):
    """False negative metric.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(tf.math.argmax(y_pred, axis=-1))
        super(SparseCategoricalFN, self).update_state(y_true, y_pred, sample_weight)

        
class SparseCategoricalFP(tf.keras.metrics.FalsePositives):
    """False positive metric.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(tf.math.argmax(y_pred, axis=-1))
        super(SparseCategoricalFP, self).update_state(y_true, y_pred, sample_weight)

        
class SparseCategoricalTN(tf.keras.metrics.TrueNegatives):
    """True negative metric.
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(tf.math.argmax(y_pred, axis=-1))
        super(SparseCategoricalTN, self).update_state(y_true, y_pred, sample_weight)
        
        
def plot_binary_confusion_matrix(conf_mat: np.ndarray):
    """ Plots confusion matrix for binary classification.
    """
    conf_mat = conf_mat / np.sum(conf_mat)
    plt.title("Confusion Matrix", color="gray", fontsize=18)
    plt.imshow(conf_mat, cmap="Blues")
    ax = plt.gca()
    ax.set_xlabel("Prediction", color="gray", fontsize=14)
    ax.set_ylabel("Condition", color="gray", fontsize=14)
    ax.text(0.0, 0.0, "TN\n" + str(round(conf_mat[0, 0], 2)), 
            ha="center", va="center", color="royalblue", fontsize=18, transform=ax.transData)
    ax.text(1.0, 0.0, "FP\n" + str(round(conf_mat[0, 1], 2)), 
            ha="center", va="center", color="orange", fontsize=18, transform=ax.transData)
    ax.text(0.0, 1.0, "FN\n" + str(round(conf_mat[1, 0], 2)), 
            ha="center", va="center", color="orange", fontsize=18, transform=ax.transData)
    ax.text(1.0, 1.0, "TP\n" + str(round(conf_mat[1, 1], 2)), 
            ha="center", va="center", color="royalblue", fontsize=18, transform=ax.transData)
    ax.set_xlabel("Prediction", color="gray", fontsize=14)
    ax.set_ylabel("Condition", color="gray", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    

def evaluate_model(model, features, labels, show_confusion_matrix=True):
    """A helper class for evaluating a trained TF Keras model with above
    custom metrics (TP, FP, FN, and TN) attached.
    
    Args:
        model: The trained TF Keras model.
        features: Inputs or features. Could be an array, a generator, 
            an instance of tf.keras.sequence, or even a tf.data.Dataset
            instance.
        labels: The array of labels if features are provided as an array,
        show_confusion_matrix: If `True` (default) will plot the confusion
            matrix after evaluation.
    """
    eval_results = model.evaluate(
    features, labels, return_dict=True, verbose=0)

    # print("Evaluation results:")
    for name, val in eval_results.items():
        print(f"\t{name:32}{round(float(val), 5)}")
    
    if show_confusion_matrix:
        conf_mat = np.array([
            [eval_results["TN"], eval_results["FP"]],
            [eval_results["FN"], eval_results["TP"]]])
        plot_binary_confusion_matrix(conf_mat)