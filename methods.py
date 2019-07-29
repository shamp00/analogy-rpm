import numpy as np
from numba import jit, njit

import np_clip_fix

# Numerical methods
@njit
def sigmoid(x, k = 1):
    """Sigmoid logistic function"""
    return 1 / (1 + np.exp(-x * k))

@njit
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

@njit
def cross_entropy(predictions: np.ndarray, targets: np.ndarray):
    """ Computes cross entropy between two distributions.
    Input: x: iterabale of N non-negative values
           y: iterabale of N non-negative values
    Returns: scalar
    """

    if np.any(predictions < 0) or np.any(targets < 0):
        raise ValueError('Negative values exist.')

    if not np.any(predictions):
        predictions = np.full(predictions.shape, 1 / len(predictions))
    
    # Force to proper probability mass function.
    #predictions = np.array(predictions, dtype=np.float)
    #targets = np.array(targets, dtype=np.float)
    predictions /= np.sum(predictions)
    targets /= np.sum(targets)

    # Ignore zero 'y' elements.
    mask = targets > 0
    x = predictions[mask]
    y = targets[mask]    
    ce = -np.sum(x * np.log(y)) 
    return ce    

@njit
def mean_squared_error(p1, p2):
    """Calculates the mean squared error between two vectors'"""
    return np.square(np.subtract(p1, p2)).mean()
    #return 0.5 * np.sum(((p1 - p2) ** 2))

@njit
def add_noise(p: np.ndarray, noise: float):
    """Add normally distributed noise to a vector."""
    if noise > 0:
        noise_vector = np.random.normal(1, noise, p.size)
        p = p * noise_vector
        p = np.clip(p, 0., 1.)    
    return p    
