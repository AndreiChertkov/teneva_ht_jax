"""teneva_ht_jax.vis: visualization methods for tensors.

This module contains the functions for visualization of HT-tensors.

"""
import jax.numpy as jnp


def show(Y):
    """Display mode size and ranks of the given HT-tensor.

    Args:
        Y (list): HT-tensor.

    Todo:
        Add more accurate and informative visualization.

    """
    if not isinstance(Y, list):
        raise ValueError('Invalid HT-tensor')

    d = Y[0].shape[0]

    print(f'HT-tensor  | d={d:-3d}')
    for q, Yl in enumerate(Y):
        print(f'Level: {q+1:-3d} | Shape: {Yl.shape}')
