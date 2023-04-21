"""teneva_ht_jax.transformation: transformation of HT-tensors.

This module contains the function for transformation of the HT-tensor into full
format.

"""
import jax
import jax.numpy as jnp


def full(Y):
    """Export HT-tensor to the full format.

    Args:
        Y (list): HT-tensor.

    Returns:
        jnp.ndarray: multidimensional array related to the given HT-tensor.

    Note:
         This function can only be used for relatively small tensors, because
         the resulting tensor will have n^d elements and may not fit in memory
         for large dimensions. And this function does not take advantage of
         jax's ability to speed up the code and can be slow, but it should only
         be meaningfully used for tensors of small dimensions.

    """
    # TODO !
    return 42.
