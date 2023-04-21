"""teneva_ht_jax.tensors: various useful HT-tensors.

This module contains the collection of functions for explicit construction of
various useful HT-tensors (only random tensor for now).

"""
import jax
import jax.numpy as jnp


def rand(d, n, r, key, a=-1., b=1.):
    """Construct a random HT-tensor from the uniform distribution.

    Args:
        d (int): number of tensor dimensions.
        n (int): mode size of the tensor.
        r (int): TT-ranks of the tensor. It should be a number (if all ranks
            are equal) or list of the length q-1, where q is a number of levels.
        key (jax.random.PRNGKey): jax random key.
        a (float): minimum value for random items of the HT-cores.
        b (float): maximum value for random items of the HT-cores.

    Returns:
        list: HT-tensor.

    """
    q = d.bit_length() # Full number of levels (e.g., d=8 -> q=4)

    if isinstance(r, int):
        r = [r] * (q-1)

    if len(r) != (q-1):
        raise ValueError('Invalid length of ranks list')

    Y = []

    def _rand_level(key, sh):
        key, key_cur = jax.random.split(key)
        Yl = jax.random.uniform(key_cur, sh, minval=a, maxval=b)
        return Yl, key

    # Build the first level (leafs):
    Yl, key = _rand_level(key, sh=(d, n, r[0]))
    Y.append(Yl) # 3D tensor (len, n, r_up)

    # Build the inner levels:
    for k in range(1, q-1):
        dl = 2**(q-k-1) # Length of the current layer
        Yl, key = _rand_level(key, sh=(dl, r[k-1], r[k], r[k-1]))
        Y.append(Yl) # 4D tensor (len, r_down, r_up, r_down)

    # Build the last level (root):
    Yl, key = _rand_level(key, sh=(r[-1], r[-1]))
    Y.append(Yl) # 2D tensor (r_down, r_down)

    return Y
