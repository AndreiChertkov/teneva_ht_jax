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
    q = d.bit_length() - 1

    if isinstance(r, int):
        r = [r] * q

    if len(r) != q:
        raise ValueError('Invalid length of ranks list')

    Y = []

    key, key_cur = jax.random.split(key)
    Y_cur = jax.random.uniform(key_cur, (2**q, n, r[0]), minval=a, maxval=b)
    Y.append(Y_cur)

    for k in range(1, q+1):
        key, key_cur = jax.random.split(key)

        if k < q:
            Y_cur = jax.random.uniform(key_cur,
                (2**(q-k), r[k-1], r[k], r[k-1]), minval=a, maxval=b)
        else:
            Y_cur = jax.random.uniform(key_cur,
                (r[k-1], r[k-1]), minval=a, maxval=b)

        Y.append(Y_cur)

    return Y
