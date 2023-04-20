"""teneva_ht_jax.act_one: single HT-tensor operations.

This module contains the basic operations with one HT-tensor (Y), including
"get", etc.

"""
import jax
import jax.numpy as jnp


def get(Y, k):
    """Compute the element of the HT-tensor.

    Args:
        Y (list): d-dimensional HT-tensor.
        k (np.ndarray): the multi-index for the tensor of the length d.

    Returns:
        float: the element of the HT-tensor.

    """
    def body_leaf(q, data):
        i1, i2, G1, G2, G = data
        q = jnp.einsum('r,q,rsq->s', G1[i1], G2[i2], G)
        return None, q

    def body(q, data):
        G1, G2, G = data
        q = jnp.einsum('r,q,rsq->s', G1, G2, G)
        return None, q

    # Compute for the first level (leafs):
    _, Q = jax.lax.scan(body_leaf, None, (k[0::2], k[1::2], Y[0][0::2], Y[0][1::2], Y[1]))

    # Compute for the inner levels:
    for k in range(1, len(Y)-2):
        _, Q = jax.lax.scan(body, None, (Q[0::2], Q[1::2], Y[k+1]))

    # Compute for the last level (root):
    q = jnp.einsum('r,q,rq->', Q[0], Q[1], Y[-1])

    return q
