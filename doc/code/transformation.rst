Module transformation: orthogonalization, truncation and other transformations of the HT-tensors
------------------------------------------------------------------------------------------------


.. automodule:: teneva_ht_jax.transformation


-----




|
|

.. autofunction:: teneva_ht_jax.transformation.full

  **Examples**:

  .. code-block:: python

    d = 8          # Dimension of the tensor
    n = 10         # Mode size for the tensor
    r = [3, 4, 5]  # Ranks for tree layers
    
    # Build the random HT-tensor:
    rng, key = jax.random.split(rng)
    Y = tnv.rand(d, n, r, key)

  .. code-block:: python

    # TODO!
    
    Y_full = tnv.full(Y) # This function is not ready!
    print(Y_full.shape)




|
|

