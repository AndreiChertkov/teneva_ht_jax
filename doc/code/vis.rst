Module vis: visualization methods for tensors
---------------------------------------------


.. automodule:: teneva_ht_jax.vis


-----




|
|

.. autofunction:: teneva_ht_jax.vis.show

  **Examples**:

  .. code-block:: python

    # Build the random HT-tensor:
    rng, key = jax.random.split(rng)
    Y = tnv.rand(d=8, n=10, r=[3, 4, 5], key=key)
    
    # Print the resulting HT-tensor:
    tnv.show(Y)                   

    # >>> ----------------------------------------
    # >>> Output:

    # HT-tensor  | d=  8
    # Level:   1 | Shape: (8, 10, 3)
    # Level:   2 | Shape: (4, 3, 4, 3)
    # Level:   3 | Shape: (2, 4, 5, 4)
    # Level:   4 | Shape: (5, 5)
    # 




|
|

