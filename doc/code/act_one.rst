Module act_one: single HT-tensor operations
-------------------------------------------


.. automodule:: teneva_ht_jax.act_one


-----




|
|

.. autofunction:: teneva_ht_jax.act_one.get

  **Examples**:

  .. code-block:: python

    d = 8          # Dimension of the tensor
    n = 10         # Mode size for the tensor
    r = [3, 4, 5]  # Ranks for tree layers
    
    # Build the random HT-tensor:
    rng, key = jax.random.split(rng)
    Y = tnv.rand(d, n, r, key)
    
    # Select some tensor element and compute the value:
    k = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = tnv.get(Y, k)
    print(y)

    # >>> ----------------------------------------
    # >>> Output:

    # -1.5443088
    # 

  We may transform the HT-tensor into full format and check the result:

  .. code-block:: python

    # TODO!
    
    Y_full = tnv.full(Y) # This function is not ready!
    
    y_full = Y_full[tuple(k)]
    print(y_full)
    
    # Let compare values:
    e = np.abs(y - y_full)
    print(f'Error : {e:7.1e}')




|
|

