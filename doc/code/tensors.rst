Module tensors: collection of explicit useful HT-tensors
--------------------------------------------------------


.. automodule:: teneva_ht_jax.tensors


-----




|
|

.. autofunction:: teneva_ht_jax.tensors.rand

  **Examples**:

  .. code-block:: python

    d = 8         # Dimension of the tensor
    n = 10        # Mode size for the tensor
    r = [3, 4, 5] # Ranks for tree layers
    
    # Build the random HT-tensor:
    rng, key = jax.random.split(rng)
    Y = tnv.rand(d, n, r, key)
    
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

  If all ranks are equal, we may set it is a number:

  .. code-block:: python

    d = 8   # Dimension of the tensor
    n = 10  # Mode size for the tensor
    r = 4   # Ranks for tree layers
    
    # Build the random HT-tensor:
    rng, key = jax.random.split(rng)
    Y = tnv.rand(d, n, r, key)
    
    # Print the resulting HT-tensor:
    tnv.show(Y)                   

    # >>> ----------------------------------------
    # >>> Output:

    # HT-tensor  | d=  8
    # Level:   1 | Shape: (8, 10, 4)
    # Level:   2 | Shape: (4, 4, 4, 4)
    # Level:   3 | Shape: (2, 4, 4, 4)
    # Level:   4 | Shape: (4, 4)
    # 

  We may also use custom limits for the uniform destribution:

  .. code-block:: python

    a = 0.99  # Minimum value
    b = 1.    # Maximum value
    
    # Build the random HT-tensor:
    rng, key = jax.random.split(rng)
    Y = tnv.rand(d, n, r, key, a, b)
    
    # Print the first HT-core of first level (leafs):
    print(Y[0][0])                      

    # >>> ----------------------------------------
    # >>> Output:

    # [[0.99490545 0.99509254 0.99896936 0.9990159 ]
    #  [0.99463993 0.99899983 0.99047322 0.99053174]
    #  [0.99541567 0.99509018 0.99153423 0.99793196]
    #  [0.99355632 0.99384485 0.9993778  0.99203122]
    #  [0.99295211 0.99767601 0.990772   0.99722857]
    #  [0.99141873 0.99057659 0.99087454 0.99856481]
    #  [0.99690017 0.99883118 0.99864542 0.99090145]
    #  [0.99295839 0.99265914 0.99846659 0.99330255]
    #  [0.99348785 0.9949604  0.99655398 0.99307878]
    #  [0.99094869 0.99198819 0.99494987 0.9941452 ]]
    # 




|
|

