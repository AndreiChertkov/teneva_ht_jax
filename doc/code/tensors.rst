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
    
    # Print the first HT-core:
    print(Y[0][0])                      

    # >>> ----------------------------------------
    # >>> Output:

    # [[0.9963801  0.9902207  0.9933224  0.99755406]
    #  [0.9954925  0.999137   0.9916649  0.991549  ]
    #  [0.99372023 0.9969657  0.9979555  0.9964686 ]
    #  [0.99506474 0.99021447 0.99620223 0.9943191 ]
    #  [0.9987403  0.99158955 0.9911972  0.99271756]
    #  [0.99949336 0.99380195 0.99738765 0.9994141 ]
    #  [0.992728   0.9925216  0.99771726 0.99352854]
    #  [0.99336684 0.99970996 0.9904865  0.99945676]
    #  [0.9913098  0.9926554  0.99574786 0.99213517]
    #  [0.9915967  0.99672365 0.99495906 0.9987823 ]]
    # 




|
|

