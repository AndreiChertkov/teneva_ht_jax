Module gd: gradient descent for HT-tensors
------------------------------------------


.. automodule:: teneva_ht_jax.gd


-----




|
|

.. autofunction:: teneva_ht_jax.gd.gd_appr

  **Examples**:

  First, we set the shape and ranks of the tensor:

  .. code-block:: python

    d = 8          # Dimension of the tensor
    n = 10         # Mode size for the tensor
    r = [7, 8, 9]  # Ranks for tree layers

  We set the target (discretized) function, for which we will try to build the HT-approximation:

  .. code-block:: python

    def func_build(d, n):
        """Ackley function. See https://www.sfu.ca/~ssurjano/ackley.html."""
    
        a = -32.768         # Grid lower bound
        b = +32.768         # Grid upper bound
    
        par_a = 20.         # Standard parameter values for Ackley function
        par_b = 0.2
        par_c = 2.*jnp.pi
    
        def func(I):
            """Target function: y=f(I); [samples,d] -> [samples]."""
            X = I / (n - 1) * (b - a) + a
    
            y1 = jnp.sqrt(jnp.sum(X**2, axis=1) / d)
            y1 = - par_a * jnp.exp(-par_b * y1)
    
            y2 = jnp.sum(jnp.cos(par_c * X), axis=1)
            y2 = - jnp.exp(y2 / d)
    
            y3 = par_a + jnp.exp(1.)
    
            return y1 + y2 + y3
    
        return func
    
    func = func_build(d, n)

  Then we generate train and validation data:

  .. code-block:: python

    m_trn = 1.E+5 # Number of train items
    
    rng, key = jax.random.split(rng)
    I_trn = jax.random.choice(key, np.arange(n), (int(m_trn), d), replace=True)
    y_trn = func(I_trn)

  .. code-block:: python

    m_vld = 1.E+3 # Number of validation items
    
    rng, key = jax.random.split(rng)
    I_vld = jax.random.choice(key, np.arange(n), (int(m_trn), d), replace=True)
    y_vld = func(I_vld)

  Let build the random HT-tensor (initial approximation):

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = tnv.rand(d, n, r, key)

  Next we set the parameters of the gradient descent method:

  .. code-block:: python

    epochs = 30     # Number of train epochs
    batch  = 100    # Size of the batch for train
    lr     = 1.E-2  # Learning rate for training

  And now we can construct the HT-approximation using gradient descent:

  .. code-block:: python

    rng, key = jax.random.split(rng)
    Y = tnv.gd_appr(Y, I_trn, y_trn, I_vld, y_vld, epochs, batch, lr, key, log=True)

    # >>> ----------------------------------------
    # >>> Output:

    # #   1 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    2.35
    # #   2 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    3.24
    # #   3 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    4.14
    # #   4 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    5.06
    # #   5 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    5.97
    # #   6 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    6.82
    # #   7 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    7.65
    # #   8 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    8.49
    # #   9 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :    9.32
    # #  10 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :   10.21
    # #  11 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :   11.07
    # #  12 | e_trn : 1.0e+00 | e_vld : 1.0e+00 | t :   11.98
    # #  13 | e_trn : 2.1e-02 | e_vld : 2.1e-02 | t :   12.88
    # #  14 | e_trn : 2.8e-02 | e_vld : 2.8e-02 | t :   13.82
    # #  15 | e_trn : 1.4e-02 | e_vld : 1.4e-02 | t :   14.86
    # #  16 | e_trn : 1.6e-02 | e_vld : 1.6e-02 | t :   15.74
    # #  17 | e_trn : 2.9e-02 | e_vld : 2.9e-02 | t :   16.70
    # #  18 | e_trn : 2.7e-02 | e_vld : 2.7e-02 | t :   17.59
    # #  19 | e_trn : 1.3e-02 | e_vld : 1.3e-02 | t :   18.48
    # #  20 | e_trn : 1.4e-02 | e_vld : 1.4e-02 | t :   19.40
    # #  21 | e_trn : 3.1e-02 | e_vld : 3.1e-02 | t :   20.28
    # #  22 | e_trn : 1.0e-02 | e_vld : 1.0e-02 | t :   21.16
    # #  23 | e_trn : 2.3e-02 | e_vld : 2.3e-02 | t :   22.04
    # #  24 | e_trn : 3.1e-02 | e_vld : 3.1e-02 | t :   22.95
    # #  25 | e_trn : 1.1e-02 | e_vld : 1.1e-02 | t :   23.84
    # #  26 | e_trn : 1.5e-02 | e_vld : 1.5e-02 | t :   24.75
    # #  27 | e_trn : 2.5e-02 | e_vld : 2.5e-02 | t :   25.66
    # #  28 | e_trn : 1.2e-02 | e_vld : 1.2e-02 | t :   26.63
    # #  29 | e_trn : 1.3e-02 | e_vld : 1.3e-02 | t :   27.57
    # #  30 | e_trn : 1.4e-02 | e_vld : 1.4e-02 | t :   28.49
    # 

  Let select some tensor element and compute the value from the constracted by GD approximation:

  .. code-block:: python

    k = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    y = tnv.get(Y, k)
    print(y)

    # >>> ----------------------------------------
    # >>> Output:

    # 20.84243
    # 

  We can compute the same element from the target function:

  .. code-block:: python

    y_real = func(k.reshape(1, -1))[0]
    print(y_real)

    # >>> ----------------------------------------
    # >>> Output:

    # 21.170488
    # 

  .. code-block:: python

    # Let compare approximated and exact values:
    e = np.abs(y - y_real)
    print(f'Error : {e:7.1e}')

    # >>> ----------------------------------------
    # >>> Output:

    # Error : 3.3e-01
    # 




|
|

