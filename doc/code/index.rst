Full docs and demo
==================

Below, we provide a brief description and demonstration of the capabilities of each function from the package. Most functions take "Y" - a list of the HT-cores for each level.

Please, note that all demos assume the following imports:

  .. code-block:: python

    from jax.config import config            # Optional
    config.update('jax_enable_x64', True)    # Optional
    import os                                # Optional
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Optional

    import jax
    import jax.numpy as jnp
    import numpy as np
    import teneva_ht_jax as tnv
    from time import perf_counter as tpc
    rng = jax.random.PRNGKey(42)

-----

.. toctree::
  :maxdepth: 4

  act_one
  gd
  tensors
  transformation
  vis
