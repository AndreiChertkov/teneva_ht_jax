"""teneva_ht_jax.gd: build HT-tensor with gradient descent.

This module contains the function "gd_appr", which computes the HT-tensor by
gradient descent method using the provided train dataset.

"""
import jax
import jax.numpy as jnp
import optax
import teneva_ht_jax as tnv
from time import perf_counter as tpc


def gd_appr(Y, I_trn, y_trn, I_vld=None, y_vld=None, epochs=100, batch=100,
            lr=1.E-4, seed=42, log=True, log_trn=True):
    """Build HT-tensor with gradient descent.

    Args:
        Y (list): HT-tensor, which is the initial approximation for algorithm.
        I_trn (np.ndarray): multi-indices for the tensor in the form of array
            of the shape [samples, d], where d is a number of tensor's
            dimensions and samples is a size of the train dataset.
        y_trn (np.ndarray): values of the tensor for multi-indices I_trn in
            the form of array of the shape [samples].
        I_vld (np.ndarray): optional multi-indices for items of validation
            dataset in the form of array of the shape [samples_vld, d], where
            samples_vld is a size of the validation dataset.
        y_vld (np.ndarray): optional values of the tensor for multi-indices
            I_vld of validation dataset in the form of array of the shape
            [samples_vld].
        epochs (int): number of train epochs.
        batch (int): size of the batch for train.
        lr (float): learning rate for training.
        key (jax.random.PRNGKey): jax random key.
        log (bool): if flag is set, then the information about the progress of
            the algorithm will be printed after each epoch.
        log_trn (bool): if flag is set (and log is also set), then the
            accuracy on the train dataset will be presented after each epoch
            (note that for large training datasets this can take a significant
            amount of time, and it may be better to turn off this flag).

    Returns:
        list: HT-tensor, which is the constructed approximation.

    """
    time = tpc()
    rng = jax.random.PRNGKey(seed)
    get = jax.jit(jax.vmap(tnv.get, (None, 0)))
    optim = optax.adam(lr)

    Y = [Yl.copy() for Yl in Y]
    state = optim.init(Y)

    @jax.jit
    def loss(Y, I, y_real):
        y = get(Y, I)
        # jax.debug.print('{v}', v=y)
        l = jnp.mean(jnp.linalg.norm(y_real - y))
        return l

    loss_grad = jax.jit(jax.grad(loss))

    @jax.jit
    def optimize(Y, state, I_cur, y_cur):
        grads = loss_grad(Y, I_cur, y_cur)
        updates, state = optim.update(grads, state)
        Y = jax.tree_util.tree_map(lambda u, y: y + u, updates, Y)
        return Y, state

    for epoch in range(epochs):
        rng, key = jax.random.split(rng)
        perm = jax.random.permutation(key, I_trn.shape[0])
        I_trn_cur = I_trn[perm]
        y_trn_cur = y_trn[perm]

        for j in range(len(I_trn_cur) // batch):
            Y, state = optimize(Y, state,
                I_trn_cur[j * batch:(j+1)*batch],
                y_trn_cur[j * batch:(j+1)*batch])

        if log:
            text = f'# {epoch+1:-3d} | '
            if log_trn:
                y_our = get(Y, I_trn)
                e_trn = jnp.linalg.norm(y_trn - y_our) / jnp.linalg.norm(y_trn)
                text += f'e_trn : {e_trn:-7.1e} | '
            if I_vld is not None and y_vld is not None:
                y_our = get(Y, I_vld)
                e_vld = jnp.linalg.norm(y_vld - y_our) / jnp.linalg.norm(y_vld)
                text += f'e_vld : {e_vld:-7.1e} | '
            text += f't : {tpc()-time:-9.2f}'
            print(text)

    return Y
