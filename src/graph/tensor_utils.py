from typing import Optional, Union
import jax.numpy as jnp
from jax import Array


def _find_rank(
    lmbd: Array, accuracy: Optional[Union[float, Array]], rank: Optional[int]
) -> Array:
    if accuracy is None:
        if rank is None:
            return jnp.array(lmbd.shape[0])
        return jnp.array(rank)
    return lmbd.shape[0] - (jnp.sqrt(jnp.cumsum(lmbd[::-1] ** 2)) < accuracy).sum()
