from typing import Dict
import jax.numpy as jnp
from jax import Array
from ..graph import MessageID

"""Computes averaged Frobenius distance between two dicts of messages.
Args:
    lhs: fist dict of messages;
    rhs: second dict of messages.
Returns:
    averaged Frobenius distance."""


def messages_frob_distance(
    lhs: Dict[MessageID, Array], rhs: Dict[MessageID, Array]
) -> Array:
    messages_number = len(lhs)
    assert messages_number == len(rhs), f"{messages_number}, {len(rhs)}"
    dist = jnp.array(0.0)
    for key, lhs_arr in lhs.items():
        dist += jnp.linalg.norm(lhs_arr - rhs[key])
    return dist / messages_number
