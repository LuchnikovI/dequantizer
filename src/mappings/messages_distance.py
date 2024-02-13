from typing import Dict
from jax import Array
from ..graph import MessageID

"""Computes averaged logarithmic Frobenius distance between two sets of messages.
Args:
    lhs: fist set of messages;
    rhs: second set of messages.
Returns:
    averaged logarithmic Frobenius distance."""


def messages_frob_distance(
    lhs: Dict[MessageID, Array], rhs: Dict[MessageID, Array]
) -> Array:
    raise NotImplementedError()
