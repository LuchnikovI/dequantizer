from typing import Callable
from jax import Array
from jax.random import normal
from .edge import Edge
from .element import MessageDir

"""Returns an initializer that initializes a message given the message direction.
Args:
    message_direction: direction of the message.
Returns:
    Initializer."""


def get_message_random_nonnegative_initializer(
    key: Array,
) -> Callable[[MessageDir], Array]:
    def initializer(direction: MessageDir) -> Array:
        if isinstance(direction[0], Edge):
            d = direction[0].dimension
        else:
            d = direction[1].dimension
        message_sq = normal(key, (d, d, 2))
        message_sq = message_sq[..., 0] + 1j * message_sq[..., 1]
        return message_sq @ message_sq.conj().T

    return initializer
