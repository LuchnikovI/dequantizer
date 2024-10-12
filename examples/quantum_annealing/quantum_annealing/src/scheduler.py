from typing import Tuple, Union, Callable, Iterable
from jax import Array

"""Returns a scheduler given its parameters."""


def get_scheduler(
    total_time_step_size: Union[Array, float],
    schedule: Callable[[Union[Array, float]], Union[Array, float]],
    steps_number: int,
) -> Iterable[Tuple[Union[Array, float], Union[Array, float]]]:
    for i in range(steps_number):
        fraction = schedule(i / (steps_number - 1))
        yield (total_time_step_size * fraction, total_time_step_size * (1 - fraction))
