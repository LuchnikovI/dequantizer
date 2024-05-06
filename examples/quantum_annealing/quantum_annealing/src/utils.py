import functools
from typing import Callable, Any


def none_wrap(callable: Callable[[Any], Any]) -> Callable:
    @functools.wraps(callable)
    def wrapped_callable(x):
        if x is None:
            return None
        else:
            return callable(x)

    return wrapped_callable
