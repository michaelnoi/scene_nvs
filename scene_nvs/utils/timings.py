import time
from functools import wraps
from typing import Callable

from scene_nvs.utils.distributed import rank_zero_print


def log_time(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()
        print(f"Function {f.__name__} took {(end_time - start_time) * 1000:.2f} ms.")
        return result

    return wrap


def rank_zero_print_log_time(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()
        rank_zero_print(
            f"Function {f.__name__} took {(end_time - start_time) * 1000:.2f} ms."
        )
        return result

    return wrap
