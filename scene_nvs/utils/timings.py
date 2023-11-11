import time
from functools import wraps
from typing import Callable


def log_time(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time.time()
        result = f(*args, **kw)
        end_time = time.time()
        print(f"Function {f.__name__} took {(end_time - start_time) * 1000:.2f} ms.")
        return result

    return wrap
