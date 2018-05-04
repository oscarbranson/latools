from functools import wraps

# Logging Function
def _log(func):
    """
    Function for logging method calls and parameters
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        a = func(self, *args, **kwargs)
        self.log.append(func.__name__ + ' :: args={} kwargs={}'.format(args, kwargs))
        return a
    return wrapper