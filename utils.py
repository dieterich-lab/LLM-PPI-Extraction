import signal


class Timeout:
    """Timeout class using ALARM signal"""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


def attempt(tries, seconds, func, args=[], kwargs={}):
    c = 0
    res = None
    while c < tries:
        try:
            with Timeout(seconds):
                res = func(*args, **kwargs)
                break
        except Timeout.Timeout:
            print("Timeout")
            c += 1
    return res
