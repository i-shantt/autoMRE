from c import call_c


def call_b(n):
    return call_c(n) * 2


def unused_in_b():
    pass
