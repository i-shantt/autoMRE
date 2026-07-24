from b import call_b


def call_a(n):
    return call_b(n) + 1


def unused_in_a():
    return "never used"
