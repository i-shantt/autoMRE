def call_c(n):
    # The bug: passing a list to a function that expects a string.
    return "abc" + [n]


def unused_in_c():
    return 0
