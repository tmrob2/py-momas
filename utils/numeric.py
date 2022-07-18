from ast import arg


def mult(*args):
    result = 1.0
    for x in args:
        result *= x
    return x