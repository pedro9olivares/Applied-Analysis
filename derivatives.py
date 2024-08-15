import numpy as np
import copy

# Partial derivative of f with respect to x_i, evaluated on x0
def partial(f, x0, i, h):
    n = len(x0)
    ei = np.zeros(n)
    ei[i] = 1

    return (f(x0 + h*ei) - f(x0))/h

# Approximation of gradient via diferencia centrada
def gradient(f, x0):
    h = 10 ** -5 # Our bound is the cube root of machine's epsilon (which is 10 ** -16)
    n = len(x0)
    grad = np.zeros(n)

    for i in range(n):
        grad[i] = partial(f, x0, i, h)

    return grad
