import numpy as np

# Partial derivative of f with respect to x_i, evaluated on x0. 
# This one uses "Diferencia finita hacia adelante"
def partial_forward(f, x0, i, h):
    n = len(x0)
    ei = np.zeros(n)
    ei[i] = 1

    return (f(x0 + h*ei) - f(x0))/h

# Partial derivative of f with respect to x_i, evaluated on x0. 
# This one uses "Diferencia finita centrada"
def partial_centered(f, x0, i, h):
    n = len(x0)
    ei = np.zeros(n)
    ei[i] = 1

    return (f(x0 + h*ei) - f(x0 - h*ei))/(2*h)

# Approximation of gradient
def gradient(f, x0, method='centrada'):
    h = 10 ** -5 # Our bound is the cube root of machine's epsilon (which is 10 ** -16)
    n = len(x0)
    grad = np.zeros(n)

    for i in range(n):
        if method == 'adelante':
            grad[i] = partial_forward(f, x0, i, h)
        elif method == 'centrada':
            grad[i] = partial_centered(f, x0, i, h)

    return grad

# Approximation of hessian evaluated at x0
def hessian(f, x0):
    h = 10 ** -5
    n = len(x0)
    hess = np.zeros((n,n))

    for k in range(n):
        for j in range(n):
            hess[k,j] = f(x0 + h*e(k,n) + h*e(j,n)) - f(x0 + h*e(k,n)) - f(x0 + h*e(j,n)) + f(x0)
            hess[k,j] = hess[k,j]/(h**2)

    return hess

# Functions for getting the i-th standard basis vector of dimension n
def e(i, n):
    e_i = np.zeros(n)
    e_i[i] = 1
    return e_i
