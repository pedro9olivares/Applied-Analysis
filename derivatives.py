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
    ek = np.zeros(n)
    ej = np.zeros(n)

    for k in range(n):
        ek[k] = 1
        for j in range(n):
            ej[j] = 1
            hess[k,j] = f(x0 + h*ek + h*ej) - f(x0 + h*ek) - f(x0 + h*ej) + f(x0)
            hess[k,j] = hess[k,j]/(h**2)
            ej[j] = 0
        ek[k] = 0

    return hess
