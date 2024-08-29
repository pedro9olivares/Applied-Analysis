import numpy as np
from derivatives import gradient, hessian

# Function that implements maximum descent (naively)
def naive_max_descent(f, xi, maxiter = 50):
    tol = 10 ** -5
    k = 0   # Iteration counter

    grad = gradient(f, xi)
    grad_norm = np.linalg.norm(grad)
    
    while grad_norm > tol and k < maxiter:
        p = -(grad)
        a = 1/2 # p / a, a in (0,1], i.e., recorte de paso

        xi = xi + a*p
        grad = gradient(f, xi)
        grad_norm = np.linalg.norm(grad)

        k += 1

    return xi, k

# Function that implements maximum descent
def max_descent(f, xi, maxiter = 200):
    tol = 10 ** -5
    k = 0   # Iteration counter
    
    c1 = 0.1

    grad = gradient(f, xi)
    grad_norm = np.linalg.norm(grad)
    
    while grad_norm > tol and k < maxiter:
        p = -grad
        a = 1

        fx = f(xi)
        slope = c1*(grad.T@p)

        xi = xi + a*p
        fxi = f(xi)
        ka_max = 10
        ka = 0
        while fxi > fx + a*slope and ka < ka_max:
            a = a/2
            xi = xi + a*p
            fxi = f(xi)
            ka += 1
            
        grad = gradient(f, xi)
        grad_norm = np.linalg.norm(grad)

        k += 1

    return xi, k

# Function that implements Newton's descent method
# OJO: it is not always the case that the hessian is symmetric and positive definite
# Observation: Newton metho converges quadratically (the error decreases quadratically)
def newton_descent(f, xi, maxiter = 50):
    tol = 10 ** -5
    k = 0   # Iteration counter
    
    c1 = 0.1

    grad = gradient(f, xi)
    grad_norm = np.linalg.norm(grad)
    
    while grad_norm > tol and k < maxiter:
        B = hessian(f, xi)
        p = np.linalg.solve(B, -grad) # Newtons system
        a = 1

        fx = f(xi)
        slope = c1*(np.dot(grad, p))

        xi = xi + a*p
        fxi = f(xi)
        ka_max = 10
        ka = 0
        while fxi > fx + a*slope and ka < ka_max:
            a = a/2
            xi = xi + a*p
            fxi = f(xi)
            ka += 1
            
        # xi = xi + a*p
        grad = gradient(f, xi)
        grad_norm = np.linalg.norm(grad)

        k += 1

    return xi, k