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
    
    c1 = .1

    grad = gradient(f, xi)
    grad_norm = np.linalg.norm(grad)
    
    while grad_norm > tol and k < maxiter:
        p = -grad
        a = 1

        x_i_unmodified = xi.copy()
        g0 = f(x_i_unmodified)
        slope = c1*(grad.T@p)

        xi = xi + a*p
        ga = f(xi)
        
        ka_max = 10
        ka = 0
        while ga > g0 + a*slope and ka < ka_max:
            a = a/2
            xi = x_i_unmodified + a*p
            ga = f(xi)
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

# Function that implements Newton's descent method that works on hessian with negative eigenvalues
def newton_descent_unrestricted(f, xi, maxiter = 50):
    tol = 10 ** -5
    k = 0   # Iteration counter
    
    c1 = 0.1

    grad = gradient(f, xi)
    grad_norm = np.linalg.norm(grad)
    
    while grad_norm > tol and k < maxiter:
        B = hessian(f, xi)
        v = np.linalg.eigvals(B)
        v_min = min(v)
        if v_min < 0:
            mu = -(v_min - 1)
            B = B + mu*np.eye(len(xi))
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

def get_positive_root(p_cauchy, p_newton, delta):
    p_aux = p_newton - p_cauchy
    A_coef=  np.dot(p_aux, p_aux)
    B_coef = 2*np.dot(p_aux, p_cauchy)
    C_coef = np.dot(p_cauchy, p_cauchy) - delta

    Disc = B_coef**(2)-4*A_coef*C_coef
    Disc = np.sqrt(Disc)
    t_1 = (-B_coef-Disc)/(2*A_coef)
    t_2 = (-B_coef+Disc)/(2*A_coef)

    if t_1 > 0:
        return t_1
    elif t_2 > 0:
        return t_2
    else:
        raise('Upsis')

# Doblez para región de confianza
def doblez_rc_(B,g,delta):
    v = B*g
    p_cauchy = -(np.dot(g,g)/np.dot(g,v))*g
    p_newton = np.linalg.solve(B, -g)

    norma_cauchy = np.linalg.norm(p_cauchy)
    norma_newton = np.linalg.norm(p_newton)

    if norma_newton <= delta:
        ps = p_newton
    else:
        if norma_cauchy >= delta:
            ps = delta * p_cauchy/norma_cauchy
        else:
            t = get_positive_root(p_cauchy, p_newton, delta)
            ps = p_cauchy+ t*(p_newton - p_cauchy)
            pass

# Doblez para región de confianza
def doblez_rc(B,g,delta):
    v = B*g
    p_cauchy = -(np.dot(g,g)/np.dot(g,v))*g
    p_newton = np.linalg.solve(B, -g)

    norma_cauchy = np.linalg.norm(p_cauchy)
    norma_newton = np.linalg.norm(p_newton)

    if norma_newton <= delta:
        ps = p_newton
        print('Dirección de Newton', end='\t')
    else:
        if norma_cauchy >= delta:
            ps = delta * p_cauchy/norma_cauchy
            print('Punto de Cauchy', end='\t')
        else:
            p_aux = p_newton - p_cauchy
            A_coef=  np.dot(p_aux, p_aux)
            B_coef = 2*np.dot(p_aux, p_cauchy)
            C_coef = np.dot(p_cauchy, p_cauchy) - delta
            t_sol = np.roots([A_coef, B_coef, C_coef])
            ts = np.amax(t_sol)
            ps = p_cauchy+ ts*p_aux
            print('Dirección de Cauchy', end='\t')

    return ps

def regioncon(f, x, maxiter=200, 
              tol=10**-4, delta_min = 10**-5, delta_max=10):
    g = gradient(f, x)
    k = 0

    while np.linalg.norm(g) > tol and k < maxiter:
        pk = 0

        k += 1

    return xf, k

# (Gradiente conjugado funcionando)
# Código del gradiente conjugado
def mi_gc2(A, b):
    tol = 10**-3
    maxiterk = A.shape[1]

    k = 0
    x = np.zeros(A.shape[1])
    r = A@x - b
    p = -r

    norma = np.linalg.norm(r)
    while norma > tol and k < maxiterk:
        alpha = (np.dot(-r,p))/((p.T@A)@p)
        x = x + np.dot(alpha, p)
        r = A@x - b
        beta = ((r.T@A)@p)/((p.T@A)@p)
        p = -r + np.dot(beta, p)
        norma = np.linalg.norm(r)
        k += 1

    return x, k

# Doblez modificado
def doblez_rc_modificado(B,g,delta):
    v = B*g
    p_cauchy = -(np.dot(g,g)/np.dot(g,v))*g
    p_newton = mi_gc2(B, -g)

    norma_cauchy = np.linalg.norm(p_cauchy)
    norma_newton = np.linalg.norm(p_newton)

    if norma_newton <= delta:
        ps = p_newton
    else:
        if norma_cauchy >= delta:
            ps = delta * p_cauchy/norma_cauchy
        else:
            t = get_positive_root(p_cauchy, p_newton, delta)
            ps = p_cauchy+ t*(p_newton - p_cauchy)
            pass

def regionconf_modificado(f, x, maxiter=200, 
              tol=10**-4, delta_min = 10**-5, delta_max=10):
    g = gradient(f, x)
    k = 0

    while np.linalg.norm(g) > tol and k < maxiter:
        pk = 0

        k += 1

    return xf, k