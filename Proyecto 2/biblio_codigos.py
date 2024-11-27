# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:10:25 2024

@author: Análisis Aplicado
            ITAM
            22 de agosto de 2024
            
     Biblioteca de códigos de optimización. 
   Máximo descenso: descenso_max.py
   Descenso de Newton: descenso_newton.py      
          
"""
#---------------------------------------------------

def descenso_max(fun,x):
    #
    #
    #
    import numpy as np
    from biblio_derivadas import Gradiente_cen
    #------------------------
    # Parámetros del código
    tol = 10**(-5)          # tolerancia  al norma del gradiente
    maxkiter = 100
    kiter = 0
    c1 = 0.1
    grad_x = Gradiente_cen(fun,x)
    norma_grad = np.linalg.norm(grad_x)
    
    while(norma_grad > tol and kiter < maxkiter):
        p = -grad_x
        # recortar el vector p / alfa en( 0, 1]
        alfa = 1.0
        fx = fun(x)
        x_trial = x + alfa*p
        fx_trial = fun(x_trial)
        pend = c1*np.inner(grad_x,p)
        jmax = 10
        j = 0
        while(fx_trial > fx + alfa*pend and j < jmax):
            alfa = alfa/2
            x_trial = x + alfa*p
            fx_trial = fun(x_trial)
            j = j + 1
        #-----------Fin del while de búsqueda de línea--------------------
        x = x + alfa*p
        grad_x = Gradiente_cen(fun,x)
        norma_grad = np.linalg.norm(grad_x)
        kiter = kiter + 1
        #print(kiter, norma_grad)
        
    # -------------Fin del while principal---------------   
    return x, kiter
#----------------------------------------------------------------

def descenso_newton(fun,x):
    #
    #
    #
    import numpy as np
    from biblio_derivadas import Gradiente_cen, hessiana
    #------------------------
    # Parámetros del código
    tol = 10**(-5)          # tolerancia  al norma del gradiente
    maxkiter = 100
    kiter = 0
    c1 = 0.1
    grad_x = Gradiente_cen(fun,x)
    norma_grad = np.linalg.norm(grad_x)
    
    while(norma_grad > tol and kiter < maxkiter):
        B = hessiana(fun,x)
        vp = np.linalg.eigvalsh(B)
        vp_min = np.min(vp)
        if(vp_min <= 0):
            B = B +(np.abs(vp_min) +1)*np.eye(len(x))
        
        
        p = np.linalg.solve(B, -grad_x)
        # recortar el vector p / alfa en( 0, 1]
        alfa = 1.0
        fx = fun(x)
        x_trial = x + alfa*p
        fx_trial = fun(x_trial)
        pend = c1*np.inner(grad_x,p)
        jmax = 10
        j = 0
        while(fx_trial > (fx + alfa*pend) and j < jmax):
            alfa = alfa/2
            x_trial = x + alfa*p
            fx_trial = fun(x_trial)
            j = j + 1
        #-----------Fin del while de búsqueda de línea--------------------
        x = x + alfa*p
        grad_x = Gradiente_cen(fun,x)
        norma_grad = np.linalg.norm(grad_x)
        kiter = kiter + 1
        #print(kiter, norma_grad)
        
    # -------------Fin del while principal---------------   
    return x, kiter
#----------------------------------------------------------------

#-----------------------------------------------------------------
# código del doblez para región de confianza
def doblez_rc(B,g,Delta):
    # B es simétrica y psositiva definida en R^n
    # g es un vector diferente de cero
    # Delta es un número real positivo y cercano a cero
    import numpy as np
    v = B*g
    p_cauchy = -( np.dot(g,g)/ np.dot(g, v))*g
    p_newton = np.linalg.solve(B,-g)
    
    norma_cauchy = np.linalg.norm(p_cauchy)
    norma_newton = np.linalg.norm(p_newton)
    if(norma_newton <= Delta):
        ps = p_newton
        #print(" Dirección de Newton")
    else:
        if (norma_cauchy >= Delta):
            ps = (Delta/norma_cauchy )*p_cauchy
            #print("Punto de Cauchy")
        else:
            p_aux = p_newton-p_cauchy
            A_coef = np.dot(p_aux, p_aux)
            B_coef = 2*np.dot(p_aux, p_cauchy)
            C_coef = np.dot(p_cauchy, p_cauchy)-Delta
            t_sol = np.roots([A_coef, B_coef, C_coef])
            ts = np.amax(t_sol) 
            ps = p_cauchy + ts*p_aux
            #print(" doblez")
                
    return ps            
 #-----------------Fin de doblez --------------------------     


def regioncon(f, x):

    # M etodo de regi ́on de confianza para f : Rn → R,
    # donde x es el punto inicial.
    # return: xf aproximacio ́n la mınimo local y
    # k el numero de iteraciones.
    # Parametros:
    # maxiter = 200, numero maximo de iteraciones
    # tol = 10−4, tolerancia a la norma del gradiente #∆min=10−5, ∆max=10
    import numpy as np
    from biblio_derivadas import Gradiente_cen, hessiana
    
    tol = 10**-4
    maxiter = 100
    deltaMin = 10**-5
    deltaMax = 10
    delta = 1
    k = 0
    phi = 1/8
    
    g = Gradiente_cen(f, x)
    B = hessiana(f, x)
    
    while (np.linalg.norm(g) > tol and k < maxiter):
        
        pk = doblez_rc(B, g, delta)
    
        mc = (1/2)*((pk.T@B)@pk) + (g.T@pk) + f(x)
        
        rhok = (f(x) - f(x+pk)) / (f(x) - mc)
    
        if rhok < (1/4):
            delta = max(deltaMin, (1/4) * np.linalg.norm(pk))
            
        elif (rhok > (3/4) and np.linalg.norm(pk) == delta):
            delta = min(2*delta, deltaMax)
            
        if rhok > phi:
            x = x + pk
        
        g = Gradiente_cen(f, x)
        B = hessiana(f, x)
        k = k + 1
        
        """
        print("x1 = ", x[0])
        print("x2 = ", x[1])
        print(k)
        """
        
    return x, k

def mi_gc(A, b):
# Metodo de gradiente conjugado para resolver el sistema lineal
# Ax=b donde A es nxn simetrica y positiva definida
# Return: x solucion al sistema lineal
# k numero de iteraciones
# tolerancia para la norma del residual es tol = 10 ∗ ∗(−8) 
# El punto inicial es x = np.zeros(len(b)).
    import numpy as np
    x = np.zeros(len(b))
    tol = 10**(-8) 
    r = A@x - b
    p = -r
    k = 0
    while(np.linalg.norm(r) > tol):
        # alpha = -( (r.T@p) / (p.T@A@p))
        alpha = -( (np.dot(r, p)) / np.dot(np.dot(p,A),p))
        x = x + alpha*p
        r = A@x - b
        beta = (r.T@A@p) / (p.T@A@p)
        p = -r + beta*p 
        k = k + 1
        print(k,np.linalg.norm(r))
    return x, k

def mi_gc2(A, b):
# Metodo de gradiente conjugado para resolver el sistema lineal
# Ax=b donde A es nxn simetrica y positiva definida
# Return: x solucion al sistema lineal
# k numero de iteraciones
# tolerancia para la norma del residual es tol = 10 ∗ ∗(−8) 
# El punto inicial es x = np.zeros(len(b)).
    import numpy as np    
    x = np.zeros(len(b))
    tol = 10**(-8) 
    r = A@x - b
    p = -r
    k = 0
    while(np.linalg.norm(r) > tol):
        # alpha = -( (r.T@p) / (p.T@A@p))
        alpha = (np.dot(r, r)) / np.dot(np.dot(p,A),p)
        x = x + alpha*p
        rPrevio = r
        r = A@x - b
        # beta = (r.T@r) / (rPrevio.T@rPrevio)
        beta = (np.dot(r, r)) / np.dot(rPrevio, rPrevio)
        p = -r + beta*p 
        k = k + 1
        #print(k,np.linalg.norm(r))
    return x, k

def jacobiana(F,x):
    import numpy as np
    import copy
    step = 10**(-5)
    n = len(x)
    A = np.zeros((n,n))
    for i in range(n):
        x_1 = copy.copy(x)
        x_2 = copy.copy(x)
        x_1[i] = x_1[i]+step
        x_2[i] = x_2[i]-step
        F_x1 = F(x_1)
        F_x2 = F(x_2)
        A[0:n,i] = F_x1-F_x2
    A = A/(2*step)
    return A

def met_newton(F,x):
    # Metodo de Newton para aproximar un cero de
    # F:R^n --> R^n continuamente diferenciable
    import numpy as np
    tol = 10**(-5)
    maxk = 50
    k = 0
    Fx = F(x)
    Fx_norma = np.linalg.norm(Fx)
    while(Fx_norma > tol and k < maxk):
        Jx = jacobiana(F,x)
        s = np.linalg.solve(Jx, -Fx)
        x = x + s
        k = k + 1
        Fx = F(x)
        Fx_norma = np.linalg.norm(Fx)
        print(k,Fx_norma)
    return x, k 


def met_broyden(F,x):
    # Metodo de Broyden para aproximar un cero de
    # F:R^n --> R^n continuamente diferenciable
    import numpy as np
    #-----Parametros Iniciales------
    tol = 10**(-5)
    maxk = 200
    k = 0
    #-----Valores Iniciales------
    #n = len(x)
    #J = np.eye(n)
    J = jacobiana(F,x)
    Fx = F(x)
    Fx_norma = np.linalg.norm(Fx)
    while(Fx_norma > tol and k < maxk):
        s = np.linalg.solve(J, -Fx)
        x = x + s
        #-------Actualizacion de Broyden----------
        Fx1 = F(x)
        y = Fx1 - Fx
        w = y - np.dot(J,s)
        #J = J + ((y-(J@s))@s.T)/(s.T@s)
        J = J + np.outer(w,s)/np.inner(s,s)
        k = k + 1
        Fx = Fx1
        Fx_norma = np.linalg.norm(Fx)
        print(k,Fx_norma)
    return x, k

def descenso_bfgs(fun,x):
    # Método de cuasi-Newton con actualización BFGS
    #
    #
    import numpy as np
    from biblio_derivadas import Gradiente_cen
    from biblio_derivadas import hessiana
    #------------------------
    # Parámetros del código
    tol = 10**(-5)          # tolerancia  al norma del gradiente
    maxkiter = 12
    kiter = 0
    c1 = 0.1
    #-------------------------------------------
    n = len(x)
    B = np.eye(n)
    #B = hessiana(fun, x)
    grad_x = Gradiente_cen(fun,x)
    norma_grad = np.linalg.norm(grad_x)
    
    while(norma_grad > tol and kiter < maxkiter):  
        (p,jk) = mi_gc2(B, -grad_x)
        # recortar el vector p / alfa en( 0, 1]
        alfa = 1.0
        fx = fun(x)
        x_trial = x + alfa*p
        fx_trial = fun(x_trial)
        pend = c1*np.inner(grad_x,p)
        jmax = 10
        j = 0
        while(fx_trial > (fx + alfa*pend) and j < jmax):
            alfa = alfa/2
            x_trial = x + alfa*p
            fx_trial = fun(x_trial)
            j = j + 1
        #-----------Fin del while de búsqueda de línea--------------------
        #  -------------Actaulización BFGS----------------
        s = alfa*p
        x = x + s
        grad_x1 = Gradiente_cen(fun,x)
        y = grad_x1-grad_x
        w = B@s
        B = B + np.outer(y,y)/np.inner(s,y) - np.outer(w,w)/np.inner(s,w)
        #-------------------------------------------------------
        grad_x = grad_x1
        norma_grad = np.linalg.norm(grad_x)
        kiter = kiter + 1
        #print(kiter, norma_grad)
        
    # -------------Fin del while principal---------------   
    return x, kiter
#----------------------------------------------------------------