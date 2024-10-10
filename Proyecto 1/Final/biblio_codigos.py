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
    maxkiter = 50
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
    maxkiter = 200
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

