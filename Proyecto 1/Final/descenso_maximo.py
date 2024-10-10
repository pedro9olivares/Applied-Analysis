#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:31:03 2024

@author: erick
"""

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
