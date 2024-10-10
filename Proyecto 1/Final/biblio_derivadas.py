#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:20:17 2024

@author: erick
"""

import numpy as np
import copy

# Se aproxima el gradiente de fname en el vector a por medio de diferencias 
# hacia adelante en el vector a.
# In:
# fname.- cadena de caracteres con el nombre de la funcion.
# a.- vector columna de dimension n.
# Out:
# g.- vector columna de n componentes con la aproximacio ́n a las derivadas parciales.
# Internamente se usa h = 10−5.


def Gradiente(fname, x):
    n = len(x)
    grad_x = np.zeros(n)
    h = 10**(-5)
    fx = fname(x)
    for i in range(n):
        x_copy = copy.copy(x)
        x_copy[i] = x_copy[i] + h
        fx_copy = fname(x_copy)
        grad_x[i] = (fx_copy - fx)/h
    return grad_x


def Gradiente_cen(fname, x):
    n = len(x)
    grad_x = np.zeros(n)
    h = 10**(-5)
    for i in range(n):
        x_copy = copy.copy(x)
        x_copy2 = copy.copy(x)
        x_copy[i] = x_copy[i] + h
        x_copy2[i] = x_copy2[i] - h
        fx_copy = fname(x_copy)
        fx_copy2 = fname(x_copy2)
        grad_x[i] = (fx_copy - fx_copy2)/(2*h)
    return grad_x


def Hessiano(fname, x):
    n = len(x)
    Hess_x = np.zeros([n,n])
    h = 10**(-5)
    fx = fname(x)
    for i in range(n):
        for j in range(n):
            x_copy = copy.copy(x)
            x_copy2 = copy.copy(x)
            x_copy3 = copy.copy(x)
            
            x_copy[i] = x_copy[i] + h
            x_copy2[j] = x_copy2[j] + h
            x_copy3[i] = x_copy3[i] + h
            x_copy3[j] = x_copy3[j] + h
            
            
            fx_copy = fname(x_copy)
            fx_copy2 = fname(x_copy2)
            fx_copy3 = fname(x_copy3)
            
            
            Hess_x[i,j] = (fx_copy3 - fx_copy - fx_copy2 + fx)/(h*h)
    return Hess_x

def hessiana(fun,x):
#
# Aproximación a la matriz hessiana de la función
# fun:R^n --> R en el punto x

    h = 10**(-5)
    n = len(x)
    H = np.zeros((n,n))
    f_x = fun(x)
    for i in range(n):
        x_i = x.copy()
        x_i[i]= x_i[i] + h
        f_x_i = fun(x_i)
        for j in range(i+1):
            x_j = x.copy()
            x_ij = x_i.copy()
            x_j[j] =x_j[j] + h
            x_ij[j] =x_ij[j] + h
            f_x_j = fun(x_j)
            f_x_ij = fun(x_ij)
            H[i,j]= (f_x_ij - f_x_i -f_x_j + f_x)/(h**2)
            if(i!=j):
                H[j,i] = H[i,j]
    return H


def hessiana_sara(fun, x):
    n=len(x);
    hess= np.zeros((n,n));
    h=10**(-5)
    ek=np.zeros(n)
    ej=np.zeros(n)
    for k in range(n):
        ek[k]=1
        for j in range (n):
            ej[j]=1
            hess[k,j]=(fun(x+(h*ek)+(h*ej))-fun(x+h*ek)-fun(x+h*ej)+fun(x))/(h**2)
            ej[j]=0
        ek[k]=0   
    
    return hess