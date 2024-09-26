#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:06:47 2024

@author: erick
"""

import numpy as np
import matplotlib.pyplot as plt
from biblio_codigos import descenso_max,  descenso_newton
from biblio_derivadas import Gradiente_cen
import pandas as pd


Datos_f = pd.read_csv("/Users/erick/11. Analisis Aplicado/population-and-demography.csv")

Datos_f = Datos_f.loc[Datos_f['Entity'] == 'China']

Datos_f.drop(['Entity', 'Code'], axis=1, inplace=True)

Datos_f.rename(columns={"Population - Sex: all - Age: all - Variant: estimates": "Population"}, inplace = True)

Datos_f["Population"] = Datos_f["Population"]/(10**6)

Datos_f["Year"] = Datos_f["Year"]-1950.0

p0 = Datos_f.iloc[0][1]

def funcion_P(t,k,r):
    return k/(1+(k/p0-1)*np.e**(-r*t))

def funcion(x):
    fx = 0
    for i in range(len(Datos_f)):
        fx = fx + (funcion_P(Datos_f.iloc[i][0], x[1], x[0]) - Datos_f.iloc[i][1])**2
    return fx/2

#Descenso Newton
    
x_n = np.array([1.7, 1000])

[x_new, kiter] = descenso_newton(funcion, x_n)

print("Kiter Newton = ", kiter)
print("Solucion Newton: ") 
for i in range(len(x_new)):
    print("x[", i ,"]=" , x_new[i])

print("Gradiente Newton: ", np.linalg.norm(Gradiente_cen(funcion, x_new)))
print("================================")
    
#Descenso Max

[x_max, kiter] = descenso_max(funcion, x_n)

print("Kiter Max = ", kiter)
print("Solucion Max: ")
for i in range(len(x_max)):
    print("x[", i ,"]=" , x_max[i])
    
print("Gradiente Max: ", np.linalg.norm(Gradiente_cen(funcion, x_max)))