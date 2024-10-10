#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 07:21:25 2024

@authors: Erick Martinez Hernandez, 191821
          Sara Visoso Gunther, 196079
          Pedro Olivares Sanchez, 190198
"""

import numpy as np
import pandas as pd

#Importamos y arreglamos los datos para poder usarlos

Datos_f = pd.read_csv("population-and-demography.csv") 
Datos_f = Datos_f.loc[Datos_f['Entity'] == 'China'] #Solo nos importa China
Datos_f.drop(['Entity', 'Code'], axis=1, inplace=True) #Eliminamos columnas inecesarias
Datos_f.rename(columns={"Population - Sex: all - Age: all - Variant: estimates": "Population"}, inplace = True) #Renombramos columna
Datos_f["Population"] = Datos_f["Population"]/(10**6) #Escalamos los datos
Datos_f["Year"] = Datos_f["Year"]-1950.0 #Restamos 1950 para que el primer año sea el año 0

p0 = Datos_f.iloc[0][1] #P0, la poblacion inicial


#Ecuacion diferencial

def funcion_P(t,k,r):
    return k/(1+((k/p0)-1)*np.e**(-r*t))

#Modelo logistico

def funcion(x):
    fx = 0
    for i in range(len(Datos_f)):
        fx = fx + (funcion_P(Datos_f.iloc[i][0], x[1], x[0]) - Datos_f.iloc[i][1])**2
    return fx/2

#Importamos y arreglamos otra vez los datos pero ahora solo para numeros pares

Datos_f_Pares = pd.read_csv("population-and-demography.csv")
Datos_f_Pares = Datos_f_Pares.loc[Datos_f_Pares['Entity'] == 'China'] #Solo nos importa China
Datos_f_Pares.drop(['Entity', 'Code'], axis=1, inplace=True) #Eliminamos columnas inecesarias
Datos_f_Pares.rename(columns={"Population - Sex: all - Age: all - Variant: estimates": "Population"}, inplace = True) #Renombramos columna
Datos_f_Pares["Population"] = Datos_f_Pares["Population"]/(10**6) #Escalamos los datos
Datos_f_Pares["Year"] = Datos_f_Pares["Year"]-1950.0 #Restamos 1950 para que el primer año sea el año 0
Datos_f_Pares = Datos_f_Pares[Datos_f_Pares["Year"] % 2 == 0] #Filtramos y nos quedamos solo con años pares


#Ecuacion diferencial

def funcion_P_Pares(t,k,r):
    return k/(1+((k/p0)-1)*np.e**(-r*t))

#Modelo logistico (solo años pares)

def funcion_Pares(x):
    fx = 0
    for i in range(len(Datos_f_Pares)):
        fx = fx + (funcion_P(Datos_f_Pares.iloc[i][0], x[1], x[0]) - Datos_f_Pares.iloc[i][1])**2
    return fx/2