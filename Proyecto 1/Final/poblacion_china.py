#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 07:21:28 2024

@authors: Erick Martinez Hernandez, 191821
          Sara Visoso Gunther, 196079
          Pedro Olivares Sanchez, 190198
"""

#Importamos todas las bibliotecas necesarias

import numpy as np
import matplotlib.pyplot as plt
from biblio_codigos import descenso_newton, descenso_max
from biblio_derivadas import Gradiente_cen
import pandas as pd
from funcionmincuadnl import funcion_P, funcion, funcion_P_Pares, funcion_Pares

#Importamos y arreglamos los datos para poder usarlos

Datos_f = pd.read_csv("population-and-demography.csv") 
Datos_f = Datos_f.loc[Datos_f['Entity'] == 'China'] #Solo nos importa China
Datos_f.drop(['Entity', 'Code'], axis=1, inplace=True) #Eliminamos columnas inecesarias
Datos_f.rename(columns={"Population - Sex: all - Age: all - Variant: estimates": "Population"}, inplace = True) #Renombramos columna
Datos_f["Population"] = Datos_f["Population"]/(10**6) #Escalamos los datos
Datos_f["Year"] = Datos_f["Year"]-1950.0 #Restamos 1950 para que el primer año sea el año 0

p0 = Datos_f.iloc[0][1] #P0, la poblacion inicial

#China parece tenener un crecimiento de el 5% y ponemos como tope 1400 millones

x_n = np.array([np.log(1.05), 1450.0]) #Valores iniciales

#Descenso Newton

[x_new, kiter] = descenso_newton(funcion, x_n) #Minimizamos la funcion con metodo de Newton

"""
print("Kiter Newton = ", kiter)
print("Solucion Newton: ") 
for i in range(len(x_new)):
    print("x[", i ,"]=" , x_new[i])

print("Gradiente Newton: ", np.linalg.norm(Gradiente_cen(funcion, x_new)))
print("================================")
"""

#Imprimimos resultados

print("================================================")
print("         Todos los años")
print("        Método de Newton")
print("================================================")
print("r = ", x_new[0])
print("K = ", x_new[1])
print("Gradiente Newton = ",np.linalg.norm(Gradiente_cen(funcion, x_new)) )
print("iter = ", kiter)
print("P(2024) = ", funcion_P(2024-1950, x_new[1], x_new[0]))
print("P(2030) = ", funcion_P(2030-1950, x_new[1], x_new[0]))

#Graficamos los datos poblacionales y las estimaciones de el modelo con los valores optimos de r y K

Datos_f.set_index("Year",drop=True,inplace=True)
Datos_f.plot(title="China Population")

m = 100

F = np.zeros([m,2])

for i in range(m):
    F[i,0] = int(i)
    F[i,1] = funcion_P(i,x_new[1],x_new[0])

years = F[0:m,0]
pob = F[0:m,1]

plt.plot(years, pob)
plt.scatter(years, pob, marker="+", color = "g")
plt.xlabel("Años")
plt.ylabel("Poblacion (Millones)")
plt.title("Estimacion China todos los años")
plt.show()

#Ahora hacemos lo mismo pero solo para los años pares

Datos_f_Pares = pd.read_csv("population-and-demography.csv")
Datos_f_Pares = Datos_f_Pares.loc[Datos_f_Pares['Entity'] == 'China'] #Solo nos importa China
Datos_f_Pares.drop(['Entity', 'Code'], axis=1, inplace=True) #Eliminamos columnas inecesarias
Datos_f_Pares.rename(columns={"Population - Sex: all - Age: all - Variant: estimates": "Population"}, inplace = True) #Renombramos columna
Datos_f_Pares["Population"] = Datos_f_Pares["Population"]/(10**6) #Escalamos los datos
Datos_f_Pares["Year"] = Datos_f_Pares["Year"]-1950.0 #Restamos 1950 para que el primer año sea el año 0

Datos_f_Pares = Datos_f_Pares[Datos_f_Pares["Year"] % 2 == 0] #Filtramos y nos quedamos solo con años pares

x_npar = np.array([np.log(1.05), 1450.0]) #Mismos valores iniciales
[x_newpar, kiterpar] = descenso_newton(funcion_Pares, x_npar) #Minimizamos la funcion con metodo de Newton

#print(x_newpar)
#print(Datos_f)

#Imprimimos resultados para años pares

print("================================================")
print("         Años Pares")
print("        Método de Newton")
print("================================================")
print("r = ", x_newpar[0])
print("K = ", x_newpar[1])
print("Gradiente Newton = ",np.linalg.norm(Gradiente_cen(funcion_Pares, x_newpar)) )
print("iter = ", kiterpar)
print("P(2024) = ", funcion_P_Pares(2024-1950, x_newpar[1], x_newpar[0]))
print("P(2030) = ", funcion_P_Pares(2030-1950, x_newpar[1], x_newpar[0]))


#Graficamos los datos poblacionales y las estimaciones de el modelo con los valores optimos de r y K, para años pares

Datos_f_Pares.set_index("Year",drop=True,inplace=True)
Datos_f_Pares.plot(title="China Population")

m = 100

F = np.zeros([m,2])

for i in range(m):
    F[i,0] = int(i)
    F[i,1] = funcion_P_Pares(i,x_newpar[1],x_newpar[0])

years = F[0:m,0]
pob = F[0:m,1]

plt.plot(years, pob)
plt.scatter(years, pob, marker="+", color = "g")
plt.xlabel("Años")
plt.ylabel("Poblacion (Millones)")
plt.title("Estimacion China años pares")
plt.show()


# ===============================================
# Descenso Máximo - Todos los años
# ===============================================

x_n = np.array([np.log(1.05), 1400.0])

[x_new, kiter] = descenso_max(funcion, x_n)

print("================================================")
print("         Todos los años")
print("        Método de Descenso Máximo")
print("================================================")
print("r = ", x_new[0])
print("K = ", x_new[1])
print("Gradiente Max = ",np.linalg.norm(Gradiente_cen(funcion, x_new)) )
print("iter = ", kiter)
print("P(2024) = ", funcion_P(2024-1950, x_new[1], x_new[0]))
print("P(2030) = ", funcion_P(2030-1950, x_new[1], x_new[0]))

# ===============================================
# Descenso Máximo - Años pares
# ==============================================
Datos_f = pd.read_csv("population-and-demography.csv")
Datos_f = Datos_f.loc[Datos_f['Entity'] == 'China']
Datos_f.drop(['Entity', 'Code'], axis=1, inplace=True)
Datos_f.rename(columns={"Population - Sex: all - Age: all - Variant: estimates": "Population"}, inplace = True)
Datos_f["Population"] = Datos_f["Population"]/(10**6)
Datos_f["Year"] = Datos_f["Year"]-1950.0

Datos_f = Datos_f[Datos_f["Year"] % 2 == 0]

x_npar = np.array([np.log(1.05), 1400.0])
[x_newpar, kiterpar] = descenso_max(funcion, x_npar)

print("================================================")
print("         Años Pares")
print("        Método Descenso Máximo")
print("================================================")
print("r = ", x_newpar[0])
print("K = ", x_newpar[1])
print("Gradiente Newton = ",np.linalg.norm(Gradiente_cen(funcion, x_newpar)) )
print("iter = ", kiterpar)
print("P(2024) = ", funcion_P(2024-1950, x_newpar[1], x_newpar[0]))
print("P(2030) = ", funcion_P(2030-1950, x_newpar[1], x_newpar[0]))