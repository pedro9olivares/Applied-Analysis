#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:39:50 2024

@authors: Erick Martinez Hernandez, 191821
          Sara Visoso Gunther, 196079
          Pedro Olivares Sanchez, 190198
"""

import numpy as np
import pandas as pd
from biblio_codigos import descenso_newton, descenso_max, regioncon, descenso_bfgs
import matplotlib.pyplot as plt
from biblio_derivadas import Gradiente_cen

# Crear los datos para las columnas
Datos_f = {
    'y': [0.591, 1.547, 2.902, 2.894, 4.703, 6.307, 7.03, 7.898, 9.470, 9.484,
          10.072, 10.163, 11.615, 12.005, 12.478, 12.982, 12.970, 13.926, 14.452, 14.404,
          15.190, 15.550, 15.528, 15.499, 16.131, 16.438, 16.387, 16.549, 16.872, 16.830,
          16.926, 16.907, 16.966, 17.060, 17.122, 17.311, 17.355, 17.668, 17.767, 17.803,
          17.765, 17.768, 17.736, 17.858, 17.877, 17.912, 18.046, 18.085, 18.291, 18.357,
          18.426, 18.584, 18.610, 18.870, 18.795, 19.111, 0.367, 0.796, 0.892, 1.903,
          2.150, 3.697, 5.870, 6.421, 7.422, 9.944, 11.023, 11.87, 12.786, 14.067,
          13.974, 14.462, 14.464, 15.381, 15.483, 15.59, 16.075, 16.347, 16.181, 16.915,
          17.003, 16.978, 17.756, 17.808, 17.868, 18.481, 18.486, 19.090, 16.062, 16.337,
          16.345, 16.388, 17.159, 17.116, 17.164, 17.123, 17.979, 17.974, 18.007, 17.993,
          18.523, 18.669, 18.617, 19.371, 19.330, 0.080, 0.248, 1.089, 1.418, 2.278,
          3.624, 4.574, 5.556, 7.267, 7.695, 9.136, 9.959, 9.957, 11.600, 13.138,
          13.564, 13.871, 13.994, 14.947, 15.473, 15.379, 15.455, 15.908, 16.114, 17.071,
          17.135, 17.282, 17.368, 17.483, 17.764, 18.185, 18.271, 18.236, 18.237, 18.523,
          18.627, 18.665, 19.086, 0.214, 0.943, 1.429, 2.241, 2.951, 3.782, 4.757,
          5.602, 7.169, 8.920, 10.055, 12.035, 12.861, 13.436, 14.167, 14.755, 15.168,
          15.651, 15.746, 16.216, 16.445, 16.965, 17.121, 17.206, 17.250, 17.339, 17.793,
          18.123, 18.49, 18.566, 18.645, 18.706, 18.924, 19.1, 0.375, 0.471, 1.504,
          2.204, 2.813, 4.765, 9.835, 10.040, 11.946, 12.596, 13.303, 13.922, 14.440,
          14.951, 15.627, 15.639, 15.814, 16.315, 16.334, 16.430, 16.423, 17.024, 17.009,
          17.165, 17.134, 17.349, 17.576, 17.848, 18.090, 18.276, 18.404, 18.519, 19.133,
          19.074, 19.239, 19.280, 19.101, 19.398, 19.252, 19.89, 20.007, 19.929, 19.268,
          19.324, 20.049, 20.107, 20.062, 20.065, 19.286, 19.972, 20.088, 20.743, 20.83,
          20.935, 21.035, 20.93, 21.074, 21.085, 20.935],
    'x': [24.41, 34.82, 44.09, 45.07, 54.98, 65.51, 70.53, 75.70, 89.57, 91.14,
          96.40, 97.19, 114.26, 120.25, 127.08, 133.55, 133.61, 158.67, 172.74, 171.31,
          202.14, 220.55, 221.05, 221.39, 250.99, 268.99, 271.80, 271.97, 321.31, 321.69,
          330.14, 333.03, 333.47, 340.77, 345.65, 373.11, 373.79, 411.82, 419.51, 421.59,
          422.02, 422.47, 422.61, 441.75, 447.41, 448.7, 472.89, 476.69, 522.47, 522.62,
          524.43, 546.75, 549.53, 575.29, 576.00, 625.55, 20.15, 28.78, 29.57, 37.41,
          39.12, 50.24, 61.38, 66.25, 73.42, 95.52, 107.32, 122.04, 134.03, 163.19,
          163.48, 175.70, 179.86, 211.27, 217.78, 219.14, 262.52, 268.01, 268.62, 336.25,
          337.23, 339.33, 427.38, 428.58, 432.68, 528.99, 531.08, 628.34, 253.24, 273.13,
          273.66, 282.10, 346.62, 347.19, 348.78, 351.18, 450.10, 450.35, 451.92, 455.56,
          552.22, 553.56, 555.74, 652.59, 656.20, 14.13, 20.41, 31.30, 33.84, 39.70,
          48.83, 54.50, 60.41, 72.77, 75.25, 86.84, 94.88, 96.40, 117.37, 139.08,
          147.73, 158.63, 161.84, 192.11, 206.76, 209.07, 213.32, 226.44, 237.12, 330.90,
          358.72, 370.77, 372.72, 396.24, 416.59, 484.02, 495.47, 514.78, 515.65, 519.47,
          544.47, 560.11, 620.77, 18.97, 28.93, 33.91, 40.03, 44.66, 49.87, 55.16,
          60.90, 72.08, 85.15, 97.06, 119.63, 133.27, 143.84, 161.91, 180.67, 198.44,
          226.86, 229.65, 258.27, 273.77, 339.15, 350.13, 362.75, 371.03, 393.32, 448.53,
          473.78, 511.12, 524.70, 548.75, 551.64, 574.02, 623.86, 21.46, 24.33, 33.43,
          39.22, 44.18, 55.02, 94.33, 96.44, 118.82, 128.48, 141.94, 156.92, 171.65,
          190.00, 223.26, 223.88, 231.50, 265.05, 269.44, 271.78, 273.46, 334.61, 339.79,
          349.52, 358.18, 377.98, 394.77, 429.66, 468.22, 487.27, 519.54, 523.03, 612.99,
          638.59, 641.36, 622.05, 631.50, 663.97, 646.9, 748.29, 749.21, 750.14, 647.04,
          646.89, 746.9, 748.43, 747.35, 749.27, 647.61, 747.78, 750.51, 851.37, 845.97,
          847.54, 849.93, 851.61, 849.75, 850.98, 848.23]
}

# Crear el DataFrame

df = pd.DataFrame(Datos_f)

df = df.sort_values(by = 'x')

# Mostrar las primeras filas del DataFrame
print(df.head())

# Mostrar información básica del DataFrame
print("\nInformación del DataFrame:")
print(df.info())

dfN = df.to_numpy()

#Modelo

def model(x,b):
    arriba = b[0] + b[1]*x + b[2]*(x**2) + b[3]*(x**3)
    abajo =   1   + b[4]*x + b[5]*(x**2) + b[6]*(x**3)
    return arriba / abajo

#Funcion error

def funcion(b):
    fx = 0
    for i in range(len(df)):
        fx = fx + (dfN[i,0] - model(dfN[i,1],b))**2
    return fx

#Puntos iniciales

random = True

if(random == False):
    b1 = np.array([1.0000000000E+01, -1.0000000000E+00, 5.0000000000E-02, -1.0000000000E-05, -5.0000000000E-02, 1.0000000000E-03, -1.0000000000E-06])
    b2 = np.array([1.0000000000E+00, -1.0000000000E-01, 5.0000000000E-03, -1.0000000000E-06, -5.0000000000E-03, 1.0000000000E-04, -1.0000000000E-07])
    b3 = np.array([1.0776351733E+00, -1.2269296921E-01, 4.0863750610E-03, -1.4262662514E-06, -5.7609940901E-03, 2.4053735503E-04, -1.2314450199E-07,])
if(random == True):
    b1 = np.array([1.0000000000E+01 + np.random.randn(), -1.0000000000E+00 + np.random.randn(), 5.0000000000E-02 + np.random.randn(), -1.0000000000E-05 + np.random.randn(), -5.0000000000E-02 + np.random.randn(), 1.0000000000E-03 + np.random.randn(), -1.0000000000E-06 + np.random.randn()])
    b2 = np.array([1.0000000000E+00 + np.random.randn(), -1.0000000000E-01 + np.random.randn(), 5.0000000000E-03 + np.random.randn(), -1.0000000000E-06 + np.random.randn(), -5.0000000000E-03 + np.random.randn(), 1.0000000000E-04 + np.random.randn(), -1.0000000000E-07 + np.random.randn()])
    b3 = np.array([1.0776351733E+00 + np.random.randn(), -1.2269296921E-01 + np.random.randn(), 4.0863750610E-03+ np.random.randn(), -1.4262662514E-06 + np.random.randn(), -5.7609940901E-03 + np.random.randn(), 2.4053735503E-04 + np.random.randn(), -1.2314450199E-07 + np.random.randn(),])

m = 800

print('\n----------------------------Newton-----------------------------------\n')
[b_new1, kiter] = descenso_newton(funcion, b1) 
[b_new2, kiter] = descenso_newton(funcion, b2) 
[b_new3, kiter] = descenso_newton(funcion, b3) 
print('INICIO 1:')
print(b_new1, kiter)
print('INICIO 2:')
print(b_new2, kiter)
print('INICIO 3:')
print(b_new3, kiter)

F1 = np.zeros([m,2])

for i in range(m):
    F1[i,0] = int(i)
    F1[i,1] = model(i,b_new1)

F3 = np.zeros([m,2])

for i in range(m):
    F3[i,0] = int(i)
    F3[i,1] = model(i,b_new3)

F2 = np.zeros([m,2])

for i in range(m):
    F2[i,0] = int(i)
    F2[i,1] = model(i,b_new2)
    
temperature3 = F3[0:m,0]
coefficient3 = F3[0:m,1]
temperature2 = F2[50:m,0]
coefficient2 = F2[50:m,1]
temperature = F1[0:m,0]
coefficient = F1[0:m,1]

plt.close('all')

df.plot(x='x', y='y',title="Coeficiente de expansion termal")
plt.xlabel("Temperatura (Kelvin)")
plt.ylabel("Coeficiente de expansion termal")
plt.title("Estimacion: Newton")
if(random == True):
    plt.plot(temperature, coefficient, color = "y", label = "Inicio 1")
plt.plot(temperature2, coefficient2, color = "r", label = "Inicio 2")
plt.plot(temperature3, coefficient3, color = "g", label = "Inicio 3")
plt.legend()
plt.show()

print('\n----------------------------Maximo-----------------------------------\n')
[b_max1, kiter] = descenso_max(funcion, b1) 
[b_max2, kiter] = descenso_max(funcion, b2) 
[b_max3, kiter] = descenso_max(funcion, b3) 
print('INICIO 1:')
print(b_max1, kiter)
print('INICIO 2:')
print(b_max2, kiter)
print('INICIO 3:')
print(b_max3, kiter)

F1 = np.zeros([m,2])

for i in range(m):
    F1[i,0] = int(i)
    F1[i,1] = model(i,b_max1)

F3 = np.zeros([m,2])

for i in range(m):
    F3[i,0] = int(i)
    F3[i,1] = model(i,b_max3)

F2 = np.zeros([m,2])

for i in range(m):
    F2[i,0] = int(i)
    F2[i,1] = model(i,b_max2)

temperature = F1[0:m,0]
coefficient = F1[0:m,1]
temperature2 = F2[50:m,0]
coefficient2 = F2[50:m,1]
temperature3 = F3[0:m,0]
coefficient3 = F3[0:m,1]

df.plot(x='x', y='y',title="Coeficiente de expansion termal")
plt.xlabel("Temperatura (Kelvin)")
plt.ylabel("Coeficiente de expansion termal")
plt.title("Estimacion: Maximo Descenso")
plt.plot(temperature, coefficient, color = "y", label = "Inicio 1")
plt.plot(temperature2, coefficient2, color = "r", label = "Inicio 2")
plt.plot(temperature3, coefficient3, color = "g", label = "Inicio 3")
plt.legend()
plt.show()


print('\n-----------------------Region de Confianza---------------------------\n')
[b_con1, kiter] = regioncon(funcion, b1) 
[b_con2, kiter] = regioncon(funcion, b2) 
[b_con3, kiter] = regioncon(funcion, b3) 
print('INICIO 1:')
print(b_con1, kiter)
print('INICIO 2:')
print(b_con2, kiter)
print('INICIO 3:')
print(b_con3, kiter)

#b_mejor = np.array([ 7.76890265e-02, -1.31594120e-01,  4.03952964e-03, -1.43241592e-06,
# -4.25186829e-04,  2.33685018e-04, -1.43132284e-07])

F1 = np.zeros([m,2])

for i in range(m):
    F1[i,0] = int(i)
    F1[i,1] = model(i,b_con1)

F3 = np.zeros([m,2])

for i in range(m):
    F3[i,0] = int(i)
    F3[i,1] = model(i,b_con3)

F2 = np.zeros([m,2])

for i in range(m):
    F2[i,0] = int(i)
    F2[i,1] = model(i,b_con2)

temperature = F1[0:m,0]
coefficient = F1[0:m,1]

temperatureM = F3[0:m,0]
coefficientM = F3[0:m,1]

temperature21 = F2[0:50,0]
coefficient21 = F2[0:50,1]

temperature22 = F2[200:m,0]
coefficient22 = F2[200:m,1]

temperature2 = np.concatenate((temperature21 , temperature22))
coefficient2 = np.concatenate((coefficient21 , coefficient22))

df.plot(x='x', y='y',title="Coeficiente de expansion termal")
plt.xlabel("Temperatura (Kelvin)")
plt.ylabel("Coeficiente de expansion termal")
plt.title("Estimacion: Region de Confianza")
plt.plot(temperature, coefficient, color = "y", label = "Inicio 1")
plt.plot(temperature2, coefficient2, color = "r", label = "Inicio 2")
plt.plot(temperatureM, coefficientM, color = "g", label = "Inicio 3")
plt.legend()
plt.show()

print('\n----------------------------Broyden----------------------------------\n')
[b_bfg1, kiter] = descenso_bfgs(funcion, b1) 
[b_bfg2, kiter] = descenso_bfgs(funcion, b2)
[b_bfg3, kiter] = descenso_bfgs(funcion, b3)
print('INICIO 1:')
print(b_bfg1, kiter)
print('INICIO 2:')
print(b_bfg2, kiter)
print('INICIO 3:')
print(b_bfg3, kiter)
print('\n---------------------------------------------------------------------\n')

F1 = np.zeros([m,2])

for i in range(m):
    F1[i,0] = int(i)
    F1[i,1] = model(i,b_bfg1)

F3 = np.zeros([m,2])

for i in range(m):
    F3[i,0] = int(i)
    F3[i,1] = model(i,b_bfg3)

F2 = np.zeros([m,2])

for i in range(m):
    F2[i,0] = int(i)
    F2[i,1] = model(i,b_bfg2)

temperature = F1[0:m,0]
coefficient = F1[0:m,1]

temperature3 = F3[0:m,0]
coefficient3 = F3[0:m,1]

temperature2 = F2[0:m,0]
coefficient2 = F2[0:m,1]

df.plot(x='x', y='y',title="Coeficiente de expansion termal")
plt.xlabel("Temperatura (Kelvin)")
plt.ylabel("Coeficiente de expansion termal")
plt.title("Estimacion: BFGS")
plt.plot(temperature, coefficient, color = "y", label = "Inicio 1")
plt.plot(temperature2, coefficient2, color = "r", label = "Inicio 2")
plt.plot(temperature3, coefficient3, color = "g", label = "Inicio 3")
plt.legend()
plt.show()

if(random == False):
    df.plot(x='x', y='y',title="Coeficiente de expansion termal")
    plt.xlabel("Temperatura (Kelvin)")
    plt.ylabel("Coeficiente de expansion termal")
    plt.title("Estimacion: Region de Confianza mejor")
    plt.plot(temperatureM, coefficientM, color = "g", label = "Inicio 3")
    plt.legend()
    plt.show()
    print("Suma de errores para mejor estimacion:", funcion(b_con3))
    print("Norma de la solucion:", np.linalg.norm(Gradiente_cen(funcion, b_con3)))
