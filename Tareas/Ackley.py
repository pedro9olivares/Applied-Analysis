import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Implementación de la función multimodal de Ackley
# Del artículo "Test functions for optimization needs",
# por Marcin Molga, Czeslaw Smutnicki
def ackley(x, a=20, b=0.2, c=2*math.pi):
    sum1 = 0
    sum2 = 0
    n = len(x)

    for i in range(n):
        sum1 += x[i] ** 2

    for i in range(n):
        sum2 = math.cos(c * x[i])

    fx = (-a) * math.exp((-b) * math.sqrt((1/n) * sum1)) - math.exp((1/n) * sum2) + a + math.exp(1)
    return fx

# Prueba: Graficando en el hipercubo -32.768 < x_i < 32.768 de dos dimensiones
# Ojo: el mínimo global f(x) = 0 se alcanza con con x = (0, 0, ..., 0)
f = ackley

m = 1000 # Density of axes
p = 32.768 # Limit of the hipercube
# p = 2
x_axis = np.linspace(-p, p, m)
y_axis = np.linspace(-p, p, m)
F = np.zeros((m,m))

for i in range(m):
    x0 = x_axis[i]
    for j in range(m):
        y0 = y_axis[j]
        w = np.array([x0, y0])
        F[i, j] = f(w)

[X, Y] = np.meshgrid(x_axis, y_axis)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,F, cmap='viridis')

plt.show()