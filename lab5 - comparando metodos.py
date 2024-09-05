from functions import ackley, rosenbrock
from derivatives import gradient
from descentMethods import max_descent, newton_descent_unrestricted
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================
# 1) Cálculo de la tabla en la función de ackley
# ==================================================================

# Variables para medir tiempo
t_max = 0
t_newt = 0

# Selección del punto inicial xi 
xi = [1,-1]

f = ackley

start = time.time()
minimum_max, k_max = max_descent(f, xi, maxiter=900)
end = time.time()
t_max = end-start

start = time.time()
minimum_newt, k_newt = newton_descent_unrestricted(f, xi)
end = time.time()
t_newt = end-start

g1 = np.linalg.norm(gradient(f, minimum_max))
g2 = np.linalg.norm(gradient(f, minimum_newt))

print('TABLA 1: Comparación en la función de Ackley')
print(f'                |=======================|=========================|')
print(f'                |====Descenso Máximo====|=====Descenso Newton=====|')
print(f'                |=======================|=========================|')
print(f'   |∇f(x_min)|  |   {g1}  | {g2}   |')
print(f'   iteraciones  |          {k_max}          |            {k_newt}           |')
print(f'     cpu time   | {t_max}  |  {t_newt}  |')


# ==================================================================
# 2) Graficando la función de ackley
# ==================================================================
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


# ==================================================================
# 3) Cálculo de la tabla en la función de rosenbrock
# ==================================================================
# Variables para medir tiempo
t_max = 0
t_newt = 0

# Selección del punto inicial xi 
xi = [1,-1]

f = rosenbrock

start = time.time()
minimum_max, k_max = max_descent(f, xi, maxiter=900)
end = time.time()
t_max = end-start

start = time.time()
minimum_newt, k_newt = newton_descent_unrestricted(f, xi)
end = time.time()
t_newt = end-start

g1 = np.linalg.norm(gradient(f, minimum_max))
g2 = np.linalg.norm(gradient(f, minimum_newt))

print()
print('TABLA 2: Comparación en la función de Rosenbrock')
print(f'                |=======================|=========================|')
print(f'                |====Descenso Máximo====|=====Descenso Newton=====|')
print(f'                |=======================|=========================|')
print(f'   |∇f(x_min)|  |   {g1} | {g2}  |')
print(f'   iteraciones  |          {k_max}          |            {k_newt}            |')
print(f'     cpu time   | {t_max}  |  {t_newt} |')