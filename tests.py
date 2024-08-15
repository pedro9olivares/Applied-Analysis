import functions
from derivatives import gradient
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Evaluating rosenbrock's function at (1,1), which we KNOW is a local minimum
f = functions.rosenbrock
x0 = np.array([1.,1.])

print('========================')
print('Rosenbrock\'s function (f)')
print('========================')

print(f'\nf({x0}) = {f(x0)}')
grad = gradient(f,x0,'adelante')
print(f'\n∇f({x0})={grad} via forward differences...')
grad = gradient(f,x0,'centrada')
print(f'\n∇f({x0})={grad} via centered differences...')

# Now, let's plot on the interval [0,2]x[0,2]
m = 50 # Density of axes
x_axis = np.linspace(0, 2, m)
y_axis = np.linspace(0, 2, m)
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


