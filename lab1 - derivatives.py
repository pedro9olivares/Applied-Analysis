import functions
from derivatives import gradient, hessian
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================================
# Rosenbrock
# ==========================================================================
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

# Evaluating rosenbrock's hessian at (1,1) (which we KNOW is a local minimum)
x0 = np.array([1.,1.])
hess = hessian(f,x0)
print(f'\n∇2f({x0}) [hessian] is \n{hess}')

print(f'\nThe eigenvalues of the hessian are {np.linalg.eigvals(hess)}.\n')

plt.show()

# ==========================================================================
# Fermat-Weber
# ==========================================================================
# Evaluating Fermat-Weber's function at (2,2)
f = functions.fermat_weber
x0 = np.array([2.,2.])

print('========================')
print('Fermat-Weber\'s function (f)')
print('========================')

print(f'\nf({x0}) = {f(x0)}')
grad = gradient(f,x0,'adelante')
print(f'\n∇f({x0})={grad} via forward differences...')
grad = gradient(f,x0,'centrada')
print(f'\n∇f({x0})={grad} via centered differences...')

# Now, let's plot
m = 50 # Density of axes
x_axis = np.linspace(-.5, 4.5, m)
y_axis = np.linspace(-.5, 3.5, m)
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

# Evaluating Fermat-Weber's hessian at (2,2)
x0 = np.array([2.,2.])
hess = hessian(f,x0)
print(f'\n∇2f({x0}) [hessian] is \n{hess}')

print(f'\nThe eigenvalues of the hessian are {np.linalg.eigvals(hess)}.\n')

plt.show()