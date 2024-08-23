import numpy as np

# Rosenbrock's function in two dims f(x,y)
def rosenbrock(x):
    x1 = x[0]
    x2 = x[1]

    fx = 100*(x2-x1**2)**2 + (1-x2)**2
    return fx

# Parabaloid in two dims
def paraboloid(x):
    x1 = x[0]
    x2 = x[1]

    fx = x1**2 + x2**2
    return fx

# fermat-Weber's functions
def fermat_weber(x):
    x1 = x[0]
    x2 = x[1]

    fx = np.sqrt(0) + np.sqrt(0) + np.sqrt(0)

# Parabaloid with uneven curvature
def uneven_paraboloid(x):
    x1 = x[0]
    x2 = x[1]

    fx = x1**2 +100*x2**2
    return fx



