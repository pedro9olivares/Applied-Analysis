import numpy as np
import math

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

    fx = np.sqrt(math.sqrt(x1**2 + x2**2)) + np.sqrt(math.sqrt((x1 - 4)**2 + x2**2)) + np.sqrt(math.sqrt((x1 - 1)**2 + (x2 - 3)**2))
    return fx

# Parabaloid with uneven curvature
def uneven_paraboloid(x):
    x1 = x[0]
    x2 = x[1]

    fx = x1**2 +10*x2**2
    return fx

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



