import math

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