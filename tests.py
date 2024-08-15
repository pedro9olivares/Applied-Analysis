import functions
from derivatives import gradient

# Evaluating rosenbrock's function at (1,1), which we KNOW is a local minimum
f = functions.rosenbrock
x0 = [1.,1.]

print(f(x0))
print(gradient(f, x0))