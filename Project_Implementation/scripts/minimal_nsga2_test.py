import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, xl=np.array([0.0,0.0]), xu=np.array([1.0,1.0]))
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_1d(x)
        out["F"] = np.atleast_2d([x[0], 1.0 - x[1]])

problem = MyProblem()
algorithm = NSGA2(pop_size=10)
res = minimize(problem, algorithm, termination=("n_gen", 2), seed=42, verbose=False)
print('Minimal NSGA-II solutions:', None if res.X is None else (res.X.shape, res.F.shape))
