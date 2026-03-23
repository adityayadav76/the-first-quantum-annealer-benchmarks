import genosolver
import numpy as np

def geno_fg(Q):
    def _fg(x):
        Qx = np.dot(Q,x)
        return np.dot(x,Qx), 2*Qx
    return _fg

def geno_minimize(Q, time_limit=None):
    n = len(Q)
    fg = geno_fg(Q)
    best = np.zeros(n)
    for _i in range(10):
        x0 = np.random.rand(n)
        res = genosolver.minimize(fg, x0, lb=np.zeros_like(x0), ub=np.ones_like(x0), np=np, options={'ls':2, 'max_iter': 200})
        x = res.x
        x = np.round(x)
        if np.dot(x, np.dot(Q, x)) < np.dot(best, np.dot(Q, best)):
            best = x
    return best

import dimod 
from dimod import BQM 
# Solve the QUBO problem using Automatski's QUBO Solvers 
from AutomatskiInitium import *

def automatski_minimize(Q, time_limit=None, host=None, port=None):

    try:
        bqm = BQM.from_qubo(Q)
        qubo, offset = bqm.to_qubo()
        
        #solver = AutomatskiInitiumTabuSolver(host=host, port=port, max_iter=1000, tabu_tenure=10)
        solver = AutomatskiInitiumSASolver(host=host, port=port, max_iter=1000, temp=10.0, cooling_rate=0.01, num_reads=10)

        best_state, best_cost = solver.solve(qubo)
        res = np.array([ x[1] for x in sorted(best_state.items()) ])
        assert np.all((res == 0) | (res == 1))
    except ValueError as e:
        print(e)
        res = np.zeros(Q.shape[0])-1
    return res

if __name__ == '__main__':
    pass
