import numpy as np
from scipy.optimize import brentq

"""Solving for betaL roots based on clamped-clamped beam vibration"""
n_betaL_roots = 10
f = lambda x: np.cosh(x)*np.cos(x) - 1.0

xs = np.linspace(0.0, 100.0, 200_001)
vals = f(xs)
edges = np.flatnonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)

betaL_roots = np.array([brentq(f, xs[i], xs[i+1]) for i in edges if xs[i+1] > 1e-12])[:n_betaL_roots]
print(betaL_roots)