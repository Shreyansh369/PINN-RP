"""Mathematically equivalent formulations of the damped Euler-Bernoulli beam (axis 15.2/15.3).

strong : c2 u_xxxx + u_tt + gamma u_t = 0                     (Batch-1; beampinn)
mixed  : v = u_xx,  c2 v_xx + u_tt + gamma u_t = 0            (mixed.py; max derivative order 2)
modal  : u = A0 phi_1(x) q(t),  q'' + gamma q' + w1^2 q = 0     (modal.py; Galerkin reduction)
"""
