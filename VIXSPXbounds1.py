import src.svi_lib as svi_lib
import src.black_lib as bs_lib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy import sparse
import math
import mosek
import mosek.fusion as mf

#CBOE on 21st June 2024
T1=20/251
S1_forw=5489.83
S1_strikes=np.linspace(4500,5800,27)
S1_norm=S1_strikes/S1_forw
S1_market_vol=np.array([0.30833965626324, 0.295476337981542, 0.283463677875459, 0.271111343370089, 0.258451404309255, 0.245507644865208, 0.233694271876973, 0.221358004433688, 0.209097873714426, 0.197249412151294, 0.186334324110301, 0.174675684455109, 0.164666282229566, 0.15488794443707, 0.145823620475546, 0.137304198281544, 0.129231575271196, 0.121548584918863, 0.114417742487412, 0.108019587373161, 0.102797547251224, 0.0988783071539835, 0.0959001836718852, 0.0934990753558453, 0.0918556359282457, 0.0918871975138243, 0.0934062296063838])

T2=40/251
S2_forw=5509.62
S2_strikes=np.linspace(4500,5800,27)
S2_norm=S2_strikes/S2_forw
S2_market_vol=np.array([0.256853, 0.247760, 0.239149, 0.230164, 0.221471, 0.213216, 0.204941, 0.196783, 0.188797, 0.180978, 0.173596, 0.166311, 0.159104, 0.152338, 0.145637, 0.139202, 0.132974, 0.127012, 0.121383, 0.116176, 0.111556, 0.108040, 0.105025, 0.102493, 0.100738, 0.099360, 0.098370])

T_V=20/251
V_forw=0.1437955868051
V_strikes=np.array([0.1200,  0.1250,    0.1300,    0.1350,    0.1400,    0.1450,    0.1500,    0.1550,    0.1600,  0.17, 0.18, 0.19, 0.2, 0.21])
V_norm=V_strikes/V_forw
V_market_vol=np.array([0.502489195, 0.531472613, 0.580483519, 0.63710948, 0.68409345, 0.736859132, 0.787044999, 0.827805687, 0.870005448, 0.94647844, 1.020997407, 1.09475647, 1.167988237, 1.219598942])

tau = T2 - T1

"""Selection of the middle 7 option prices"""
S1_strikes=S1_strikes[12:19]
S1_market_vol=S1_market_vol[12:19]
S2_strikes=S2_strikes[12:19]
S2_market_vol=S2_market_vol[12:19]
V_strikes=V_strikes[2:9]
V_market_vol=V_market_vol[2:9]

"""Market Prices of those options"""
C1 = np.array([bs_lib.black_scholes_call(S1_forw, K, T1, vol) for K,vol in zip(S1_strikes, S1_market_vol)])
C2 = np.array([bs_lib.black_scholes_call(S2_forw, K, T2, vol) for K,vol in zip(S2_strikes, S2_market_vol)])
CV= np.array([bs_lib.black_scholes_call(V_forw, K, T_V, vol) for K,vol in zip(V_strikes, V_market_vol)])

"""L func for dispersion"""
def L(r): return -2.0 / tau * np.log(r)

"""pmf grid"""
grid_size =75
s1_vals = np.linspace(0.75*S1_forw, 1.25*S1_forw, grid_size)
s2_vals = np.linspace(0.75*S2_forw, 1.25*S2_forw, grid_size)
v_vals = np.linspace(0.5*V_forw, 1.5*V_forw, grid_size)
S1, S2, V = np.meshgrid(s1_vals, s2_vals, v_vals, indexing='ij')
num_vars = grid_size ** 3

"""payoff for use in objective function"""
payoff_grid = np.maximum(V - V_forw, 0.0) #ATM VIX call
payoff_flat = payoff_grid.flatten()

"""Constraints built up and put into sparse matrix notation"""
# SPX T1 calls
A_S1 = sparse.lil_matrix((len(S1_strikes), num_vars))
for k, K1 in enumerate(S1_strikes):
    for i in range(grid_size):
        for j in range(grid_size):
            for l in range(grid_size):
                idx = i * grid_size * grid_size + j * grid_size + l
                payoff = s1_vals[i] - K1
                if payoff > 0:
                    A_S1[k, idx] = payoff
A_S1 = A_S1.tocsr()

# SPX T2 calls
A_S2 = sparse.lil_matrix((len(S2_strikes), num_vars))
for k, K2 in enumerate(S2_strikes):
    for i in range(grid_size):
        for j in range(grid_size):
            for l in range(grid_size):
                idx = i * grid_size * grid_size + j * grid_size + l
                payoff = s2_vals[j] - K2
                if payoff > 0:
                    A_S2[k, idx] = payoff
A_S2 = A_S2.tocsr()

# VIX T1 calls
A_V = sparse.lil_matrix((len(V_strikes), num_vars))
for k, Kv in enumerate(V_strikes):
    for i in range(grid_size):
        for j in range(grid_size):
            for l in range(grid_size):
                idx = i * grid_size * grid_size + j * grid_size + l
                payoff = v_vals[l] - Kv
                if payoff > 0:
                    A_V[k, idx] = payoff
A_V = A_V.tocsr()

# Martingale constraint
rows, cols, data = [], [], []
for i in range(grid_size):
    for l in range(grid_size):
        s1 = s1_vals[i]
        for j in range(grid_size):
            idx = i * grid_size * grid_size + j * grid_size + l
            rows.append(i * grid_size + l)
            cols.append(idx)
            data.append(s2_vals[j] - s1)
A_mart = sparse.csr_matrix((data, (rows, cols)), shape=(grid_size*grid_size, num_vars))
b_mart = np.zeros(grid_size * grid_size)

# Dispersion constraint
rows, cols, data = [], [], []
for i in range(grid_size):
    for l in range(grid_size):
        s1 = s1_vals[i]
        v = v_vals[l]
        for j in range(grid_size):
            idx = i * grid_size * grid_size + j * grid_size + l
            r = max(s2_vals[j] / s1, 1e-8)
            rows.append(i * grid_size + l)
            cols.append(idx)
            data.append(L(r) - v**2)
A_disp = sparse.csr_matrix((data, (rows, cols)), shape=(grid_size*grid_size, num_vars))
b_disp = np.zeros(grid_size * grid_size)

# Normalisation
A_norm = sparse.csr_matrix(np.ones((1, num_vars)))
b_norm = np.array([1.0])

"""Constraints stacked into sparse matrix"""
A_eq = sparse.vstack([A_S1, A_S2, A_V, A_norm, A_mart, A_disp]).tocsr()
b_eq = np.concatenate([C1, C2, CV, b_norm, b_mart, b_disp])
bounds = [(0, None)] * num_vars

"""Change constraints to Mosek fusion matrix"""
A_mosek = mf.Matrix.sparse(
    A_eq.shape[0],
    A_eq.shape[1],
    A_eq.tocoo().row,
    A_eq.tocoo().col,
    A_eq.tocoo().data
)

"""Lower bounds optimisation problem"""
M_min = mf.Model("min_price")
x = M_min.variable("x", num_vars, mf.Domain.greaterThan(0.0))
M_min.constraint("eqs", mf.Expr.mul(A_mosek, x), mf.Domain.equalsTo(b_eq))
M_min.objective("obj", mf.ObjectiveSense.Minimize, mf.Expr.dot(payoff_flat, x))
M_min.solve()
min_price = M_min.primalObjValue()
print("Min price:", min_price)

"""Lower bounds optimisation problem"""
M_max = mf.Model("max_price")
x_max = M_max.variable("x", num_vars, mf.Domain.greaterThan(0.0))
M_max.constraint("eqs", mf.Expr.mul(A_mosek, x_max), mf.Domain.equalsTo(b_eq))
M_max.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.dot(payoff_flat, x_max))
M_max.solve()
max_price = M_max.primalObjValue()
print("Max price:", max_price)

violMax = M_max.getConstraint("eqs").level() - b_eq
print("Max abs violation for upper bound:", abs(violMax).max())

violMin = M_min.getConstraint("eqs").level() - b_eq
print("Max abs violation for lower bound:", abs(violMin).max())

print("Primal solution status:", M_max.getPrimalSolutionStatus())
print("Dual solution status:  ", M_max.getDualSolutionStatus())
print("Problem status:        ", M_max.getProblemStatus())

pmf_opt = x.level().reshape(grid_size, grid_size, grid_size)
martingale_error = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for l in range(grid_size):
        martingale_error[i, l] = np.sum((s2_vals - s1_vals[i]) * pmf_opt[i, :, l])

dispersion_error = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for l in range(grid_size):
        s1 = s1_vals[i]
        v = v_vals[l]
        dispersion_error[i, l] = np.sum((L(np.maximum(s2_vals / s1, 1e-8)) - v**2) * pmf_opt[i, :, l])

print("Max abs dispersion error:", np.max(np.abs(dispersion_error)))

print("Max abs martingale error:", np.max(np.abs(martingale_error)))