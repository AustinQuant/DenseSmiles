import src.black_lib as bs_lib
import numpy as np
import mosek.fusion as mf
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy.ma as ma

T1=20/251
S1_forw=5489.83
S1_strikes=np.linspace(4500,5800,27)
S1_market_vol=np.array([0.30833965626324, 0.295476337981542, 0.283463677875459, 0.271111343370089,
                        0.258451404309255, 0.245507644865208, 0.233694271876973, 0.221358004433688,
                        0.209097873714426, 0.197249412151294, 0.186334324110301, 0.174675684455109,
                        0.164666282229566, 0.15488794443707, 0.145823620475546, 0.137304198281544, 
                        0.129231575271196, 0.121548584918863, 0.114417742487412, 0.108019587373161,
                        0.102797547251224, 0.0988783071539835, 0.0959001836718852, 0.0934990753558453,
                         0.0918556359282457, 0.0918871975138243, 0.0934062296063838])

T2=40/251
S2_forw=5509.62
S2_strikes=np.linspace(4500,5800,27)
S2_market_vol=np.array([0.256853, 0.247760, 0.239149, 0.230164,
                        0.221471, 0.213216, 0.204941, 0.196783, 
                        0.188797, 0.180978, 0.173596, 0.166311, 
                        0.159104, 0.152338, 0.145637, 0.139202, 
                        0.132974, 0.127012, 0.121383, 0.116176, 
                        0.111556, 0.108040, 0.105025, 0.102493, 
                        0.100738, 0.099360, 0.098370])
"""Choosing a subset for calibration"""
S1_strikes = S1_strikes[7:]
S1_market_vol = S1_market_vol[7:]
S2_strikes = S2_strikes[7:]
S2_market_vol = S2_market_vol[7:]

#Call price calculation
C1 = np.array([bs_lib.black_scholes_call(S1_forw, K, T1, vol) for K,vol in zip(S1_strikes, S1_market_vol)])
C2 = np.array([bs_lib.black_scholes_call(S2_forw, K, T2, vol) for K,vol in zip(S2_strikes, S2_market_vol)])

#grid for the PMFs
grid_size = 100
s1_vals = np.linspace(0.85*S1_forw, 1.1*S1_forw, grid_size)
s2_vals = np.linspace(0.8*S2_forw, 1.1*S2_forw, grid_size)
nvar = grid_size**2
#Sparse matrix construction arrays
rows, cols, data = [], [], []
b_eq = []
rowcount = 0

# S1 call constraints
for K1, C in zip(S1_strikes, C1):
    for i, s1 in enumerate(s1_vals):
        payoff_s1 = max(s1 - K1, 0)
        if payoff_s1 > 0:
            for j in range(grid_size):
                idx = i*grid_size + j
                rows.append(rowcount)
                cols.append(idx)
                data.append(payoff_s1)
    b_eq.append(C)
    rowcount += 1

# S2 call constraints
for K2, C in zip(S2_strikes, C2):
    for i in range(grid_size):
        for j, s2 in enumerate(s2_vals):
            payoff_s2 = max(s2 - K2, 0)
            if payoff_s2 > 0:
                idx = i*grid_size + j
                rows.append(rowcount)
                cols.append(idx)
                data.append(payoff_s2)
    b_eq.append(C)
    rowcount += 1

# Martingale constraints
for i, s1 in enumerate(s1_vals):
    for j, s2 in enumerate(s2_vals):
        idx = i*grid_size + j
        rows.append(rowcount)
        cols.append(idx)
        #Forward ratio accounts for the discounting in the E value
        data.append(s2 - s1*(S2_forw / S1_forw))  
    b_eq.append(0.0)
    rowcount += 1

# Normalisation constraint for PMF
for idx in range(nvar):
    rows.append(rowcount)
    cols.append(idx)
    data.append(1.0)
b_eq.append(1.0)
rowcount += 1
# Sparse constraint matrix put together for Mosek solver
A_sparse = coo_matrix((data, (rows, cols)), shape=(rowcount, nvar))
b_eq = np.array(b_eq)
A_mosek = mf.Matrix.sparse(
    A_sparse.shape[0],
    A_sparse.shape[1],
    A_sparse.row,
    A_sparse.col,
    A_sparse.data
)
#payoff grid for objective function (here a straddle)
S1_grid, S2_grid = np.meshgrid(s1_vals, s2_vals, indexing="ij")
payoff_flat = np.abs(S2_grid-S1_grid).flatten()#(np.log(S2_grid / S1_grid) ** 2).flatten()

#Minimise over admissible prices (dual and primal simultaneously with interior point method)
M_min = mf.Model("min_price")
x_min = M_min.variable("x", nvar, mf.Domain.greaterThan(0.0))
M_min.constraint("eqs", mf.Expr.mul(A_mosek, x_min), mf.Domain.equalsTo(b_eq))
M_min.objective("obj", mf.ObjectiveSense.Minimize, mf.Expr.dot(payoff_flat, x_min))
M_min.solve()
min_price = M_min.primalObjValue()

#Max over admissible prices
M_max = mf.Model("max_price")
x_max = M_max.variable("x", nvar, mf.Domain.greaterThan(0.0))
M_max.constraint("eqs", mf.Expr.mul(A_mosek, x_max), mf.Domain.equalsTo(b_eq))
M_max.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.dot(payoff_flat, x_max))
M_max.solve()
max_price = M_max.primalObjValue()

"""OUTPUT 1 - Price bounds"""
print("Min price:", min_price)
print("Max price:", max_price)

#extremal probability distributions
pmf_opt = np.array(x_max.level()).reshape((grid_size, grid_size))
pmf_min = np.array(x_min.level()).reshape((grid_size, grid_size))
pmf_opt = np.clip(pmf_opt, 0, None)
pmf_min = np.clip(pmf_min, 0, None)

#calculation of conditional probabilities 
tol = 1e-12 #(avoids division by near zero vals)
row_sums_opt = pmf_opt.sum(axis=1, keepdims=True)
row_sums_min = pmf_min.sum(axis=1, keepdims=True)
pmf_opt_cond = np.divide(pmf_opt, row_sums_opt, out=np.zeros_like(pmf_opt), where=row_sums_opt > tol)
pmf_min_cond = np.divide(pmf_min, row_sums_min, out=np.zeros_like(pmf_min), where=row_sums_min > tol)

#dual variables for the upper bound
dual_vars = M_max.getConstraint("eqs").dual().flatten()
#duals for martingale constraints in maximal model
s1_call_duals = dual_vars[:len(C1)]
s2_call_duals = dual_vars[len(C1):len(C1) + len(C2)]
mart_duals = dual_vars[len(C1)+len(C2):len(C1)+len(C2)+grid_size]
norm_dual = dual_vars[-1]
#dual variables for the lower bound
dual_vars_min = M_min.getConstraint("eqs").dual().flatten()
#duals for the martingale constraints in minimal model
mart_duals_min = dual_vars_min[len(C1)+len(C2):len(C1)+len(C2)+grid_size]

"""OUTPUT 2 - Heatmaps of the conditional probabilities showing sparseness"""
#heatmap of maximal solution
plt.figure(figsize=(10, 8))
masked = ma.masked_where(pmf_opt_cond.T <=1e-12, pmf_opt_cond.T)
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='white') 
plt.imshow(masked,aspect='auto',origin='lower',extent=[s1_vals[0], s1_vals[-1], s2_vals[0], s2_vals[-1]],cmap=cmap)
plt.colorbar(label="P(S2 | S1)")
plt.xlabel("S1")
plt.ylabel("S2")
plt.title("Conditional distribution P(S2 | S1) for the maximal price PMF\n (Exact Zeros in white)")
plt.show()

# Heatmap for the minimal solution
plt.figure(figsize=(10, 8))
masked_min = ma.masked_where(pmf_min_cond.T <= 1e-12, pmf_min_cond.T)
cmap_min = plt.cm.viridis.copy()
cmap_min.set_bad(color='white')  # Masked zero values
plt.imshow(masked_min,aspect='auto',origin='lower',extent=[s1_vals[0], s1_vals[-1], s2_vals[0], s2_vals[-1]],cmap=cmap_min)
plt.colorbar(label="P(S2 | S1)")
plt.xlabel("S1")
plt.ylabel("S2")
plt.title("Conditional distribution P(S2 | S1) for the minimal price PMF\n (Exact Zeros in white)")
plt.show()

"""OUTPUT 3 - Dual variables in the call options"""
fig1, axs1 = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

# Top row: Superhedge
# S1 call weights (superhedge)
axs1[0, 0].bar(S1_strikes, dual_vars[:len(C1)], width=(S1_strikes[1]-S1_strikes[0])*0.8, color='tab:blue')
axs1[0, 0].set_xlabel("Strike")
axs1[0, 0].set_ylabel("S1 Call Weights")
axs1[0, 0].set_title("Superhedging S1 Call Weights")

# S2 call weights (superhedge)
axs1[0, 1].bar(S2_strikes, dual_vars[len(C1):len(C1)+len(C2)], width=(S2_strikes[1]-S2_strikes[0])*0.8, color='tab:green')
axs1[0, 1].set_xlabel("Strike")
axs1[0, 1].set_ylabel("S2 Call Weights")
axs1[0, 1].set_title("Superhedging S2 Call Weights")

# Bottom row: Subhedge
# S1 call weights (subhedge)
axs1[1, 0].bar(S1_strikes, dual_vars_min[:len(C1)], width=(S1_strikes[1]-S1_strikes[0])*0.8, color='tab:blue')
axs1[1, 0].set_xlabel("Strike")
axs1[1, 0].set_ylabel("S1 Call Weights")
axs1[1, 0].set_title("Subhedging S1 Call Weights")

# S2 call weights (subhedge)
axs1[1, 1].bar(S2_strikes, dual_vars_min[len(C1):len(C1)+len(C2)], width=(S2_strikes[1]-S2_strikes[0])*0.8, color='tab:green')
axs1[1, 1].set_xlabel("Strike")
axs1[1, 1].set_ylabel("S2 Call Weights")
axs1[1, 1].set_title("Subhedging S2 Call Weights")

# Consistent y scaling for each row
for row in [0, 1]:
    ymin = min(axs1[row, 0].get_ylim()[0], axs1[row, 1].get_ylim()[0])
    ymax = max(axs1[row, 0].get_ylim()[1], axs1[row, 1].get_ylim()[1])
    axs1[row, 0].set_ylim(ymin, ymax)
    axs1[row, 1].set_ylim(ymin, ymax)

fig1.suptitle("Call Option Weights: Superhedging (Top) vs Subhedging (Bottom)", fontsize=16)
fig1.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

"""OUTPUT 4 - Dual Variables for the Delta Hedging Strategy"""
plt.figure(figsize=(10, 6))
plt.plot(s1_vals, mart_duals, '-', color='tab:orange', label='Superhedge Delta')
plt.plot(s1_vals, mart_duals_min, '-', color='tab:purple', label='Subhedge Delta')
plt.xlabel("Observed value of S1")
plt.ylabel("Quantity of S2 future")
plt.title("Optimal Delta Hedge at T1: Superhedge vs Subhedge")
plt.legend()
plt.grid(True)
plt.show()

"""OUTPUT 5 - Plot of the full Lagrangian for upper bound showing duality via support points"""
# First compute Lagrangian for all S1, S2 pairs
lagrangian_3d = np.zeros((len(s1_vals), len(s2_vals)))

for i, s1 in enumerate(s1_vals):
    # Constant part from S1 calls
    s1_constant = 0
    for k, K in enumerate(S1_strikes):
        s1_constant += s1_call_duals[k] * max(s1 - K, 0)
    
    # S2 call payoffs
    s2_payoffs = np.zeros_like(s2_vals)
    for k, K in enumerate(S2_strikes):
        s2_payoffs += s2_call_duals[k] * np.maximum(s2_vals - K, 0)
    
    # Forward payoff
    forward_payoff = mart_duals[i] * (s2_vals - s1 * (S2_forw / S1_forw))
    
    # Total Lagrangian for this S1
    lagrangian_3d[i, :] = s1_constant + s2_payoffs + forward_payoff + norm_dual

payoff_3d = payoff_flat.reshape((len(s1_vals), len(s2_vals)))
difference_upper = lagrangian_3d - payoff_3d

# heatmap
significant_diff_upper = np.where(np.abs(difference_upper) >= 1e-6, difference_upper, np.nan)
original_cmap = plt.cm.viridis.copy()
original_cmap.set_bad(color='white')
plt.figure(figsize=(12, 8))
im = plt.imshow(significant_diff_upper.T, aspect='auto', origin='lower', 
                extent=[s1_vals[0], s1_vals[-1], s2_vals[0], s2_vals[-1]],
                cmap=original_cmap)
plt.colorbar(im, label='Lagrangian - Payoff')
plt.xlabel('S1')
plt.ylabel('S2')
plt.title('Lagrangian in Upper Dual minus Payoff\n Exact zeros in white (support points)')
plt.show()

"""OUTPUT 6 - Cross-sections of the Lagrangian for Superhedging"""
# select out some cross sections from the heatmap
s1_indices = [20,54,85] #Good for VolSwap payoff -> [29,65,85]
fig, axes = plt.subplots(len(s1_indices), 1, figsize=(10, 8), sharex=True)
for idx, ax in zip(s1_indices, axes):
    s1 = s1_vals[idx]
    lagrangian = lagrangian_3d[idx, :]
    payoff = payoff_3d[idx, :]
    ax.plot(s2_vals, payoff, label="Payoff", linewidth=2)
    ax.plot(s2_vals, lagrangian, label="Lagrangian", linewidth=2)
    cond = pmf_opt[idx, :]
    s2_support = s2_vals[cond > 1e-8]
    ax.scatter(s2_support, payoff[cond > 1e-8], color='red', s=50, zorder=5, label="Support points")
    ax.set_title(f"S1 = {s1:.2f}")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
plt.xlabel("S2")
plt.tight_layout()
plt.show()
