import src.svi_lib as svi_lib
import numpy as np
import mosek.fusion as mf
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import numpy.ma as ma

T1 = 20/251
S1_forw = 5489.83
S1_strikes = np.linspace(4500,5800,27)
S1_market_vol=np.array([0.30833965626324, 0.295476337981542, 0.283463677875459, 0.271111343370089,
                        0.258451404309255, 0.245507644865208, 0.233694271876973, 0.221358004433688,
                        0.209097873714426, 0.197249412151294, 0.186334324110301, 0.174675684455109,
                        0.164666282229566, 0.15488794443707, 0.145823620475546, 0.137304198281544, 
                        0.129231575271196, 0.121548584918863, 0.114417742487412, 0.108019587373161,
                        0.102797547251224, 0.0988783071539835, 0.0959001836718852, 0.0934990753558453,
                         0.0918556359282457, 0.0918871975138243, 0.0934062296063838])

T2 = 40/251
S2_forw = 5509.62
S2_strikes = np.linspace(4500,5800,27)
S2_market_vol=np.array([0.256853, 0.247760, 0.239149, 0.230164,
                        0.221471, 0.213216, 0.204941, 0.196783, 
                        0.188797, 0.180978, 0.173596, 0.166311, 
                        0.159104, 0.152338, 0.145637, 0.139202, 
                        0.132974, 0.127012, 0.121383, 0.116176, 
                        0.111556, 0.108040, 0.105025, 0.102493, 
                        0.100738, 0.099360, 0.098370])

S1_strikes = S1_strikes[5:22]
S1_market_vol = S1_market_vol[5:22]
S2_strikes = S2_strikes[5:22]
S2_market_vol = S2_market_vol[5:22]
S1_norm= S1_strikes/S1_forw
S2_norm= S2_strikes/S2_forw

# Fit SVI and build marginal densities
s1opt = svi_lib.svi_fit(np.log(S1_norm), S1_market_vol, [0.01]*5, T1)
s2opt = svi_lib.svi_fit(np.log(S2_norm), S2_market_vol, [0.01]*5, T2)

def mu_s1(s):
    return svi_lib.SVIDensity(s,s1opt.x)

def mu_s2(s):
    return svi_lib.SVIDensity(s,s2opt.x)

# Discretisation grid in normalised space
grid_size = 200
s1_vals = np.linspace(0.9, 1.1, grid_size)
s2_vals = np.linspace(0.9, 1.1, grid_size)

ds1 = s1_vals[1] - s1_vals[0]
ds2 = s2_vals[1] - s2_vals[0]

nvar = grid_size**2
tol = 0.00001

# Discretised marginals
mu1_disc = np.array([mu_s1(s) for s in s1_vals]) * ds1
mu2_disc = np.array([mu_s2(s) for s in s2_vals]) * ds2
mu1_disc /= mu1_disc.sum()
mu2_disc /= mu2_disc.sum()
mu1_disc = np.clip(mu1_disc, 0, None)
mu2_disc = np.clip(mu2_disc, 0, None)

# Constraints
rows, cols, data = [], [], []
b_eq = []
rowcount = 0

# S1 marginal constraints
for i, prob in enumerate(mu1_disc):
    for j in range(grid_size):
        idx = i * grid_size + j
        rows.append(rowcount)
        cols.append(idx)
        data.append(1.0)
    b_eq.append(prob)
    rowcount += 1

# S2 marginal constraints
for j, prob in enumerate(mu2_disc):
    for i in range(grid_size):
        idx = i * grid_size + j
        rows.append(rowcount)
        cols.append(idx)
        data.append(1.0)
    b_eq.append(prob)
    rowcount += 1

# Martingale constraints
for i, s1 in enumerate(s1_vals):
    for j, s2 in enumerate(s2_vals):
        idx = i * grid_size + j
        rows.append(rowcount)
        cols.append(idx)
        data.append(s2 - s1)
    b_eq.append(0.0)
    rowcount += 1

A_sparse = coo_matrix((data, (rows, cols)), shape=(rowcount, nvar))
b_eq = np.array(b_eq)
b_lower = b_eq - tol
b_upper = b_eq + tol

A_mosek = mf.Matrix.sparse(
    A_sparse.shape[0],
    A_sparse.shape[1],
    A_sparse.row,
    A_sparse.col,
    A_sparse.data
)
S1_grid, S2_grid = np.meshgrid(s1_vals, s2_vals, indexing="ij")
payoff_flat =(np.log((S2_grid/S1_grid)*(S1_forw/S2_forw))**2).flatten()# np.abs(S2_grid*S2_forw-S1_grid*S1_forw).flatten()#

# Minimize price
M_min = mf.Model("min_price")
x_min = M_min.variable("x", nvar, mf.Domain.greaterThan(0.0))
M_min.constraint("eqs", mf.Expr.mul(A_mosek, x_min), mf.Domain.inRange(b_lower, b_upper))
M_min.objective("obj", mf.ObjectiveSense.Minimize, mf.Expr.dot(payoff_flat, x_min))
M_min.solve()
min_price = M_min.primalObjValue()

# Maximize price
M_max = mf.Model("max_price")
x_max = M_max.variable("x", nvar, mf.Domain.greaterThan(0.0))
M_max.constraint("eqs", mf.Expr.mul(A_mosek, x_max), mf.Domain.inRange(b_lower, b_upper))
M_max.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.dot(payoff_flat, x_max))
M_max.solve()
max_price = M_max.primalObjValue()

"""OUTPUT 1 - Price bounds"""
print("Min price:", min_price)
print("Max price:", max_price)
print("Size of Bounded Interval: ", (max_price-min_price))
print("Relative percentage: ", (max_price-min_price)/(0.5*(max_price+min_price)))

# Extract PMFs
pmf_opt = np.array(x_max.level()).reshape((grid_size, grid_size))
pmf_min = np.array(x_min.level()).reshape((grid_size, grid_size))
pmf_opt = np.clip(pmf_opt, 0, None)
pmf_min = np.clip(pmf_min, 0, None)

tol_cond = 1e-12
row_sums_opt = pmf_opt.sum(axis=1, keepdims=True)
row_sums_min = pmf_min.sum(axis=1, keepdims=True)
pmf_opt_cond = np.divide(pmf_opt, row_sums_opt, out=np.zeros_like(pmf_opt), where=row_sums_opt > tol_cond)
pmf_min_cond = np.divide(pmf_min, row_sums_min, out=np.zeros_like(pmf_min), where=row_sums_min > tol_cond)

"""OUTPUT 2 - Heatmaps of the conditional probabilities showing sparseness"""
plt.figure(figsize=(10, 8))
masked = ma.masked_where(pmf_opt_cond.T <= 1e-12, pmf_opt_cond.T)
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='white')
plt.imshow(masked, aspect='auto', origin='lower', extent=[s1_vals[0], s1_vals[-1], s2_vals[0], s2_vals[-1]], cmap=cmap)
plt.colorbar(label="P(S2 | S1)")
plt.xlabel("S1")
plt.ylabel("S2")
plt.title("Conditional distribution P(S2 | S1) for the maximal price PMF\n (Exact Zeros in white)")
plt.show()

plt.figure(figsize=(10, 8))
masked_min = ma.masked_where(pmf_min_cond.T <= 1e-12, pmf_min_cond.T)
cmap_min = plt.cm.viridis.copy()
cmap_min.set_bad(color='white')
plt.imshow(masked_min, aspect='auto', origin='lower', extent=[s1_vals[0], s1_vals[-1], s2_vals[0], s2_vals[-1]], cmap=cmap_min)
plt.colorbar(label="P(S2 | S1)")
plt.xlabel("S1")
plt.ylabel("S2")
plt.title(f"Conditional distribution P(S2 | S1) for the minimal price PMF\n (Exact Zeros in white)")
plt.show()

"""OUTPUT 3 & 4 - Dual variables for the constraints (Superhedge & Subhedge)"""
dual_vars = M_max.getConstraint("eqs").dual().flatten()
dual_vars_min = M_min.getConstraint("eqs").dual().flatten()

# S1 marginal duals, S2 marginal duals, martingale duals
s1_duals = dual_vars[:grid_size]
s2_duals = dual_vars[grid_size:2*grid_size]
mart_duals = dual_vars[2*grid_size:]
s1_duals_min = dual_vars_min[:grid_size]
s2_duals_min = dual_vars_min[grid_size:2*grid_size]
mart_duals_min = dual_vars_min[2*grid_size:]

# Superhedge
plt.figure(figsize=(8, 5))
plt.plot(s1_vals, s1_duals, label="S1 Marginal Dual", color="tab:blue")
plt.plot(s2_vals, s2_duals, label="S2 Marginal Dual", color="tab:green")
plt.title("Superhedge Marginals (Upper Bound)")
plt.xlabel("S1 / S2")
plt.ylabel("Dual Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Subhedge
plt.figure(figsize=(8, 5))
plt.plot(s1_vals, s1_duals_min, label="S1 Marginal Dual", color="tab:blue")
plt.plot(s2_vals, s2_duals_min, label="S2 Marginal Dual", color="tab:green")
plt.title("Subhedge Marginals (Lower Bound)")
plt.xlabel("S1 / S2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Martingale duals
plt.figure(figsize=(8, 5))
plt.plot(s1_vals, mart_duals, label="Martingale Dual (Superhedge)", color="tab:orange")
plt.plot(s1_vals, mart_duals_min, label="Martingale Dual (Subhedge)", color="tab:red", linestyle="--")
plt.title("Martingale Duals")
plt.xlabel("S1")
plt.ylabel("Dual Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""OUTPUT 5 - Plot of the full Lagrangian for upper bound showing duality via support points"""
lagrangian_3d = np.zeros((len(s1_vals), len(s2_vals)))
for i, s1 in enumerate(s1_vals):
    s1_constant = s1_duals[i]
    s2_contribution = s2_duals
    martingale_contribution = mart_duals[i] * (s2_vals - s1)
    lagrangian_3d[i, :] = s1_constant + s2_contribution + martingale_contribution

payoff_3d = payoff_flat.reshape((len(s1_vals), len(s2_vals)))
difference_upper = lagrangian_3d - payoff_3d

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
s1_indices = [int(grid_size*0.3), int(grid_size*0.5), int(grid_size*0.7)]
fig, axes = plt.subplots(len(s1_indices), 1, figsize=(10, 8), sharex=True)
for idx, ax in zip(s1_indices, axes):
    s1 = s1_vals[idx]
    lagrangian = lagrangian_3d[idx, :]
    payoff = payoff_3d[idx, :]
    cond = pmf_opt[idx, :]
    s2_support = s2_vals[cond > 1e-8]
    ax.plot(s2_vals, payoff, label="Payoff", linewidth=2)
    ax.plot(s2_vals, lagrangian, label="Lagrangian", linewidth=2)
    ax.scatter(s2_support, payoff[cond > 1e-8], color='red', s=50, zorder=5, label="Support points")
    ax.set_title(f"S1 = {s1:.2f}")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
plt.xlabel("S2")
plt.tight_layout()
plt.show()