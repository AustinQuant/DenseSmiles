import numpy as np
from scipy.stats import lognorm
from scipy.sparse import coo_matrix
import mosek.fusion as mf
import matplotlib.pyplot as plt
import numpy.ma as ma

# Synthetic lognormal parameters
S0 = 5500
T1 = 20 / 251
T2 = 40 / 251
sigma = 0.2

grid_size = 400
s1_vals = np.linspace(0.8 * S0, 1.2 * S0, grid_size)
s2_vals = np.linspace(0.8 * S0, 1.2 * S0, grid_size)
nvar = grid_size ** 2

# Allows some slackness in fitting the martingale condition
tol = 0.00001

# Marginal density masses for S1 and S2 (lognormal)
ds1 = s1_vals[1] - s1_vals[0]
ds2 = s2_vals[1] - s2_vals[0]
s1_pdf_vals = lognorm.pdf(s1_vals, s=sigma * np.sqrt(T1), scale=S0)
s2_pdf_vals = lognorm.pdf(s2_vals, s=sigma * np.sqrt(T2), scale=S0)
s1_pmf = s1_pdf_vals * ds1
s2_pmf = s2_pdf_vals * ds2
s1_pmf /= s1_pmf.sum()
s2_pmf /= s2_pmf.sum()

# Constraints
rows, cols, data = [], [], []
b_eq = []
rowcount = 0

# S1 marginal constraints
for i, prob in enumerate(s1_pmf):
    for j in range(grid_size):
        idx = i * grid_size + j
        rows.append(rowcount)
        cols.append(idx)
        data.append(1.0)
    b_eq.append(prob)
    rowcount += 1

# S2 marginal constraints
for j, prob in enumerate(s2_pmf):
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

# Objective: log-contract payoff (variance swap)
S1_grid, S2_grid = np.meshgrid(s1_vals, s2_vals, indexing="ij")
payoff_flat =(np.log(S2_grid / S1_grid) ** 2).flatten()#(np.abs(S2_grid - S1_grid)).flatten()#

M_min = mf.Model("min_price")
x_min = M_min.variable("x", nvar, mf.Domain.greaterThan(0.0))
M_min.constraint("eqs", mf.Expr.mul(A_mosek, x_min), mf.Domain.inRange(b_lower, b_upper))
M_min.objective("obj", mf.ObjectiveSense.Minimize, mf.Expr.dot(payoff_flat, x_min))
M_min.solve()
min_price = M_min.primalObjValue()

M_max = mf.Model("max_price")
x_max = M_max.variable("x", nvar, mf.Domain.greaterThan(0.0))
M_max.constraint("eqs", mf.Expr.mul(A_mosek, x_max), mf.Domain.inRange(b_lower, b_upper))
M_max.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.dot(payoff_flat, x_max))
M_max.solve()
max_price = M_max.primalObjValue()

"""OUTPUT 1 - Price bounds"""
print("Min price:", min_price)
print("Max price:", max_price)

# Extract PMFs
pmf_opt = np.array(x_max.level()).reshape((grid_size, grid_size))
pmf_min = np.array(x_min.level()).reshape((grid_size, grid_size))
pmf_opt = np.clip(pmf_opt, 0, None)
pmf_min = np.clip(pmf_min, 0, None)

# Conditional probabilities
tol = 1e-12
row_sums_opt = pmf_opt.sum(axis=1, keepdims=True)
row_sums_min = pmf_min.sum(axis=1, keepdims=True)
pmf_opt_cond = np.divide(pmf_opt, row_sums_opt, out=np.zeros_like(pmf_opt), where=row_sums_opt > tol)
pmf_min_cond = np.divide(pmf_min, row_sums_min, out=np.zeros_like(pmf_min), where=row_sums_min > tol)

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
plt.title(f"Conditional distribution P(S2 | S1) for the minimal price PMF\n (Exact Zeros in white, Sigma = {sigma})")
plt.show()

"""OUTPUT 3 & 4 - Dual variables for the constraints (Superhedge & Subhedge)"""
dual_vars = M_max.getConstraint("eqs").dual().flatten()
dual_vars_min = M_min.getConstraint("eqs").dual().flatten()

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

# Martingale
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