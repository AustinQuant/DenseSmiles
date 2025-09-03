import src.black_lib as bs_lib
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import matplotlib.cm as cm
import mosek.fusion as mf
from matplotlib.colors import Normalize

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

S1_strikes=S1_strikes[7:]
S1_market_vol=S1_market_vol[7:]
S2_strikes=S2_strikes[7:]
S2_market_vol=S2_market_vol[7:]
V_strikes=V_strikes[1:9]
V_market_vol=V_market_vol[1:9]

"""Market Prices of those options"""
C1 = np.array([bs_lib.black_scholes_call(S1_forw, K, T1, vol) for K,vol in zip(S1_strikes, S1_market_vol)])
C2 = np.array([bs_lib.black_scholes_call(S2_forw, K, T2, vol) for K,vol in zip(S2_strikes, S2_market_vol)])
CV= np.array([bs_lib.black_scholes_call(V_forw, K, T_V, vol) for K,vol in zip(V_strikes, V_market_vol)])

"""L func for dispersion"""
def L(r): return -2.0 / tau * np.log(r)

"""pmf grid"""
grid_size =50
s1_vals = np.linspace(0.8*S1_forw, 1.1*S1_forw, grid_size)
s2_vals = np.linspace(0.8*S2_forw, 1.1*S2_forw, grid_size)
v_vals = np.linspace(0.5*V_forw, 1.5*V_forw, grid_size)
S1, S2, V = np.meshgrid(s1_vals, s2_vals, v_vals, indexing='ij')
num_vars = grid_size ** 3

"""payoff for use in objective function"""
payoff_grid =  np.abs(S2-S1)
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
            data.append(s2_vals[j]/S2_forw - s1/S1_forw)
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
            data.append(L(r*(S1_forw/S2_forw)) - v**2)
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

pmf_min = x.level().reshape(grid_size, grid_size, grid_size)
pmf_max = x_max.level().reshape(grid_size, grid_size, grid_size)

# Tests on constraint violations
martingale_error = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for l in range(grid_size):
        martingale_error[i, l] = np.sum((s2_vals - s1_vals[i]*(S2_forw/S1_forw)) * pmf_min[i, :, l])

dispersion_error = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for l in range(grid_size):
        s1 = s1_vals[i]
        v = v_vals[l]
        dispersion_error[i, l] = np.sum((L(np.maximum((s2_vals/S2_forw) / (s1/S1_forw), 1e-8)) - v**2) * pmf_min[i, :, l])

assert np.max(np.abs(dispersion_error)) < 1e-8, "Dispersion constraint violated"
assert np.max(np.abs(martingale_error)) < 1e-8, "Martingale constraint violated"

"""OUTPUT 1 - marginal PMF for (S1, S2) by summing over the VIX axis (axis=2)"""
pmf_s1_s2_marginal = pmf_max.sum(axis=2)
s1_marginal_sums = pmf_s1_s2_marginal.sum(axis=1, keepdims=True)
pmf_s2_cond_on_s1 = np.divide(pmf_s1_s2_marginal, s1_marginal_sums,
                              out=np.zeros_like(pmf_s1_s2_marginal),
                              where=s1_marginal_sums > 1e-12)
plt.figure(figsize=(10, 8))
masked = np.ma.masked_where(pmf_s2_cond_on_s1.T <= 1e-12, pmf_s2_cond_on_s1.T)
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='white')
plt.imshow(masked, aspect='auto', origin='lower',
           extent=[s1_vals[0], s1_vals[-1], s2_vals[0], s2_vals[-1]],
           cmap=cmap)
plt.colorbar(label="P(S2 | S1) [VIX Integrated]")
plt.xlabel("S1")
plt.ylabel("S2")
plt.title("VIX-Integrated Conditional Distribution P(S2 | S1) (Maximal Price PMF)")
plt.show()

"""OUTPUT 2: Point cloud of extremal solution"""
pmf_to_plot = pmf_max
tol = 1e-8 
voxel_grid = pmf_to_plot > tol
colors = np.zeros(voxel_grid.shape + (4,))
probs = pmf_to_plot[voxel_grid]
norm = Normalize(vmin=probs.min(), vmax=probs.max())
cmap = plt.cm.viridis
colors[voxel_grid] = cmap(norm(probs))
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')
ax.voxels(voxel_grid, facecolors=colors, edgecolor='k', linewidth=0.2, alpha=0.7)
ax.set_xticks(np.linspace(0, grid_size, num=5))
ax.set_xticklabels([f"{val:.0f}" for val in np.linspace(s1_vals[0], s1_vals[-1], num=5)])
ax.set_yticks(np.linspace(0, grid_size, num=5))
ax.set_yticklabels([f"{val:.0f}" for val in np.linspace(s2_vals[0], s2_vals[-1], num=5)])
ax.set_zticks(np.linspace(0, grid_size, num=5))
ax.set_zticklabels([f"{val:.2%}" for val in np.linspace(v_vals[0], v_vals[-1], num=5)])
ax.set_xlabel('S1 Value')
ax.set_ylabel('S2 Value')
ax.set_zlabel('VIX Level')
ax.set_title('Maximal PMF for the Straddle')
mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(mappable, shrink=0.5, aspect=10, ax=ax)
cbar.set_label('Probability Mass')
plt.show()


"""OUTPUT 3 - The upper/lower VIX smile consistent with SPX and VIX options"""
vix_strikes_fine = np.linspace(V_strikes[0], V_strikes[-1], 25)
prices_min_smile = []
prices_max_smile = []

# Loop through each strike to plot upper/lower smile
for K in vix_strikes_fine:
    vix_call_payoff_grid = np.maximum(V - K, 0)
    vix_call_payoff_flat = vix_call_payoff_grid.flatten()

    M_min_smile = mf.Model("min_vix_call")
    x_min_smile = M_min_smile.variable("x", num_vars, mf.Domain.greaterThan(0.0))
    M_min_smile.constraint("eqs", mf.Expr.mul(A_mosek, x_min_smile), mf.Domain.equalsTo(b_eq))
    M_min_smile.objective("obj", mf.ObjectiveSense.Minimize, mf.Expr.dot(vix_call_payoff_flat, x_min_smile))
    M_min_smile.solve()
    prices_min_smile.append(M_min_smile.primalObjValue())
    
    M_max_smile = mf.Model("max_vix_call")
    x_max_smile = M_max_smile.variable("x", num_vars, mf.Domain.greaterThan(0.0))
    M_max_smile.constraint("eqs", mf.Expr.mul(A_mosek, x_max_smile), mf.Domain.equalsTo(b_eq))
    M_max_smile.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.dot(vix_call_payoff_flat, x_max_smile))
    M_max_smile.solve()
    prices_max_smile.append(M_max_smile.primalObjValue())

prices_min_smile = np.array(prices_min_smile)
prices_max_smile = np.array(prices_max_smile)

iv_min = np.array([bs_lib.implied_vol(V_forw, K, T_V, p) for p, K in zip(prices_min_smile, vix_strikes_fine)])
iv_max = np.array([bs_lib.implied_vol(V_forw, K, T_V, p) for p, K in zip(prices_max_smile, vix_strikes_fine)])
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('Jointly Calibrated Upper and Lower VIX Smile incuding No-Arbitrage Region', fontsize=16)
axs[0].plot(vix_strikes_fine, iv_max, color='red', lw=1.5, label='Upper VIX Smile')
axs[0].plot(vix_strikes_fine, iv_min, color='blue', lw=1.5, label='Lower VIX Smile')
axs[0].fill_between(vix_strikes_fine, iv_min, iv_max, color='gray', alpha=0.5, label='No-Arbitrage Region')
axs[0].plot(V_strikes, V_market_vol, 'go', markersize=8, label='Market VIX Smile (Constraints)')
axs[0].set_ylabel('Implied Volatility')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].set_title('Implied VIX Smile Bounds')
iv_spread_bps = (iv_max - iv_min) * 10000
axs[1].plot(vix_strikes_fine, iv_spread_bps, color='black', lw=2)
axs[1].fill_between(vix_strikes_fine, iv_spread_bps, color='gray', alpha=0.5)
axs[1].set_xlabel('Strike')
axs[1].set_ylabel('Spread (bps)')
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].set_title('Width of No-Arbitrage Region (Upper Smile - Lower Smile)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
