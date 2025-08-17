"""Demonstration of LP bounds for the FX calibration problem"""
import src.ot_lib as ot_lib
import src.black_lib as bs_lib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from tabulate import tabulate

"""CALIBRATION DATA"""
T=1/12 #Maturity for FX data
X_strikes = np.array([1.0681,1.0791,1.0904,1.1014,1.1119]) #EUR/USD strikes
X_market_vols = np.array([0.0554,0.053115,0.0516,0.051435,0.0523]) #EUR/USD implied vol
X_forw=1.0903 #EUR/USD forward rate
X_norm=X_strikes/X_forw
X_strikes_lognrm = np.log(X_norm)
X_market_prices=np.zeros(len(X_strikes))

Y_strikes = np.array([1.2456,1.2595,1.274,1.2883,1.3017]) #GBP/USD strikes
Y_market_vols = np.array([0.06055,0.058665,0.0573,0.057185,0.05765]) #GBP/USD implied vol
Y_forw=1.2738 #GBP/USD forward rate
Y_norm=Y_strikes/Y_forw
Y_strikes_lognrm = np.log(Y_norm)
Y_market_prices=np.zeros(len(Y_strikes))

Z_strikes = np.array([0.84386,0.84969,0.85585,0.86234,0.86875]) #EUR/GBP strikes
Z_market_vols = np.array([0.03809,0.03725,0.037225,0.03825,0.03986]) #EUR/GBP implied vol
Z_forw=0.8559 #EUR/GBP forward rate
Z_norm=Z_strikes/Z_forw
Z_strikes_lognrm = np.log(Z_norm)
Z_market_prices=np.zeros(len(Z_strikes))

for i in range(len(X_strikes)): #Calculation of Market Prices from vol
    X_market_prices[i]=bs_lib.black_scholes_call(X_forw, X_strikes[i], T, X_market_vols[i])
    Y_market_prices[i]=bs_lib.black_scholes_call(Y_forw, Y_strikes[i], T, Y_market_vols[i])
    Z_market_prices[i]=bs_lib.black_scholes_call(Z_forw, Z_strikes[i], T, Z_market_vols[i])

"""PMF grid for LP"""
grid_size = 50
x_vals = np.linspace(0.8, 1.2, grid_size)
y_vals = np.linspace(0.8, 1.2, grid_size)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')


"""CONSTRAINT MATRICES for Primal LP"""
#Normalisation constraint of PMF
A_norm=np.ones((1,grid_size**2))
b_norm=np.array([1.0])

#Flattened version of X call option price constraints
x_price = X_market_prices / X_forw
A_X = np.zeros((len(X_norm), grid_size**2))
for k, K_x in enumerate(X_norm):
    for i in range(grid_size):
        for j in range(grid_size):
            A_X[k, i * grid_size + j] = max(x_vals[i] - K_x, 0)

#Flattened version of Y call option price constraints
y_price = Y_market_prices / Y_forw
A_Y = np.zeros((len(Y_norm), grid_size**2))
for k, K_y in enumerate(Y_norm):
    for i in range(grid_size):
        for j in range(grid_size):
            A_Y[k, i * grid_size + j] = max(y_vals[j] - K_y, 0)

#Flattened version of Y call option price constraints
z_price = Z_market_prices / Z_forw
A_Z = np.zeros((len(Z_norm), grid_size**2))
for k, K_z in enumerate(Z_norm):
    for i in range(grid_size):
        for j in range(grid_size):
            A_Z[k, i * grid_size + j] = max(x_vals[i] - K_z * y_vals[j], 0)

#Flattened constraint that E[X]=1, E[Y]=1 (forward normalisation of X&Y)
row_fX = np.zeros(grid_size**2)
row_fY = np.zeros(grid_size**2)
for i in range(grid_size):
    for j in range(grid_size):
        idx = i * grid_size + j
        row_fX[idx] = x_vals[i]
        row_fY[idx] = y_vals[j]
A_forwards = []
b_forwards = []
#E[X]=1
A_forwards.append(row_fX)
b_forwards.append(1.0)
#E[Y]=1
A_forwards.append(row_fY)
b_forwards.append(1.0)

A_forwards = np.vstack(A_forwards)      
b_forwards = np.array(b_forwards)

"""Stacked A_eq for primal problem"""
A_eq = np.vstack([A_X, A_Y, A_Z, A_norm, A_forwards])
b_eq = np.concatenate([x_price, y_price, z_price, b_norm, b_forwards])
bounds = [(0, None)] * (grid_size**2)

"""GENERIC PAYOFF FUNCTION - EDIT AS NEEDED"""
def payoff_gen(X,Y,K=None):
    #return (X-Y)**2 #Quadratic payoff
    #return np.maximum(np.maximum(X,Y)-K,0) #Best-Of Call payoff
    #return np.maximum(X/Y - K,0)  #Quanto call payoff
    #return np.maximum(0.5*(X+Y)-K,0) #basket option payoff
    return np.maximum(0.5*(X+Y)-K,0)

K=1
payoff_grid = payoff_gen(X, Y, K)
payoff_flat = payoff_grid.flatten()



"""Primal Problem Solver"""
#primal upper
res_min = linprog(c=payoff_flat, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
min_price = res_min.fun if res_min.success else np.nan

#primal lower
res_max = linprog(c=-payoff_flat, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
max_price = -res_max.fun if res_max.success else np.nan

"""Sinkhorn density (for comparison against LP bounds)"""
#Quadrature for sinkhorn pricing
quadPoints = 400
[u, w] = np.polynomial.legendre.leggauss(quadPoints)
a = 0.9
b = 1.1
quadgrid = 0.5*(b-a)*u + 0.5*(a+b)
quadweights = 0.5*(b-a)*w
Xsink, Ysink = np.meshgrid(quadgrid, quadgrid)
#sinkhorn pricing density
sinkhorn_density = ot_lib.sink_density(Xsink,Ysink,T,X_strikes_lognrm,X_market_vols,Y_strikes_lognrm,Y_market_vols,Z_strikes_lognrm,Z_market_vols)
#sinkhorn price for comparison
weights_2d = quadweights[:, np.newaxis] * quadweights[np.newaxis, :]
sinkhorn_price = np.sum(payoff_gen(Xsink, Ysink,K) * sinkhorn_density * weights_2d)

"""Dual Problem Solver"""
# upper dual
res_dual_super = linprog(
    c=b_eq,
    A_ub=-A_eq.T,
    b_ub=-payoff_flat,
    bounds=[(None, None)] * A_eq.shape[0],
    method='highs'
)
hedge_price_super = res_dual_super.fun
weights_super = res_dual_super.x
#lower dual
res_dual_sub = linprog(
    c=-b_eq,
    A_ub=A_eq.T,
    b_ub=payoff_flat,
    bounds=[(None, None)] * A_eq.shape[0],
    method='highs'
)
hedge_price_sub = -res_dual_sub.fun
weights_sub = res_dual_sub.x

# duality gap
upper_duality_gap = max_price - hedge_price_super
lower_duality_gap = min_price-hedge_price_sub  

"""OUTPUT 1 - The Price Bounds"""
table = [
    ["Upper Bound", max_price],
    ["Sinkhorn Price", sinkhorn_price],
    ["Lower Bound", min_price],
    ["Upper Duality Gap", upper_duality_gap],
    ["Lower Duality Gap", lower_duality_gap]
]
print(tabulate(table, headers=["Quantity", "Value"], floatfmt=".12f"))

"""OUTPUT 2 - The Superhedging Portfolio Composition"""
n_X = len(X_norm)
n_Y = len(Y_norm)
n_Z = len(Z_norm)

weights_X_super = weights_super[:n_X]
weights_Y_super = weights_super[n_X:n_X + n_Y]
weights_Z_super = weights_super[n_X + n_Y:n_X + n_Y + n_Z]
weight_fX_super = weights_super[-2]
weight_fY_super = weights_super[-1]

fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

#to scale y axis
max_weight = max(
    np.max(np.abs(weights_X_super)),
    np.max(np.abs(weights_Y_super)),
    np.max(np.abs(weights_Z_super))
)

# Plot X vanilla calls
axs[0].bar(np.arange(len(weights_X_super)), weights_X_super / max_weight, width=0.1)
axs[0].set_title(f"X vanilla calls\nForward weight: {weight_fX_super:.4f}")
axs[0].set_xticks(np.arange(len(X_norm)))
axs[0].set_xticklabels([f"{k:.3f}" for k in X_norm], rotation=45)
axs[0].set_xlabel("Strike")
axs[0].set_ylabel("Normalised Weight")

# Plot Y vanilla calls
axs[1].bar(np.arange(len(weights_Y_super)), weights_Y_super / max_weight, width=0.1, color='orange')
axs[1].set_title(f"Y vanilla calls\nForward weight: {weight_fY_super:.4f}")
axs[1].set_xticks(np.arange(len(Y_norm)))
axs[1].set_xticklabels([f"{k:.3f}" for k in Y_norm], rotation=45)
axs[1].set_xlabel("Strike")

# Plot Z vanilla calls
axs[2].bar(np.arange(len(weights_Z_super)), weights_Z_super / max_weight, width=0.1, color='green')
axs[2].set_title("Z vanilla calls")
axs[2].set_xticks(np.arange(len(Z_norm)))
axs[2].set_xticklabels([f"{k:.3f}" for k in Z_norm], rotation=45)
axs[2].set_xlabel("Strike")

plt.tight_layout()
plt.show(block=True)

"""OUTPUT 3 - Heatmap of Performance of the Superhedge"""

#payoff of the superhedge
superhedge_grid = np.zeros_like(payoff_grid)

for idx, K_x in enumerate(X_norm):
    superhedge_grid += weights_X_super[idx] * np.maximum(X - K_x, 0)

for idx, K_y in enumerate(Y_norm):
    superhedge_grid += weights_Y_super[idx] * np.maximum(Y - K_y, 0)

for idx, K_z in enumerate(Z_norm):
    superhedge_grid += weights_Z_super[idx] * np.maximum(X - K_z * Y, 0)

superhedge_grid += weight_fX_super * (X - 1.0)
superhedge_grid += weight_fY_super * (Y - 1.0)

diff_super = superhedge_grid - payoff_grid

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

im0 = axs[0].imshow(payoff_grid, extent=[Y.min(), Y.max(), X.min(), X.max()],
                    origin='lower', aspect='equal', cmap='viridis')
axs[0].set_title("Payoff function")
axs[0].set_xlabel("Y")
axs[0].set_ylabel("X")
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(diff_super, extent=[Y.min(), Y.max(), X.min(), X.max()],origin='lower', aspect='equal', cmap='coolwarm')
axs[1].set_title("Superhedge minus Payoff")
axs[1].set_xlabel("Y")
axs[1].set_ylabel("X")
fig.colorbar(im1, ax=axs[1])

plt.tight_layout()
plt.show()