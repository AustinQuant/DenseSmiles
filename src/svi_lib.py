"""Code for handling implied vol data, fitting SVI smiles, transforming the smiles into probability densities"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
from scipy.optimize import minimize
import math

def black_scholes_call(F, K, T, sigma):
    """Black-Scholes price for a call option given forward F, strike K, expiry T, and implied vol sigma."""
    """F is forward price; K is the raw strike; T is maturity; sigma is the volatility"""
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = F * norm.cdf(d1) - K * norm.cdf(d2)
    return price

def implied_vol(F, K, T, price):
    """Computes the Black-Scholes implied volatility given the call price"""
    def obj(sigma):
        return black_scholes_call(F, K, T, sigma) - price
    result = root_scalar(obj, bracket=[-5, 5.0], method='brentq')
    return result.root if result.converged else np.nan

def svi_var(k_lognrm,params):
    """
    Implements Gatheral's SVI parameterisation of the volatility smile
    
    Inputs are:
    k: array of strikes (LOG-NORMALISED)
    and following params
    +++++++
    a: base volatility
    b: slope in wings
    sigma: curvature of minimum
    rho: skew
    m: location of minimum volatility
    """
    a, b, sigma, rho, m = params[0], params[1], params[2], params[3], params[4]
    
    #No arbitrage conditions on the parameters
    rho=min(max(rho,-1.0),1.0)
    b=min(b,2/(1+np.abs(rho)))
    a=max(a,-b*sigma*np.sqrt(1-rho**2))
    
    x=k_lognrm
    return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))

def svi_vol(k_lognrm,params,T):
    """SVI parameterisation of the implied volatility
    
    Ensure that strikes are log-normalised!
    """
    return np.sqrt(svi_var(k_lognrm,params)/T)

def svi_rss(k_lognrm,params,obs_v,T):
    """A sum of squares loss function used to fit the SVI parameters in svi_fit"""
    return np.sum((svi_vol(k_lognrm,params,T)-obs_v)**2)

def svi_fit(k_lognrm,obs_vars,bestp,T):
    """Uses an initial guess, svi_rss loss function and scipy optimiser to fit an svi smile to implied volatility data"""
    params_init = bestp  # initial guess for [a, b, sigma, rho, m] {"a": 0.05125, "b": 0.09, "rho": -0.3, "m": 1.0975, "sigma": 0.001}
    
    def objective(params):
        return svi_rss(k_lognrm,params,obs_vars,T)
    
    result = minimize(
        fun=objective, 
        x0=params_init,
        method='Nelder-Mead', 
        tol=1E-14, 
        options={"maxiter":2000}
    )
    
    return result

def SVIDensity(K_norm,params):
    """Closed form expression for the density function associated to an SVI smile,
        derived using Breeden-Litzenberger Rule
        
        NB: Strike input is to be normalised by the forward price but NOT in log-space
        """

    a=params[0];b=params[1];sigma=params[2];rho=params[3];m=params[4];
    k=np.log(K_norm)
    V=a + b*(rho*(k - m) + np.sqrt((k - m)**2 + sigma**2))
    V1=b*(rho + (k-m)/np.sqrt((m - k)**2 + sigma**2))
    V2=b*sigma**2/(m**2 - 2*m*k + k**2 + sigma**2)**1.5
    tmp=-np.exp(-(4*k**2 + V**2)/(8*V))
    tmp2=-4*k**2*V1**2 +4*V*V1*(4*k +V1) +  V**2*(V1**2 -8*(2 + V2))
    tmp3=16*K_norm**1.5*np.sqrt(2*math.pi)*V**2.5
    return tmp*tmp2/tmp3