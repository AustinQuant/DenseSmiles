"""Code for implementing conversion between implied vol and price"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar


"""LIBRARY FUNCTIONS"""

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


"""LIBRARY TESTS"""
#Test that takes implied vol data for various strikes and applies call price and then inverts
def test_black_scholes_inversion():
    T=1/12
    X_strikes = np.array([1.0681,1.0791,1.0904,1.1014,1.1119])
    X_market_vols = np.array([0.0554,0.053115,0.0516,0.051435,0.0523])
    X_forw=1.0903

    Y_strikes = np.array([1.2456,1.2595,1.274,1.2883,1.3017])
    Y_market_vols = np.array([0.06055,0.058665,0.0573,0.057185,0.05765])
    Y_forw=1.2738

    Z_strikes = np.array([0.84386,0.84969,0.85585,0.86234,0.86875])
    Z_market_vols = np.array([0.03809,0.03725,0.037225,0.03825,0.03986])
    Z_forw=0.8559
    # EUR/USD
    for i in range(len(X_strikes)):
        price = black_scholes_call(X_forw, X_strikes[i], T, X_market_vols[i])
        iv = implied_vol(X_forw, X_strikes[i], T, price)
        assert abs(iv - X_market_vols[i]) < 1e-8, f"EUR/USD failed at {i}: {iv} vs {X_market_vols[i]}"
    # GBP/USD
    for i in range(len(Y_strikes)):
        price = black_scholes_call(Y_forw, Y_strikes[i], T, Y_market_vols[i])
        iv = implied_vol(Y_forw, Y_strikes[i], T, price)
        assert abs(iv - Y_market_vols[i]) < 1e-8, f"GBP/USD failed at {i}: {iv} vs {Y_market_vols[i]}"
    # EUR/GBP
    for i in range(len(Z_strikes)):
        price = black_scholes_call(Z_forw, Z_strikes[i], T, Z_market_vols[i])
        iv = implied_vol(Z_forw, Z_strikes[i], T, price)
        assert abs(iv - Z_market_vols[i]) < 1e-8, f"EUR/GBP failed at {i}: {iv} vs {Z_market_vols[i]}"

if __name__ == "__main__":
    print("--- Running Black Scholes tests ---")
    test_black_scholes_inversion()
    print("--- Tests Passed ---")