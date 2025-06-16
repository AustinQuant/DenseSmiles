#Martin's Optimisation code
# error should be the should be the sum of the squares of the SVI implied vol models 
# minus the market implied vols, and you should add these constrains into your SVI function

import numpy as np 
import scipy.optimize
result = scipy.optimize.minimize(Error, params,method='Nelder-Mead', tol=1E-14, options={"maxiter":2000})
params=result.x
print("Calibrated paramaters=",params)

#constraints to be added
rho=min(max(rho,-1.0),1.0);
b=min(b,2/(1+np.abs(rho)));
a=max(a,-b*sigma*np.sqrt(1-rho**2));