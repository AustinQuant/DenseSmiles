"""Code for implementing the Sinkhorn Algorithm for optimal transport"""

from scipy.interpolate import CubicSpline
import numpy as np
from scipy.optimize import bisect
import src.svi_lib as svi_lib

"""Sinkhorn Calibration function"""
def sinkhorn(sinkgrid,sinkweights,mu_X,mu_Y,mu_Z,iterations):
    """Input: Accepts a Gaussian Quadrature grid, the three density functions for calibration
        and the desired number of iterations
       OUTPUT: The three spline functions needed for the gibbs ansatz for FX density 
        """  
    u_x= np.zeros(len(sinkgrid))
    v_y= np.zeros(len(sinkgrid))
    w_z= np.zeros(len(sinkgrid))
    x_points=sinkgrid
    y_points=sinkgrid
    z_points = np.linspace(np.min(x_points)/np.max(y_points), np.max(x_points)/np.min(y_points), len(sinkgrid))
    for it in range(iterations):
        w_spline= CubicSpline(z_points, w_z)
        #u update
        for i in range(u_x.shape[0]):
            yw_z = y_points*w_spline(x_points[i]/y_points)
            intx=np.exp(np.clip(v_y+yw_z,-700,700))*mu_Y(y_points)
            u_x[i] = -np.log(np.sum(intx * sinkweights)) 
        #v update
        for j in range(v_y.shape[0]):
            yw_z = y_points[j]*w_spline(x_points/y_points[j])
            inty=np.exp(np.clip(u_x+yw_z,-700,700))*mu_X(x_points)
            v_y[j] = -np.log(np.sum(inty * sinkweights))
        #w update using bisection for w
        v_spline = CubicSpline(y_points, v_y)
        for k in range(len(z_points)):
            zk = z_points[k]
            def root_func(wz):
                x_over_z = x_points / zk
                exp_term = u_x + v_spline(x_over_z) + x_over_z * wz
                exp_term_clipped = np.clip(exp_term, -700, 700)
                intz = np.exp(exp_term_clipped) * (x_points**2 / zk**3) * mu_X(x_points) * mu_Y(x_points/zk)
                return np.sum(intz * sinkweights) - mu_Z(zk)
            w_z[k] = bisect(root_func, -5000, 100,rtol=1e-4)
    return u_x, v_y, w_z

"""Sinkhorn density grid function from market data"""
def sink_density(Xmesh, Ymesh,T,X_strikes_lognrm,X_market_vols,Y_strikes_lognrm,Y_market_vols,Z_strikes_lognrm,Z_market_vols):
    Xopt = svi_lib.svi_fit(X_strikes_lognrm,X_market_vols,[0.0001, 0.002, 0.0014, 0, 0.005],T)
    Yopt = svi_lib.svi_fit(Y_strikes_lognrm,Y_market_vols,[0.0001, 0.002, 0.0014, 0, 0.005],T)
    Zopt = svi_lib.svi_fit(Z_strikes_lognrm,Z_market_vols,[0.00001, 0.0005, 0.001, 0, 0.0],T)

    def mu_X(xvar):
        return svi_lib.SVIDensity(xvar,Xopt.x)
    def mu_Y(yvar):
        return svi_lib.SVIDensity(yvar,Yopt.x)
    def mu_Z(zvar):
        return svi_lib.SVIDensity(zvar,Zopt.x)
    def opt_mu(x,y, u_spline, v_spline, w_spline):
        return np.exp(u_spline(x)+v_spline(y)+y*w_spline(x/y))*mu_X(x)*mu_Y(y)
    
    #Quadrature for use in sinkhorn calibration
    sinkPoints = 400
    [u_s, w_s] = np.polynomial.legendre.leggauss(sinkPoints)
    a_s = 0.8
    b_s = 1.2
    sinkgrid = 0.5*(b_s-a_s)*u_s + 0.5*(a_s+b_s)
    sinkweights = 0.5*(b_s-a_s)*w_s
    z_points = np.linspace(np.min(sinkgrid)/np.max(sinkgrid), np.max(sinkgrid)/np.min(sinkgrid), len(sinkgrid))

    opt_u, opt_v, opt_w = sinkhorn(sinkgrid,sinkweights,mu_X,mu_Y,mu_Z,30)
    u_spline = CubicSpline(sinkgrid, opt_u)
    v_spline = CubicSpline(sinkgrid, opt_v)
    w_spline = CubicSpline(z_points, opt_w)
    return opt_mu(Xmesh, Ymesh, u_spline, v_spline, w_spline)

print()