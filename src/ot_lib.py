from scipy.interpolate import CubicSpline
import numpy as np
from scipy.optimize import bisect

def sinkhorn(sinkgrid,sinkweights,mu_X,mu_Y,mu_Z,iterations):
    """Accepts a Gaussian Quadrature grid, the three density functions for calibration
        and the desired number of iterations"""  
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
