import numpy as np 
import math

NumWeights=100
[u,w]=np.polynomial.legendre.leggauss(NumWeights)

b=10;a=-b;
#transform standardized u-vals and weights for the interval [-1,1] to the interval [a,b] under consideration
x=.5*(b-a)*u+.5*(a+b);w_1=(b-a)/2*w
y=.5*(b-a)*u+.5*(a+b);w_2=(b-a)/2*w

f1=np.exp(-.5*x*x)/np.sqrt(2*math.pi);
I1=np.sum(tmp1)

print(I1)

x_=x.reshape((NumWeights,1))
y_=y.reshape((1,NumWeights))
f2=np.exp(-.5*(x_*x_+y_*y_))/(2*math.pi);

tmp1=f1*w_1
tmp2=f2*w_1*w_2

I2=tmp2.sum(axis=(0,1))
print(I2)