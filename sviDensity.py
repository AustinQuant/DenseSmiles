def SVIDensity(K,params):
    a=params[0];b=params[1];sigma=params[2];rho=params[3];m=params[4];
    k=np.log(K)
    V=a + b*(rho*(k - m) + np.sqrt((k - m)**2 + sigma**2))
    V1=b*(rho + (k-m)/np.sqrt((m - k)**2 + sigma**2))
    V2=b*sigma**2/(m**2 - 2*m*k + k**2 + sigma**2)**1.5
    tmp=-np.exp(-(4*k**2 + V**2)/(8*V))
    tmp2=-4*k**2*V1**2 +4*V*V1*(4*k +V1) +  V**2*(V1**2 -8*(2 + V2))
    tmp3=16*K**1.5*np.sqrt(2*math.pi)*V**2.5
    return tmp*tmp2/tmp3