import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
from scipy.optimize import root
from matplotlib import rc
from scipy.interpolate import interp1d
rc('text', usetex=True)

cosmo = {'omega_m' : 0.3,
'omega_R' : 0,
'omega_k' : 1 - 0.3 - 0.7,
'omega_lambda' : 0.7,
'h' : 0.7,
'n' : 1,
'sigma_8' : 0.8,
'c' : 3e8,
'G' : 4.302e-9, #Mpc/Msun (km/s)^2
'mu' : 0.59,
'm_p' : 1.6726219e-27, #kg
'k_b' : 1.38064852e-23 #(m/s)^2 kg / K
}

###Question 1: Angular diameter distance, growth factor, variance & Press-Schechter
def scale(z):
    return (1+z)**-1

def Hubble(z):
    return 100*cosmo['h']*np.sqrt(cosmo['omega_m']*scale(z)**-3 + \
    cosmo['omega_R']*scale(z)**-4 + cosmo['omega_lambda'])

def dH(z):
    return cosmo['c']/Hubble(z) #kpc

def calc_ang_diam_dist(z):
    return integrate.quad(dH,0,z)[0]/(1+z) #kpc


if __name__=='__main__':
    	  ##Question 1: Angular diameter distance, growth factor, variance & Press-Schechter
        zs=np.linspace(0,8,50)
        da=np.zeros(len(zs))
        growth=np.zeros(len(zs))
        jj=0
        for ii in zs:
            da[jj]=calc_ang_diam_dist(ii)
            jj+=1

        plt.plot(zs,1/da*206265,label='Constant size')
        plt.plot(zs,1*(1+zs)**(-0.5)/da*206265,label='m=0.5')
        plt.plot(zs,1*(1+zs)**(-0.7)/da*206265,label='m=0.7')
        plt.plot(zs,1*(1+zs)**(-1)/da*206265,label='m=1.0')
        plt.plot(zs,1*(1+zs)**(-1.5)/da*206265,label='m=1.5')
        #plt.plot(zs,da/(1+zs))
        #plt.plot(zs,da*(1+zs)**(-1.5))
        #plt.plot(zs,1/(1+zs))
        #plt.plot(zs,(1+zs)**(-1.5))
        #plt.yscale('log')
        #plt.ylim([0,2e3])
        plt.legend()
        plt.ylabel('Angular extent (arcsec)')
        plt.xlabel('Redshift')
        plt.show()
