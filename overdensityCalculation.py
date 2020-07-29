import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from astropy.cosmology import FlatLambdaCDM
from scipy import integrate
import astropy.units as u

plt.rcParams["figure.figsize"] = [4.2,4]
plt.rcParams['font.size'] = (9)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)


cosmo = FlatLambdaCDM(67, 0.3)
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff7f00','#f781bf','#a65628','#98ff98']

def z6_LF(M):
  #z=6 Finkelstein+15
  Mstar=-21.13
  alpha=-2.02
  psi_0=1.86e-4
  return 0.4*np.log(10)*psi_0 * 10**(-0.4*(M-Mstar)*(alpha+1))*np.exp(-10**(-0.4*(M-Mstar)))

def calc_abs_mag(obs_mag, z):
    return obs_mag - cosmo.distmod(z).value + 2.5*np.log10(1+z)

def calc_obs_mag(abs_mag, z):
    return abs_mag + cosmo.distmod(z).value - 2.5*np.log10(1+z)

def ref25_CIILF(ax):
  #From Decarli+17 fig 3b, webplotdigitizer
  rad=np.array([1.482e-1,3.686e+1, 9.167e+0, 2.169e+0,6.341e-1]) #Mpc
  area_CII=rad**3
  n=np.array([2.754e-4,1.784e+1,1.096e+0, 6.146e-2,5.248e-3])

  NCII=np.mean(n/(area_CII))

  ax.plot(rad,n,'-')
  overdensity_garcia(NCII,rad,ax)
  return


def overdensity_garcia(N_Mpc,r_Mpc,ax):
  #Decarli use r0=12.61
  over= N_Mpc*4*np.pi*r_Mpc*(r_Mpc**2/3+8.83/0.7)*redshift_depth
  over_low= N_Mpc*4*np.pi*r_Mpc*(r_Mpc**2/3+(8.83-1.51)/0.7)*redshift_depth
  over_high= N_Mpc*4*np.pi*r_Mpc*(r_Mpc**2/3+(8.83+1.39)/0.7)*redshift_depth
  ax.fill_between(r_Mpc,over_low,over_high,color=colors[3],alpha=0.5)
  ax.plot(r_Mpc,over,'--',label='F15, Garcia-Vergara et al. (2017) $z\simeq4$ quasar--galaxy clustering',color=colors[3])
  return


def overdensity_qiu(N_Mpc,r_Mpc,ax):
  #Yisheng has r0=5.3/h Mpc, gamma=1.6, integral solved with Wolfram
  over= N_Mpc*4*np.pi * 0.714286 *r_Mpc**3 *(0.466667+((5.3/0.7)/r_Mpc)**1.6)*redshift_depth
  over_low= N_Mpc*4*np.pi * 0.714286 *r_Mpc**3 *(0.466667+(((5.3-2.6)/0.7)/r_Mpc)**1.6)*redshift_depth
  over_high= N_Mpc*4*np.pi * 0.714286 *r_Mpc**3 *(0.466667+(((5.3+2.3)/0.7)/r_Mpc)**1.6)*redshift_depth
  ax.fill_between(r_Mpc,over_low,over_high,color=colors[1],alpha=0.5)
  ax.plot(r_Mpc,over,'-.',label='F15, Qiu et al. (2018) $z=5.9$ galaxy--galaxy clustering',color=colors[1])
  #companion_radii=np.sort(np.array([17.6,15.9,16.0,8.4,15.7,19.4,14.8,16.7,17.3]))
  return


def plot_Decarli(ax):
  #Decarli2017 plot 3b
  x=np.array([0.042,0.064,0.320,0.434])
  y=np.array([0.04,0.08,0.12,0.16])
  y_low=np.array([0.007,0.028,0.055,0.084])
  y_up=np.array([0.131,0.185,0.236,0.282])

  y_low_err=y-y_low
  y_up_err=y_up-y

  ax.errorbar(x,y,yerr=[y_low_err,y_up_err],color=[0.6,0.6,0.6],marker='o',
  linestyle='',label=r'Decarli et al. (2017), [CII]-selected companions at $z\geq6$')
  return


def plot_overdensity():
  fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios':[3.5,1]})

  #radius, area of circle for our observations
  radius=np.linspace(0.5,80) #arcsec
  radius_Mpc=radius/3600/Mpc_to_deg
  area=np.pi*radius**2
  area_Mpc=np.pi*radius_Mpc**2

  plotline1, caplines1, barlinecols1= ax[0].errorbar(3.2/3600/Mpc_to_deg,9/6,
  yerr=[[(9-1)/6],[(20-9)/6]],
  #yerr=6/6,
  marker='o',color='k',capsize=3,
  label='Our quasar fields',markersize=8,markerfacecolor=colors[0])
  caplines1[0].set_marker('v')
  caplines1[0].set_markersize(4)

  #ax[0].plot(3.2/3600/Mpc_to_deg,9/6,marker='o',color='k',label='Our quasar fields',markersize=8,markerfacecolor=colors[0],zorder=100,linestyle='')

  ax[0].plot(radius_Mpc,N_Mpc*redshift_depth*area_Mpc,'-',label='Finkelstein et al. (2015; F15), random sightline',color=colors[4])
  overdensity_garcia(N_Mpc,radius_Mpc,ax[0])
  overdensity_qiu(N_Mpc,radius_Mpc,ax[0])
  #ref25_CIILF(ax[0])

  #plt.plot(companion_radii*(1+6)/1000,np.array([1,2,3,4,5,6,7,8,9])/6,'rs',label='Our Quasar Fields')
  plot_Decarli(ax[0])

  ax[0].set_xlabel('Projected distance (co-moving Mpc)')
  ax[0].set_ylabel(r'Average number of $z=6$ galaxies in field')
  ax[0].set_xlim(0.5/3600/Mpc_to_deg,80/3600/Mpc_to_deg)
  ax[0].set_ylim(10**-3.5,1e2)
  ax[0].set_xscale('log')
  ax[0].set_yscale('log')

  ax[1].axis('off')
  ax[0].legend(loc=(-0.13,-0.6),fontsize='small')
  plt.subplots_adjust(left=0.15)
  plt.savefig('plots/clustering.pdf')

def func(m):
  return 10**p(m)


###############################################################################
####ALL FOREGROUND GALAXIES, WINDHORST+11
print('ALL FOREGROUND GALAXIES')

#data=np.genfromtxt('./F125W_LF.csv', delimiter=',', names=True, missing_values=' ')
#m=data['M'][::-1]
#NdM=data['NdM'][::-1]*2 #*2 to convert from /0.5 dex units
#p = np.poly1d(np.polyfit(m, np.log10(NdM), 3)) #fit polynomial to data
#N=integrate.quad(func,13,26.5)[0]

#plt.plot(m,NdM)
#plt.plot(m,10**p(m))
#plt.plot(calc_obs_mag(M,6),psi)
#plt.yscale('log')
#plt.show()

#From Rogier's calculations
N=517852*0.72
print('Number per sq deg: ',N)
print('Number per sq arcsec: ',N/3600**2)
FOVarea=6.5*6.5#np.pi*3.2**2
print(6.5*6.5,np.pi*3.2**2)
lamb=N/(3600**2)*FOVarea
print('Number per FOV: ',lamb)
print('Number per 6 FOV: ',lamb*6)
print('Number per 6 FOV with 15% variance:',lamb*6*1.15)
##Binomial
#prob=1-lamb**0*np.exp(-1*lamb)/math.factorial(0)
#print('Prob of seeing at least 1 galaxy: ',prob)
#n=6
#k=6
#P=math.factorial(n)/(math.factorial(k)*math.factorial(n-k)) * prob**k *(1-prob)**(n-k)
#print('Prob of seeing at least 1 galaxy in {} of {} FOVs: {}'.format(k,n,P))

#Poisson prob of seeing 20
lamb*=6 #all FOVS
n=20
prob=lamb**n*np.exp(-1*lamb)/math.factorial(n)
print('Stdevs above mean of finding 20: ',(20-lamb)/np.sqrt(lamb))
print('Poisson prob of seeing 20: ',prob)
#prob=0
#for ii in range(0,n):
#  prob+=lamb**ii*np.exp(-1*lamb)/math.factorial(ii)
#print('Poisson prob of seeing 20 or more: ',1-prob)

lamb*=1.15
n=20
prob=lamb**n*np.exp(-1*lamb)/math.factorial(n)
print('Stdevs above mean of finding 20 with 15% variance: ',(20-lamb)/np.sqrt(lamb))
print('Poisson prob of seeing 20 with 15% variance: ',prob)
#prob=0
#for ii in range(0,n):
#  prob+=lamb**ii*np.exp(-1*lamb)/math.factorial(ii)
#print('Poisson prob of seeing 20 or more with 15% variance: ',1-prob)



###############################################################################
####Z=6 GALAXIES, Finkelstein+15

redshift_depth=(cosmo.comoving_distance(6.5)-cosmo.comoving_distance(5.5))
Mpc_to_deg=(1000/(cosmo.kpc_comoving_per_arcmin(6).value))/(60)
Mpc2_to_sqdeg=Mpc_to_deg**2
Mpc3_to_sqdeg=(redshift_depth/Mpc2_to_sqdeg).value #From delta z = 1.


print('Z=6 GALAXIES')

#m=np.linspace(-23,-20)
M=np.linspace(-23,-20)#calc_abs_mag(m,6)
psi=z6_LF(M)*Mpc3_to_sqdeg #to /sq deg



N_Mpc=integrate.quad(z6_LF, -23,-20)[0]
print('Number per Mpc^3: ',N_Mpc)

N=integrate.quad(z6_LF, -23,-20)[0]*Mpc3_to_sqdeg
print('Number per sq deg: ',N)
print('Number per sq arcsec: ',N/3600**2)

#plt.plot(radius,(N/3600**2)*area,label='Finkelstein et al. (2015), Random Sightline')
#plt.plot(radius,overdensity(N_Mpc,radius/3600*Mpc_to_deg)*Mpc3_to_sqdeg)
#plt.plot(3.2,9/6,'ko',label='Our Quasar Fields')
#plt.xlabel('Radius of FOV (arcsec)')
#print(3.2/3600/Mpc_to_deg)

plot_overdensity()


lamb=N/(3600**2)*FOVarea
print('Number per FOV: ',lamb)
print('Number per 6 FOVs: ',lamb*6)

##Binomial
#prob=1-lamb**0*np.exp(-1*lamb)/math.factorial(0)
#print('Prob of seeing at least 1 galaxy in a FOV: ',prob)
#n=6
#k=5
#P=math.factorial(n)/(math.factorial(k)*math.factorial(n-k)) * prob**k *(1-prob)**(n-k)
#print('Prob of seeing at least 1 galaxy in {} of {} FOVs: {}'.format(k,n,P))

#Poisson
lamb*=6 #FOVS
#prob=0
#for ii in range(0,9):
#  prob+=lamb**ii*np.exp(-1*lamb)/math.factorial(ii)
n=9
prob=lamb**n*np.exp(-1*lamb)/math.factorial(n)
print('Poisson prob of seeing 9: ',prob)

plt.show()
