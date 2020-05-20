import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(67, 0.3)
plt.rcParams["figure.figsize"] = [3.6,3.4]
plt.rcParams['font.size'] = (9)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

def calc_obs_mag(abs_mag, z):
    return abs_mag + cosmo.distmod(z).value - 2.5*np.log10(1+z)

def calc_abs_mag(obs_mag, z):
    return obs_mag - cosmo.distmod(z).value + 2.5*np.log10(1+z)

def calc_mag_to_mJy(mag):
  return  10**(-0.4*(np.array(mag)-8.90))*1e3

#Wang2010=np.genfromtxt('./Wang2010_detections.csv',delimiter=',', names=True)
#Wang2011=np.genfromtxt('./Wang2011_detections.csv',delimiter=',', names=True)

#LUV_2010=10**(-0.4*(Wang2010['m_1450']-8.90))*1e3 #mJy
#LUV_2011=10**(-0.4*(Wang2011['m_1450']-8.90))*1e3 #mJy

#M1450_2010=calc_abs_mag(Wang2010['m_1450'], Wang2010['z'])
#M1450_2011=calc_abs_mag(Wang2011['m_1450'], Wang2011['Redshift'])

#plt.plot(M1450_2010,Wang2010['S_250GHz_mJy']/LUV_2010,'gs')
#plt.plot(M1450_2011,Wang2011['S_250GHz_mJy']/LUV_2011,'gd')


#From Mira's UV mags and Wang+2011 & 2013
our_M1450=[-25.14,-23.89,-26.26,-26.47,-26.21,-25.73]
our_redshift=[6.13,5.78,5.72,5.89,6.04,5.85]

our_m1450=calc_obs_mag(np.array(our_M1450),np.array(our_redshift))
#our_m1450=[21.78,22.28,20.97,20.62,20.60]
#print(our_m1450)
our_SUV=[1.13,2.57,1.85,2.27,2.98,0.36]
err_SUV=[0.36,0.06,0.46,0.51,0.05,0.48]

our_UVflux=calc_mag_to_mJy(our_m1450) #mJy
#our_FIR_err=
#plt.yscale('log')
#plt.show()


##From Mira's data file
data=np.genfromtxt('./uv_ir_props.csv',delimiter=',', names=True)
m1450=calc_obs_mag(data['M1450'],data['z'])
UVflux=calc_mag_to_mJy(m1450) #mJy

limits=[]
detects=[]
for ii in range(0,len(data)):
  if data['E250'][ii]>data['S250'][ii]:
    limits.append(ii)
  else:
    detects.append(ii)


plt.errorbar(data['M1450'][limits],data['S250'][limits]/UVflux[limits],yerr=0.5*data['S250'][limits]/UVflux[limits],marker='o',linestyle='',uplims=True,color=[0.6,0.6,0.6],markerfacecolor='w',label=r'$z>5.6$ Quasars, Limits',markersize=5)
plt.errorbar(data['M1450'][detects],data['S250'][detects]/UVflux[detects],yerr=data['E250'][detects]/UVflux[detects],marker='o',linestyle='',color=[0.6,0.6,0.6],label=r'$z>5.6$ Quasars, Detections',markersize=5)#,markerfacecolor='w')
#plt.errorbar(data['M1450'],data['S250']/UVflux,yerr=data['E250']/UVflux,marker='o',linestyle='')

#All but 0005
plt.errorbar(our_M1450[:-1],our_SUV[:-1]/our_UVflux[:-1],yerr=err_SUV[:-1]/our_UVflux[:-1],color='m',marker='o',linestyle='',markersize=8,label='This Sample')
#0005
plt.errorbar(our_M1450[-1],our_SUV[-1]/our_UVflux[-1],yerr=0.5*our_SUV[-1]/our_UVflux[-1],color='m',marker='o',linestyle='',uplims=True,markersize=8)

plt.xlabel(r'$M_{1450\AA}$ (AB mag)')
plt.ylabel(r'$F_{250GHz}/F_{1\mu m}$')
plt.legend(loc='upper left', fontsize='small')
plt.yscale('log')
plt.tight_layout()
plt.savefig('FIRvsFUV.pdf')
plt.show()
