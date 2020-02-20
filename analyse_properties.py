##Load properties of quasar 'companions' from output residual fits file
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from beta_mag_plot import calc_mag_beta
import pandas as pd

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
markers=['o','s','d','<','p','v','h','D']

markers_nums=['$1$','$2$','$3$','$4$','$5$','$6$','$7$','$8$']
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff7f00','#f781bf','#a65628','#98ff98']



quasars = ['NDWFS-J1425+3254','SDSS-J0005-0006','SDSS-J0203+0012','CFHQS-J0033-0125','SDSS-J0129-0035','SDSS-J2054-0005']
n_quasars = len(quasars)

redshift={'NDWFS-J1425+3254':5.89,'SDSS-J0005-0006':5.85,'SDSS-J0203+0012':5.72,
'CFHQS-J0033-0125':6.13,'SDSS-J0129-0035':5.78,'SDSS-J2054-0005':6.04}
angular_scale={'NDWFS-J1425+3254':6.026,'SDSS-J0005-0006':6.049,'SDSS-J0203+0012':6.122,
'CFHQS-J0033-0125':5.895,'SDSS-J0129-0035':6.088,'SDSS-J2054-0005':5.922} #kpc/pixel.
#Assumes h=0.67, om_m=0.3, flat Universe in New Wright Cosmo Calc

wcs = WCS(fits.getheader('NDWFS-J1425+3254/sci_NDWFS-J1425+3254_J.fits.gz'))
pxscale = 3600*np.abs(np.linalg.eigvals(wcs.pixel_scale_matrix))

mag={}
mag_err={}
mag_H={}
mag_H_err={}
index={}
index_err={}
reff={}
reff_err={}
b_on_a={}
b_on_a_err={}
xy={}
xy_err={}
xy_sep={}
flag={}

abs_mag={}
err={} #Pretty sure this is error in beta
beta={}

quasar_xy={}
quasar_mag={}

load_properties()
print(flag)

calc_mag_beta(mag,mag_err,mag_H,mag_H_err,abs_mag,err,beta,xy,flag)

save_table()

fig,ax=plt.subplots(1,2,figsize=(10,4))
#Plot Jiang+20 data
plot_colour_mag_obs(ax[0])



for jj,quasar in enumerate(quasars):
  f = fits.open('{}/out_{}_J_residual.fits'.format(quasar,quasar))

  n_companions=len(f[0].header['*SER_ANG'])
  if n_companions>0:
    leg=0
    for ii in range(0,n_companions):
      if (abs_mag[quasar][ii]<-23) or (beta[quasar][ii]>0):
        edge_col=colors[jj]
        facecolor='w'
      else:
        edge_col=colors[jj]#'k'
        facecolor=colors[jj]


      if flag[quasar][ii]==0 or flag[quasar][ii]==2:
        if leg==0:
          #Plot size-mag
          ax[1].errorbar(-50,-50,xerr=mag_err[quasar][ii],
          yerr=reff_err[quasar][ii]*pxscale[0]*angular_scale[quasar],marker='o',
          linestyle='',label=quasar,markeredgecolor='k',color=colors[jj],ms=8)

          ax[1].errorbar(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],xerr=mag_err[quasar][ii],
          yerr=reff_err[quasar][ii]*pxscale[0]*angular_scale[quasar],marker='o',
          linestyle='',label='_nolabel_',markeredgecolor=edge_col,color=colors[jj],ms=14,markerfacecolor=facecolor,alpha=0.8)

          ax[1].plot(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],marker=markers_nums[ii],color='k',zorder=100,ms=7)
          leg=1
        else:
          #Plot colour-mag
          ax[1].errorbar(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],xerr=mag_err[quasar][ii],
          yerr=reff_err[quasar][ii]*pxscale[0]*angular_scale[quasar],marker='o',
          linestyle='',label='__nolabel__',markeredgecolor=edge_col,color=colors[jj],ms=14,markerfacecolor=facecolor,alpha=0.8)

          ax[1].plot(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],marker=markers_nums[ii],color='k',zorder=100,ms=7)

      if flag[quasar][ii]==0:
        ax[0].errorbar(mag[quasar][ii],mag[quasar][ii]-mag_H[quasar][ii],xerr=mag_err[quasar][ii],
        yerr=np.sqrt(mag_err[quasar][ii]**2+mag_H_err[quasar][ii]**2),marker='o',
        linestyle='',markeredgecolor=edge_col,color=colors[jj],ms=14,markerfacecolor=facecolor,alpha=0.8)

        ax[0].plot(mag[quasar][ii],mag[quasar][ii]-mag_H[quasar][ii],marker=markers_nums[ii],color='k',zorder=100,ms=7)

      elif flag[quasar][ii]==2:
        mag_tot=-2.5*np.log10(10**(-0.4*mag[quasar][ii])+10**(-0.4*mag[quasar][ii-1]))
        mag_tot_H=-2.5*np.log10(10**(-0.4*mag_H[quasar][ii])+10**(-0.4*mag_H[quasar][ii-1]))
        ax[0].errorbar(mag_tot,mag_tot-mag_tot_H,xerr=max(mag_err[quasar][ii],mag_err[quasar][ii-1]),
        yerr=np.sqrt(max(mag_err[quasar][ii],mag_err[quasar][ii-1])**2+max(mag_H_err[quasar][ii],mag_H_err[quasar][ii-1])),
        marker='o',linestyle='',markeredgecolor=edge_col,color=colors[jj],ms=14,markerfacecolor=facecolor,alpha=0.8)

        ax[0].plot(mag_tot,mag_tot-mag_tot_H,marker=markers_nums[ii],color='k',zorder=100,ms=7)

#Plot Kawamata+18 size-M relation. (from fit)
kaw_mass=np.array([-21,-18])
ax[1].plot(kaw_mass,10**(-0.4*0.46*(kaw_mass+21)+np.log10(0.94)),'--',color=colors[2],label='Kawamata et al. (2018)')

ax[0].set_ylabel('J-H')
ax[0].set_xlabel('J')
ax[1].set_ylabel("Radius (arcsec)")
ax[1].set_xlabel(r'$M_{1500}$')
ax[1].legend(fontsize='small',ncol=1,loc=(0.7,0.68))
ax[1].axis([-24,-18.2,0,4.5])
#ax[1].set_yscale('log')
plt.tight_layout()
plt.savefig('mag_colour_size.pdf')
plt.show()
