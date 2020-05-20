import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit
from astropy import wcs
from astropy.wcs import WCS
from astropy.io import fits
import pandas as pd
from astropy import units as u
from astropy.coordinates import Angle

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['font.size'] = (9)
plt.rcParams["figure.figsize"] = [8,5]
markers_nums=['$1$','$2$','$3$','$4$','$5$','$6$','$7$','$8$']
markers=['o','s','d','<','p','v','h','D']
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff7f00','#f781bf','#a65628','#98ff98']

cosmo = FlatLambdaCDM(67, 0.3)
lams = np.array((1.25, 1.55))
quasars = ['NDWFS-J1425+3254','SDSS-J0005-0006','SDSS-J0203+0012','CFHQS-J0033-0125','SDSS-J0129-0035','SDSS-J2054-0005']
n_quasars = len(quasars)
redshift={'NDWFS-J1425+3254':5.89,'SDSS-J0005-0006':5.85,'SDSS-J0203+0012':5.72,
'CFHQS-J0033-0125':6.13,'SDSS-J0129-0035':5.78,'SDSS-J2054-0005':6.04}
angular_scale={'NDWFS-J1425+3254':6.026,'SDSS-J0005-0006':6.049,'SDSS-J0203+0012':6.122,
'CFHQS-J0033-0125':5.895,'SDSS-J0129-0035':6.088,'SDSS-J2054-0005':5.922} #kpc/''
#Assumes h=0.67, om_m=0.3, flat Universe in New Wright Cosmo Calc

wcs1425 = WCS(fits.getheader('NDWFS-J1425+3254/sci_NDWFS-J1425+3254_J.fits.gz'))
pxscale = 3600*np.abs(np.linalg.eigvals(wcs1425.pixel_scale_matrix))


def calc_ra_dec(file,position):
    # Load the FITS hdulist using astropy.io.fits, from https://docs.astropy.org/en/stable/wcs/
    w = WCS(file.header)
    #w.wcs.print_contents()
    pixcrd = np.array([position], dtype=np.float64)
    #world = w.wcs_pix2world(pixcrd, 0)
    rd=wcs.utils.pixel_to_skycoord(*position, w)
    #pixcrd2 = w.wcs_world2pix(world, 0)
    #assert np.max(np.abs(pixcrd - pixcrd2)) < 1e-6
    #x = 0
    #y = 0
    #origin = 0
    #assert (w.wcs_pix2world(x, y, origin) ==
    #        w.wcs_pix2world(x + 1, y + 1, origin + 1))
    #print(world)
    print(rd)
    return rd



def mag_vs_lam(lam, beta, mag0):
    return -2.5*(beta+2)*np.log10(lam) + mag0

def calc_beta(mags,mags_err,lam): #Exact solution
  bb=-0.4*(mags[0]-mags[1])/(np.log10(lams[0])-np.log10(lams[1]))-2
  bb_err=np.abs(-0.4/(np.log10(lams[0])-np.log10(lams[1])))*np.sqrt(mags_err[0]**2+mags_err[1]**2)
  return bb,bb_err

def calc_m0(mags,mags_err,lams,beta,beta_err): #Exact solution
  mm=mags[0]+2.5*(beta+2)*np.log10(lams[0])
  mm_err=np.sqrt(mags_err[0]**2+(2.5*np.log10(lams[0])*beta_err)**2)
  return mm,mm_err


def calc_abs_mag(obs_mag, z):
    return obs_mag - cosmo.distmod(z).value + 2.5*np.log10(1+z)
# Test with Linhua's object 44
# mags = np.array([25.10, 25.12, 24.99])
# magerrs = np.array([0.05, 0.05, 0.06])
# lams = np.array([1.0, 1.25, 1.55])
# parL, covL = curve_fit(mag_vs_lam, xdata=lams, ydata=mags, sigma=magerrs)
#
# z = 6.543
# abs_m1500 = abs_mag(mag_vs_lam((1+z)*0.15, *parL), z)


def plot_beta_mag_obs(ax):
    # Data from Linhua's paper
    jiang = np.genfromtxt('./beta-mag.txt', names=True, missing_values='--')

    ax.errorbar(jiang['m1500'], jiang['beta'],  yerr=jiang['betaerr'],
                color='gray', ls='none', marker='o', capthick=0,markerfacecolor='w',
                label='Jiang et al. 2013')

    #Jiang 2020
    mag_jiang = np.array([-21.66,-21.42,-20.95,-20.81,-20.72,-20.58,-20.15])

    beta_jiang = np.array([-2.89,-2.77,-2.57,-1.41,-3.39,-2.57,-3.38])
    beta_jiang_err = np.array([0.16,0.15,0.20,0.30,0.41,0.20,0.35])

    ax.errorbar(mag_jiang, beta_jiang,  yerr=beta_jiang_err,
                color='gray', ls='none', marker='x', capthick=0,
                label='Jiang et al. 2020')


    mag_bouw = [-21.5, -20.5, -19.5, -18.5, -17.5]
    beta_bouw = [-1.78, -2.08, -2.30, -2.30, -2.54]
    err_bouw = [0.14, 0.14, 0.14, 0.14, 0.14]
    ax.errorbar(mag_bouw, beta_bouw, yerr=err_bouw,
                markerfacecolor=[0.7,0.7,0.7], marker='h', capthick=0, ms=8,color=[0.2,0.2,0.2],ls='--',
                label='Bouwens et al. 2012')

    mag_dun = [-21.5, -20.5, -19.5, -18.5]
    beta_dun = [-2.07, -2.02, -2.00, -2.14]
    err_dun = [0.09, 0.10, 0.09, 0.16]
    ax.errorbar(mag_dun, beta_dun, yerr=err_dun,
                markerfacecolor=[0.7,0.7,0.7], marker='d', capthick=0, ms=8,color=[0.2,0.2,0.2],ls=':',
                label='Dunlop et al. 2012')

    mag_fink = [-20.5, -19.6, -18.1]
    beta_fink = [-2.05, -2.01, -2.32]
    err_fink = np.array([[0.11, 0.11], [0.13, 0.13], [0.15, 0.22]]).T
    ax.errorbar(mag_fink, beta_fink, yerr=err_fink,
                markerfacecolor=[0.7,0.7,0.7], marker='s', capthick=0, ms=7,color=[0.2,0.2,0.2],ls='-',
                label='Finkelstein et al. 2012')

    # Abs. Mag Uplter Limit
    x = -21.539723754
    y = -3.5
    #plt.plot(x,y,'cs', ms=10, label = 'z~6 Hosts (Avg)')

    #plt.arrow(x,y, 0.3, 0, fc='m', ec = 'm', head_width = 0.1, head_length = 0.1, lw = 1.5)
    #plt.axvline(x, ls='--', color='m', zorder = -10, lw = 1.5)
    ax.axvline(-20.79, ls='--', color='black', lw = 1.5, zorder = -10) #Finkelstein +16

    ax.set_xlabel('M$_{1500}$ (AB mag)')#, fontsize=12)
    ax.set_ylabel(r'UV Slope ($\beta$)')#, fontsize=12)
    ax.axis([-24, -17.3, -5.5, 2.2])


def plot_colour_mag_obs(axes):
  #Jiang 2013
  jiang = np.genfromtxt('./jiang_2013.txt', delimiter=',',names=True)
  axes.errorbar(jiang['F125W'], jiang['F125W']-jiang['F160W'], xerr=jiang['F125W_err'], yerr=np.sqrt(jiang['F125W_err']**2+jiang['F160W_err']**2),
    color='gray', ls='none', marker='o', capthick=0,markerfacecolor='w',
    label='Jiang et al. 2013')

  #Jiang 2020
  J_jiang = np.array([26.74,26.30,25.94,26.29,26.01,25.19,25.36])
  J_jiang_err = np.array([0.12,0.11,0.11,0.07,0.12,0.05,0.06])
  H_jiang = np.array([27.10,26.40,26.04,26.55,25.90,25.42,25.69])
  H_jiang_err = np.array([0.17,0.10,0.13,0.11,0.07,0.08,0.07])

  axes.errorbar(J_jiang, J_jiang-H_jiang, xerr=J_jiang_err, yerr=np.sqrt(J_jiang_err**2+H_jiang_err**2),
              color='gray', ls='none', marker='x', capthick=0,
              label='Jiang et al. 2020')


def calc_mag_beta(mag,mag_err,mag_H,mag_H_err,obs_UV_mag,abs_mag,abs_mag_err,beta,beta_err,xy,flag,ind_offset):

  for jj,quasar in enumerate(quasars):
    ##J-band
    print(quasar)
    f = fits.open('{}/out_{}_J_residual.fits'.format(quasar,quasar))

    n_companions=len(f[0].header['*SER_ANG'])
    if n_companions>0:
      mag[quasar]=np.zeros((n_companions))
      mag_err[quasar]=np.zeros((n_companions))
      abs_mag[quasar]=np.zeros((n_companions))
      obs_UV_mag[quasar]=np.zeros((n_companions))
      abs_mag_err[quasar]=np.zeros((n_companions))
      beta[quasar]=np.zeros((n_companions))
      beta_err[quasar]=np.zeros((n_companions))
      flag[quasar]=np.zeros((n_companions))
      ind_offset[quasar]=np.zeros((n_companions), dtype=int)
      xy[quasar]=np.zeros((2,n_companions))


      for ii in range(0,n_companions):
        mag[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[0])
        mag_err[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[-1])

        xy[quasar][:,ii]=eval(f[0].header['{}SER_XY'.format(ii+2)].split(' ')[0])
        #check for double sersics
        if ii>0:
          for tt in range(0,ii):
            if (np.abs(xy[quasar][0,ii]-xy[quasar][0,tt])<5) and (np.abs(xy[quasar][1,ii]-xy[quasar][1,tt])<5):
              flag[quasar][ii]=2
              flag[quasar][tt]=1
              ind_offset[quasar][tt:]+=1


      f = fits.open('{}/out_{}_H_residual.fits'.format(quasar,quasar))
      mag_H[quasar]=np.zeros((n_companions))
      mag_H_err[quasar]=np.zeros((n_companions))

      for ii in range(0,n_companions):
        mag_H[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[0])
        mag_H_err[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[-1])
        z=redshift[quasar]

        if flag[quasar][ii]==0:
          #par, cov = curve_fit(mag_vs_lam, xdata=lams, ydata=[mag[quasar][ii],mag_H[quasar][ii]])
          beta[quasar][ii],beta_err[quasar][ii]=calc_beta([mag[quasar][ii],mag_H[quasar][ii]],[mag_err[quasar][ii],mag_H_err[quasar][ii]],lams)
          m0,m0_err=calc_m0([mag[quasar][ii],mag_H[quasar][ii]],[mag_err[quasar][ii],mag_H_err[quasar][ii]],lams,beta[quasar][ii],beta_err[quasar][ii])

          obs_UV_mag[quasar][ii]=mag_vs_lam((1+z)*0.15, beta[quasar][ii],m0)
          abs_mag[quasar][ii] = calc_abs_mag(obs_UV_mag[quasar][ii], z)

          abs_mag_err[quasar][ii] = np.sqrt((2.5*np.log10((1+z)*0.15)*beta_err[quasar][ii])**2+m0_err**2)

        elif flag[quasar][ii]==2:
          #print(flag[quasar][ii-1])
          mag_tot=-2.5*np.log10(10**(-0.4*mag[quasar][ii])+10**(-0.4*mag[quasar][ii-1]))
          mag_tot_H=-2.5*np.log10(10**(-0.4*mag_H[quasar][ii])+10**(-0.4*mag_H[quasar][ii-1]))
          mag_tot_err = np.sqrt(mag_err[quasar][ii]**2+mag_err[quasar][ii-1]**2)
          mag_tot_H_err = np.sqrt(mag_H_err[quasar][ii]**2+mag_H_err[quasar][ii-1]**2)

          beta[quasar][ii],beta_err[quasar][ii]=calc_beta([mag_tot,mag_tot_H],[mag_tot_err,mag_tot_H_err],lams)
          m0,m0_err=calc_m0([mag_tot,mag_tot_H],[mag_tot_err,mag_tot_H_err],lams,beta[quasar][ii],beta_err[quasar][ii])

          obs_UV_mag[quasar][ii]=mag_vs_lam((1+z)*0.15, beta[quasar][ii],m0)
          abs_mag[quasar][ii] = calc_abs_mag(obs_UV_mag[quasar][ii], z)

          abs_mag_err[quasar][ii] = np.sqrt((2.5*np.log10((1+z)*0.15)*beta_err[quasar][ii])**2+m0_err**2)

        else:
          obs_UV_mag[quasar][ii]=np.nan
          abs_mag[quasar][ii]=np.nan
          abs_mag_err[quasar][ii]=np.nan
          beta[quasar][ii]=np.nan
          beta_err[quasar][ii]=np.nan



def load_properties():
  #Load properties of each galaxy around each quasar
  for jj,quasar in enumerate(quasars):
    ##J-band
    print(quasar)
    f = fits.open('{}/out_{}_J_residual.fits'.format(quasar,quasar))

    n_companions=len(f[0].header['*SER_ANG'])
    if n_companions>0:
      mag[quasar]=np.zeros((n_companions))
      mag_err[quasar]=np.zeros((n_companions))
      index[quasar]=np.zeros((n_companions))
      index_err[quasar]=np.zeros((n_companions))
      reff[quasar]=np.zeros((n_companions))
      reff_err[quasar]=np.zeros((n_companions))
      b_on_a[quasar]=np.zeros((n_companions))
      b_on_a_err[quasar]=np.zeros((n_companions))
      ra_dec[quasar]=[0 for ii in range(0,n_companions)]
      xy[quasar]=np.zeros((2,n_companions))
      xy_err[quasar]=np.zeros((2,n_companions))
      xy_sep[quasar]=np.zeros((2,n_companions))
      flag[quasar]=np.zeros((n_companions))
      ind_offset[quasar]=np.zeros((n_companions), dtype=int)

      quasar_xy[quasar]=eval(f[0].header['1PS_XY'].split(' ')[0])
      #print(calc_ra_dec(f[0],quasar_xy[quasar]).ra.to_string(unit=u.hour),calc_ra_dec(f[0],quasar_xy[quasar]).dec.to_string())
      quasar_mag[quasar]=float(f[0].header['1PS_MAG'].split(' ')[0])

      for ii in range(0,n_companions):
        mag[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[0])
        mag_err[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[-1])

        index[quasar][ii]=float(f[0].header['{}SER_N'.format(ii+2)].split(' ')[0])
        index_err[quasar][ii]=float(f[0].header['{}SER_N'.format(ii+2)].split(' ')[-1])

        reff[quasar][ii]=float(f[0].header['{}SER_RE'.format(ii+2)].split(' ')[0])
        reff_err[quasar][ii]=float(f[0].header['{}SER_RE'.format(ii+2)].split(' ')[-1])
        r_b=float(f[0].header['{}SER_REB'.format(ii+2)].split(' ')[0])
        r_b_err=float(f[0].header['{}SER_REB'.format(ii+2)].split(' ')[-1])
        b_on_a[quasar][ii]=r_b/reff[quasar][ii]
        b_on_a_err[quasar][ii]=np.sqrt((r_b_err/r_b)**2+(reff_err[quasar][ii]/reff[quasar][ii])**2)*b_on_a[quasar][ii]


        xy[quasar][:,ii]=eval(f[0].header['{}SER_XY'.format(ii+2)].split(' ')[0])
        #check for double sersics
        if ii>0:
          for tt in range(0,ii):
            if (np.abs(xy[quasar][0,ii]-xy[quasar][0,tt])<5) and (np.abs(xy[quasar][1,ii]-xy[quasar][1,tt])<5):
              flag[quasar][ii]=2
              flag[quasar][tt]=1
              ind_offset[quasar][tt:]+=1

        xy_err[quasar][:,ii]=eval(f[0].header['{}SER_XY'.format(ii+2)].split(' ')[-1])
        xy_sep[quasar][:,ii]=xy[quasar][:,ii]-quasar_xy[quasar]
        ra_dec[quasar][ii]= calc_ra_dec(f[0],xy_sep[quasar][:,ii])
      ##H-band
      f = fits.open('{}/out_{}_H_residual.fits'.format(quasar,quasar))
      mag_H[quasar]=np.zeros((n_companions))
      mag_H_err[quasar]=np.zeros((n_companions))
      for ii in range(0,n_companions):
        mag_H[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[0])
        mag_H_err[quasar][ii]=float(f[0].header['{}SER_MAG'.format(ii+2)].split(' ')[-1])


def plot_colour_mag(ax):
  for jj,quasar in enumerate(quasars):
    f = fits.open('{}/out_{}_J_residual.fits'.format(quasar,quasar))

    n_companions=len(f[0].header['*SER_ANG'])
    if n_companions>0:
      leg=0
      for ii in range(0,n_companions):
        if (abs_mag[quasar][ii]<-22.1) or (beta[quasar][ii]>0) or (beta[quasar][ii]<-4):
          edge_col=colors[jj]
          facecolor='w'
        else:
          edge_col=colors[jj]#'k'
          facecolor=colors[jj]


        if flag[quasar][ii]==0:
          ax.errorbar(mag[quasar][ii],mag[quasar][ii]-mag_H[quasar][ii],xerr=mag_err[quasar][ii],
          yerr=np.sqrt(mag_err[quasar][ii]**2+mag_H_err[quasar][ii]**2),marker='o',
          linestyle='',markeredgecolor=edge_col,color=colors[jj],ms=12,markerfacecolor=facecolor,alpha=alpha)

          if numbers:
            ax.plot(mag[quasar][ii],mag[quasar][ii]-mag_H[quasar][ii],marker=markers_nums[ii-ind_offset[quasar][ii]],color='k',zorder=100,ms=7)

        elif flag[quasar][ii]==2:
          mag_tot=-2.5*np.log10(10**(-0.4*mag[quasar][ii])+10**(-0.4*mag[quasar][ii-1]))
          mag_tot_H=-2.5*np.log10(10**(-0.4*mag_H[quasar][ii])+10**(-0.4*mag_H[quasar][ii-1]))
          ax.errorbar(mag_tot,mag_tot-mag_tot_H,xerr=max(mag_err[quasar][ii],mag_err[quasar][ii-1]),
          yerr=np.sqrt(max(mag_err[quasar][ii],mag_err[quasar][ii-1])**2+max(mag_H_err[quasar][ii],mag_H_err[quasar][ii-1])),
          marker='o',linestyle='',markeredgecolor=edge_col,color=colors[jj],ms=12,markerfacecolor=facecolor,alpha=alpha)

          if numbers:
            ax.plot(mag_tot,mag_tot-mag_tot_H,marker=markers_nums[ii-ind_offset[quasar][ii]],color='k',zorder=100,ms=7)
  ax.set_ylabel('$m_J-m_H$ (AB mag)')
  ax.set_xlabel('$m_J$ (AB mag)')


def plot_size_mag(ax):

  for jj,quasar in enumerate(quasars):
    f = fits.open('{}/out_{}_J_residual.fits'.format(quasar,quasar))

    n_companions=len(f[0].header['*SER_ANG'])
    if n_companions>0:
      leg=0
      for ii in range(0,n_companions):
        if (abs_mag[quasar][ii]<-22.1) or (beta[quasar][ii]>0) or (beta[quasar][ii]<-4):
          edge_col=colors[jj]
          facecolor='w'
        else:
          edge_col=colors[jj]#'k'
          facecolor=colors[jj]


        if flag[quasar][ii]==0 or flag[quasar][ii]==2:
          if leg==0:
            #Plot size-mag
            ax.errorbar(-50,-50,xerr=mag_err[quasar][ii],
            yerr=reff_err[quasar][ii]*pxscale[0]*angular_scale[quasar],marker='o',
            linestyle='',label=quasar,markeredgecolor=edge_col,color=colors[jj],ms=8)

            ax.errorbar(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],xerr=mag_err[quasar][ii],
            yerr=reff_err[quasar][ii]*pxscale[0]*angular_scale[quasar],marker='o',
            linestyle='',label='_nolabel_',markeredgecolor=edge_col,color=colors[jj],ms=12,markerfacecolor=facecolor,alpha=alpha)

            if numbers:
              ax.plot(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],marker=markers_nums[ii-ind_offset[quasar][ii]],color='k',zorder=100,ms=7)
            leg=1
          else:
            #Plot colour-mag
            ax.errorbar(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],xerr=mag_err[quasar][ii],
            yerr=reff_err[quasar][ii]*pxscale[0]*angular_scale[quasar],marker='o',
            linestyle='',label='__nolabel__',markeredgecolor=edge_col,color=colors[jj],ms=12,markerfacecolor=facecolor,alpha=alpha)

            if numbers:
              ax.plot(abs_mag[quasar][ii],reff[quasar][ii]*pxscale[0]*angular_scale[quasar],marker=markers_nums[ii-ind_offset[quasar][ii]],color='k',zorder=100,ms=7)

  ax.set_ylabel("Radius (kpc)")
  ax.set_xlabel(r'$M_{1500}$ (AB mag)')
  ax.legend(fontsize='small',ncol=1,loc=(0.56,0.75))
  ax.axis([-24,-18,0,4.5])


def plot_size_mag_obs(axes):
  # Data from Linhua's paper
  #shibuya = np.genfromtxt('./Shibuya_2015_Tab5.txt', names=True, missing_values='--')

  #axes.errorbar(shibuya['UVmag'], shibuya['Reff'],  yerr=shibuya['e_Reff'],
  #            color=[0.8,0.8,0.8], ls='none', marker='.', capthick=0,
  #            label='Shibuya et al. 2015')

  #Plot Shibuya 2015, relation for z~6 LBGs
  shibuya_muv=[-22,-21,-20,-19,-18]
  shibuya_size=[1.053,0.635,0.565,0.584,0.371]
  shibuya_up=[0.841,0.717,0.400,0.424,0.222]
  shibuya_low=[0.696,0.274,0.287,0.327,0.198]
  ax.errorbar(shibuya_muv,shibuya_size,yerr=[shibuya_low,shibuya_up],color=[0.3,0.3,0.3],label='Shibuya et al. (2015)',marker='o',linestyle='')

  #Plot Kawamata+18 size-M relation. (from fit)
  kaw_muv=np.array([-21,-18])
  ax.plot(kaw_muv,10**(-0.4*0.46*(kaw_muv+21)+np.log10(0.94)),'--',color=[0.5,0.5,0.5],label='Kawamata et al. (2018)')


def plot_beta_mag(ax):
  #plot beta vs abs mag
  for jj,quasar in enumerate(quasars):
    f = fits.open('{}/out_{}_J_residual.fits'.format(quasar,quasar))

    n_companions=len(f[0].header['*SER_ANG'])
    if n_companions>0:
      for ii in range(0,n_companions):

        if (abs_mag[quasar][ii]<-22.1) or (beta[quasar][ii]>0) or (beta[quasar][ii]<-4):
          edge_col=colors[jj]
          facecolor='w'
        else:
          edge_col=colors[jj]
          facecolor=colors[jj]

        if ii==0:
            ax.errorbar(-50, -50, yerr=beta_err[quasar][ii],xerr=abs_mag_err[quasar][ii],
                    color=colors[jj], ls='none', marker=markers[ii], capthick=0, ms=8,
                    markeredgecolor=edge_col,label=quasar,markerfacecolor=colors[jj])
            ax.errorbar(abs_mag[quasar][ii], beta[quasar][ii], yerr=beta_err[quasar][ii],xerr=abs_mag_err[quasar][ii],
                    color=colors[jj], ls='none', marker='o', capthick=0, ms=12,
                    markeredgecolor=edge_col,label='__nolabel__',markerfacecolor=facecolor,alpha=alpha)
            if numbers:
              ax.plot(abs_mag[quasar][ii], beta[quasar][ii],marker=markers_nums[ii-ind_offset[quasar][ii]],color='k',zorder=100,ms=7)
        else:
            ax.errorbar(abs_mag[quasar][ii], beta[quasar][ii], yerr=beta_err[quasar][ii],xerr=abs_mag_err[quasar][ii],
                    color=colors[jj], ls='none', marker='o', capthick=0, ms=12,
                    markeredgecolor=edge_col,label='__nolabel__',markerfacecolor=facecolor,alpha=alpha)
            if numbers:
              ax.plot(abs_mag[quasar][ii], beta[quasar][ii],marker=markers_nums[ii-ind_offset[quasar][ii]],color='k',zorder=100,ms=7)



def save_table():
  total_df=pd.DataFrame(columns=['Quasar','J','J-H','RA','Dec','Distance',r'$n$',r'$R_e$',r'$b/a$',r'$M_{1500}$',r'$\beta$'])

  for quasar in quasars:
    individual_df=pd.DataFrame(columns=['Quasar','J','J-H','RA','Dec','Distance',r'$n$',r'$R_e$',r'$b/a$',r'$M_{1500}$',r'$\beta$'])
    for ii in range(0,len(mag[quasar])):
      individual_df.loc[ii]=[quasar,
      r"${:.1f}\pm{:.1f}$".format(mag[quasar][ii],mag_err[quasar][ii]),
      r"${:.1f}\pm{:.1f}$".format(mag[quasar][ii]-mag_H[quasar][ii],mag_H_err[quasar][ii]),
      ra_dec[quasar][ii].ra.to_string(unit=u.hour,precision=2),
      ra_dec[quasar][ii].dec.to_string(precision=2),
      r"${:.1f}\pm{:.1f}$".format(np.sqrt(xy_sep[quasar][0,ii]**2+xy_sep[quasar][1,ii]**2)*pxscale[0]*angular_scale[quasar],
      np.sqrt(xy_err[quasar][0,ii]**2+xy_err[quasar][1,ii]**2)*pxscale[1]*angular_scale[quasar]),
      r"${:.1f}\pm{:.1f}$".format(index[quasar][ii],index_err[quasar][ii]),
      r"${:.1f}\pm{:.1f}$".format(reff[quasar][ii]*pxscale[0]*angular_scale[quasar],reff_err[quasar][ii]*pxscale[0]*angular_scale[quasar]),
      r"${:.2f}\pm{:.2f}$".format(b_on_a[quasar][ii],b_on_a_err[quasar][ii]),
      r"${:.1f}\pm{:.1f}$".format(abs_mag[quasar][ii],abs_mag_err[quasar][ii]),
      r"${:.1f}\pm{:.1f}$".format(beta[quasar][ii],beta_err[quasar][ii])]

    total_df=pd.concat([total_df,individual_df])
  with open('galaxy_properties.tex', 'w') as tf:
     tf.write(total_df.to_latex(escape=False))
  print(total_df)


if __name__=="__main__":

  # Calculate UV slope for each quasar from MCMC fits, plot against observations
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
  ra_dec={}
  flag={} #flag to recognise double sersic fits to single galaxy
  ind_offset={} #make correct label given some are flagged as doubles

  abs_mag={}
  abs_mag_err={}
  obs_UV_mag={}
  beta={}
  beta_err={}

  quasar_xy={}
  quasar_mag={}

  numbers=True #Plot numbers on companion symbols
  if numbers:
    alpha=1#0.8
  else:
    alpha=1

  load_properties()

  calc_mag_beta(mag,mag_err,mag_H,mag_H_err,obs_UV_mag,abs_mag,abs_mag_err,beta,beta_err,xy,flag,ind_offset)

  fig,ax=plt.subplots(1,3,figsize=(10,4),gridspec_kw={"width_ratios":[1,1,0.2],'wspace':0.3})
  ax[2].axis('off')
  plot_beta_mag_obs(ax[1])
  plot_beta_mag(ax[1])
  ax[1].legend(loc=(1.05,0.2),fontsize='small')

  plot_colour_mag_obs(ax[0])
  plot_colour_mag(ax[0])
  if numbers:
    plt.savefig('Plots/colour_mag.pdf')
  else:
    plt.savefig('Plots/colour_mag_nonumbers.pdf')
  plt.show()

  print(ra_dec)
  save_table()

  fig,ax=plt.subplots(1,1,figsize=(4,4))
  plot_size_mag_obs(ax)
  plot_size_mag(ax)

  #ax[1].set_yscale('log')
  #plt.tight_layout()
  if numbers:
    plt.savefig('Plots/size_mag.pdf')
  else:
    plt.savefig('Plots/size_mag_nonumbers.pdf')
  plt.show()

  numbers=False
  fig_b,ax_b=plt.subplots(1,2,figsize=(7,3),gridspec_kw={"width_ratios":[1,0.2],'wspace':0.3})
  ax_b[1].axis('off')
  plot_beta_mag_obs(ax_b[0])
  plot_beta_mag(ax_b[0])
  ax_b[0].legend(loc=(1.05,0.2),fontsize='small')
  plt.subplots_adjust(left=0.1, bottom=0.14, top=0.95)
  plt.savefig('Plots/beta_mag.pdf')
  plt.show()
