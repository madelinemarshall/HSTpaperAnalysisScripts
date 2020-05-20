##Load properties of quasar 'companions' from output residual fits file
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import pandas as pd
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from matplotlib.colors import SymLogNorm
from scipy.special import gamma, gammainc, gammaincinv
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Sersic_parameter_investigation import find_mag_distribution
from MUV_mass_conversion import upper_mass_limit
from scipy.stats import binned_statistic

plt.rcParams["figure.figsize"] = [10,2]
plt.rcParams['font.size'] = (9)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
markers=['o','s','d','<','x','>']
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff7f00','#f781bf','#a65628','#98ff98']
cosmo = FlatLambdaCDM(67, 0.3)

#mag_zeropoint: Magnitude zeropoint, i.e. the magnitude of one
#            ADU, whether in electrons per second (as with published HST
#            zeropoints) or whatever funky units the data use.

def plot_literature_observations(ax):
  #Kormendy & Ho (2013):
  #MBH/10^9=(0.49\pm0.6)(Mbulge/10^11)^(1.17\pm0.08), intrinsic scatter 0.28 dex (p571)
  x=[10**9,10**12]
  logMBH=np.log10(0.49)+1.17*np.log10(np.array(x)/10**11)+9
  ax.errorbar(np.log10(x),logMBH,yerr=0.28,label='Kormendy \& Ho (2013)\n- Fit',capsize=3,linewidth=2.5, color='k',zorder=0)
  ax.set_xlim([9,12])
  ax.set_xlabel(r'$M_\ast/M_\odot$')
  ax.set_ylabel(r'$M_{\mathrm{BH}}/M_\odot$')

  #Willot+17
  Mdyn=[10.10,10.11,10.64,10.62,10.55,10.59,10.60,10.78,10.76,10.83,10.86,10.86,
  10.98,11.05,11.13,11.17,11.16,11.31,11.36,10.64,11.08]
  MBH=[7.90,8.23,8.08,8.38,8.98,9.38,10.09,9.70,9.28,8.97,8.93,9.32,9.45,9.04,9.43,9.23,9.17,8.91,8.69,7.70,8.39]
  MBH_err=0.45
  ax.errorbar(Mdyn,MBH,yerr=MBH_err,
    marker='o',color=[0.7,0.7,0.7],linestyle='',
    linewidth=0.8,markersize=5,label='Willott et al. 2017')

  #Izumi+18a
  Mdyn=np.array([5.6,1.4,8.2,4.4])*1e10
  MBH=np.array([0.2,4.3,6.1,0.4])*1e8
  MBH_up=np.array([0.2,7.8,6.1,0.7])*1e8
  MBH_low=np.array([0.1,2.8,3.0,0.3])*1e8
  logMBH_low=np.log10(MBH)-np.log10(MBH-MBH_low)
  logMBH_up=np.log10(MBH_up+MBH)-np.log10(MBH)

  ax.errorbar(np.log10(Mdyn),np.log10(MBH),yerr=[logMBH_low,logMBH_up],
    marker='p',color=[0.7,0.7,0.7],linestyle='',
    linewidth=0.8,markersize=5,label='Izumi et al. 2018')


  #Izumi+19, Using MgII
  Mdyn=np.array([1.3e10,29e10])
  MBH=np.array([7.1e8,11e8])
  MBH_low=np.array([2.4e8,3e8])
  MBH_up=np.array([5.2e8,2e8])
  logMBH_low=np.log10(MBH)-np.log10(MBH-MBH_low)
  logMBH_up=np.log10(MBH_up+MBH)-np.log10(MBH)

  ax.errorbar(np.log10(Mdyn),np.log10(MBH),yerr=[logMBH_low,logMBH_up],
    marker='s',color=[0.7,0.7,0.7],linestyle='',
    linewidth=0.8,markersize=5,label='Izumi et al. 2019')

  Mdyn=2.0e10
  MBH=1.1e8
  ax.errorbar(np.log10(Mdyn),np.log10(MBH),yerr=0.25,lolims=True,
    marker='s',color=[0.7,0.7,0.7],linestyle='',
    linewidth=0.8,markersize=5,label='__nolabel__')

  ##Pensabene+20
  Mdyn=np.array([0.26,6.0,3.7,1.8,3.3,0.7,16.9])*1e10
  Mdyn_up=np.array([0.03,16,3.2,0.9,0.9,4.5,1.4])*1e10
  Mdyn_low=np.array([0.02,3.0,0.3,0.4,0.9,0.3,1.1])*1e10

  MBH=np.array([8e8,3.3e8,5.6e9,6.3e8,2.0e9,0.9e9,2.7e8])
  MBH_up=np.array([4e8,1.0e8,0.5e9,1.5e8,0.6e9,1.6e9,1.4e8])
  MBH_low=np.array([4e8,1.0e8,0.5e9,1.5e8,0.6e9,0.6e9,1.4e8])

  logMBH_low=np.log10(MBH)-np.log10(MBH-MBH_low)
  logMBH_up=np.log10(MBH_up+MBH)-np.log10(MBH)

  logMdyn_low=np.log10(Mdyn)-np.log10(Mdyn-Mdyn_low)
  logMdyn_up=np.log10(Mdyn_up+Mdyn)-np.log10(Mdyn)

  ax.errorbar(np.log10(Mdyn),np.log10(MBH),yerr=[logMBH_low,logMBH_up],
  xerr=[logMdyn_low,logMdyn_up],marker='d',color=[0.7,0.7,0.7],linestyle='',
  linewidth=0.8,markersize=5,label='Pensabene et al. 2020')

  #2054
  ax.errorbar(np.log10(Mdyn)[5],np.log10(MBH)[5],yerr=[[logMBH_low[5]],[logMBH_up[5]]],
  xerr=[[logMdyn_low[5]],[logMdyn_up[5]]],marker='o',color='k',markerfacecolor=colors[5],
  linewidth=1,markersize=6,capsize=3,label=r'$M_{\mathrm{dyn}}$ measurement')

  #0129
  ax.errorbar(np.log10(0.78e11),np.log10(1.7e8),yerr=0.45,xerr=0.1,
  xlolims=True,color='k',marker='o',markerfacecolor=colors[4],capsize=3,
  linewidth=1,markersize=6,zorder=200)#,label='Pensabene et al. 2020')

  #1425
  ax.errorbar(np.log10(1.56e11),9.41,yerr=0.11,xerr=0.1,
  xlolims=True,color='k',marker='o',markerfacecolor=colors[0],capsize=3,
  linewidth=1,markersize=6,zorder=200)#,label='Wang et al. 2010')

def calc_sersic_mag(flux_in_annulus,r_inner,r_outer,nn,Re,zeropoint):
  bn=gammaincinv(2*nn,0.5)

  gamma_diff=(gammainc(2*nn,bn*r_outer/Re)-gammainc(2*nn,bn*r_inner/Re))/gamma(2*nn)

  fe = flux_in_annulus/(2*np.pi*nn) * bn**(2*nn)/(Re**2*np.exp(bn)) / gamma_diff

  flux_in_r_outer=2*np.pi*nn*fe*Re**2*np.exp(bn)/bn**(2*nn)*gammainc(2*nn,bn*r_outer/Re)/gamma(2*nn)
  #gamma_2nx=gammainc(2*nn,bn*(radius/Re)**(1/nn))/gamma(2*nn)

  #mag = SB - 2.5*bn/np.log(10)*((radius/Re)**(1/nn)-1) - 5*np.log10(Re) -2.5*np.log10(2*np.pi*nn*np.exp(bn)/bn**(2*nn) * gamma_2nx)
  mag = -2.5*np.log10(flux_in_r_outer)+zeropoint
  #print(mag)
  return mag


def calc_phot(band,mag,SB,mag_sersic,quasar_mags):
  #Load properties of each galaxy around each quasar
  fig,axes=plt.subplots(1,int(np.ceil(n_quasars/1))+1,gridspec_kw={'hspace':0.25,'wspace':0.1,'width_ratios':[1,1,1,1,1,1,0.1]})#,sharex=True,sharey=True)
  ax=axes.flat
  for jj,quasar in enumerate(quasars):
    ##J-band
    if test:
      f = fits.open('{}/out_{}_{}_point_source_subtracted.fits'.format(quasar,quasar,band))
      quasar_loc=np.array(eval(f[0].header['2Ser_XY'].split(' ')[0]))
    else:
      f = fits.open('{}/out_{}_{}_residual.fits'.format(quasar,quasar,band))
      quasar_loc=np.array(eval(f[0].header['1PS_XY'].split(' ')[0]))
    counts=f[0].data

    ivm = fits.open('{}/out_{}_{}_composite_ivm.fits'.format(quasar,quasar,band))
    noise=np.sqrt(1/ivm[0].data)

    #_, _, noise_sma = sigma_clipped_stats(counts, sigma=3.0) #1 std
    #fig2,ax2=plt.subplots()
    #im=ax2.imshow(noise/noise_sma,cmap='coolwarm')#,norm=SymLogNorm(3,vmin=-10**2.5,vmax=10**2.5))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    #plt.colorbar(im)

    sky=float(f[0].header['0SKY_ADU'].split(' ')[0])

    quasar_loc[0]+=shift[quasar][0]
    quasar_loc[1]+=shift[quasar][1]

    if test and quasar in ['SDSS-J0005-0006']:
      quasar_loc[1]+=1
      quasar_loc[0]-=0.5

    r_in=r_inner
    r_out=r_outer

    if test and quasar=='NDWFS-J1425+3254':
      r_out=8
    if test and quasar=='SDSS-J0005-0006':
        r_out=15.5

    if test:
      aperture = CircularAperture(quasar_loc, r=r_out)
    else:
      aperture = CircularAnnulus(quasar_loc, r_in=r_in, r_out=r_out)
      aperture_in = CircularAperture(quasar_loc, r=r_in)

    #Ignore in the photometry all pixels with S/N<-10 (i.e. just artifacts of quasar subtraction)
    condition=np.zeros_like(counts)#(((counts-sky))/noise <-10)
    counts_masked=np.ma.masked_where(condition, counts)
    frac_masked_inside=float(aperture_photometry(np.ma.masked_where(condition,np.ones_like(counts)), aperture_in)['aperture_sum'])/(np.pi*r_in**2)

    phot_table = aperture_photometry(counts_masked-sky, aperture)
    tot_counts=float(phot_table['aperture_sum'])
    noise_ap=float(aperture_photometry(noise, aperture)['aperture_sum'])/aperture.area() #avg noise per pixel in the aperture
    avg_counts=tot_counts/aperture.area()/pxscale[0]/pxscale[1] #flux/arcsec^2

    SNR=tot_counts/(noise_ap*aperture.area())

    annulus_masks = aperture_in.to_mask(method='center')

    phot_table_in = aperture_photometry(counts_masked-sky, aperture_in)
    tot_counts_in=float(phot_table_in['aperture_sum'])#/frac_masked_inside

    zeropoint=-2.5*np.log10(f[0].header['PHOTFLAM'])-5*np.log10(f[0].header['PHOTPLAM'])-2.408 #checked, correct

    if SNR<3:
      #Calculate magnitude from 2 sigma noise limits, assuming Sersic profile that allows for maximum brightness given the SB at that annuli.
      SB[quasar]=-2.5*np.log10(2*noise_ap/pxscale[0]/pxscale[1])+zeropoint #2 sigma
      #mag_sersic[quasar]=calc_sersic_mag(2*noise_ap*aperture.area(),r_in*pxscale[0],r_out*pxscale[0],0.5,1/angular_scale[quasar],zeropoint) #values in ''. Re=1kpc, n=0.5 (disk)
      mag_sersic[quasar]=find_mag_distribution(2*noise_ap*aperture.area(),r_in*pxscale[0],r_out*pxscale[0],zeropoint,angular_scale[quasar],quasar_mags[quasar],plot=False)
    else:
      print('Host Detection')
      SB[quasar]=-2.5*np.log10(avg_counts)+zeropoint


    if test:
      magnitude=-2.5*np.log10(tot_counts)+zeropoint
    else:
      #magnitude=-2.5*np.log10(tot_counts/aperture.area()*np.pi*r_out**2)+zeropoint
      magnitude=-2.5*np.log10(tot_counts_in)+zeropoint
    #print("{}: SB={}, m={}".format(quasar,SB[quasar],magnitude)) #area less than 1 arcsec^2 -> mag fainter than SB
    mag[quasar]=magnitude

    #plot
    annulus_masks = aperture.to_mask(method='center')
    annulus_data = annulus_masks[0].multiply(counts-sky)

    #im=ax[jj].imshow((counts-sky)/noise,cmap='coolwarm',norm=SymLogNorm(3,vmin=-10**2.5,vmax=10**2.5))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    im=ax[jj].imshow((counts-sky)/noise,cmap='coolwarm',norm=SymLogNorm(2,vmin=-50,vmax=50))#norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
    #ax[jj].imshow(SNR,cmap='coolwarm',norm=SymLogNorm(1),vmin=-100,vmax=100)
    if ~test:
      circle1=plt.Circle(quasar_loc,r_in, fill=False,color='k',linestyle='--')
      ax[jj].add_artist(circle1)
    circle2=plt.Circle(quasar_loc,r_out, fill=False,color='k',linestyle='--')
    ax[jj].add_artist(circle2)
    ax[jj].set_xlim([quasar_loc[0]-14,quasar_loc[0]+14])
    ax[jj].set_ylim([quasar_loc[1]-14,quasar_loc[1]+14])
    #ax[0].plot(quasar_loc[0],quasar_loc[1],'ko')
    #ax[1].imshow(np.log10(annulus_data+1),cmap='inferno',vmin=0,vmax=0.02)
    #ax[1].set_ylim([0,14])
    #if jj>=3:
    ax[jj].set_xticks([-0.5/pxscale[0]+64,64,0.5/pxscale[0]+64])
    ax[jj].set_xticklabels(["-0.5''","0","0.5''"])
    #else:
    #    ax[jj].set_xticks([])
    if jj==0:# or jj==3:
      ax[jj].set_yticks([-0.5/pxscale[0]+64,64,0.5/pxscale[0]+64])
      ax[jj].set_yticklabels(["-0.5''","0","0.5''"])
    else:
      ax[jj].set_yticks([])
    #ax[jj].set_yticks([])
    ax[jj].set_title(quasar)
    if band=='H':
      ax[0].set_ylabel('F160W')
    if band=='J':
      ax[0].set_ylabel('F125W')

  cbar = plt.colorbar(im,cax=ax[-1],fraction=0.15,pad=0)
  cbar.set_ticks([-30,-3,0,3,30])
  cbar.set_ticklabels(['-30','-3','0','3','30'])
  cbar.set_label('S/N')
  #plt.tight_layout()
  plt.savefig('Plots/quasar_host_photometry_{}.pdf'.format(band))
  plt.show()


def mag_vs_lam(lam, beta, mag0):
    return -2.5*(beta+2)*np.log10(lam) + mag0


def calc_abs_mag(obs_mag, z):
    #print(cosmo.distmod(z).value)
    #print(obs_mag)
    #print(obs_mag - cosmo.distmod(z).value + 2.5*np.log10(1+z))
    return obs_mag - cosmo.distmod(z).value + 2.5*np.log10(1+z)


def calc_mag_beta(mag_J,mag_H):
  lams = np.array((1.25, 1.55))
  for jj,quasar in enumerate(quasars):
    z=redshift[quasar]
    par, cov = curve_fit(mag_vs_lam, xdata=lams, ydata=[mag_J[quasar],mag_H[quasar]],
    sigma=[0.1,0.1])
    abs_mag[quasar] = calc_abs_mag(mag_vs_lam((1+z)*0.15, *par), z)
    beta[quasar] = par[0]


def save_table():
  total_df=pd.DataFrame(columns=["SB_J (${:3g}-{:3g}'', 2\sigma$)".format(r_inner*pxscale[0],r_outer*pxscale[0]),
  'M_J (Sersic fit, 2$\sigma$)',
  "SB_H (${:3g}-{:3g}'', 2\sigma$)".format(r_inner*pxscale[0],r_outer*pxscale[0]),
  'M_H (Sersic fit, 2$\sigma$)'])

  #columns=["M_J ($<{:3g}''$)".format(7*pxscale[0]),
  #"SB_J (${:3g}-{:3g}'', 2\sigma$)".format(7*pxscale[0],11*pxscale[0]),'M_J (Sersic fit, 2$\sigma$)',
  #"M_H ($<{:3g}''$)".format(7*pxscale[0]),"SB_H (${:3g}-{:3g}'', 2\sigma$)".format(7*pxscale[0],11*pxscale[0]),
  #'M_H (Sersic fit, 2$\sigma$)'])


  for quasar in quasars:
    total_df.loc[quasar]=[
      #r"${:.1f}$".format(mag_J_in[quasar]),
      r"${:.1f}$".format(SB_J_out[quasar]),
      r"${:.1f}$".format(mag_sersic_J[quasar]),
      #r"${:.1f}$".format(mag_H_in[quasar]),
      r"${:.1f}$".format(SB_H_out[quasar]),
      r"${:.1f}$".format(mag_sersic_H[quasar])]

  with open('quasar_photometry.tex', 'w') as tf:
     tf.write(total_df.to_latex(escape=False))
  print(total_df)


if __name__=="__main__":
  quasars = ['NDWFS-J1425+3254','SDSS-J0005-0006','SDSS-J0203+0012','CFHQS-J0033-0125','SDSS-J0129-0035','SDSS-J2054-0005']
  n_quasars = len(quasars)

  redshift={'NDWFS-J1425+3254':5.89,'SDSS-J0005-0006':5.85,'SDSS-J0203+0012':5.72,
  'CFHQS-J0033-0125':6.13,'SDSS-J0129-0035':5.78,'SDSS-J2054-0005':6.04}
  angular_scale={'NDWFS-J1425+3254':6.026,'SDSS-J0005-0006':6.049,'SDSS-J0203+0012':6.122,
  'CFHQS-J0033-0125':5.895,'SDSS-J0129-0035':6.088,'SDSS-J2054-0005':5.922} #kpc/''

  quasar_J={'NDWFS-J1425+3254':20.6,'SDSS-J0005-0006':20.94,'SDSS-J0203+0012':19.96,
  'CFHQS-J0033-0125':21.4,'SDSS-J0129-0035':22.03,'SDSS-J2054-0005':20.68}
  quasar_H={'NDWFS-J1425+3254':20.57,'SDSS-J0005-0006':20.75,'SDSS-J0203+0012':19.46,
  'CFHQS-J0033-0125':21.24,'SDSS-J0129-0035':21.24,'SDSS-J2054-0005':20.58}

  bh_masses={'NDWFS-J1425+3254':9.408219991184932,#Shen19
  'SDSS-J0005-0006':8.02, #Trakhtenbrot. MgII
  'SDSS-J0203+0012':10.72480968733483,#Shen19
  'CFHQS-J0033-0125':9.515635218774843, #Shen19
  'SDSS-J0129-0035':8.23, #Willott15
  'SDSS-J2054-0005':8.95} #Willott15
  bh_masses_err={'NDWFS-J1425+3254':0.11160163206471481, #Shen19. Mg ii observations if possible, else from CIV.
  'SDSS-J0005-0006':0,
  'SDSS-J0203+0012':0.25792698077188814,#Shen19
  'CFHQS-J0033-0125':0.8745788548443557, #Shen19
  'SDSS-J0129-0035':0.45, #Willott15
  'SDSS-J2054-0005':0.47} #Willott15 #Mg ii $\lambda 2799$ observations if possible, else from Eddington luminosity assumption.

  wcs = WCS(fits.getheader('NDWFS-J1425+3254/sci_NDWFS-J1425+3254_J.fits.gz'))
  pxscale = 3600*np.abs(np.linalg.eigvals(wcs.pixel_scale_matrix))

  mag_H_in={}
  mag_J_in={}

  SB_H_out={}
  SB_J_out={}

  mag_sersic_J={}
  mag_sersic_H={}

  test=False

  """
  ##J-band
  r_outer={'NDWFS-J1425+3254':6.5,'SDSS-J0005-0006':6.5,'SDSS-J0203+0012':6,
  'CFHQS-J0033-0125':6,'SDSS-J0129-0035':5,'SDSS-J2054-0005':6.5}
  #r_inner={'NDWFS-J1425+3254':5.5,'SDSS-J0005-0006':5,'SDSS-J0203+0012':4.5,
  #'CFHQS-J0033-0125':4.5,'SDSS-J0129-0035':4,'SDSS-J2054-0005':5}
  r_inner={'NDWFS-J1425+3254':0,'SDSS-J0005-0006':0,'SDSS-J0203+0012':0,
  'CFHQS-J0033-0125':0,'SDSS-J0129-0035':0,'SDSS-J2054-0005':0}
  shift={'NDWFS-J1425+3254':[0,-1],'SDSS-J0005-0006':[0,0],'SDSS-J0203+0012':[0,0],
  'CFHQS-J0033-0125':[0,-1],'SDSS-J0129-0035':[0.5,-0.5],'SDSS-J2054-0005':[-0.5,-0.5]}
  calc_phot('J',mag_J,SB_J,two_sigma_J)
  print(two_sigma_J) """

  ##J-band, r_outer
  #r_inner={'NDWFS-J1425+3254':7,'SDSS-J0005-0006':7,'SDSS-J0203+0012':7,
  #'CFHQS-J0033-0125':7,'SDSS-J0129-0035':7,'SDSS-J2054-0005':7}
  r_outer=10
  r_inner=7
  shift={'NDWFS-J1425+3254':[0,-1],'SDSS-J0005-0006':[0,0],'SDSS-J0203+0012':[0,0],
  'CFHQS-J0033-0125':[0,-1],'SDSS-J0129-0035':[0.5,-0.5],'SDSS-J2054-0005':[-0.5,-0.5]}
  calc_phot('J',mag_J_in,SB_J_out,mag_sersic_J,quasar_J)

  """
  ##H-band
  r_outer={'NDWFS-J1425+3254':7,'SDSS-J0005-0006':7,'SDSS-J0203+0012':8,
  'CFHQS-J0033-0125':6.5,'SDSS-J0129-0035':6,'SDSS-J2054-0005':7}
  r_inner={'NDWFS-J1425+3254':0,'SDSS-J0005-0006':0,'SDSS-J0203+0012':0,
  'CFHQS-J0033-0125':0,'SDSS-J0129-0035':0,'SDSS-J2054-0005':0}
  #r_inner={'NDWFS-J1425+3254':5.5,'SDSS-J0005-0006':5.5,'SDSS-J0203+0012':6.5,
  #'CFHQS-J0033-0125':5,'SDSS-J0129-0035':4.5,'SDSS-J2054-0005':5}
  shift={'NDWFS-J1425+3254':[-1,0.5],'SDSS-J0005-0006':[0,0],'SDSS-J0203+0012':[0,0],
  'CFHQS-J0033-0125':[-0.5,0],'SDSS-J0129-0035':[0,0],'SDSS-J2054-0005':[-0.5,-0.5]}
  calc_phot('H',mag_H,SB_H,two_sigma_H)
  print(two_sigma_H)"""

  ##H-band, r_outer
  #r_inner={'NDWFS-J1425+3254':7,'SDSS-J0005-0006':7,'SDSS-J0203+0012':7,
  #'CFHQS-J0033-0125':7,'SDSS-J0129-0035':7,'SDSS-J2054-0005':7}
  r_outer=10
  r_inner=7
  shift={'NDWFS-J1425+3254':[-1,0.5],'SDSS-J0005-0006':[0,0],'SDSS-J0203+0012':[0,0],
  'CFHQS-J0033-0125':[-0.5,0],'SDSS-J0129-0035':[0,0],'SDSS-J2054-0005':[-0.5,-0.5]}
  calc_phot('H',mag_H_in,SB_H_out,mag_sersic_H,quasar_H)

  print(mag_J_in,mag_H_in)

  abs_mag={}
  host_mass={}
  beta={}
  calc_mag_beta(mag_sersic_J,mag_sersic_H)
  host_mass = {qq: upper_mass_limit(M,-20) for qq, M in abs_mag.items()} #Use -20 as faintest BT z=7 hosts


  print(abs_mag)
  print(host_mass)

  save_table()

  ##BH-stellar mass relation
  fig,ax=plt.subplots(2,1,figsize=(4.2,5),gridspec_kw={'height_ratios':[1,0.2]})
  for ii,qq in enumerate(quasars):
    ax[0].errorbar(host_mass[qq],bh_masses[qq],yerr=bh_masses_err[qq],xerr=0.5,
    xuplims=True,label=qq,capsize=4,marker='o',markersize=8,zorder=100,color=colors[ii])
  ax[1].axis('off')

  plot_literature_observations(ax[0])

  ax[0].legend(fontsize='small',ncol=2,loc=[0.02,-0.49])

  plt.savefig('Plots/BHmassRelation.pdf')
  plt.show()
