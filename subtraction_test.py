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

def mag_to_flux(mag,zeropoint):
  return 10**(-0.4 * (mag-zeropoint))

def flux_to_mag(flux,zeropoint):
  return -2.5*np.log10(flux) + zeropoint

quasar = 'SDSS-J0203+0012'
band='H'
quasar_loc=[63.5,63.5]

psf = fits.open('{}/sci_psf_02hr_{}.fits'.format(quasar,band))
psf=psf[0].data

image = fits.open('{}/sci_{}_{}.fits'.format(quasar,quasar,band))
zeropoint=-2.5*np.log10(image[0].header['PHOTFLAM'])-5*np.log10(image[0].header['PHOTPLAM'])-2.408

ivm = fits.open('{}/out_{}_{}_composite_ivm.fits'.format(quasar,quasar,band))
noise=np.sqrt(1/ivm[0].data)

resid = fits.open('{}/out_{}_{}_residual.fits'.format(quasar,quasar,band))
PS_mag=float(resid[0].header['1PS_MAG'].split()[0])#19.46
PS_flux=mag_to_flux(PS_mag,zeropoint)


norm_psf=psf/np.sum(psf)
scaled_psf=norm_psf*PS_flux

subtraction = image[0].data - scaled_psf #This is not the same as the residual image,
#as this uses the 'best' parameter value and the residual image is weighted by all the mcmc points

#Look at noise in middle from noise map and induced by subtraction
aperture = CircularAperture(quasar_loc, r=7)
mask = aperture.to_mask(method='center')
inner_sub=mask[0].multiply(subtraction)
inner_noise=mask[0].multiply(noise)

_, _, noise_sma3 = sigma_clipped_stats(inner_sub, sigma=3.0)
_, _, noise_sma10 = sigma_clipped_stats(inner_sub, sigma=10.0)
print(noise_sma3,noise_sma10,np.std(inner_sub),np.median(inner_noise))

total_noise=noise_sma3*aperture.area()
print(flux_to_mag(2*total_noise,zeropoint))
print(flux_to_mag(2*float(aperture_photometry(noise, aperture)['aperture_sum']),zeropoint))

fig,ax=plt.subplots()
im=ax.imshow((image[0].data/np.max(image[0].data)-psf/np.max(psf)),cmap='coolwarm')
plt.colorbar(im)
#im=ax[1].imshow(resid[0].data/noise,cmap='coolwarm')
plt.show()


if True:  ##Test subtraction by looking at flux left in inner annulus
  counts=resid[0].data
  sky=float(resid[0].header['0SKY_ADU'].split(' ')[0])

  indxs=np.indices(np.shape(counts),dtype='float')
  indxs[0]-=quasar_loc[0]
  indxs[1]-=quasar_loc[1]
  dist=np.sqrt(indxs[0]**2+indxs[1]**2)

  sig=np.array((counts-sky))
  dist_flat=np.reshape(dist,np.shape(sig)[0]*np.shape(sig)[0])
  noise_flat=np.reshape(noise,np.shape(sig)[0]*np.shape(sig)[0])
  sig=np.reshape(sig,np.shape(sig)[0]*np.shape(sig)[0])

  mm,bin,_=binned_statistic(dist_flat,sig/noise_flat,statistic='std',bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

  #Plot median signal
  fig2,ax2=plt.subplots()
  ax2.plot(bin[:-1],mm)
  plt.show()

  aa=np.array(mask[0].multiply(counts-sky))
  aa=np.reshape(aa,np.shape(aa)[0]*np.shape(aa)[0])
  #fig,ax=plt.subplots()
  #plt.hist(aa,range=(-2,2))
  print(np.median(mask[0].multiply(counts-sky)))
  print(np.mean(mask[0].multiply(counts-sky)))
  print(np.sum(mask[0].multiply(counts-sky)))


  #plt.show()

  #ax[jj].text(50,50,'SNR = {}'.format(np.round(SNR,2)))
