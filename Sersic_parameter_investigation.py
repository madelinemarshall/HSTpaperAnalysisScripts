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
import scipy.stats

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = [4,4]
plt.rcParams['font.size'] = (9)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
markers=['o','s','d','<','x','>']
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff7f00','#f781bf','#a65628','#98ff98']
cosmo = FlatLambdaCDM(67, 0.3)

def calc_sersic_mag(flux_in_annulus,r_inner,r_outer,nn,Re,zeropoint):
  bn=gammaincinv(2*nn,0.5)

  gamma_diff=(gammainc(2*nn,bn*(r_outer/Re)**(1/nn))-gammainc(2*nn,bn*(r_inner/Re)**(1/nn)))/gamma(2*nn)

  fe = flux_in_annulus/(2*np.pi*nn) * bn**(2*nn)/(Re**2*np.exp(bn)) / gamma_diff

  flux_in_r_outer=2*np.pi*nn*fe*Re**2*np.exp(bn)/bn**(2*nn)*gammainc(2*nn,bn*(r_outer/Re)**(1/nn))/gamma(2*nn)
  #gamma_2nx=gammainc(2*nn,bn*(radius/Re)**(1/nn))/gamma(2*nn)

  #mag = SB - 2.5*bn/np.log(10)*((radius/Re)**(1/nn)-1) - 5*np.log10(Re) -2.5*np.log10(2*np.pi*nn*np.exp(bn)/bn**(2*nn) * gamma_2nx)
  mag = -2.5*np.log10(flux_in_r_outer)+zeropoint
  #print(mag)
  return mag


def calc_sersic_flux(flux_in_annulus,r_inner,r_outer,nn,Re,zeropoint):
  bn=gammaincinv(2*nn,0.5)

  gamma_diff=(gammainc(2*nn,bn*(r_outer/Re)**(1/nn))-gammainc(2*nn,bn*(r_inner/Re)**(1/nn)))/gamma(2*nn)

  fe = flux_in_annulus/(2*np.pi*nn) * bn**(2*nn)/(Re**2*np.exp(bn)) / gamma_diff

  flux_in_r_outer=2*np.pi*nn*fe*Re**2*np.exp(bn)/bn**(2*nn)*gammainc(2*nn,bn*(r_outer/Re)**(1/nn))/gamma(2*nn)

  return flux_in_r_outer


def plot_obs(axes):
  shibuya = np.genfromtxt('./Shibuya_2015_Tab4.csv', delimiter=',', names=True, missing_values=' ')

  #axes.errorbar(shibuya['nUV'], shibuya['ReffUV'],  yerr=shibuya['e_ReffUV'], xerr=shibuya['e_nUV'],
  #            color=[0.8,0.8,0.8], ls='none', marker='.', capthick=0,
  #            label='Shibuya et al. 2015')
  print(np.nanmin(shibuya['nOpt']))
  ax.plot(shibuya['nOpt']/(indices[1]-indices[0])-indices[0], shibuya['ReffOpt']/(radii[1]-radii[0])-radii[0], 'k.')


def calc_size_PDF(radii):
  mu=0.95
  sigma=0.55
  x = radii
  pdf = (np.exp(-(np.log(x/mu))**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))
  #axes.plot(pdf*1000,'k')
  return pdf


def calc_n_PDF(indices):
  mu=1.5
  sigma=0.45
  x = indices
  pdf = (np.exp(-(np.log(x/mu))**2 / (2 * sigma**2))/ (x * sigma * np.sqrt(2 * np.pi)))
  return pdf


def calc_2D_PDF(radius,index):
  mu1=0.95
  sigma1=0.55
  x = radius

  mu2=1.5
  sigma2=0.45
  y = index

  pdf = (np.exp(-(np.log(x/mu1))**2 / (2 * sigma1**2))/ (x * sigma1 * np.sqrt(2 * np.pi))) *\
  (np.exp(-(np.log(y/mu2))**2 / (2 * sigma2**2))/ (y * sigma2 * np.sqrt(2 * np.pi)))
  #axes.plot(pdf*1000,'k')
  return pdf


def make_magnitude_grid(ax):
  n_cells=1000 #1000 to look smooth
  indices=np.linspace(0.5,5,n_cells)
  radii=np.linspace(0,4,n_cells)

  mag=np.zeros((len(indices),len(radii)))
  flux=np.zeros((len(indices),len(radii)))
  pdf2d=np.zeros((len(indices),len(radii)))

  #Calculate & plot magnitude at each grid cell
  for ii in range(0,len(indices)):
    for jj in range(0,len(radii)):
      mag[ii,jj]=calc_sersic_mag(flux_in_annulus,r_inner,r_outer,indices[ii],radii[jj]/angular_scale,zeropoint)
      flux[ii,jj]=calc_sersic_flux(flux_in_annulus,r_inner,r_outer,indices[ii],radii[jj]/angular_scale,zeropoint)
      if (mag[ii,jj]<20.6):#22.5) and (radii[jj]/angular_scale<0.3):
        mag[ii,jj]=np.nan
        flux[ii,jj]=np.nan
      pdf2d[ii,jj]=calc_2D_PDF(radii[jj],indices[ii])
    #print(flux[ii,:])
    #flux_avg[ii]=np.average(flux[ii,:][np.logical_not(np.isnan(flux[ii,:]))],weights=size_pdf[np.logical_not(np.isnan(flux[ii,:]))])
  im=ax.imshow(mag)
  #im=ax.imshow(flux)
  ax.invert_yaxis()
  ax.set_xticks([0,n_cells//4,n_cells//2,3*n_cells//4,n_cells])
  ax.set_xticklabels([str(np.round(radii[i],1)) for i in [0,n_cells//4,n_cells//2,3*n_cells//4,-1]])
  ax.set_yticks([0,n_cells//4,n_cells//2,3*n_cells//4,n_cells])
  ax.set_yticklabels([str(np.round(indices[i],1)) for i in [0,n_cells//4,n_cells//2,3*n_cells//4,-1]])
  ax.set_xlabel('Effective Radius $R_e$ (kpc)')
  ax.set_ylabel(r'S\'ersic Index $n$')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cb=plt.colorbar(im, cax=cax)
  cb.ax.set_ylabel('mag')
  plt.subplots_adjust(top=0.98,bottom=0.02,left=0.155,right=0.865,hspace=0.2,wspace=0.2)

  ax.set_xlim(0,n_cells)
  ax.set_ylim(0,n_cells)
  #plot_obs(ax)
  cset = ax.contour(np.linspace(0,n_cells,n_cells),np.linspace(0,n_cells,n_cells), pdf2d, colors='k')

  plt.savefig('SersicParameterInvestigation.pdf')


def additional_plots():
  #pdf2d_size=np.tile(size_pdf,(n_cells,1))
  #avg2d=np.average(flux[np.logical_not(np.isnan(flux))],weights=pdf2d_size[np.logical_not(np.isnan(flux))])
  #print(-2.5*np.log10(np.average(flux_avg))+zeropoint)
  #print(-2.5*np.log10(avg2d)+zeropoint)

  #fig,ax=plt.subplots()
  #plt.plot(radii,size_pdf,'r')
  #plt.hist(size_samp,density=True,bins=40)
  #plt.xscale('log')

  #fig,ax=plt.subplots()
  #plt.plot(indices,n_pdf)
  #plt.hist(index_samp,density=True,bins=40)
  #plt.yscale('log')

  #fig,ax=plt.subplots()
  #ax.plot(size_samp,index_samp,'r.')
  #cset = ax.contour(radii,indices, pdf2d, colors='k',zorder=100)

  #flux_bins=np.logspace(np.log10(np.nanmin(flux)),np.log10(np.nanmax(flux)),50)
  #pdf_bins=np.zeros(50)
  #for ii in range(0,49):
  #  condition=(flux<flux_bins[ii+1])&(flux>flux_bins[ii])
  #  pdf_bins[ii]=np.sum(pdf2d[condition])/np.sum(pdf2d)
  return


def find_mag_distribution(flux_in_annulus,r_inner,r_outer,zeropoint,angular_scale,quasar_mag,plot=False):
  #n_cells=400
  #indices=np.linspace(0.5,5,n_cells)
  #radii=np.linspace(0.1,4,n_cells)

  ##Calculate PDFs of r,n and flux
  #size_pdf=calc_size_PDF(radii)
  #n_pdf=calc_n_PDF(indices)
  size_samp=np.random.lognormal(mean=np.log(0.95), sigma=0.55, size=10000000)
  index_samp=np.random.lognormal(mean=np.log(1.5), sigma=0.45, size=10000000)
  flux_samp=calc_sersic_flux(flux_in_annulus,r_inner,r_outer,index_samp,size_samp/angular_scale,zeropoint)
  #flux_samp[(-2.5*np.log10(flux_samp)+zeropoint<22.5) & (size_samp/angular_scale<0.3)]=np.nan
  flux_samp[(-2.5*np.log10(flux_samp)+zeropoint<quasar_mag)]=np.nan
  mags_samp=-2.5*np.log10(flux_samp)+zeropoint
  mags_samp=mags_samp[mags_samp>0]


  #Plot magnitude distribution calculated from Monte Carlo sampling
  #ax.plot(flux_bins,pdf_bins,'.')
  #x = np.linspace(np.nanmin(flux_samp), np.nanmax(flux_samp), 100)
  #x_mag = -2.5*np.log10(x)+zeropoint
  x = np.linspace(np.nanmin(mags_samp), np.nanmax(mags_samp), 100)

  counts,bin=np.histogram(mags_samp,range=(20,25),bins=50,normed=True)

  if plot:
    fig,ax=plt.subplots()
    plt.hist(mags_samp,range=(20,25),bins=50,normed=True)
    plt.axvline(x=bin[np.argmax(counts)]+(bin[1]-bin[0])/2,color='g')

  if False:
    #Fit lognormal dist to magnitude distribution
    dist = getattr(scipy.stats, 'lognorm')
    param = dist.fit(-mags_samp)
    pdf_fitted = dist.pdf(-x,*param)

    plt.plot(x,pdf_fitted)
    plt.axvline(x=-1*dist.median(*param),color='r',linestyle='-')
    plt.axvline(x=-1*dist.mean(*param),color='b',linestyle='-')


    print(dist.mean(*param),dist.median(*param),dist.var(*param))

  if plot:
    plt.show()

  return bin[np.argmax(counts)]+(bin[1]-bin[0])/2 #Return the most common magnitude (bin center)



if __name__=="__main__":
  quasars = ['NDWFS-J1425+3254','SDSS-J0005-0006','SDSS-J0203+0012','CFHQS-J0033-0125','SDSS-J0129-0035','SDSS-J2054-0005']

  quasar_flux_in_annulus=[3.2723372065326037,2.8130612348936332,3.9825667251608814,\
  3.7105709727709257,3.976330398841131,3.4554154103220993]

  quasar_J=[20.6,20.94,19.96,21.4,22.03,20.68]

  flux_in_annulus=quasar_flux_in_annulus[0]
  quasar_mag=quasar_J[0]
  r_inner=0.41999999999997506
  r_outer=0.6599999999999608
  zeropoint=26.23021712375807
  angular_scale=6.026 #kpc/''

  #Plot R vs n vs magnitude grid
  fig,ax=plt.subplots()#,gridspec_kw={'width_ratios':[1,0.2]})
  make_magnitude_grid(ax)

  mode=find_mag_distribution(flux_in_annulus,r_inner,r_outer,zeropoint,angular_scale,quasar_mag,plot=True)
  print(mode)
