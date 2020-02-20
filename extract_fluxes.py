##Load properties of quasar 'companions' from output residual fits file
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from companion_plots import calc_mag_beta
import pandas as pd
from photutils import CircularAperture,CircularAnnulus
from photutils import aperture_photometry, background
from astropy.stats import sigma_clipped_stats
from photutils import make_source_mask
from matplotlib.colors import SymLogNorm

plt.rcParams["figure.figsize"] = [10,2]
plt.rcParams['font.size'] = (9)
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
markers=['o','s','d','<','x','>']
colors         = ['#e41a1c','#377eb8','#4daf4a','#984ea3',\
                  '#ff7f00','#a65628','#f781bf','#98ff98']

#mag_zeropoint: Magnitude zeropoint, i.e. the magnitude of one
#            ADU, whether in electrons per second (as with published HST
#            zeropoints) or whatever funky units the data use.

def calc_phot(band,mag,SB):
  #Load properties of each galaxy around each quasar
  fig,axes=plt.subplots(1,int(np.ceil(n_quasars/1)),gridspec_kw={'hspace':0.25,'wspace':0.1})#,sharex=True,sharey=True)
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

    _, _, noise = sigma_clipped_stats(counts, sigma=3.0) #1 std
    sky=float(f[0].header['0SKY_ADU'].split(' ')[0])

    quasar_loc[0]+=shift[quasar][0]
    quasar_loc[1]+=shift[quasar][1]

    if test and quasar in ['SDSS-J0005-0006']:
      quasar_loc[1]+=1
      quasar_loc[0]-=0.5




    r_in=r_inner[quasar]
    r_out=r_outer[quasar]


    if test and quasar=='NDWFS-J1425+3254':
      r_out=8
    if test and quasar=='SDSS-J0005-0006':
        r_out=15.5

    if test:
      aperture = CircularAperture(quasar_loc, r=r_out)
    else:
      aperture = CircularAnnulus(quasar_loc, r_in=r_in, r_out=r_out)
      aperture_in = CircularAperture(quasar_loc, r=r_in)

    phot_table = aperture_photometry(counts-sky, aperture)
    tot_counts=float(phot_table['aperture_sum'])
    avg_counts=tot_counts/aperture.area()/pxscale[0]/pxscale[1] #flux/arcsec^2
    SNR=tot_counts/aperture.area()/noise
    #ax[jj].text(50,50,'SNR = {}'.format(np.round(SNR,2)))

    phot_table_in = aperture_photometry(counts-sky, aperture_in)
    tot_counts_in=float(phot_table_in['aperture_sum'])


    zeropoint=-2.5*np.log10(f[0].header['PHOTFLAM'])-5*np.log10(f[0].header['PHOTPLAM'])-2.408 #checked, correct

    if SNR<3:
      SB[quasar]=-2.5*np.log10(2*noise/pxscale[0]/pxscale[1])+zeropoint #2 sigma
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

    ax[jj].imshow(counts-sky,cmap='coolwarm',norm=SymLogNorm(0.01,vmin=-0.2,vmax=0.2))#cmap='inferno',vmin=0,vmax=0.02)
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
  #plt.tight_layout()
  plt.savefig('Plots/quasar_host_photometry_{}.pdf'.format(band))
  plt.show()


def save_table():
  total_df=pd.DataFrame(columns=['SB_J','M_J','SB_H','M_H'])

  for quasar in quasars:
    total_df.loc[quasar]=[
      r"${:.1f}$".format(SB_J_out[quasar]),
      r"${:.1f}$".format(mag_J_in[quasar]),
      r"${:.1f}$".format(SB_H_out[quasar]),
      r"${:.1f}$".format(mag_H_in[quasar])]

  with open('quasar_photometry.tex', 'w') as tf:
     tf.write(total_df.to_latex(escape=False))
  print(total_df)

if __name__=="__main__":
  quasars = ['NDWFS-J1425+3254','SDSS-J0005-0006','SDSS-J0203+0012','CFHQS-J0033-0125','SDSS-J0129-0035','SDSS-J2054-0005']
  n_quasars = len(quasars)

  redshift={'NDWFS-J1425+3254':5.89,'SDSS-J0005-0006':5.85,'SDSS-J0203+0012':5.72,
  'CFHQS-J0033-0125':6.13,'SDSS-J0129-0035':5.78,'SDSS-J2054-0005':6.04}
  angular_scale={'NDWFS-J1425+3254':6.026,'SDSS-J0005-0006':6.049,'SDSS-J0203+0012':6.122,
  'CFHQS-J0033-0125':5.895,'SDSS-J0129-0035':6.088,'SDSS-J2054-0005':5.922} #kpc/pixel

  wcs = WCS(fits.getheader('NDWFS-J1425+3254/sci_NDWFS-J1425+3254_J.fits.gz'))
  pxscale = 3600*np.abs(np.linalg.eigvals(wcs.pixel_scale_matrix))

  mag_H_in={}
  mag_J_in={}


  SB_H_out={}
  SB_J_out={}

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
  r_inner={'NDWFS-J1425+3254':7,'SDSS-J0005-0006':7,'SDSS-J0203+0012':7,
  'CFHQS-J0033-0125':7,'SDSS-J0129-0035':7,'SDSS-J2054-0005':7}
  r_outer={'NDWFS-J1425+3254':11,'SDSS-J0005-0006':11,'SDSS-J0203+0012':11,
  'CFHQS-J0033-0125':11,'SDSS-J0129-0035':11,'SDSS-J2054-0005':11}
  shift={'NDWFS-J1425+3254':[0,-1],'SDSS-J0005-0006':[0,0],'SDSS-J0203+0012':[0,0],
  'CFHQS-J0033-0125':[0,-1],'SDSS-J0129-0035':[0.5,-0.5],'SDSS-J2054-0005':[-0.5,-0.5]}
  calc_phot('J',mag_J_in,SB_J_out)

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
  r_inner={'NDWFS-J1425+3254':7,'SDSS-J0005-0006':7,'SDSS-J0203+0012':7,
  'CFHQS-J0033-0125':7,'SDSS-J0129-0035':7,'SDSS-J2054-0005':7}
  r_outer={'NDWFS-J1425+3254':11,'SDSS-J0005-0006':11,'SDSS-J0203+0012':11,
  'CFHQS-J0033-0125':11,'SDSS-J0129-0035':11,'SDSS-J2054-0005':11}
  shift={'NDWFS-J1425+3254':[-1,0.5],'SDSS-J0005-0006':[0,0],'SDSS-J0203+0012':[0,0],
  'CFHQS-J0033-0125':[-0.5,0],'SDSS-J0129-0035':[0,0],'SDSS-J2054-0005':[-0.5,-0.5]}
  calc_phot('H',mag_H_in,SB_H_out)

  save_table()
