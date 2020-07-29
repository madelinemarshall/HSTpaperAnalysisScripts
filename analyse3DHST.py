import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
import astropy.coordinates as cs
from scipy.stats import binned_statistic
f=fits.open('Data/3DHST_GOODSS/Catalog/goodss_3dhst.v4.1.cat.fits')
z=fits.open('Data/3DHST_GOODSS_GRISM/goodss_3dhst.v4.1.5.zbest.fits')

#print(f[1].header)
#print(z[1].header)

RA = cs.Longitude(f[1].data['ra'],u.deg)
dec = cs.Latitude(f[1].data['dec'],u.deg)
coord = cs.SkyCoord(RA, dec, unit='deg')

redshift = z[1].data['z_best']
#Best available redshift measurement (-1 for stars)
redshift_low = z[1].data['z_best_l95']
#Lower 95% confidence limit derived form the z_best
redshift_up = z[1].data['z_best_l95']
source_redshift = z[1].data['z_best_s']
#Source of the best redshift: 1 = ground-based spectroscopy; 2 = grism; 3 = photometry; 0 = star
err=(redshift-redshift_low)

##Looking at the redshift distribution
#plt.hist(err[redshift>5],range=(0,1))

#plt.hist(redshift[source_redshift==3],histtype='step',range=(5,6),log=True)
#plt.hist(redshift[(source_redshift==3)&(err<1)],histtype='step',range=(5,6),log=True)
#plt.show()

##Deciding on z=6 galaxy sample
#print(len(redshift[(redshift>5)&(err<0.5)]))
#print(len(redshift[(redshift>5.5)&(err<0.5)]))
#print(len(redshift[(redshift_low>5)]))
#print(len(redshift[(redshift>5.5)&(redshift_low>5)]))]
sample=(redshift>5.5)&(redshift_low>5)
host_sample=sample[0:33879] #VDW catalog only has first 33879 galaxies

#Find companions within 3'' of a z~6 galaxy
companions=np.zeros((len(sample[sample]),len(RA)),dtype='bool')
companions_full_list=np.zeros(len(RA),dtype='bool')
dist=np.zeros((len(sample[sample]),len(RA)))
for ss in range(0,len(sample[sample])):
  dist[ss]=coord[sample][ss].separation(coord).arcsec
  companions[ss]=(dist[ss]<3)&(dist[ss]>0)
  companions_full_list+=companions[ss]
#checked that search works correctly - seems to.

all_companions=companions_full_list[0:33879]

'''
plt.plot(RA.arcsec,dec.arcsec,'y.')
plt.plot(RA[companions_full_list].arcsec,dec[companions_full_list].arcsec,'k.')
plt.plot(RA[sample].arcsec,dec[sample].arcsec,'r.')
plt.axis('equal')
plt.show()
'''
#checked that search works correctly - seems to.


###SIZES
vdw=np.loadtxt('Data/VanDerWel/cosmos/cosmos_3dhst.v4.1_f125w.galfit')

#print(vdw[33800:33879,0],f[1].data['id'][33800:33879])
##Same ids, just shorter array

Re=vdw[:,6]
#re: semi-major axis in arcsec of the ellipse that contains half of the total light in the best fitting Sersic model
fit_flag=vdw[:,3]
#f: FLAG value (0: good fit; 1: suspicious fit; 2: bad fit; 3: no fit -- see van der Wel et al. 2012)
sersic=vdw[:,8]
#n: Sersic index of the best-fitting Sersic model
mag=vdw[:,4]
#mag: total AB magnitude from best-fitting Sersic model (GALFIT)

'''
fig,ax=plt.subplots(1,2)
ax[0].hist(Re[fit_flag==0],range=(0,10),log=True,histtype='step')
ax[0].hist(Re[host_sample],range=(0,10),log=True,histtype='step')
ax[1].hist(sersic[fit_flag==0],range=(0,9),log=True,histtype='step')
ax[1].hist(sersic[host_sample],range=(0,9),log=True,histtype='step')
plt.show()
'''

#plt.hexbin(Re[fit_flag==0],sersic[fit_flag==0],extent=(0,10,0,8),cmap='Blues',mincnt=1,gridsize=35,bins='log')
#plt.show()


def plot_VDW_sizes(ax):
  ax[0,0].plot(mag[(host_sample)&(fit_flag==0)],Re[(host_sample)&(fit_flag==0)],'.',color=[0.6,0.6,0.6],label='van der Wel et al. (2012)')
  ax[1,0].plot(mag[(host_sample)&(fit_flag==0)],sersic[(host_sample)&(fit_flag==0)],'.',color=[0.6,0.6,0.6],label='van der Wel et al. (2012)')

  ax[0,1].plot(mag[(all_companions)&(fit_flag==0)],Re[(all_companions)&(fit_flag==0)],'.',color=[0.6,0.6,0.6],label='van der Wel et al. (2012)')
  ax[1,1].plot(mag[(all_companions)&(fit_flag==0)],sersic[(all_companions)&(fit_flag==0)],'.',color=[0.6,0.6,0.6],label='van der Wel et al. (2012)')

  med,bin_edge,_=binned_statistic(mag[(host_sample)&(fit_flag==0)],Re[(host_sample)&(fit_flag==0)],statistic='median',bins=6,range=(21,27))
  median=med
  magns=bin_edge
  ax[0,0].plot(bin_edge[:-1]+(bin_edge[1]-bin_edge[0])/2,med,color=[0.4,0.4,0.4],label='van der Wel et al. (2012), Median')
  med,bin_edge,_=binned_statistic(mag[(host_sample)&(fit_flag==0)],sersic[(host_sample)&(fit_flag==0)],statistic='median',bins=6,range=(21,27))
  ax[1,0].plot(bin_edge[:-1]+(bin_edge[1]-bin_edge[0])/2,med,color=[0.4,0.4,0.4],label='van der Wel et al. (2012), Median')
  medianS=med
  magnsS=bin_edge

  med,bin_edge,_=binned_statistic(mag[(all_companions)&(fit_flag==0)],Re[(all_companions)&(fit_flag==0)],statistic='median',bins=6,range=(21,27))
  ax[0,1].plot(bin_edge[:-1]+(bin_edge[1]-bin_edge[0])/2,med,color=[0.4,0.4,0.4],label='van der Wel et al. (2012), Median')

  med,bin_edge,_=binned_statistic(mag[(all_companions)&(fit_flag==0)],sersic[(all_companions)&(fit_flag==0)],statistic='median',bins=6,range=(21,27))
  ax[1,1].plot(bin_edge[:-1]+(bin_edge[1]-bin_edge[0])/2,med,color=[0.4,0.4,0.4],label='van der Wel et al. (2012), Median')
  return median,magns,medianS,magnsS

if __name__=='__main__':
  fig,ax=plt.subplots(2,2,sharex=True)

  print(np.median([2.2,4.2,4.7,3.6,7.0,4.6,5.6,4.8,2.1]))
  print(np.mean([2.2,4.2,4.7,3.6,7.0,4.6,5.6,4.8,2.1]))
  print(np.shape(mag[(host_sample)&(fit_flag==0)]))
  print(np.shape(mag[(all_companions)&(fit_flag==0)]))
  print(np.shape(mag[(host_sample)&(all_companions)&(fit_flag==0)]))

  print(np.shape(mag[(host_sample)]))
  print(np.shape(mag[(all_companions)]))
  print(np.shape(mag[(host_sample)&(all_companions)]))
  print(np.shape(mag[(all_companions)&(redshift[0:33879]>5.5)]))

  print(len(mag[(redshift[0:33879]>5.5)])/len(mag))


  plot_VDW_sizes(ax)

  ax[0,0].set_ylim(0,1.5)
  ax[1,0].set_ylim(0,1.5)
  plt.show()

  plt.figure()
  plt.hexbin(Re,sersic,extent=(0,1,0,8),bins='log')
  plt.show()
