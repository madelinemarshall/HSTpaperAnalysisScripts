import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def printname(name):
  print(name)

f = fits.open('NDWFS-J1425+3254/out_NDWFS-J1425+3254_J_db.fits')
#f = fits.open('NDWFS-J1425+3254/out_NDWFS-J1425+3254_J_residual.fits')

n_companions=int((f[1].header['TFIELDS']-6)/6)
n_chains=int((f[1].header['NAXIS2']))

mag=np.zeros((n_companions,n_chains))
index=np.zeros((n_companions,n_chains))
reff=np.zeros((n_companions,n_chains))
b_on_a=np.zeros((n_companions,n_chains))
xy=np.zeros((n_companions,n_chains,2))
xy_sep=np.zeros((n_companions,n_chains,2))

quasar_xy=f[1].data['1_PointSource_xy']
quasar_mag=f[1].data['1_PointSource_mag']

for ii in range(0,n_companions):
  mag[ii,:]=f[1].data['{}_Sersic_mag'.format(ii+2)]
  index[ii,:]=f[1].data['{}_Sersic_index'.format(ii+2)]
  reff[ii,:]=f[1].data['{}_Sersic_reff'.format(ii+2)]
  b_on_a[ii,:]=f[1].data['{}_Sersic_reff_b'.format(ii+2)]/reff[ii,:]
  xy[ii,:]=f[1].data['{}_Sersic_xy'.format(ii+2)]
  xy_sep[ii,:]=xy[ii,:]-quasar_xy

#plt.hist(mag[0,:],bins=20)
#plt.show()
plt.plot(mag[0,:],reff[0,:],'.')
plt.show()
