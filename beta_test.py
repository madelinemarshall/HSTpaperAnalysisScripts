import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit
from astropy.wcs import WCS
import pandas as pd

def mag_vs_lam(lam, beta, mag0):
    return -2.5*(beta+2)*np.log10(lam) + mag0#-2.5*(beta+2)*np.log10(lam) + mag0

lams = np.array((1.25, 1.55))
mags=[24.6,24.2]

par, cov = curve_fit(mag_vs_lam, xdata=lams, ydata=mags)

print(par)

plt.plot(lams,mags,'ko')
plt.plot(lams,mag_vs_lam(lams,*par))
plt.plot(np.linspace(1,1.6),mag_vs_lam(np.linspace(1,1.6),*par))
plt.xscale('log')
plt.show()
