import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def mag_to_mass_rand(abs_mag):
  #Using  Song+16  z=6 relation
  mass = 9.53-0.50*(abs_mag+21)
  sigma = 0.36 #Could add this if wanted
  return np.random.normal(loc=mass, scale=sigma, size=1)


def mag_to_mass_determ(abs_mag):
  #Using  Song+16  z=6 relation
  mass = 9.53-0.50*(abs_mag+21)
  sigma = 0.36 #Could add this if wanted
  return mass

def upper_mass_limit(mag_lim,mag_lower):
  mags=np.random.uniform(low=mag_lim,high=mag_lower,size=100000)

  mass=np.zeros_like(mags)
  for ii in range(0,len(mags)):
    mass[ii]=mag_to_mass_rand(mags[ii])

  dist = getattr(scipy.stats, 'norm')
  param = dist.fit(mass)
  pdf_fitted = dist.pdf(np.linspace(9,12,100),*param)
  return dist.mean(*param)+2*dist.std(*param) #2 sigma mass limit on resulting mass distribution


if __name__=="__main__":

  mag_lim=-23.7
  mag_lower=-20#mag_lim+3
  mags={}
  mags['flux']=-2.5*np.log10(np.random.uniform(low=10**(-0.4*mag_lim),high=10**(-0.4*mag_lower),size=100000)) #uniform in flux
  mags['mag']=np.random.uniform(low=mag_lim,high=mag_lower,size=100000) #uniform in magnitude

  for type in ['flux','mag']:

    magnitudes=mags[type]
    mass=np.zeros_like(magnitudes)
    for ii in range(0,len(magnitudes)):
      mass[ii]=mag_to_mass_rand(magnitudes[ii])

    dist = getattr(scipy.stats, 'norm')
    param = dist.fit(mass)
    pdf_fitted = dist.pdf(np.linspace(9,12,100),*param)

    fig,ax=plt.subplots()

    ax.plot(np.linspace(9,12,100),pdf_fitted,'k')

    ax.axvline(x=dist.mean(*param),color='g',linestyle='-')
    ax.axvline(x=dist.mean(*param)+dist.std(*param),color='g',linestyle='--')
    ax.axvline(x=dist.mean(*param)+2*dist.std(*param),color='g',linestyle='--')
    print(dist.mean(*param)+2*dist.std(*param))
    #print(upper_mass_limit(mag_lim,mag_lower))

    plt.hist(mass,normed=True)
    plt.axvline(mag_to_mass_determ(mag_lim),color='r')
    plt.axvline(mag_to_mass_determ(mag_lim)+0.36,color='r',linestyle='--')
    plt.axvline(mag_to_mass_determ(mag_lim)+0.72,color='r',linestyle='--')

    plt.title('Uniform in {}'.format(type))


  plt.show()
