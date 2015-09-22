import numpy as np
from scipy.special import lambertw
from scipy.interpolate import interp1d

OmegaM=0.25
OmegaL=0.75
G=43007.1
H0=0.1 #km/s/(kpc/h)
RhoCrit=3*H0**2/8/np.pi/G
DeltaC=200. #virial overdensity

LudlowData=np.loadtxt('Ludlow_WMAP1_cMz.dat')
LudlowSpline=interp1d(LudlowData[:,0],LudlowData[:,1], bounds_error=False)

def mean_concentration(m200, model='Ludlow'):
  ''' average concentration at z=0 
  m200 in units of 1e10msun/h '''
  if model=='Prada':
	'''according to Sanchez-Conde&Prada14, with scatter of 0.14dex.'''
	x=np.log(m200*1e10)
	pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
	return np.polyval(pars, x)
  
  if model=='MaccioW1':
	#return 7.57*(m200/1e2)**-0.119 #Maccio08 WMAP1, C200, z=0, all; sigma(log10)=0.13
	return 8.26*(m200/1e2)**-0.104 #Maccio08 WMAP1, C200, z=0, relaxed; sigma(log10)=0.11
  
  if model=='Ludlow':
	return LudlowSpline(np.log10(m200)) #Ludlow14, z=0
  
  print "Unknown model!"
  raise
	  
def NFWFunc(x):
  return np.log(1+x)-x/(1+x)

class NFWHalo:
  '''NFW halo'''
  def __init__(self,m=100.,c=None):
	'''initialize 
	input mass m200 in unit of 1e10msun/h
	if concentration c is not given, determine c from the default mass-concentration relation (Ludlow14 model)
	all radius in unit of kpc/h'''
	self.M=m
	if c is None:
	  self.C=mean_concentration(self.M)
	else:
	  self.C=c
	self.Rhos=DeltaC/3.*self.C**3/NFWFunc(self.C)*RhoCrit
	self.Rv=(self.M/DeltaC/RhoCrit/(4*np.pi/3))**(1./3)
	self.Rs=self.Rv/self.C
	self.Ms=4*np.pi*self.Rhos*self.Rs**3
	self.Pots=4*np.pi*self.Rhos*self.Rs**2
	self.Ls=4*np.pi/3.*self.Rhos**2*self.Rs**3
	
  def radius(self, m):
	''' inversion of mass(r)'''
	return (-1./lambertw(-np.exp(-(1.+m/self.Ms)))-1.).real*self.Rs
  
  def mass(self,r):
	'''cumulative mass profile'''
	return self.Ms*NFWFunc(r/self.Rs)
  
  def luminosity(self, r):
	'''luminosity profile'''
	return (1.-(1.+r/self.Rs)**-3)*self.Ls #this is quite insensitive to truncation outside Rs.
  
  def density(self,r):
	'''density'''
	x=r/self.Rs
	return self.Rhos/x/(1+x)**2