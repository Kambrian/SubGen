import numpy as np
from scipy.optimize import fsolve
from scipy.special import lambertw
from scipy.interpolate import interp1d

OmegaM=0.25
OmegaL=0.75
G=43007.1
H0=0.1 #km/s/(kpc/h)
RhoCrit=3*H0**2/8/np.pi/G
#z=0.
#scaleF=1./(1+z);
#Hz=H0 * sqrt(OmegaM/scaleF**3+ (1 -OmegaM -OmegaL) / scaleF**2 +OmegaL)
#Hratio=Hz/HUBBLE0
#OmegaZ=OmegaM/scaleF**3./Hratio**2

#rootdir='/gpfs/data/jvbq85/SubProf/'
rootdir='/work/Projects/SubProf/'
datadir=rootdir+'data/'
LudlowData=np.loadtxt(datadir+'Ludlow/WMAP1_cMz.dat')
LudlowSpline=interp1d(LudlowData[:,0],LudlowData[:,1], bounds_error=False)

def virial_density(virtype='c200', scaleF=1., Omega0=0.25):
  '''return virial density'''
  OmegaL=1.-Omega0
  G=43007.1
  HUBBLE0=0.1
  Hz=HUBBLE0 * np.sqrt(Omega0 /scaleF**3+ (1. -Omega0 -OmegaL) / scaleF**2 +OmegaL);
  Hratio=Hz/HUBBLE0
  OmegaZ=Omega0/scaleF**3/Hratio**2

  virialF={'tophat': 18.0*np.pi**2+82.0*(OmegaZ-1)-39.0*(OmegaZ-1)**2,
		  'c200': 200,
		  'b200': 200*OmegaZ}[virtype]

  RhoCrit=3*Hz**2/8./np.pi/G
  #print virialF
  return RhoCrit*virialF

def mean_concentration(m200, model='Ludlow'):
  ''' average concentration at z=0 
  m200 in units of 1e10msun/h '''
  if model=='Prada':
	'''according to Sanchez-Conde&Prada14, with scatter of 0.14dex.'''
	x=np.log(m200*1e10)
	pars=[37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7][::-1]
	return np.polyval(pars, x)
  
  if model=='Duffy':
	#return 5.74*(m200/2e2)**-0.097 #Duffy08, C200, at z=0, all
	return 6.67*(m200/2e2)**-0.092 #relaxed
	#return 5.71*(m200/2e2)**-0.084/2**0.47 #z=1, all
  
  if model=='DuffyB200':
	return 12.*(m200/2e2)**-0.087 #B200, z=0, relaxed
  
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
  def __init__(self,m=100.,c=None,rhos=None,rs=None, DeltaC=200.):
	'''initialize '''
	self.M=m
	if c is None:
	  self.C=mean_concentration(self.M)
	else:
	  self.C=c
	self.Rhos=DeltaC/3.*self.C**3/NFWFunc(self.C)*RhoCrit
	self.Rv=(self.M/DeltaC/RhoCrit/(4*np.pi/3))**(1./3)
	self.Rs=self.Rv/self.C
	if rhos is not None:
	  self.Rhos=rhos
	  self.Rs=rs
	  #TODO: complete this
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
  
  def density_cum(self,r):
	'''cumulative density, inside r'''
	x=r/self.Rs
	return 3*self.Rhos*NFWFunc(x)/x**3
  
  def strip_func(self, sat, r, k=1):
	''' m/m_0 for a subhalo (sat) inside the current halo'''
	x=np.array(r,ndmin=1)
	y=np.zeros_like(x)
	for i,R in enumerate(x):
	  rhs=(2+k)*self.density_cum(R)-3*self.density(R)
	  func=lambda a:np.log(sat.density_cum(np.exp(a)*sat.Rs))-np.log(rhs)
	  result=fsolve(func, 1.)
	  y[i]=sat.mass(np.exp(result[0])*sat.Rs)/sat.M
	if np.isscalar(r):
	  return y[0]
	return y
	
	