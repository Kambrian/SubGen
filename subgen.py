''' generate Monte-Carlo samples of subhaloes.
'''
import numpy as np
import matplotlib.pyplot as plt
from nfw import NFWHalo,mean_concentration
import emcee

def InfallMass2StellarMass(mAcc):
  '''mAcc: infall mass, in 1e10Msun/h
  output: stellar mass, in 1e10Msun/h
  reference: Wang, De Lucia, Weinmann 2013, MNRAS 431, for satellite galaxies.'''
  M0=5.23e1 #unit:1e10Msun/h
  k=10**0.30*0.73 #unit:1e10Msun/h
  alpha=0.298
  beta=1.99
  
  MvcInMvb=0.733 #correction from 200*rho_crit mass to 200*rho_background mass, which is close to bound mass
  m=mAcc*MvcInMvb/M0
  
  return 2*k/(m**-alpha+m**-beta)

class ModelParameter:
  ''' container for model parameters. contains fs, A, alpha, mustar, beta parameters.'''
  def __init__(self,M):
	''' model parameters for a given host mass M, in units of 1e10Msun/h '''
	self.fs=0.55 #fraction of survived subhaloes
	self.A=0.1*M**-0.02 #infall mass function amplitude
	self.alpha=0.95 #infall mass function slope
	self.mustar=0.5*M**-0.03 #stripping function amplitude
	self.beta=1.7*M**-0.04 #stripping function slope
	self.sigma_mu=1.1
	
class SubhaloSample:
  ''' a sample of subhaloes '''
  def __init__(self,M,N=1e4,MMinInfall=1e-5,MMaxInfall=1e-1,Rmax=2,C=None,include_disruption=True,weighted_sample=True):
	''' initialize the sampling parameters.
  
	M: host mass in 1e10Msun/h
	N: number of subhaloes to generate
	MMinInfall: minimum infall mass, in unit of host mass
	MMaxInfall: maximum infall mass, in unit of host mass
	Rmax: maximum radius to sample, in unit of host virial radius
	C: host concentration. If None, it will be determined by the mean mass-concentration relation.
	include_disruption: whether to include disrupted subhaloes in the sample.
	weighted_sample: whether to sample the mass in a weighted way, so that different mass ranges are sampled equally well. this is useful if 				you have a large dynamic range in mass, e.g., from 10^-6 to 10^12 Msun.
	'''
	self.M=M
	self.Host=NFWHalo(M,C) 
	#self.Host.density=my_density_func #to use custom density, overwrite Host.density() function.
	self.C=self.Host.C
	self.HOD=ModelParameter(self.M)
		
	self.Rmax=Rmax
	self.Msample=self.Host.mass(Rmax*self.Host.Rv) #mass inside Rmax
	
	self.mAccMin=MMinInfall*M
	self.mAccMax=MMaxInfall*M
	self.n=N #sample size
	self.nPred=self.HOD.A*self.Msample*(self.mAccMin**-self.HOD.alpha-self.mAccMax**-self.HOD.alpha)/self.HOD.alpha #expected number of subhaloes per host.
	
	self.include_disruption=include_disruption
	if include_disruption:
	  self.nSurvive=int(N*self.HOD.fs)
	  self.nDisrupt=N-self.nSurvive
	else:
	  self.nPred*=self.HOD.fs #only survived subhaloes are generated.
	  self.nSurvive=N
	  self.nDisrupt=int(N/self.HOD.fs*(1-self.HOD.fs))

	self.weighted_sample=weighted_sample
	
  def _lnPDF(self, x):
	''' R in units of Rv'''
	lnmu,lnR=x
	lnmubar=np.log(self.HOD.mustar)+self.HOD.beta*lnR
	dlnmu=lnmu-lnmubar
	if lnmu>0: #mu<1
	  return -np.inf
	if dlnmu>np.log(4.2): #mu<mumax=4.2*mubar
	  return -np.inf
	if lnR>np.log(self.Rmax):
	  return -np.inf
	
	lnPDFmu=-0.5*(dlnmu/self.HOD.sigma_mu)**2
	lnPDFR=3.*lnR+np.log(self.Host.density(np.exp(lnR)*self.Host.Rv)) #dM/dlnR=rho*R^3. 
	return lnPDFmu+lnPDFR

  def assign_mu_R(self, nwalkers=8, nburn=200, plot_chain=True):
	'''run emcee to sample mu and R '''
	nsteps=int(self.n/nwalkers+1+nburn) #one more step to make up for potential round-off in N/nwalkers
	print 'running %d steps'%nsteps
	ndim=2
	x00=np.array([-0.5,-0.5])
	x0=np.kron(np.ones([nwalkers,1]),x00)#repmat to nwalkers rows
	x0+=(np.random.rand(ndim*nwalkers).reshape(nwalkers,ndim)-0.5)*0.1 #random offset, [-0.5,0.5]*0.1
	sampler=emcee.EnsembleSampler(nwalkers,ndim,self._lnPDF)
	sampler.run_mcmc(x0,nsteps)
	if plot_chain:
	  plt.figure()
	  labels=[r"$\ln \mu$",r"$\ln R/R_{200}$"]
	  for i in range(ndim):
		plt.subplot(ndim,1,i+1)
		for j in range(nwalkers):
		  plt.plot(range(nsteps),sampler.chain[j,:,i],'.')
		plt.ylabel(labels[i])
		plt.plot([nburn,nburn],plt.ylim(),'k--')
	  plt.xlabel('Step')
	  plt.subplot(211)
	  plt.title('%d walkers, %d burn-in steps assumed'%(nwalkers,nburn), fontsize=10)
	#==========extract mu and R===========
	sample=sampler.chain[:,nburn:,:]
	flatchain=sample.reshape([-1,ndim])[-self.n:] #take the last N entries
	flatchain=np.exp(flatchain) #from log() to linear
	self.mu,self.R=flatchain.T
	#==========disruptions===============
	if self.include_disruption:
	  self.mu[self.nSurvive:]=0. #trailing masses set to 0.
	
  def project_radius(self):
	phi=np.arccos(np.random.rand(self.n)*2-1.) #random angle around the z-axis
	self.Rp=self.R*np.sin(phi) #projected radius

  def assign_mass(self):
	'''sample m and mAcc'''
	if self.weighted_sample:
	  lnmmin,lnmmax=np.log(self.mAccMin), np.log(self.mAccMax)
	  lnmAcc=np.random.rand(self.n)*(lnmmax-lnmmin)+lnmmin #uniform distribution between lnmmin and lnmmax
	  self.mAcc=np.exp(lnmAcc)
	  self.weight=self.mAcc**-self.HOD.alpha
	  self.weight=self.weight/self.weight.sum()*self.nPred #w/sum(w)*Npred, equals to dN/dlnm*[delta(lnm)/N] as N->inf
	  #self.weight=self.HOD.fs*self.HOD.A*Host.Msample*mAcc**-self.HOD.alpha*(lnmmax-lnmmin)/self.n #the weight is dN/dlnm*[delta(lnm)/N]
	  print np.sum(self.weight), self.nPred
	else:
	  mmax,mmin=self.mAccMin**(-self.HOD.alpha),self.mAccMax**(-self.HOD.alpha) 
	  mAcc=np.random.rand(self.n)*(mmax-mmin)+mmin #uniform in m**-alpha which is proportional to N
	  self.mAcc=mAcc**-(1./self.HOD.alpha)
	  self.weight=1.*self.nPred/self.n*np.ones(self.n)
	
	self.m=self.mAcc*self.mu

  def populate(self, plot_chain=True):
	'''populate the sample with m,mAcc,R and weight'''
	self.assign_mu_R(plot_chain=plot_chain)
	self.assign_mass()
	
  def assign_stellarmass(self): 
	'''generate stellar mass from infall mass, according to an abundance matching model'''
	logMstar=np.log10(InfallMass2StellarMass(self.mAcc))
	sigmaLogMstar=0.192
	deltaLogMstar=np.random.normal(0, sigmaLogMstar, self.n)
	self.mStar=10**(logMstar+deltaLogMstar)
  
  def assign_annihilation_emission(self, concentration_model='Ludlow'):
	'''generate annihilation luminosity
	concentration_model: 'MaccioW1', Maccio08 relation with WMAP 1 cosmology
						 'Ludlow', Ludlow14 relation with WMAP 1 cosmology'''
	#first generate concentration from infall mass, according to a mass-concetration relation
	logC=np.log10(mean_concentration(self.mAcc[:self.nSurvive],concentration_model)) #mean
	sigmaLogC=0.13 
	deltaLogC=np.random.normal(0, sigmaLogC, self.nSurvive) #scatter
	cAcc=10**(logC+deltaLogC)
	SatHalo=[NFWHalo(self.mAcc[i], cAcc[i]) for i in range(self.nSurvive)]
	rt=np.array([SatHalo[i].radius(self.m[i]) for i in range(self.nSurvive)]) #truncation radius
	self.L=np.zeros_like(self.m)
	self.L[:self.nSurvive]=np.array([SatHalo[i].luminosity(rt[i]) for i in range(self.nSurvive)]) #truncated luminosity

  def save(self, outfile, save_all=False):
	''' save the sample to outfile.
	if save_all=True, save all the properties; otherwise only R,m,mAcc,weight will be saved.'''
	if save_all:
	  np.savetxt(outfile, np.array([self.R,self.m,self.mAcc,self.weight,self.Rp,self.mStar,self.L]).T, header='R/R200, m/[1e10Msun/h], mAcc/[1e10Msun/h], weight, Rp/R200, mStar/[1e10Msun/h], Luminosity/[(1e10Msun/h)^2/(kpc/h)^3]')
	else:
	  np.savetxt(outfile, np.array([self.R,self.m,self.mAcc,self.weight]).T, header='R/R200, m/[1e10Msun/h], mAcc/[1e10Msun/h], weight')

