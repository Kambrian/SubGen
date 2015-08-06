''' generate Monte-Carlo samples of subhaloes.

Only survived subhaloes are generated.
If you also want the disrupted population, simply set m=0 in the survived subhaloes while keeping mAcc and R intact. Then take fs fraction of entries.

'''
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from nfw import NFWHalo,mean_concentration
from MbdIO import cross_section
import emcee
from myutils import myhist

####################### User Parameters #############################
MASS_HOST=1e6 #Host mass in unit of 1e10Msun/h
CONC_HOST=None #Host concentration. If None, determine concentration automatically from an average mass-concentration relation
NUM_SURV_SUBS=1e5 #number of survived subhaloes to sample (excluding disrupted, i.e., m=0, subhaloes)
R_MAX=2. #maximum radius to sample, in units of host virial radius
MASS_MIN_INFALL=1e-5*MASS_HOST #minimum infall mass
MASS_MAX_INFALL=MASS_HOST/10. #maximum infall mass
UNIFORM_MASS_SAMPLING=True #whether to sample the mass uniformly. If true, the mass between MASS_MIN_INFALL and MASS_MAX_INFALL will be uniformly sampled in logspace. Each datapoint in the sample does not represent a single subhalo, but N subhaloe as given by the "Multiplicity" column, so that the subhalo mass function is determined from the Multiplicity-weighted counts. This sampling scheme is useful when the dynamical range between MASS_MIN_INFALL and MASS_MAX_INFALL is huge (e.g., from 1e-6 to 1e15 Msun/h when studying subhalo annihilation emission), so that the probability to obtain a subhalo at the highest mass end is negligible compared with that at the lowest mass end. Without uniform mass sampling, the sample would then contain no high mass objects because the high mass end is so poorly sampled. When UNIFORM_MASS_SAMPLING=False, every subhalo will have the same Multiplicity. In both cases, the multiplicity-weighted total counts equal to the number of survived subhaloes between MASS_MIN_INFALL and MASS_MAX_INFALL in a single host halo of MASS_HOST, according to the infall mass function.


###################### Internal Model Parameters #####################################
class ModelParameter(obj):
  ''' container for model parameters. contains fs, A, alpha, mustar, beta parameters.'''
  def __init__(Mhost):
	''' model parameters for a given host mass Mhost, in units of 1e10Msun/h '''
	fs=0.55 #fraction of survived subhaloes
	A=0.1*Mhost**-0.02 #infall mass function amplitude
	alpha=0.95 #infall mass function slope
	mustar=0.5*Mhost**-0.03 #stripping function amplitude
	beta=1.7*Mhost**-0.04 #stripping function slope

ModelPars=ModelParameter(MASS_HOST)
Host=NFWHalo(MASS_HOST,Chost) 
#Host.density=my_density_func #to use custom density, overwrite Host.density() function.
Host.Msample=Host.mass(R_MAX*Host.Rv) #mass inside Rmax
NUM_SUBS_PRED=ModelPars.fs*ModelPars.A*Host.Msample*(MASS_MIN_INFALL**-ModelPars.alpha-MASS_MAX_INFALL**-ModelPars.alpha)/ModelPars.alpha #expected number of subhaloes per host.
#=================================Generate mu and R=======================================
def lnPDF(x):
  ''' R in units of Rv'''
  lnmu,lnR=x
  lnmubar=np.log(ModelPars.mustar)+ModelPars.beta*lnR
  dlnmu=lnmu-lnmubar
  if lnmu>0: #mu<1
	return -np.inf
  if dlnmu>np.log(4.2): #mu<mumax=4.2*mubar
	return -np.inf
  if lnR>np.log(R_MAX):
	return -np.inf
  
  lnPDFmu=-0.5*(dlnmu/sigma)**2
  lnPDFR=3.*lnR+np.log(Host.density(np.exp(lnR)*Host.Rv)) #dM/dlnR=rho*R^3. 
  return lnPDFmu+lnPDFR

####### run emcee ############
nwalkers=8
nburn=200
nsteps=int(NUM_SUBS/nwalkers+nburn)
print 'running %d steps'%nsteps
ndim=2

x00=np.array([-0.5,-0.5])
labels=[r"$\ln \mu$",r"$\ln R/R_{200}$"]
x0=np.kron(np.ones([nwalkers,1]),x00)#repmat to nwalkers rows
x0+=(np.random.rand(ndim*nwalkers).reshape(nwalkers,ndim)-0.5)*0.1 #random offset, [-0.5,0.5]*0.1
sampler=emcee.EnsembleSampler(nwalkers,ndim,lnPDF)
sampler.run_mcmc(x0,nsteps)

##examine the walk
#from matplotlib.pyplot import *
#ion()
#figure()
#for i in range(ndim):
  #subplot(ndim,1,i)
  #for j in range(nwalkers):
    #plot(range(nsteps),sampler.chain[j,:,i],'.')
  #ylabel(labels[i])
#xlabel('Step')  

sample=sampler.chain[:,nburn:,:]
flatchain=sample.reshape([-1,ndim])
flatchain=np.exp(flatchain) #from log() to linear
nsub=flatchain.shape[0]
nsub_disrupt=nsub/ModelPars.fs*(1-ModelPars.fs) #number of disrupted subhaloes.
print '%d disrupted subhaloes not sampled'%nsub_disrupt

mu,R=flatchain.T
#==========projections==========================
phi=np.arccos(np.random.rand(nsub)*2-1.) #random angle around the z-axis
Rp=R*np.sin(phi)
#==========generate Infall mass==================================
if UNIFORM_MASS_SAMPLING:
  lnmmin,lnmmax=np.log(MASS_MIN_INFALL), np.log(MASS_MAX_INFALL)
  lnmAcc=np.random.rand(NUM_SUBS)*(lnmmax-lnmmin)+lnmmin #uniform distribution between lnmmin and lnmmax
  mAcc=np.exp(lnmAcc)
  #count=mAcc**-alpha
  #count=count/count.sum()*NUM_SUBS_PRED #w/sum(w)*Npred, equals to dN/dlnm*[delta(lnm)/N] as N->inf
  Multiplicity=ModelPars.fs*ModelPars.A*Host.Msample*mAcc**-ModelPars.alpha*(lnmmax-lnmmin)/NUM_SUBS #the weight is dN/dlnm*[delta(lnm)/N]
  print np.sum(Multiplicity), NUM_SUBS_PRED
else:
  mmax,mmin=MASS_MIN_INFALL**(-ModelPars.alpha),MASS_MAX_INFALL**(-ModelPars.alpha) 
  mAcc=np.random.rand(NUM_SUBS)*(mmax-mmin)+mmin #uniform in m**-alpha
  mAcc=mAcc**-(1./ModelPars.alpha)
  Multiplicity=NUM_SUBS_PRED/NUM_SUBS*np.ones(NUM_SUBS)

#========== generate final mass =========================
m=mAcc*mu

#===============generate stellar mass according to infall mass====================
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

sigmaMstar=0.192
logMstar=np.log10(InfallMass2StellarMass(mAcc))
deltalogMstar=np.random.normal(0, sigmaMstar, NUM_SUBS)
mStar=10**(logMstar+deltalogMstar)
#============================generate concentration==========================
sigmaC=0.13 
deltalogC=np.random.normal(0, sigmaC, NUM_SUBS) #scatter
logC=np.log10(mean_concentration(mAcc,'MaccioW1')) #mean
cAcc=10**(logC+deltalogC)
#===============generate truncation radius and annihilation luminosity===========
SatHalo=[NFWHalo(mAcc[i], cAcc[i]) for i in range(NUM_SUBS)]
rt=np.array([SatHalo[i].radius(m[i]) for i in range(NUM_SUBS)]) #truncation radius
Lt=np.array([SatHalo[i].luminosity(rt[i]) for i in range(NUM_SUBS)]) #truncated luminosity

#============================save========================================

outfile=h5py.File(datadir+'MockCluster-%d.hdf5'%ithread,'w')
outfile.create_dataset('MHost',data=MASS_HOST)
dset=outfile.create_dataset('Common', data=common)
dset.attrs['rows']='count,m,mAcc,R,Rp,phi,mu'

dset=outfile.create_dataset('Maccio',data=Maccio)
dset.attrs['rows']='cAcc,rt,Lt,Lv'
dset.attrs['sigma_lnC']=sigmaC

dset=outfile.create_dataset('Ludlow',data=Ludlow)
dset.attrs['rows']='cAcc,rt,Lt,Lv'
dset.attrs['sigma_lnC']=sigmaC

dset=outfile.create_dataset('Ludlow1',data=Ludlow1)
dset.attrs['rows']='cAcc,rt,Lt,Lv'
dset.attrs['sigma_lnC']=sigmaC
outfile.close()

