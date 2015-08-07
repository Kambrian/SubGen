''' an example code to populate subhaloes in a cluster halo
Check the readme for more guidance.
'''
from subgen import *
from histogram import myhist 
plt.ion()

#============create the sample=========================
#initialize: sample 1e5 subhaloes down to 1e-5 host mass, inside a host halo of mass 1e14 Msun/h.
sample=SubhaloSample(M=1e4, N=1e5,MMinInfall=1e-5) 
#run the sampler. a plot will pop out showing you the actual MCMC steps. 
sample.populate()
#That's it! now save the data
sample.save("halo.txt")

#==========now visualize the distribution==============
Host=sample.Host
# plot the spatial distributions
def radial_PDF(R,weight,nbin=30):
  '''obtain radial distribution from input radius array R'''
  y,x=myhist(np.log(R),nbin,weights=weight)
  y/=(x[1]-x[0])
  x=np.exp(x[1:])
  p=y/4/np.pi/x**3/np.sum(weight[R<1])
  return x,p
plt.figure()
x,p=radial_PDF(sample.R,sample.weight)
plt.plot(x,p,'r-',label='Infall')
plt.plot(x, Host.density(x*Host.Rv)/Host.M*Host.Rv**3,label='Host')
mass_selection=(sample.m>sample.mAccMin) #select a resolution limited final mass sample
x,p=radial_PDF(sample.R[mass_selection], sample.weight[mass_selection])
plt.plot(x,p,'g-',label='Final')
plt.xlabel(r'$R/R_{200}$')
plt.ylabel(r'$\mathrm{d}P/\mathrm{d}^3(R/R_{200})$')
plt.legend()
plt.loglog()

#plot mass function
plt.figure()
selection=sample.R<1
y,x=myhist(np.log(sample.mAcc[selection]),50,weights=sample.weight[selection])
y=y/(x[1]-x[0]) #dN/dlnx
xmid=np.exp(x[:-1]+(x[1]-x[0])/2)
plt.plot(xmid, y, 'r-', label='Infall')
plt.plot(xmid, sample.HOD.A*Host.M*xmid**-sample.HOD.alpha, 'r--', label='Infall(Model)')
selection=(sample.R<1)&(sample.m>sample.mAccMin) 
y,x=myhist(np.log(sample.m[selection]),50,weights=sample.weight[selection])
y=y/(x[1]-x[0])
xmid=np.exp(x[:-1]+(x[1]-x[0])/2)
plt.plot(xmid, y, 'g-', label='Final')
plt.xlabel(r'$m[10^{10}M_{\odot}/h]$')
plt.ylabel(r'$dN/d\ln m$')
plt.legend()
plt.loglog()
  
##==further assign stellar mass and annihilation luminosity if needed================
sample.assign_stellarmass()
sample.assign_annihilation_emission()
#and you can generate projected radial coordinate
sample.project_radius()
#and you can save them all
sample.save("halo.txt", save_all=True)

#plot the annihilation profile
plt.figure()
selection=(sample.m>sample.mAccMin)
Lw=sample.L*sample.weight
y,x=myhist(np.log(sample.R)[selection], 50, weights=Lw[selection])
LHost=Host.Ls
x=np.exp(x)
y=y/np.diff(4./3.*np.pi*x**3)
xmid=x[:-1]*np.exp((x[1]-x[0])/2)
plt.plot(xmid,y/LHost,'r',label='Sub')
plt.plot(xmid, Host.density(xmid*Host.Rv)**2*Host.Rv**3/Host.Ls/LHost, 'g', label='Host')
plt.legend()
plt.loglog()
plt.xlabel(r'$R/R_{200}$')
plt.ylabel(r'$\mathrm{d}\tilde{L}_{\rm sub}/\mathrm{d}^3(R/R_{200})$')