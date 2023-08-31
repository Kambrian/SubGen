## `SubGen`: a fast subhalo sampler 
`SubGen` generates Monte-Carlo samples of dark matter subhaloes, according to the unified subhalo distribution model in Han et. al. (2015; http://arxiv.org/abs/1509.02175).

This sampler works only for CDM subhaloes. For an extension of the code to also work with WDM subhaloes, see [`SubGen2`](https://github.com/fhtouma/subgen2/tree/main)

### Get Started
To get started, check `example.py`. You only need to specify the host halo mass, and the number of subhaloes to sample.

### Prerequisites

You need a python installation with the core scientific packages: `numpy`, `matplotlib` and `scipy`.
You also need the `emcee` package(http://dan.iel.fm/emcee) for MCMC sampling. Try

     easy_install numpy matplotlib scipy emcee

or

     pip install numpy matplotlib scipy emcee

to install these dependences if you miss them.

### The basic sample contains:

- R:     radial coordinate of subhalo, in unit of host R200
- m:     subhalo mass, in unit of 10^10Msun/h. By default, disrupted subhaloes are also included in the sample (which may not be useful at all). You can suppress the creation of disrupted subhaloes, and only obtain survived population, by, e.g., 

        sample=SubhaloSample(M=1e4, include_disruption=False)

- mAcc:  subhalo infall mass, in unit of 10^10Msun/h
- weight: the number of appearances of this subhalo. This exists because the number of sampled subhaloes may not be the same as the expected number of subhaloes. The subhalo abundance can be correctly recovered when counting with this weight. By default, the weights are not uniform but determined by mass function in order to avoid the sample being dominated by low mass objects. If you want uniform weights, you can do, e.g., 

        sample=SubhaloSample(M=1e4, weighted_sample=False).

### optional properties:  
  - mStar/[1e10Msun/h],  the subhalo stellar mass
  - Luminosity/[(1e10Msun/h)^2/(kpc/h)^3],  annihilation luminosity (adopting Ludlow14 mass-concentration by default)
  - Rp/R200,  projected (along a line of sight) radial coordinate Rp. 

For complete features, have a look at the docstrings and the source code (they are not long~).

## Authors
Jiaxin Han (@Kambrian)[ICC, Durham]
http://kambrian.github.io/SubGen
