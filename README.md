
This repository contains code to reproduce figures and analysis from the paper "An effective medium equation for weakly nonlinear shallow water waves over periodic bathymetry".

## Files

- `reproduce_simulation_figures.ipynb`: Jupyter notebook with code to reproduce all the simulation figures.  Some of the figures depend on Clawpack simulations; you can either install Clawpack and run them yourself (instructions in the notebook) or use data files contained in this repository.

The remaining files are Mathematica notebooks with symbolic calculations:

- `full_derivation_v1.0.nb`: The main process of deriving the homogenized equations
- `linear_terms_only.nb`: Homogenization performed on the linearized shallow water equations.  This yields just the linear dispersive terms.
- `pwc_coefficient_formulas.nb`: Computes explicit formulas for some of the coefficients, in the case of piecewise-constant bathymetry
- `LeVeque-Yong redux.nb`: This isn't directly part of the paper, but shows how to reproduce the analysis from the 2003 paper of LeVeque and Yong where this technique was pioneered.  It may be useful (as a second example) for others who wish to perform similar analysis.
