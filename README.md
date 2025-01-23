# Summer '24 CNN Research

This repo contains all relevant files from my summer 2024 research into using 1D CNNs for parameter estimation in Nicholson's Blowfly model. I worked under the supervision of Dr Amanda Lenzi, and used ideas introduced in [Neural networks for parameter estimation in intractable models](https://www.sciencedirect.com/science/article/pii/S0167947323000737) by Lenzi et al. The repo is organised into two main parts: summary notes and code.

---

### Draft Paper & Notes

- `manuscript-draft` contains a (sadly unfinished) draft of a paper summarising the findings of this research, including some commentary on motivation and full referencing.
- `early-research-notes` contains initial observations I had when experimenting with using NNs for different kinds of parameter estimation, early into the project.

### Code

- `MVEs` contains code for designing a mean variance estimation NN, used for estimating parameters of a Gaussian distribution. This is not related to the Blowfly model estimation, and was just a means of familiarising myself with using NNs for parameter estimation.
- `Blowfly model (5 param)` contains all code used in `manuscript-draft`, namely the code used for estimating all 5 parameters in Nicholson's Blowfly model. In particular:
    - `CNN_designs` contains functions to define 2 different CNN designs which I experimented with for the estimation problem.
    - `functions.R` contains various functions used (i) for plotting results (for the paper) and (ii) implementing the synthetic likelihood inferential method.
    - `5_param_estimation` uses the above 2 files to design and implement a CNN for the estimation problem, and then compare it against the SL method.
    - `param_chains.Rdata` saves the particular instance of the MCMC chains whose plots are shown in `manuscript-draft`.
- `Blowfly model (3 param)` mirrors the same structure as the 5 param file above, but for the setting where only 3 out of the 5 parameters of the Blowfly model are being estimated. I didn't actually run a full (SL) MCMC algorithm for this, but the code to do it is there nonetheless.
- `sl_0.0-6.tar.gz` is the custom `R` package for implementing the SL method, provided by Professor Simon Wood. An earlier version of the package is available in the supplementary information of his paper [here](https://www.nature.com/articles/nature09319#Bib1).
