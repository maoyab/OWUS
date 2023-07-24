# Inference of Optimal plant Water Use Strategies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5518546.svg)](https://doi.org/10.5281/zenodo.5518546) 

This repository contains scripts necessary to repeat analysis in:

M Bassiouni, S Manzoni, and G Vico (2023), Optimal plant water use strategies explain soil moisture variability. Advances in Water Resources,173, 104405. https://doi.org/10.1016/j.advwatres.2023.104405


- *WUE_theory.ipynb*: executes scripts in this repository to generate theoretical figures of plant water use strategies

- *WUE_application.ipynb*: executes scripts in this repository to visualize results from the inferrence of data-driven versus optimality-based plant water use strategies at study sites.

- *SPAC_traits_formulas.nb*: provides a simple parameterization of the soil-plant-atmosphere continuum and solves for leaf water potential, well-watered transpiration rate, and soil water potential at a given transpiration rate. (see details in https://doi.org/10.1002/2014WR015375)

- *sswm.py*: contains functions for a stochastic soil water balance model.

- *param_sm_pdf.py*: contains function to estimates ecohydrological parameters using a Metropolis‐Hastings Markov chain Monte Carlo algorithm.

- *select_site_records.py*: processes original data to create model inputs using functions in *data_management.py*

- *get_WUS_sites.py*: performs the inference of plant water use strategies using functions in *param_sm_pdf.py* and *sswm.py*

- *create_figs.py*: contains functions to generate all figures in the manuscript.


The FLUXNET2015 data products used in this study are available at http://fluxnet.fluxdata.org/data/fluxnet2015-dataset/.

