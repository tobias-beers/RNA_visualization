# Models in PyStan

This directory includes several models specified in PyStan. The PyStan code is given in './StanModels'. These models were compile and saved in pickled format, the can be found in this form in './pickled_models'. Figures that were generated using these notebooks can be found in './figs'.
The iPython notebooks demonstrate the usage and performance of these models.


### Statistical models
Some of these notebooks describe the statistical model:

- Gaussian_dist_parameter_estiamtion.ipynb: Estimating parameters of a Gaussian distribution
- Gaussian_mixture_model.ipynb: Solving a GMM
- PPCA_part1.ipynb and PPCA-part2.ipynb: Solving a PPCA model
- PPCA_ARD.ipynb: Solving a PPCA model with Automatic Relevance Detection
- ZIFA.ipynb and ZIFA_simple.ipynb: Trying to solve a Zero-inflated Factor analysis by two approaches, does not work perfectly
- MoPPCAs.ipynb: Solving a MoPPCAs model
- Hierarchical_model.ipynb: Solving a HmPPCAs model
- Hierarchical_model_AutomaticClustering.ipynb: solving a HmPPCAs model with Automatic clustering

### Notebooks used for results
Other Notebooks were used to generate results:

- Hierarchical_model_Darmanis.ipynb: Evaluating the HmPPCAs model on the Darmanis Data-set
- Hierarchical_Model_nestorowa.ipynb: Evaluating the HmPPCAs model on the Nestorowa data-set
- Hierarchical_model_Splatter.ipynb: Evaluating the HmPPCAs model on the Splatter data-sets
- NUTS_VB_comparison.ipynb: A comparison of NUTS and ADVI
- UMAP.ipynb: A comparison of t-SNE and UMAP, also some comparisons with Stan
Data-sets used in these notebooks may be found in './pickled_data' and '../DataSets/'

### Others

- Samplers.ipynb shows how MC smapling, HMC and NUTS work.
- utils.py contains some functions and the complete HmPPCAs model used in some of the notebooks.

