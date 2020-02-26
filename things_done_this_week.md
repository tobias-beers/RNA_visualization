## Done for thursday (27-2)

- Latent variable models with contraints on std. dev. in attempt to make 
optimizing()-method work.
- Evaluated the difference between the mean of all samples and the 
sample with highest likelihood, wrote a little piece about it and 
applied the mean sample to all parrameters, except for parameters that 
indicate missing data such as latent data.
- Made extra ZIFA/ZIPPCA model where the model is specified instead of 
the log-likelihood. Found that the slight offset in results is due to 
the wrong assumption that x and y<sub>y!=0</sub> come from the 
same 
dsitribution.
- Generalized GMM to multivariate, discovered the importance of inital 
values  and started to initialize with k-means 
clustering algorithm
- Completed the Mixture of PPCA's optimization in pystan, works 
perfectly.
- Completed the *interactive* version of the Hierarchical latent 
variable model as described by Bishop & Tipping in PyStan.
