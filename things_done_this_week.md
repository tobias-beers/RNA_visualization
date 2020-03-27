## Done for thursday (2-04)
- Solved negative numbers in Nestorowa data
- Looked at normalization Nestorowa data, only found that nomalizationw as performed with Rpackage Combat.
- made Pystan script for gmm with diagonal cov. matrix

## Done for thursday (26-3)
- Made cluster determination by AIC and BIC
	- Solved underflow problem in cluster determination
	- evaluated performance of both, AIC is good, bic is better
- Solved MoPPCAs problem.
- Gathered and integrated labels of nestorowa and darmaris dataset
- completed fully automated hierarchical algorithm

## Done for tuesday (17-3)

- Implemented ARI and weighted accuracy on hierarchical model as 
clustering evaluation score
- Found information coding to be one of the most important heuristics of 
visualization evaluation (Forsell & Johansson, 2010) and agree. 
Considering sum of Euclidean distances between original and recreated 
projection as evaluation measure for visualization purposes.
- Corrected ARD prior with respect to latent dimensions, also made ARD 
prior on observed dimension but was not effective.
- Gathered and formatted darmaris and nestorowa dataset
- Looked into GAP-statistic, encountered problems (found always too many 
or too few clusters.
	- Finding too many clusters was due to non-round shape of 
clusters. Decided it was better to divide into max. 2 clusters anyway, 
since more separation can always be performed on a lower level, where 
the latent space has a better definition.
	- also looked into silhouette score, was not applicable as it 
doesn't fit models of only one cluster.
	- Finding too few (usually just 1) clusters was due to very latent space, as MoPPCAs-model were very 
cohesive, 
separate GMM/k-means and PPCA produce better results.
		- Tried not to specify mean and covariance matrix of 
latent data of each cluster to avoid single-clustered latent data, did 
not work.
- tried fitting models to real datasets, takes very long (days), kernel 
crashes when fitting ppca with ARD prior on N-1 latent dimensions

## Done for thursday (5-3)

- soft clustering in hierarchical model
- clustering in full space instead of in latent space
- Tried to implement ARD, noticed that some implementations implement 
this on latent dimensions and others on observed dimensions, implemented 
on both.
- read and understood papers on i-ppca and panoview.

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
- Moved Stan scripts to separate folder outside of ntoebooks and created 
a 
loader to cache and load scripts to save compiling time.
- Completed the Mixture of PPCA's optimization in pystan, works 
perfectly.
- Completed the *interactive* version of the Hierarchical latent 
variable model as described by Bishop & Tipping in PyStan.
- started this note with an overview of completed work by week.
