## Huiswerk
These are just some quick notes on what to do for the coming time for 
myself.

### 07-05
- code opruimen
	- celltypes in figuurtjes
- besluiten wat in verslag
	- data
	- experimenten
	- layout verslag
- vergelijken NUTS en VB
- uitzoeken:
	- derde persoon
	- presentatie
	- eisen lengte verslag
- misschien derde dataset

### 30-04
- replace boundaries with priors
- give sigma inverse gamma distribution as prior

### 23-04
- reparametrization pystan for underflow
- vectorization pystan
- ondergrens sigma
- for-loopje R moppcas omgooien, vector over N
- meer priors

### 16-4
- priors aan model toevoegen (sigma <- inv. gamma distr.)
- vb proberen met gewone gaussian mixture models
- kijk naar convergentie sampling en vb
- log likelihood per level
- advi/vb en nuts sampling beter begrijpen


### 9-4
- kijken of optimalizatie/vb sneller werkt
- anders parallelisatie of tensorflow proberen
- misschien vragen op forum naar pystan versnelling
- schrijven, iets meer over pystan sampling
- uitzoeken variational bayes en NUTS sampling


### 2-4
- meetbaarheid celltype separability met gewogen multinom regressie (gewogen zowel bij trainen als testen)
- data simuleren met splatter
- tensorflow bekijken moppcas


### 26-3
- transformeren data
	- log2(data+1)
	- normalisatie
- uitzoeken wat nestorowa met data heeft gedaan
- gmm met diagonale cov matrix ipv kmeans
- filteren na transformatie
- splatter bekijken
- model evaluatie door celltypes te scheiden mbv eenvoudige classifier
- evalueren impact van filtering, bottlenecks opzoeken


### 17-3
- AIC/BIC to estimate number of clusters
- apply weight to latent data moppcas
- check weighted ppca model, maybe use log of weights

### 5-3
- Look into GAP statistic to estimate number of clusters
- Obtain real scRNA-seq datasets
- Fit model on real datasets
- Try to evaluate model with and without ARD-prior
- Think about (combinations of) clustering methods
- Look into simulated dataset Philip
- Look into evaluation scores for both clustering and visualization and 
try to evaluate work

### 27-2
- ARD ppca specification
- ppca to initialize W for moppcas
- soft clustering for hierarchical mixtures
- selection in ppca dimension but clustering in original space
- modellen van perry

### 20-2
- optimizing methode, sigma z met lower bound
- ZIFA model specificatie
- mean vs best fit
- (optional) start on MoPPCAs

### older
- code werkend maken
- in de gaten houden dat data vergelijkbaar is.

- stukje afschrijven
- lees het Vpac stukje
- lees stukje over programmeertools


- git maken
- ZIFA bekijken en namaken
- stukje afschrijven (hoeft niet helemaal met herleidde formules)

- lambda schatten ipv h
- diagonale covariantie matrix

- EM algorithme in python voor ppca en ook closed form solution
- leer sampelen met stan
