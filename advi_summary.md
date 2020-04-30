### ADVI (automatic differentiation variational inference) works through these steps:

- The user inputs a model written in Stan
- ADVI transforms the support of the function to include only real-valued variables
- ADVI then minimizes KL(q(theta)||p(theta|x))
- ADVI recasts the gradient of the KL as an expectation over q
- ADVI reparameterizes the gradient in terms of a Gaussian by a transformation
- ADVI uses noisy gradients to optimize variational distribution

- minimize KL = maximize ELBO
- transform by using Weibull and Poisson distribution
- maximize elbo by stochastic gradient ascent
	- automatic differentiation to compute gradient
	- MC integration to approximate expectations
- elliptical standardization to make expectation known

### Variational Bayes

- tries to approximate the values of latent variables and parameters $Z$ given dataset $X$.
- approximates the posterior distribution of these parameters as $P(Q)$
- $P(Q) \approx P(Z|X)$, but $P(Q)$ will have a simpler form.
- Similarity between $P(Z|X)$ and $P(Q)$ si measured by Kullback-Leibler divergence of P from Q
- $KL(Q||P)$ can also be written as $KL(Q||P) = evidence - ELBO$, therefore, maximizing the ELBO minimizes the KL-divergence (the evidence is constant).


