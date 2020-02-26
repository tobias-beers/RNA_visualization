data{
    int<lower=0> N;// number  of  data  points  in  entire  dataset
    vector[N] y;//  observations
}


parameters{
    real mu;//  locations  of  mixture  components
    real<lower=0> sigma;   //  standard  deviations  of  mixture  components
}

model{
    //  priors
    mu~normal(0.0, 10.0) ;
    sigma~lognormal(0.0, 2.0) ;
    
    //  likelihood
    y~normal(mu, sigma);
}