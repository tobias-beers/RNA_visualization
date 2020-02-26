data{
    int<lower=1> N;// number  of  data  points  in  entire  dataset
    vector[N] x;//  input
    vector[N] y;//  output
}


parameters{
    real alpha;// intercept
    real beta;// slope
    real<lower=0> sigma;
}

model{

    //  priors
    y~normal(alpha + beta*x, sigma);
    
}