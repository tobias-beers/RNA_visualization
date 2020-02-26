data{
    int<lower=0> N; // number  of  datapoints
    int<lower=0> D; // number  of  dimensions  in  observed  dataset
    int<lower=0> M; // number  of  dimensions  in  latent  dataset
    matrix[N,D] y; // zero inflated observations
}

parameters{
    matrix[M,N] z;  // latent data
    matrix[D,M] A;  // factor loadings
    real<lower=0> sigma;   //  standard  deviations
    real<lower=0, upper=0.5> lambda; // zero-inflation
    vector[D] mu;   //  added means
    matrix[N,D] x; // non-inflated observations
}

transformed parameters{
    // means and covariacne matrix of z
    vector[M] mean_z;
    matrix[M,M] cov_z;
    
    for (m in 1:M){
        mean_z[m] = 0.0;
        for (n in 1:M){
            if (m==n){
                cov_z[m,n]=1.0;
            } else{
                cov_z[m,n]=0.0;
            }
        }
    }
}

model{
    
    for (n in 1:N){
        target+=multi_normal_lpdf(z[:,n]|mean_z,cov_z);
        for (d in 1:D){
            if (y[n,d]==0){
                // exact formula (Works only for Bernoulli process for some reason)
                //target+=-square(x[n,d]-row(A,d)*col(z,n)+mu[d])/(2*square(sigma)) - 0.5*log(square(sigma)) - lambda*square(x[n,d]);
                target+= -lambda*square(x[n,d]);
                
                // PyStan implementation (Works only for Gaussian distribution for some reason)
                target+=normal_lpdf(x[n,d]|row(A,d)*col(z,n)+mu[d], sigma);
                //target+=bernoulli_logit_lpmf(1|-square(x[n,d])*lambda);
                
            } else {
                
                // exact formula (Works only for Bernoulli process for some reason)
                //target+=-square(y[n,d]-row(A,d)*col(z,n)+mu[d])/(2*square(sigma)) - 0.5*log(square(sigma)) + log(1-exp(-lambda*square(y[n,d])));
                //target+=-square(x[n,d]-row(A,d)*col(z,n)+mu[d])/(2*square(sigma)) - 0.5*log(square(sigma)) + log(1-exp(-lambda*square(y[n,d])));
                target+=log(1-exp(-lambda*square(y[n,d])));
                
                // PyStan implementation (Works only for Gaussian distribution for some reason)
                target+=normal_lpdf(x[n,d]|row(A,d)*col(z,n)+mu[d], sigma);
                target+=normal_lpdf(y[n,d]|row(A,d)*col(z,n)+mu[d], sigma);
                //target+=bernoulli_logit_lpmf(1|log(1-exp(-lambda*square(y[n,d]))));
            }
        }
    }
}