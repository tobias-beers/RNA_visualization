data{
    int<lower=0> N;// number  of  datapoints
    int<lower=0> D;// number  of  dimensions  in  observed  dataset
    int<lower=0> M;// number  of  dimensions  in  latent  dataset
    vector[D] x[N];//  observations
    vector<lower=0, upper=1>[N] weights; // weights
}


parameters{
    matrix[M,N] z;  // latent data
    matrix[D,M] W;  // factor loadings
    real<lower=0> sigma;   //  standard  deviations
    vector[D] mu;   //  added means
}

transformed parameters{
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
    //  priors
    //for (d in 1:D){
    //    W[d] ~ normal(0.0,sigma);
    //    mu[d]~normal(0.0, 5.0) ;
    //    }
    //sigma~lognormal(0.0, 1.0) ;
    
    
    //  likelihood
    for (n in 1:N){
        target+=weights[n]*multi_normal_lpdf(z[:,n]|mean_z, cov_z);
        target+=weights[n]*normal_lpdf(x[n]|W*col(z,n)+mu, sigma);
        }
        
}