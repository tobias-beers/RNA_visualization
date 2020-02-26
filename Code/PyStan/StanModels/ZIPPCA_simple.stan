data{
    int<lower=0> N; // number  of  datapoints
    int<lower=0> D; // number  of  dimensions  in  observed  dataset
    int<lower=0> M; // number  of  dimensions  in  latent  dataset
    matrix[N,D] y; // zero inflated observations
}

transformed data{
    int h[N,D]; // zero-inflation
    
    for (n in 1:N){
        for (d in 1:D){
            if (y[n,d] == 0.0){
                h[n,d] = 1;
            } else {
                h[n,d] = 0;
            }
        }
    }
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
    
    matrix[D,D] W;
    for (i in 1:D){
        for (j in 1:D){
            if (i==j){
                W[i,j] = sigma;
            } else {
                W[i,j] = 0.0;
            }
        }
    }
    
    for (n in 1:N){
        z[:,n] ~ multi_normal(mean_z, cov_z);
        x[n,:] ~ multi_normal(A*z[:,n]+mu,W);
        for (d in 1:D){
            h[n,d] ~ bernoulli(exp(-lambda*square(x[n,d])));
            if (y[n,d] != 0.0){
                y[n,d] ~ normal(row(A,d)*col(z,n)+mu[d], sigma);
                } else {
                //x[n,d] ~ normal(row(A,d)*col(z,n)+mu[d], sigma);
                }
        }
    }
}