data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  int<lower=1> D;          // number of variables
  matrix[N,D] y;               // observations
  vector<lower=0,upper=1>[N] weights;     // weights
}

parameters {
  simplex[K] theta;          // mixing proportions
  vector[D] mu[K];            // locations of mixture components
  cov_matrix[D] sigma[K];  // covariance matrices of mixture components
}


model {
    vector[K] log_theta = log(theta);  // cache log calculation
    
    
    for (k in 1:K){        // punish exlcusion of clusters
        if (theta[k]<1.0/(2*K)){
            target+= negative_infinity();
        }
    }
    
    
    for (n in 1:N){
        vector[K] lps = log_theta;
        for (k in 1:K){
            lps[k] += multi_normal_lpdf(y[n,:] | mu[k], sigma[k]);
            }
            target += weights[n]*log_sum_exp(lps);
        }
}

generated quantities{
    matrix[N,K] z;
    matrix[N,K] z_weighted;
    
    for (n in 1:N){
        for (k in 1:K){
            z[n,k] = multi_normal_lpdf(y[n,:]|mu[k], sigma[k]);
            z[n,k] += log(theta[k]);
            z_weighted[n,k] = weights[n]*multi_normal_lpdf(y[n,:]|mu[k], sigma[k]);
            z_weighted[n,k] += weights[n]*log(theta[k]);
        }
    }
    
}