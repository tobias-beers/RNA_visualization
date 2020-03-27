data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  int<lower=1> D;          // number of variables
  matrix[N,D] y;               // observations
}

parameters {
  simplex[K] theta;          // mixing proportions
  vector[D] mu[K];            // locations of mixture components
  vector<lower=0>[D] sigma[K];  // covariance matrices of mixture components
}

transformed parameters {
    matrix[D,D] covs[K];
    
    for (k in 1:K){
        for (i in 1:D){
            for (j in 1:D){
                if (i==j){
                    covs[k][i,j] = sigma[k][i];
                } else {
                    covs[k][i,j] = 0.0;
                }
            }
        }
    }
}

model {
    vector[K] log_theta = log(theta);  // cache log calculation
    
    //for (k in 1:K){        // punish exlcusion of clusters
        //if (theta[k]<1.0/(2*K)){
            //target+= negative_infinity();
        //}
    //}
    
    
    for (n in 1:N){
        vector[K] lps = log_theta;
        for (k in 1:K){
            lps[k] += multi_normal_lpdf(y[n,:] | mu[k], covs[k]);
            }
            target += log_sum_exp(lps);
        }
}

generated quantities{
    matrix[N,K] z;
    
    for (n in 1:N){
        for (k in 1:K){
            z[n,k] = multi_normal_lpdf(y[n,:]|mu[k], covs[k]);
            z[n,k] += log(theta[k]);
        }
    }
    
}