data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  int<lower=2> D;          // number of observed variables
  int<lower=1> M;          // number of latent dimensions
  matrix[N,D] y;           // observations
}

parameters {
  simplex[K] theta;          // mixing proportions
  vector[D] mu[K];            // locations of mixture components
  real<lower=0,upper=5> sigma[K];  // scales of mixture components
  matrix[M,N] z[K];  // latent data
  matrix[D,M] W[K];  // factor loadings
  
}

transformed parameters{
    vector[M] mean_z;
    matrix[M,M] cov_z;
    matrix[D,D] covs[K];
    
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
    
    for (k in 1:K){
        for (i in 1:D){
            for (j in 1:D){
                if (i==j){
                    covs[k][i,j]=sigma[k];
                } else {
                    covs[k][i,j]=0.0;
                }
            }
        }
    }
    
}

model {

    vector[K] log_theta = log(theta);  // cache log calculation
    real probclus[K]; 
    real prob_in_clus;
    
    for (k in 1:K){
        if (theta[k]<1.0/(2*K)){
            target+= negative_infinity();
        }
    }    

    
    for (n in 1:N){
    
    vector[K] lps = log_theta;
        for (k in 1:K){
            probclus[k] = exp(log_theta[k]+multi_normal_lpdf(y[n,:] | W[k]*col(z[k],n)+mu[k], covs[k])+multi_normal_lpdf(z[k][:,n]|mean_z, cov_z));
        }
        for (k in 1:K){
            prob_in_clus = probclus[k]/sum(probclus);
            //target += prob_in_clus*log_theta[k];
            target += prob_in_clus*multi_normal_lpdf(y[n,:] | W[k]*col(z[k],n)+mu[k], covs[k]);
            target += prob_in_clus*multi_normal_lpdf(z[k][:,n]|mean_z, cov_z);
        }
    }
}

generated quantities{
    matrix[N,K] clusters;
    vector[K] log_theta = log(theta);  // cache log calculation
    
    for (n in 1:N){
        for (k in 1:K){
            clusters[n,k] = exp(log_theta[k]+multi_normal_lpdf(y[n,:] | W[k]*col(z[k],n)+mu[k], covs[k]));
        }
        clusters[n,:] = clusters[n,:]/sum(clusters[n,:]);
    }
    
}