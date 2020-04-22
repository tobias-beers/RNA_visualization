data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  int<lower=1> M;
  int<lower=1> D;          // number of variables
  real<lower=0> lim_sigma_up[K];   // upper limits of sigma
  vector[D] lim_mu_up[K];   // upper limits of mu
  vector[D] lim_mu_low[K];   // upper limits of mu
  matrix[N,D] y;               // observations
}

parameters {
  simplex[K] theta;          // mixing proportions
  vector<lower=0, upper=1>[D] raw_mu[K];            // locations of mixture components
  real<upper=0> raw_sigma[K];  // covariance matrices of mixture components
  matrix[N,M] z;   // latent data
  matrix[D,M] W[K];   // factor loadings
}

transformed parameters {
    matrix[N,K] R;
    vector[M] mean_z;
    matrix[M,M] cov_z;
    vector[M] std_z;
    vector[K] dir_prior;
    vector[D] mu[K];
    real<lower=0> sigma[K];
    
    for (k in 1:K){
        mu[k] = lim_mu_low[k] + (lim_mu_up[k]-lim_mu_low[k]).*raw_mu[k];
        sigma[k] = raw_sigma[k] + lim_sigma_up[k];
    }
    
    for (i in 1:M){
        mean_z[i] = 0.0;
        std_z[i] = 1.0;
        for (j in 1:M){
            if (i==j){
                cov_z[i,j] = 1.0;
            } else {
                cov_z[i,j] = 0.0;
            }
        }
    }

    
    for (n in 1:N){
        for (k in 1:K){
            R[n,k] = exp( log(theta[k]) + normal_lpdf(y[n,:] | W[k]*z[n,:]'+mu[k], sigma[k]) );
        }
        R[n,:] = R[n,:]/sum(R[n,:]);
    }

    for (k in 1:K){
        dir_prior[k]=1.0;
        }
}

model {
    //real probclus[K]; 
    //real prob_in_clus;
    vector[K] log_theta = log(theta);  // cache log calculation
    
    for (k in 1:K){
        for (d in 1:D){
            W[k][d,:] ~ normal(0,1.5);
            }
       }
    theta ~ dirichlet(dir_prior);
    
    //for (k in 1:K){        // punish exlcusion of clusters
    //    if (theta[k]<1.0/(2*K)){
    //        target+= negative_infinity();
    //    }
    //}
    
    
    for (n in 1:N){
        vector[K] lps = log_theta;
        //for (k in 1:K){
          //  probclus[k] = exp(log_theta[k]+normal_lpdf(y[n,:] | W[k]*col(z,n)+mu[k], sigma[k]));
        //}
        for (k in 1:K){
            //prob_in_clus = probclus[k]/sum(probclus);
            //lps[k] += normal_lpdf(y[n,:] | mu[k], sigma[k]);
            target += R[n,k]*(log_theta[k] + normal_lpdf(y[n,:] | mu[k], sigma[k]));
            target += R[n,k]*normal_lpdf(z[n,:] | mean_z, std_z);
            }
            //target += log_sum_exp(lps);
        }
}
