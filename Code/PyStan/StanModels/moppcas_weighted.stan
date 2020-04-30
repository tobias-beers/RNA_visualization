data {
  int<lower=1> K;          // number of mixture components
  int<lower=1> N;          // number of data points
  int<lower=1> M;
  int<lower=1> D;          // number of variables
  real<lower=0> lim_sigma_up[K];   // upper bounds of sigma
  real<lower=0> lim_sigma_low[K];   // lower bounds of sigma
  real<lower=0> mean_sigma[K];   // E[sigma]
  real<lower=0> std_sigma[K];   // std(sigma)
  vector[D] lim_mu_up[K];   // upper limits of mu
  vector[D] lim_mu_low[K];   // upper limits of mu
  vector[D] mean_mu[K];    // E[mu]
  vector[D] std_mu[K];    // std(mu)
  vector<lower=0,upper=1>[K] found_theta;
  matrix[N,D] y;               // observations
  vector[N] weights;
  matrix[N,K] found_R;
  
}

transformed data{

    vector[M] mean_z;
    matrix[M,M] cov_z;
    vector[M] std_z;
    vector[K] dir_prior;
    
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
    
    for (k in 1:K){
        dir_prior[k]=1.0;
    }
}

parameters {
  simplex[K] theta;          // mixing proportions
  vector<lower=0, upper=1>[D] raw_mu[K];            // locations of mixture components
  real<lower=0, upper=1> raw_sigma[K];  // covariance matrices of mixture components
  //real<lower=0.001> sigma[K];  // sigma
  //real<lower=0> mu[K];  // mu
  matrix[N,M] z;   // latent data
  matrix[D,M] W[K];   // factor loadings
}

transformed parameters {
    matrix[N,K] R;
    vector[D] mu[K];
    real<lower=0.0001> sigma[K];
    
    for (k in 1:K){
        mu[k] = lim_mu_low[k] + (lim_mu_up[k]-lim_mu_low[k]).*raw_mu[k];
        sigma[k] = lim_sigma_low[k] + (lim_sigma_up[k]-lim_sigma_low[k])*raw_sigma[k];
    }

    
    for (n in 1:N){
        for (k in 1:K){
            R[n,k] = log(theta[k]) + normal_lpdf(y[n,:] | W[k]*z[n,:]'+mu[k], sigma[k]);
        }
        R[n,:] = R[n,:] - max(R[n,:]);
        for (k in 1:K){
            R[n,k] = exp(R[n,k]);
        }
        R[n,:] = R[n,:]/sum(R[n,:]);
    }
    
    
}

model {
    vector[K] log_theta = log(theta);  // cache log calculation
    
    for (n in 1:N){
        vector[K] lps = log_theta;
        //target += dirichlet_lpdf(R[n,:]' | found_R[n,:]'+0.001);
        for (k in 1:K){
            target += weights[n]*R[n,k]*(log_theta[k] + normal_lpdf(y[n,:] | W[k]*z[n,:]'+mu[k], sigma[k]));
            }
        target += weights[n]*multi_normal_lpdf(z[n,:] | mean_z, cov_z);
    }
    
    // priors
    
    //for (k in 1:K){
       // for (d in 1:D){
       //     W[k][d,:] ~ normal(0,1.5);
       //     }
       //}
    theta ~ dirichlet(dir_prior);
    
    for (k in 1:K){
        target += normal_lpdf(mu[k]|mean_mu[k], std_mu[k]);
        target += normal_lpdf(sigma[k]|mean_sigma[k], 0.5*std_sigma[k]);
        theta[k] ~ normal(found_theta[k], 0.025);
        for (n in 1:N){
            target+= normal_lpdf(R[n,k] | found_R[n,k], 0.05);
        }
    }
    
    //for (k in 1:K){
        //raw_mu[k] ~ normal((mean_mu[k]-lim_mu_down[k])/(lim_mu_up[k]-lim_mu_low[k]), );
        //sigma[k] = lim_sigma_low[k] + (lim_sigma_up[k]-lim_sigma_low[k])*raw_sigma[k];
    //}
    
}
