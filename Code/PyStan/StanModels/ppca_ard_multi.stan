data{
    int<lower=0> N;// number  of  datapoints
    int<lower=0> D;// number  of  dimensions  in  observed  dataset
    int<lower=0> M;// number  of  dimensions  in  latent  dataset
    vector[D] x[N];//  observations
    int<lower=0, upper=1> latent_ard;  //pass 1 to use ARD prior on the latent dimensions
    int<lower=0, upper=1> observed_ard; // pass 1 to use ARD prior on the observed dimensions
}


parameters{
    matrix[M,N] z;  // latent data
    matrix[D,M] W;  // factor loadings
    real<lower=0> sigma;   //  standard  deviations
    vector[D] mu;   //  added means
    vector<lower=0>[M] alpha;  // variance explained by observed variables
    vector<lower=0>[D] beta;  // variance explained by observed variables
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
        
    if (latent_ard==1){
        alpha~inv_gamma(1.0,1.0);
        for (d in 1:D){
            W[d] ~ normal(0.0,alpha);
        }
    }
        
    if (observed_ard==1){
        beta~inv_gamma(1.0,1.0);
        for (m in 1:M){
            W[:,m] ~ normal(0.0,beta);
        }
    }
       
    //  likelihood
    for (n in 1:N){
        z[:,n] ~ multi_normal(mean_z, cov_z);
        x[n] ~ normal(W*col(z,n)+mu, sigma);
        }
        
}