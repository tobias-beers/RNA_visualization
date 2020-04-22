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

//transformed parameters {
//    matrix[D,D] covs[K];
    
//    for (k in 1:K){
//        for (i in 1:D){
//            for (j in 1:D){
  //              if (i==j){
 //                   covs[k][i,j] = sigma[k][i];
//                } else {
//                    covs[k][i,j] = 0.0;
//                }
//            }
//        }
//    }
//}

model {
    //real probclus[K]; 
    //real prob_in_clus;
    vector[K] log_theta = log(theta);  // cache log calculation
    
    //for (k in 1:K){        // punish exlcusion of clusters
    //    if (theta[k]<1.0/(2*K)){
    //        target+= negative_infinity();
    //    }
    //}
    
    
    for (n in 1:N){
        vector[K] lps = log_theta;
        //for (k in 1:K){
          //  probclus[k] = exp(log_theta[k]+normal_lpdf(y[n,:] | mu[k], sigma[k]));
        //}
        for (k in 1:K){
            //prob_in_clus = probclus[k]/sum(probclus);
            lps[k] += normal_lpdf(y[n,:] | mu[k], sigma[k]);
            }
            target += log_sum_exp(lps);
        }
}

generated quantities{
    matrix[N,K] z;
    vector[K] log_theta = log(theta);
    
    for (n in 1:N){

        for (k in 1:K){
            z[n,k] = exp( log_theta[k] + normal_lpdf(y[n,:]|mu[k], sigma[k]) + log(theta[k]) );
        }
        z[n,:] = z[n,:]/sum(z[n,:]);
    }
    
}