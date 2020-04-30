import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pystan
import numpy as np
import seaborn as sns
import itertools
from sklearn.cluster import KMeans
from sklearn import datasets
import os
import pickle
import scipy
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import SpectralClustering
from scipy.stats import norm,multivariate_normal
from sklearn.metrics import adjusted_rand_score
sns.set()


def loadStan(file, recompile=False, automatic_pickle = True, parallel=False):
    print('Loading model ', file)
    if parallel:
        file_p = file+'_p'
        os.environ['STAN_NUM_THREADS'] = "8"
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']
    else:
        file_p = file
    if recompile:
        try:
            if parallel:
                model = pystan.StanModel(file = 'StanModels/'+file+'.stan', extra_compile_args=extra_compile_args)
            else:
                model = pystan.StanModel(file = 'StanModels/'+file+'.stan')
            print('Model compiled succesfully.')
            if automatic_pickle:
                with open('pickled_models/'+file_p+'.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print('Model saved succesfully to cache.')
        except FileNotFoundError:
            print(file+'.stan not found!')
        except ValueError:
            print('Could not compile! Error in code maybe!')
    else:
        try:
            model = pickle.load(open('pickled_models/'+file_p+'.pkl', 'rb'))
            print('Model loaded succesfully from cache.')
        except:
            try:
                if parallel:
                    model = pystan.StanModel(file = 'StanModels/'+file+'.stan', extra_compile_args=extra_compile_args)
                else:
                    model = pystan.StanModel(file = 'StanModels/'+file+'.stan')
                print('Model compiled succesfully.')
                if automatic_pickle:
                    with open('pickled_models/'+file_p+'.pkl', 'wb') as f:
                        pickle.dump(model, f)
                    print('Model saved succesfully to cache.')
            except FileNotFoundError:
                print(file+'.stan not found!')
            except ValueError:
                print('Could not compile! Error in code maybe!')

    return model

def GAP(points, k_max=2, nref=10):
    gap = []
    kmeans_list = []
    n,dims = np.shape(points)
    for K_clus in range(1,k_max+1):

        kmeans_init = KMeans(K_clus).fit(points)
        kmeans_list.append(kmeans_init)
        obs = np.log(kmeans_init.inertia_)

        exp = 0
        tops = points.max(axis=0)
        bots = points.min(axis=0)
        for i in range(nref):
            points_ref = np.random.uniform(bots, tops, (n, dims))
            kmeans_ref = KMeans(K_clus).fit(points_ref)
            exp+=np.log(kmeans_ref.inertia_)
        exp = exp/nref
        gap.append(exp-obs)

    return list(range(1,k_max+1))[np.argmax(gap)], kmeans_list[np.argmax(gap)], gap

def est_k(points,k_min = 1, k_max = 2, refs = 3, retry=2, method='bic', verbose = False, gmm=False, weights=None):
    
    quit = False
    clus = []
    aics = []
    bics = []
    mods = []
    
    N, D = np.shape(points)
    if k_min>k_max:
        print('k_min must be smaller than k_max!')
        return
    if np.all(weights==None):
        weights = np.ones(N)
    for k in range(k_min,k_max+1):
        if quit:
            continue
        if k ==1:    # for k=1, the result will always be the same, no need for multiple testing
            n_ref = 1
        else:
            n_ref = refs
        for ref in range(n_ref):
            if verbose:
                print('%i clusters, model %i of %i'%(k,ref+1,n_ref))
            model = KMeans(k).fit(points)
            if verbose:
                print('Model built!')
            model_data = {}
            theta = [np.sum(model.labels_==k_i)/N for k_i in range(k)]
            if gmm:
                count = 0 
                while True:
                    if verbose:
                        print('Building GMM model with %i clusters'%k)
                    gmm_dat = {'N': N,'K': k, 'D':D, 'y':points, 'weights':weights}
                    fit = gmm_diag_weighted.sampling(data=gmm_dat, chains=1, iter=150, init=[{'mu':model.cluster_centers_, 'theta':theta}])
                    if verbose:
                        print('Model built!')
                    fit_ext = fit.extract()
                    z_sim = np.mean(fit_ext['z'],axis=0)
                    labels = np.argmax(z_sim, axis=1)
                    if np.all(np.array([sum(labels==k_i) for k_i in range(k)])>1):
                        a, b =kmeans_AIC(points,labels, k, verbose=verbose)
                        model_data['mu'] = np.mean(fit_ext['mu'],axis=0)
                        model_data['theta'] = np.mean(fit_ext['theta'],axis=0)
                        model_data['labels'] = labels
                        break
                    count+=1
                    print('Failed to find model with right number of clusters (trial %i of %i)'%(count,retry))
                    if count>=retry:
                        print('No model with ', k, ' clusters found, showing cluster estimates with k_max at ',k-1)
                        quit = True
                        break
            else:
                a, b =kmeans_AIC(points,model.labels_, k, verbose=verbose)
                model_data['mu'] = model.cluster_centers_
                model_data['theta'] = theta
                model_data['labels'] = model.labels_
                model_data['min_mus'] = [np.min(points[model.labels_==k_i],axis=0) for k_i in range(k)]
                model_data['std_mus'] = [np.std(points[model.labels_==k_i],axis=0)+0.0001 for k_i in range(k)]
                model_data['max_mus'] = [np.max(points[model.labels_==k_i],axis=0) for k_i in range(k)]
                model_data['max_sigmas'] = [np.max(np.std(points[model.labels_==k_i],axis=0)) for k_i in range(k)]
                model_data['min_sigmas'] = [np.min(np.std(points[model.labels_==k_i],axis=0))+0.0001 for k_i in range(k)]
                model_data['mean_sigmas'] = [np.mean(np.std(points[model.labels_==k_i],axis=0))+0.0001 for k_i in range(k)]
                model_data['std_sigmas'] = [np.std(np.std(points[model.labels_==k_i],axis=0))+0.0001 for k_i in range(k)]
            
            
            if quit == False:
                mods.append(model_data)
                clus.append(k)
                aics.append(a)
                bics.append(b)
    if clus[np.argmin(aics)]!=clus[np.argmin(bics)]:
        print(clus[np.argmin(aics)], ' clusters according to AIC, ', clus[np.argmin(bics)], ' clusters according to BIC.')
    if method=='aic':
        return clus[np.argmin(aics)], mods[np.argmin(aics)]
    elif method=='bic':
        return clus[np.argmin(bics)], mods[np.argmin(bics)]
    elif method=='both':
        return clus[np.argmin(aics)], mods[np.argmin(aics)],clus[np.argmin(bics)], mods[np.argmin(bics)]
    else:
        print("Choose 'aic' or 'bic' as method!")
        return 1
    
def kmeans_AIC(points, labels, K, verbose = False):
    if verbose:
        print('Evaluating %i clusters'%K)
    N,D = np.shape(points)
    probs = np.zeros((N,K))
    theta = [np.sum(labels==k)/K for k in range(K)]
    mu = [np.mean(points[labels==k],axis=0) for k in range(K)]
    sigmas = [np.std(points[labels==k],axis=0) for k in range(K)]

    for k in range(K):
        probs[:,k] = multivariate_normal.logpdf(points, mean=mu[k], cov=np.diag(sigmas[k]), allow_singular=True) 
        
    probs+=np.log(theta)
    llh = np.sum(logsumexp(probs,axis=1))

    aic = -2*llh + 2*K*D
    bic = -2*llh + np.log(N)*K*D
    if verbose:
        print('Evaluation complete: n_clusters: ',K, ', AIC: ',aic, ', BIC: ',bic)
        print()
    return aic, bic


class hierarchical_model:
    
    def __init__(self):
        
        self.latent = []
        self.mus = [[]]
        self.cats_per_lvl = []
    
    def fit(self,x, M=2, max_depth=5, k_max =3, plotting=True, min_clus_size=10, vis_threshold=0.05, gmm = False, its=300, samplingmethod='VB', n_try=3):
        
        if k_max <3:
            print("It is suggested to give 'k_max' a value of at least 3.")
        # initialize the algorithm
        self.M = M
        
        if M>3 and plotting:
            print('Latent dimensions greater than 3, plotting only the three most important dimensions.')
        moppcas_weighted = loadStan('moppcas_weighted')
        ppca_weighted = loadStan('ppca_weighted')
        
        

        N,D = np.shape(x)
        self.N = N
        self.probs = [np.ones((N,1))]
        self.colors = [np.random.uniform(size=3) for k in range(k_max**max_depth)]
        
        if gmm:
            est_ref = 1
        else:
            est_ref = 5

        # top-level latent data
        print('Latent data on top level:')
        ppca_dat = {'N':N, 'M':M, 'D':D, 'x':x, 'weights': self.probs[-1][:,0]}
        if samplingmethod == 'VB':
            fit = ppca_weighted.vb(data=ppca_dat)
            df = pd.read_csv(fit['args']['sample_file'].decode('ascii'), comment='#').dropna()
            dfmean = df.mean()
            dfz = dfmean[dfmean.index.str.startswith('z.')]
            latent_top = np.array(dfz).reshape(self.N, M)
        elif samplingmethod == 'NUTS':
            fit_top = ppca_weighted.sampling(data=ppca_dat, iter=its, chains=1)
            fitreturn_top = fit_top.extract()
            best_ind_top = np.where(fitreturn_top['lp__']==max(fitreturn_top['lp__']))[0][0]
            latent_top = fitreturn_top['z'][best_ind_top]
        else:
            print("Please use 'NUTS' or 'VB' as samplingmethod!")
            return 1

        # top-level cluster determination
        K_1, clusters_1 = est_k(x, k_max = k_max, gmm = gmm, refs = est_ref)
        self.mus[-1].append(clusters_1['mu'])
        print('Estimated number of clusters on top-level data: %i (out of a maximum of) %i'%(K_1, k_max))
        
        # in case M>2, find out which three dimensions to plot in 3D
        if M>2:
            chosen = []
            for dim in range(3):
                s = 0
                for i in range(self.M):
                    if i not in chosen:
                        if np.std(self.latent[0][0][i,:])>s:
                            s = np.std(self.latent[0][0][i,:])
                            best = i
                chosen.append(best)
            self.dimx,self.dimy,self.dimz = chosen
        
        # plot top-level latent data with coloured first clusters
        if plotting:
            fig = plt.figure(figsize=(6,6))
            rgba_colors = np.ones((N,4))
#             for k_i in range(K_1):
#                 rgba_colors[clusters_1['labels']==k_i,:3] = self.colors[k_i]
            rgba_colors[:,:3] = self.colors[0]
            if M==2:
                ax = fig.add_subplot(111)
                ax.scatter(latent_top[:,0],latent_top[:,1], c = rgba_colors, zorder=1)
            if M>2:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(latent_top[self.dimx,:],latent_top[self.dimy,:],latent_top[self.dimz,:], c = rgba_colors, zorder=1)
#             for k_i in range(K_1):
#                 if M==2:
#                     cc_x, cc_y = np.mean(latent_top[clusters_1['labels']==k_i],axis=0)
#                     ax.scatter(cc_x, cc_y, c='black', s=500, zorder=10)
#                     ax.text(cc_x, cc_y, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
#                 elif M>2:
#                     cc_x, cc_y, cc_Z = np.mean(latent_top[clusters_1['labels']==k_i],axis=0)
#                     ax.scatter(cc_x, cc_y, cc_z, c='black', s=500, zorder=10)
#                     ax.text(cc_x, cc_y, cc_z, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
            ax.set_title('Top-level latent data')
            plt.show()

            
            

        cats = np.argmax(self.probs[-1],axis=1)
        self.cats_per_lvl.append(cats.copy())
        self.latent.append(latent_top.copy())
#         self.probs.append(np.ones((N,4)))
        
        
        for lvl in range(max_depth):
            # repeat cluster detemination and MoPPCAs until max_depth is reached or until all clusters fully analyzed
            more_depth = False
            print('level %i:'%(lvl+1))
            lvl_probs = self.probs[-1].copy()
            n_clus = np.shape(lvl_probs)[1]
            count = 0
            
            levelcats = np.argmax(lvl_probs,axis=1)
            lvl_latents  = np.zeros((N,2))
            probs_round = np.zeros((N,0))
            
            
            # analyze all subclusters as found in the last level
            for cl in range(n_clus):
                print('Cluster %i:'%(cl+1))
                clus_probs = lvl_probs[:,cl]
                if plotting and lvl>0:
                    rgba_colors = 0.2*np.ones((N,4))    
                    plt.scatter(latent_top[:,0], latent_top[:,1], c=rgba_colors)
#                     rgba_colors = np.zeros((N,4))
                    rgba_colors[:,:3] = self.colors[cl]
                    rgba_colors[:,3] = clus_probs
                    plt.scatter(latent_top[:,0], latent_top[:,1], c=rgba_colors)
                    plt.title('Analysing cluster %i'%(cl+1))
                    plt.show()
                n_subc_clus = []
                
                
                
                # Dont divide clusters further if they are too small
                if sum(levelcats==cl)>k_max:
                    n_subs, subs = est_k(x[levelcats==cl], k_max = k_max, gmm = gmm, refs = est_ref, weights=clus_probs)
                    while np.any([sum(subs['labels']==k_i)<min_clus_size for k_i in range(n_subs)]):
                        if n_subs <= 2:
                            n_subs = 1
                            break
                        n_subs, subs = est_k(x[levelcats==cl], k_max = n_subs-1, gmm = gmm, refs = est_ref, weights=clus_probs)
                else:
                    n_subs = 1

                if n_subs == 1:
                    # If cluster doesnt contain more subclusters, just copy it over to the next level
                    clus_latent = self.latent[-1]
                    print('Cluster ', cl+1, ' doesnt contain any more subclusters')
                    new_probs = clus_probs[np.newaxis].T
                    lvl_latents += clus_latent*clus_probs[np.newaxis].T
                    mask = clus_probs>vis_threshold
                    
                    # And plot if chosen so
                    if plotting:
                        rgba_colors = np.zeros((N,4))
                        rgba_colors[mask,:3] = self.colors[count]
                        rgba_colors[:,3] = clus_probs
                        fig = plt.figure(figsize=(6,6))
                        if M==2:
                            ax = fig.add_subplot(111)
                            ax.scatter(clus_latent[mask,0], clus_latent[mask,1], c=rgba_colors[mask,:], zorder=0)
                        if M>2:
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(clus_latent[mask,self.dimx], clus_latent[mask,self.dimy], clus_latent[mask,self.dimz], c=rgba_colors[mask,:], zorder=0)
                        ax.set_title('Latent data of subcluster '+str(cl+1)+' (copied over from higher level)')
                        plt.show()
                    count+=1
                else:
                    # If cluster contains more subclusters, initiate MoPPCAs
                    print('First guess: cluster %i contains %i subclusters (out of a maximum of %i)'%(cl+1, n_subs,k_max))
#                     try:

                    if samplingmethod == 'VB':
                        R = np.zeros((N,n_subs))
                        R_tmp = np.zeros((len(subs['labels']),n_subs))
                        for k in range(n_subs):
                            R_tmp[subs['labels']==k,k] = 1
                        R[levelcats==cl,:] = R_tmp
#                         moppcas_dat = {'N': N, 'M': M, 'K': n_subs, 'D':D, 'y':x, 'lim_sigma_up':5*np.array(subs['sigmas']), 'lim_mu_up':subs['max_mus'],
#                                            'lim_mu_low':subs['min_mus'], 'weights':clus_probs}
                        moppcas_dat = {'N': N, 'M': M, 'K': n_subs, 'D':D, 'y':x, 'weights':clus_probs, 'mean_mu':subs['mu'], 'std_mu':subs['std_mus'], 'mean_sigma':subs['mean_sigmas'], 'std_sigma':subs['std_sigmas'], 'lim_sigma_up':1.25*np.array(subs['max_sigmas']), 'lim_sigma_low':0.75*np.array(subs['min_sigmas']),  'lim_mu_up':subs['max_mus'],'lim_mu_low':subs['min_mus'], 'found_theta':subs['theta'], 'found_R':R}
                        
                        init_dic = {'mu':subs['mu'], 'theta':subs['theta'], 'sigma':subs['mean_sigmas'], 'R':R}
                        while True:
                            R = np.zeros((N,n_subs))
                            R_tmp = np.zeros((len(subs['labels']),n_subs))
                            for k in range(n_subs):
                                R_tmp[subs['labels']==k,k] = 1
                            R[levelcats==cl,:] = R_tmp
                            tries = 0
                            df_tries = []
                            n_subs_found = []
                            
                            while tries<n_try:
                                fit = moppcas_weighted.vb(data=moppcas_dat, init=[init_dic])
                                df = pd.read_csv(fit['args']['sample_file'].decode('ascii'), comment='#').dropna()
                                dfmean = df.mean()

                                dfclus = dfmean[dfmean.index.str.startswith('R.')]
                                rawprobs = np.array(dfclus).reshape(n_subs,N).T
                                moppcas_cats = np.argmax(rawprobs,axis=1)
                                moppcas_cats_set = set(moppcas_cats)
                                n_subs2 = len(moppcas_cats_set)
                                print('Found MoPPCAs fit with %i clusters.'%n_subs2)
                                if n_subs2 == n_subs:
                                    tries = n_try
                                
                                tries +=1
                                df_tries.append(dfmean.copy())
                                n_subs_found.append(n_subs2)
                                if tries>=n_try:
                                    dfmean = df_tries[np.argmax(n_subs_found)]
                                    dfclus = dfmean[dfmean.index.str.startswith('R.')]
                                    found_W = dfmean[dfmean.index.str.startswith('W.')]
#                                     found_W_form = found_W
                                    found_z = dfmean[dfmean.index.str.startswith('z.')]
                                    rawprobs = np.array(dfclus).reshape(n_subs,N).T
                                    moppcas_cats = np.argmax(rawprobs,axis=1)
                                    moppcas_cats_set = set(moppcas_cats)
                                    n_subs2 = len(moppcas_cats_set)
                                    break
                                print('Trying again for a better fit.')

                            if n_subs2==n_subs:
                                dfz = dfmean[dfmean.index.str.startswith('z.')]
                                newfound_latent = np.reshape(np.array(dfz),(M,N)).T
                                n_subc_clus.append(n_subs)
                                break
                            else:
                                if n_subs2==1:
                                    print('MoPPCAS was looking for %i clusters, but no more subclusters were found.'%n_subs)
                                    rawprobs = np.ones((N,1))
                                    newfound_latent = np.reshape(np.array(dfz),(M,N)).T
                                    n_subs = 1
                                    n_subc_clus.append(n_subs)
        #                             dfz = dfmean[dfmean.index.str.startswith('z.')]
        #                             newfound_latent = np.reshape(np.array(dfz),(M,N)).T
                                    break
                                else:
                                    print('MoPPCAS was looking for %i clusters, but found only %i clusters.'%(n_subs, n_subs2))
                                    n_subs2_mask = np.array([i in moppcas_cats_set for i in range(np.shape(rawprobs)[1])])
                                    W_found = np.reshape(np.array(dfmean[dfmean.index.str.startswith('W.')]),(M,D,n_subs)).T[n_subs2_mask]
                                    theta_found = np.array(dfmean[dfmean.index.str.startswith('theta.')])[n_subs2_mask]
                                    theta_found = theta_found/np.sum(theta_found)
                                    R_found = rawprobs[:,n_subs2_mask]/np.sum(rawprobs,axis=1)[np.newaxis].T
                                    rawprobs = R_found
                                    z_found = np.reshape(np.array(dfmean[dfmean.index.str.startswith('z.')]),(M,N)).T
                                    newfound_latent = z_found
                                    mu_found_r = np.reshape(np.array(dfmean[dfmean.index.str.startswith('raw_mu.')]),(D,n_subs)).T[n_subs2_mask]
                                    sigma_found_r = np.array(dfmean[dfmean.index.str.startswith('raw_sigma.')])[n_subs2_mask]
                                    mu_found = np.reshape(np.array(dfmean[dfmean.index.str.startswith('mu.')]),(D,n_subs)).T[n_subs2_mask]
                                    sigma_found = np.array(dfmean[dfmean.index.str.startswith('sigma.')])[n_subs2_mask]
                                    sigma_min_found = np.array([np.min(np.std(x[moppcas_cats==k_i],axis=0)) for k_i in moppcas_cats_set])
                                    sigma_max_found = np.array([np.max(np.std(x[moppcas_cats==k_i],axis=0)) for k_i in moppcas_cats_set])
                                    sigma_std_found = np.array([np.std(np.std(x[moppcas_cats==k_i],axis=0)) for k_i in moppcas_cats_set])
                                    mu_max_found = np.array([np.max(x[moppcas_cats==k_i],axis=0) for k_i in moppcas_cats_set])
                                    mu_min_found = np.array([np.min(x[moppcas_cats==k_i],axis=0) for k_i in moppcas_cats_set])
                                    mu_std_found = np.array([np.std(x[moppcas_cats==k_i],axis=0) for k_i in moppcas_cats_set])
                                    
                                    init_dic = {'raw_mu':mu_found_r, 'theta':theta_found, 'raw_sigma':sigma_found_r, 'R':R_found, 'W':W_found, 'z':z_found}
#                                     n_subs, subs = est_k(x[levelcats==cl], k_max = n_subs2, k_min=n_subs2, gmm = gmm, refs = est_ref, weights=clus_probs)    
                                    n_subs = n_subs2
                                    moppcas_dat = {'N': N, 'M': M, 'K': n_subs2, 'D':D, 'y':x, 'weights':clus_probs, 'mean_mu':mu_found, 'std_mu':mu_std_found, 'mean_sigma':sigma_found, 'std_sigma':sigma_std_found, 'lim_sigma_up':1.25*sigma_max_found, 'lim_sigma_low':0.75*sigma_min_found,  'lim_mu_up':mu_max_found,'lim_mu_low':mu_min_found, 'found_theta':theta_found, 'found_R':R_found}
                                    break
                                    
                            print('Accepted MoPPCAs fit with %i clusters.'%n_subs)


                    elif samplingmethod == 'NUTS':
                        moppcas_dat = {'N':N, 'M':M,'K':n_subs, 'D':D, 'y':x, 'weights':clus_probs}
                        fit = moppcas_weighted.sampling(data=moppcas_dat, chains=1, iter=its, init=[{'mu':subs['mu'],
                                                                                                     'z':[np.zeros((M,N)) for i in range(n_subs)]}])
                        fit_ext_molv1 = fit.extract()
                        best_molv1 = np.where(fit_ext_molv1['lp__']==max(fit_ext_molv1['lp__']))[0][0]
                        newfound_latent = fit_ext_molv1['z'][best_molv1]
                        rawprobs = np.mean(fit_ext_molv1['clusters'],axis=0).T
                    else:
                        print("Please use 'NUTS' or 'VB' as samplingmethod!")
                        return 1

                    lvl_latents += newfound_latent*clus_probs[np.newaxis].T
                    new_probs = rawprobs*clus_probs[np.newaxis].T

                    plotcats = np.argmax(new_probs, axis=1)
                        
                    
                    # and plot latent data of all newfound subclusters if chosen so
                    if plotting:
                        
                        rgba_colors = np.zeros((N,4))
                        fig = plt.figure(figsize=(6,6))
                        if M==2:
                            ax = fig.add_subplot(111)
                        if M>2:
                            ax = fig.add_subplot(111, projection='3d')
                        for k_i in set(plotcats):
                            rgba_colors[plotcats==k_i,:3] = self.colors[k_i]
                            rgba_colors[:,3] = new_probs[:,k_i]
                            if M==2:
                                ax.scatter(self.latent[-1][new_probs[:,k_i]>vis_threshold,0],self.latent[-1][new_probs[:,k_i]>vis_threshold,1], c=rgba_colors[new_probs[:,k_i]>vis_threshold,:], zorder=1)
                                cc_x, cc_y = np.average(self.latent[-1], weights=new_probs[:,k_i], axis=0)
                                ax.scatter(cc_x, cc_y, c='black', s=500, zorder=10)
                                ax.text(cc_x, cc_y, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
                            if M>2:
                                ax.scatter(self.latent[-1][new_probs[:,k_i]>vis_threshold,self.dimx],self.latent[-1][new_probs[:,k_i]>vis_threshold,self.dimy], self.latent[-1][new_probs[:,k_i]>vis_threshold,self.dimz], c=rgba_colors[mask,:],  zorder=1)
                                cc_x, cc_y, cc_z = np.average(self.latent[-1], weights=new_probs[:,k_i], axis=0)
                                ax.scatter(cc_x, cc_y, cc_z, c='black', s=500, zorder=1)
                                ax.text(cc_x, cc_y, cc_z, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
                                

                        ax.set_title('Latent data of cluster %i with found clusters'%(cl+1))
                        plt.show()
                        
                        n_row = int(n_subs/4)+1
                        n_col = min(4,n_subs)
                        fig = plt.figure(figsize=(6*n_col,6*n_row))
                        no_plotje = 1
                        for subc in range(n_subs):
                            subprobs = new_probs[:,subc]
                            mask = subprobs>vis_threshold

                            rgba_colors = np.zeros((N,4))
                            rgba_colors[:,:3] = self.colors[count]
                            count+=1
                            rgba_colors[:,3] = subprobs
                            
                            if M==2:
                                ax = fig.add_subplot(n_row, n_col, no_plotje)
                                ax.scatter(newfound_latent[mask,0],newfound_latent[mask,1], c=rgba_colors[mask,:], zorder=0)
                            if M>2:
                                ax = fig.add_subplot(n_row, n_col, no_plotje, projection='3d')
                                ax.scatter(newfound_latent[mask,self.dimx],newfound_latent[mask,self.dimy], newfound_latent[mask,self.dimz], c=rgba_colors[mask,:], zorder=0)
                            ax.set_title('Subcluster '+str(subc+1))
                            no_plotje+=1
                        plt.suptitle('Latent data of subclusters')
                        plt.show()
                
                probs_round = np.hstack((probs_round, new_probs))
            
            cats = np.argmax(probs_round,axis=0)
            if np.any(np.array(n_subc_clus)>1):
                more_depth = True
            
            # Plotting Top-level latent data with new cluster-colouring
            if plotting:
                fig = plt.figure(figsize=(6,6))
                if M==2:
                    ax = fig.add_subplot(111)
                if M>2:
                    ax = fig.add_subplot(111, projection='3d')
                for c in range(np.shape(probs_round)[1]):
                    mask = probs_round[:,c]>vis_threshold
                    rgba_colors = np.zeros((N,4))
                    rgba_colors[:,:3] = self.colors[c]
                    rgba_colors[:,3] = probs_round[:,c]
                    if M==2:
                        ax.scatter(latent_top[:,0],latent_top[:,1], c = rgba_colors, zorder=0)
                        cc_x, cc_y = np.average(latent_top, weights=probs_round[:,c], axis=0)
                        ax.scatter(cc_x, cc_y, c='black', s=500, zorder=1)
                        ax.text(cc_x, cc_y, str(c+1),fontweight= 'bold', size=14, c = 'white', zorder=10,horizontalalignment='center',verticalalignment='center')
                    if M>2:
                        ax.scatter(latent_top[mask,self.dimx],latent_top[mask,self.dimy],latent_top[mask,self.dimz], c = rgba_colors[mask,:], zorder=0)
                        cc_x, cc_y, cc_z = np.average(latent_top, weights=probs_round[:,c], axis=0)
                        ax.scatter(cc_x, cc_y, cc_z, c='black', s=500, zorder=1)
                        ax.text(cc_x, cc_y, cc_z, str(c+1),fontweight= 'bold', size=14, c = 'white', zorder=10,horizontalalignment='center',verticalalignment='center')
                ax.set_title('Clusters after level '+str(lvl+1))
                plt.show()

            # Stop if all clusters are fully analyzed
            if more_depth == False:
                print('All clusters are fully analyzed!')
                return self.latent, self.cats_per_lvl, self.probs
            
            self.probs.append(probs_round.copy())
            next_clus = np.shape(probs_round)[1]
            self.cats_per_lvl.append(cats.copy())
            self.latent.append(lvl_latents.copy())
            
        print('Maximum depth has been reached!')
        return self.latent, self.cats_per_lvl, self.probs
    
    def visualize_tree(self,categories = None, vis_threshold=0.05):
        # plot the subdivision of clusters in hierarchical order
        if np.all(categories)==None:
            categories = self.cats_per_lvl[-1]
        for lvl in range(len(self.latent)):
            print('Level ', lvl)
            clusters = set(categories)
            n_clus = len(clusters)
            fig = plt.figure(figsize=(min(n_clus*6, 24), (int((n_clus-1)/4)+1)*6))
            for clus in clusters:
                rgba_colors = np.zeros((self.N, 4))
                for k_i in range(n_clus):
                    rgba_colors[categories==k_i,:3] = self.colors[k_i]
                
                prob_cur = self.probs[lvl]
                
                for k_i in range(n_cat):
                    rgba_colors[:,3] = prob_cur[:,clus]
                vis_mask = np.array(prob_cur[:,clus]>vis_threshold)
                if self.M==2:
                    ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus+1)
                    ax.scatter(self.latent[lvl][vis_mask,0],self.latent[lvl][vis_mask,1],c = rgba_colors[vis_mask,:])
                if self.M>2:
                    ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus, projection='3d')
                    ax.scatter(self.latent[lvl][vis_mask,self.dimx],self.latent[lvl][vis_mask,self.dimy],self.latent[lvl][vis_mask,self.dimz],c = rgba_colors[vis_mask,:])
            plt.suptitle('Clusters on level '+str(lvl))
            plt.show()
        return
            
    def visualize_end(self,categories = None, vis_threshold=0.05):
        # plot the top-level latent data with the end-result of the clustering as colouring
        if np.all(categories)==None:
            categories = self.cats_per_lvl[-1]
            title = 'Top-level latent data coloured by guessed clusters'
        else:
            title = 'Top-level latent data coloured by given clusters'
        rgba_colors = np.ones((self.N, 4))
        for k_i in range(int(max(categories))):
            rgba_colors[categories==k_i,:3] = self.colors[k_i]
#             rgba_colors[categories==k_i,3] = self.probs[-1][categories==k_i,k_i]
        fig = plt.figure(figsize=(6,6))
        if self.M==2:
#             print('Plotting 2-dimensional latent data with final cluster colouring.')
            ax = fig.add_subplot(111)
            ax.scatter(self.latent[0][:,0],self.latent[0][:,1],c = rgba_colors)
        if self.M>2:
#             print('Plotting 3-dimensional latent data with final cluster colouring.')
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.latent[0][:,self.dimx],self.latent[0][:,self.dimy],self.latent[0][:,self.dimz],c = rgba_colors)
        ax.set_title(title)
        plt.show()
        return
    
    def visualize_latent_final(self, categories=None, vis_threshold=0.05):
        # plot the latent spaces of all found subclusters
        if np.all(categories)==None:
            title = 'Latent data coloured by guessed clusters'
            categories = self.cats_per_lvl[-1]
        else:
            title = 'Latent data coloured by given clusters'
        n_clus = len(set(categories))
        fig = plt.figure(figsize=(min(n_clus*6, 24), (int((n_clus-1)/4)+1)*6))
        for clus in range(n_clus):
            rgba_colors = np.zeros((self.N, 4))
            for k_i in range(int(max(categories))):
                rgba_colors[categories==k_i,:3] = self.colors[k_i]
            prob_cur = self.probs[-1]
            for k_i in range(n_clus):
                rgba_colors[categories==k_i,3] = prob_cur[categories==k_i,k_i]
            vis_mask = np.array(prob_cur[:,clus]>vis_threshold)
            if self.M==2:
                ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus+1)
                ax.scatter(self.latent[-1][vis_mask,0],self.latent[-1][vis_mask,1],c = rgba_colors[vis_mask,:])
            if self.M>2:
                ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus, projection='3d')
                ax.scatter(self.latent[-1][vis_mask,self.dimx],self.latent[-1][vis_mask,self.dimy],self.latent[-1][vis_mask,self.dimz],c = rgba_colors[vis_mask,:])
        plt.suptitle(title)
        plt.show()
        return
    
    def ari_per_level(self, ind):
        lvls = len(self.cats_per_lvl)
        return [adjusted_rand_score(self.cats_per_lvl[lvl], ind) for lvl in range(lvls)]
    
    def visual_score(self, ind, plot_hmppca = True, plot_hmppca_logres = False, plot_real = True, plot_logreg = True, vis_threshold=0.1):
        for lvl in range(len(self.latent)):

            lvl_i = min(len(self.latent)-1, lvl+1)
            nclus = len(set(self.cats_per_lvl[lvl]))
            print('level ', lvl)
            if plot_hmppca:

                
                
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                for clus in range(nclus):
                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>vis_threshold
                    rgba_cols = np.zeros((N,4))
                    for k_i in set(self.cats_per_lvl[lvl_i][np.argmax(self.probs[lvl],axis=1)==clus]):
                        rgba_cols[self.cats_per_lvl[lvl_i]==k_i,:3] = self.colors[k_i]
                        rgba_cols[self.cats_per_lvl[lvl_i]==k_i,3] = self.probs[lvl][self.cats_per_lvl[lvl_i]==k_i,clus]
                        if lvl<len(self.latent)-1:
                            cc_x, cc_y = np.average(self.latent[lvl], axis=1, weights = self.probs[lvl_i][:,k_i])
                            plt.scatter(cc_x, cc_y, s = 500, c = 'black', zorder=9)
                            plt.text(cc_x, cc_y, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=10,horizontalalignment='center',
                verticalalignment='center')
                    ax.scatter(self.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:], zorder=1)
                    ax.set_title('subcluster '+str(clus+1))
                plt.suptitle('HmPPCA clusters')
                plt.show()


            if plot_hmppca_logres:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                c_i = 0
                for clus in range(n_clus):

                    classifier = sklearn.linear_model.SGDClassifier(loss='log')
                    classifier.fit(self.latent[lvl], self.cats_per_lvl[lvl_i], sample_weight=self.probs[lvl][:,clus])
                    preds = classifier.predict(self.latent[lvl])
                    preds = np.unique(preds, return_inverse=True)[1]

                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>vis_threshold
                    rgba_cols = np.ones((N,4))

                    for k_i in range(len(set(preds))):
                        rgba_cols[preds==k_i,:3] = self.colors[c_i]
                        c_i+=1

                    ax.scatter(self.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:])
                plt.suptitle('log.reg. clusters (based on hmppca)')
                plt.show()

            if plot_real:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                for clus in range(n_clus):
                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>0.1
                    rgba_cols = np.ones((N,4))
                    for k_i in range(len(set(ind))):
                        rgba_cols[ind==k_i,:] = self.colors[k_i]
                    ax.scatter(self.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:])
                plt.suptitle('Real clusters')
                plt.show()

            if plot_logreg:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                c_i = 0
                w_ARI = 0
                w_ACC = 0
                for clus in range(n_clus):

                    classifier = sklearn.linear_model.SGDClassifier(loss='log')
                    classifier.fit(self.latent[lvl], ind, sample_weight=self.probs[lvl][:,clus])
                    preds = classifier.predict(self.latent[lvl])

                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>0.1
                    rgba_cols = np.zeros((N,4))

                    for k_i in set(preds):
                        rgba_cols[preds==k_i,:] = cols[int(k_i)]
                        c_i+=1
                    rgba_cols[:,3] = np.ones(N)
                    ax.scatter(model.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:])
                    ari = adjusted_rand_score(preds[mask], ind[mask])
                    acc = accuracy_score(preds, ind, sample_weight = self.probs[lvl][:,clus])
                    ax.set_title('Cluster %i - ARI: %.3f, ACC: %.3f'%(clus+1,ari,acc))
                    w_ARI+= ari*(sum(mask))/N
                    w_ACC+= acc*(sum(mask))/N
                plt.suptitle('log.reg. - w. ARI: %.3f, w. ACC: %.3f'%(w_ARI, w_ACC))
                plt.show()
            
def weighted_Accuracy(true, weights):
    
    total = 0
    correct = 0
    for i in range(len(true)):
        total+=1
        correct+=weights[i,int(true[i])]
        
    return correct/total