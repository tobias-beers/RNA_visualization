import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pystan
import numpy as np
import seaborn as sns
import itertools
import sklearn
from sklearn.cluster import KMeans
from sklearn import datasets
import os
import pickle
import scipy
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import SpectralClustering
from scipy.stats import norm,multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, accuracy_score
import warnings
import time
sns.set()


def loadStan(file, recompile=False, automatic_pickle = True, parallel=False):
    # Either loads pickled Stan model or compiles and saves new model from code
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
    # Computes GAP score for number of clusters determination
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

def est_k(points,k_min = 1, k_max = 2, refs = 3, retry=2, method='bic', verbose = False, weights=None, clustering='gmm'):
    # Wrapper for numer of clusters determination methods
    
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
            model_data = {}
            
            if clustering=='spectral':
                model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',assign_labels='kmeans')
                spec_labels = model.fit_predict(points)
                a, b = kmeans_AIC(points,spec_labels, k, verbose=verbose)
                model_data = get_parameters(points, spec_labels)
                
            elif clustering=='kmeans':
                model = KMeans(k).fit(points)
                a, b = kmeans_AIC(points,model.labels_, k, verbose=verbose)
                model_data = get_parameters(points, spec_labels)
            
            elif clustering=='gmm':
                model = GaussianMixture(k)
                labels = model.fit_predict(points)
#                 a, b = kmeans_AIC(points,labels, k, verbose=verbose)
                a = model.aic(points)
                b = model.bic(points)
                model_data = get_parameters(points, labels)
#             if gmm:
#                 count = 0 
#                 while True:
#                     if verbose:
#                         print('Building GMM model with %i clusters'%k)
#                     gmm_dat = {'N': N,'K': k, 'D':D, 'y':points, 'weights':weights}
#                     fit = gmm_diag_weighted.sampling(data=gmm_dat, chains=1, iter=150, init=[{'mu':model_data['mu'], 'theta':model_data['theta']}])
#                     if verbose:
#                         print('Model built!')
#                     fit_ext = fit.extract()
#                     z_sim = np.mean(fit_ext['z'],axis=0)
#                     labels = np.argmax(z_sim, axis=1)
#                     if np.all(np.array([sum(labels==k_i) for k_i in range(k)])>1):
#                         a, b =kmeans_AIC(points,labels, k, verbose=verbose)
#                         model_data['mu'] = np.mean(fit_ext['mu'],axis=0)
#                         model_data['theta'] = np.mean(fit_ext['theta'],axis=0)
#                         model_data['labels'] = labels
#                         break
#                     count+=1
#                     print('Failed to find model with right number of clusters (trial %i of %i)'%(count,retry))
#                     if count>=retry:
#                         print('No model with ', k, ' clusters found, showing cluster estimates with k_max at ',k-1)
#                         quit = True
#                         break
            if verbose:
                print('Model built!')
                
            
            
            if quit == False:
                mods.append(model_data)
                clus.append(k)
                aics.append(a)
                bics.append(b)
    if clus[np.argmin(aics)]!=clus[np.argmin(bics)]:
        print(clus[np.argmin(aics)], ' clusters according to AIC, ', clus[np.argmin(bics)], ' clusters according to BIC.')
    if method=='aic':
        return clus[np.argmin(aics)], mods[np.argmin(aics)]['labels']
    elif method=='bic':
        return clus[np.argmin(bics)], mods[np.argmin(bics)]['labels']
    elif method=='both':
        return clus[np.argmin(aics)], mods[np.argmin(aics)]['labels'],clus[np.argmin(bics)], mods[np.argmin(bics)]['labels']
    else:
        print("Choose 'aic' or 'bic' as method!")
        return 1
    
def kmeans_AIC(points, labels, K, verbose = False):
#     Computes AIC
    if verbose:
        print('Evaluating %i clusters'%K)
    N,D = np.shape(points)
    probs = np.zeros((N,K))
    theta = [np.sum(labels==k)/K for k in range(K)]
    mu = [np.mean(points[labels==k],axis=0) for k in range(K)]
    sigmas = [np.std(points[labels==k],axis=0) for k in range(K)]
    sigmas = np.max([sigmas, 0.00000001*np.ones_like(sigmas)],axis=0)

    for k in range(K):
        probs[:,k] = np.nansum(norm.logpdf(points, loc=mu[k], scale=sigmas[k]),axis=1)

    probs+=np.log(theta)
    try:
        llh = np.sum(logsumexp(probs,axis=1))
    except:
        for n in range(N):
            try:
                logsumexp(probs[n,:])
            except:
                print(probs[n,:])
    aic = -2*llh + 2*K*D
    bic = -2*llh + np.log(N)*K*D
    if verbose:
        print('Evaluation complete: n_clusters: ',K, ', AIC: ',aic, ', BIC: ',bic)
        print()
    return aic, bic

def get_parameters(points, labels):
    # Recomputes several statistics in a higher dimensional space given the clustering of data-points
    N, D = np.shape(points)
    K = set(labels)
    model_data = {}
    model_data['mu'] = [np.mean(points[labels==k_i],axis=0) for k_i in K]
    theta = [np.sum(labels==k_i)/N for k_i in K]
    model_data['theta'] = theta
    model_data['labels'] = labels
    model_data['min_mus'] = [np.min(points[labels==k_i],axis=0) for k_i in K]
    model_data['std_mus'] = [np.std(points[labels==k_i],axis=0)+0.0001 for k_i in K]
    model_data['max_mus'] = [np.max(points[labels==k_i],axis=0) for k_i in K]
    model_data['max_sigmas'] = [np.max(np.std(points[labels==k_i],axis=0)) for k_i in K]
    model_data['min_sigmas'] = [np.min(np.std(points[labels==k_i],axis=0))+0.0001 for k_i in K]
    model_data['mean_sigmas'] = [np.mean(np.std(points[labels==k_i],axis=0))+0.0001 for k_i in K]
    model_data['std_sigmas'] = [np.std(np.std(points[labels==k_i],axis=0))+0.0001 for k_i in K]
    return model_data

def logres_scores(points, labels, weights, K=5):
    # Get the K-fold cross-validated logistic regression score
    N,D = np.shape(points)
    folded_index = np.zeros(N)
    
    # Create random folds
    w_sort = np.argsort(-weights)
    
    for i in range(N):
        if i%K==0:
            order = np.random.permutation(range(K))
        folded_index[w_sort[i]] = order[i%K]
        
    w_acc = 0
    w_ari = 0
#     preds_final = np.zeros(N)
    preds_all = np.zeros((N,len(set(labels))))
    
    for k in range(K):
        classifier = sklearn.linear_model.SGDClassifier(loss='log')
        classifier.fit(points[folded_index!=k,:], labels[folded_index!=k], sample_weight=weights[folded_index!=k])
        preds = classifier.predict(points[folded_index==k])
        ari = adjusted_rand_score(preds, labels[folded_index==k])
        w_ari += ari
        acc = accuracy_score(preds, labels[folded_index==k], sample_weight = weights[folded_index==k])
        w_acc += acc
        preds_tmp = np.zeros((sum(folded_index==k),len(set(labels))))
        for k_i in set(preds):
            preds_tmp[preds==k_i,k_i] += np.sum(weights[folded_index!=k])
        preds_all[folded_index==k,:] += preds_tmp
    
    preds_final = np.argmax(preds_all, axis=1)
    
#         preds_final[folded_index==k] += preds*np.sum(weights[folded_index!=k])
#     preds_final = np.round(preds_final/sum(weights))
    w_ari = w_ari/K
    w_acc = w_acc/K
    
#     w_ari = adjusted_rand_score(preds_final, labels)
#     w_acc = accuracy_score(preds_final, labels, sample_weight = weights)
    
    return preds_final, w_acc, w_ari
    
class hierarchical_model:
    # HmPPCAs model
    
    def __init__(self):
        
        self.latent = []
        self.mus = [[]]
        self.cats_per_lvl = []
        self.times = []
        self.knots_tried = []
        self.knots_found = []
    
    def fit(self,x, M=2, max_depth=5, k_max =3, plotting=True, min_clus_size=10, vis_threshold=0.05, its=300, samplingmethod='VB', n_try=3, n_cluster='latent', plot_kmeans=True, init_cluster='gmm', savefigs=False):
        # Fit to data
        
        if k_max <3:
            print("It is suggested to give 'k_max' a value of at least 3.")
        # initialize the algorithm
        self.M = M
        
        self.samplingmethod = samplingmethod
        
        if M>3 and plotting:
            print('Latent dimensions greater than 3, plotting only the three most important dimensions.')
        moppcas_weighted = loadStan('moppcas_weighted')
        ppca_weighted = loadStan('ppca_weighted')
        
        

        N,D = np.shape(x)
        self.N = N
        self.probs = [np.ones((N,1))]
        self.colors = [np.random.uniform(size=3) for k in range(k_max**max_depth)]
        
#         if gmm:
#             est_ref = 1
#         else:
        est_ref = 5
    
        dead_end = [[False]]

        # top-level latent data
        ppca_dat = {'N':N, 'M':M, 'D':D, 'x':x, 'weights': self.probs[-1][:,0]}
        starttime = time.time()
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
#             latent_top = fitreturn_top['z'][best_ind_top]
            latent_top = np.mean(fitreturn_top['z'], axis=0).T
        else:
            print("Please use 'NUTS' or 'VB' as samplingmethod!")
            return 1
        self.latent.append(latent_top.copy())
        self.times.append(time.time()-starttime)
        self.knots_tried.append(0)
        self.knots_found.append(0)
        
        # top-level cluster determination
        if n_cluster=='full_data':
            K_1, labels_1 = est_k(x, k_max = k_max, k_min = 2, refs = est_ref, clustering=init_cluster)
#             clusters_1 = get_parameters(x, labels_1)
        elif n_cluster=='latent':
            K_1, labels_1 = est_k(latent_top, k_max = k_max, k_min = 2, refs = est_ref, clustering=init_cluster)
#             clusters_1 = get_parameters(latent_top, labels_1)
        clusters_1 = get_parameters(x, labels_1)
        
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
            self.dim_mask = np.array([i in chosen for i in range(self.M)])
        
        # plot top-level latent data with coloured first clusters
        if plotting:
            fig = plt.figure(figsize=(6,6))
            rgba_colors = np.ones((N,4))
            if plot_kmeans:
                for k_i in range(K_1):
                    rgba_colors[clusters_1['labels']==k_i,:3] = self.colors[k_i]
            else:
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
#                     cc_x, cc_y, cc_Z = np.mean(latent_top[clusters_1['labels']==k_i],axis=0)[self.dim_mask]
#                     ax.scatter(cc_x, cc_y, cc_z, c='black', s=500, zorder=10)
#                     ax.text(cc_x, cc_y, cc_z, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
            if plot_kmeans:
                ax.set_title('Top-level latent data\n(initial clustering by %s)'%init_cluster)
            else:
                ax.set_title('Top-level latent data')
            if savefigs:
                plt.savefig('toplevel_%s.png'%samplingmethod)
            plt.show()

            
            

        cats = np.argmax(self.probs[-1],axis=1)
        self.cats_per_lvl.append(cats.copy())
        
#         self.probs.append(np.ones((N,4)))
        
        
        for lvl in range(max_depth):
            # repeat cluster detemination and MoPPCAs until max_depth is reached or until all clusters fully analyzed
            more_depth = False
            dead_end.append([])
            print('level %i:'%(lvl+1))
            lvl_probs = self.probs[-1].copy()
            n_clus = np.shape(lvl_probs)[1]
            count = 0
            
            levelcats = np.argmax(lvl_probs,axis=1)
            lvl_latents  = np.zeros((N,2))
            probs_round = np.zeros((N,0))
            
            i_c = 1
            # analyze all subclusters as found in the last level
            for cl in range(n_clus):
                
                print('Cluster %i:'%(cl+1))
                clus_probs = lvl_probs[:,cl]
                
                # Dont retry the dead ends
                if dead_end[-2][cl] or sum(levelcats==cl)<min_clus_size:
#                     n_subs, labels = est_k(self.latent[-1][levelcats==cl], k_max = 1, k_min = 1, refs = est_ref, weights=clus_probs, clustering=init_cluster)
                    n_subs = 1
                    labels = np.zeros(np.shape(self.latent[-1][levelcats==cl])[0])
                    subs = get_parameters(x[levelcats==cl], labels)
                else:
                    # Dont divide clusters further if they are too small
                    
                    k_max = np.max([np.min([sum(levelcats==cl)-1, k_max]),1])

                    if lvl==0:
                        k_min = 2
                    else:
                        k_min = 1
                    if n_cluster=='full_data':
                        n_subs, labels = est_k(x[levelcats==cl], k_max = k_max, k_min = k_min, refs = est_ref, weights=clus_probs, clustering=init_cluster)
                    elif n_cluster=='latent':
                        n_subs, labels = est_k(self.latent[-1][levelcats==cl], k_max = k_max, k_min = k_min, refs = est_ref, weights=clus_probs, clustering=init_cluster)
                    subs = get_parameters(x[levelcats==cl], labels)
                    while np.any([sum(subs['labels']==k_i)<min_clus_size for k_i in range(n_subs)]):
                        if n_subs <= 2:
                            n_subs = 1
                            break
                        if n_cluster=='full_data':
                            n_subs, labels = est_k(x[levelcats==cl], k_max = n_subs-1, refs = est_ref, weights=clus_probs, clustering=init_cluster)
                        elif n_cluster=='latent':
                            n_subs, labels = est_k(self.latent[-1][levelcats==cl], k_max = n_subs-1, refs = est_ref, weights=clus_probs, clustering=init_cluster)
                        subs = get_parameters(x[levelcats==cl], labels)


                if plotting and lvl>0:
                    rgba_colors = 0.2*np.ones((N,4))    
                    plt.scatter(latent_top[:,0], latent_top[:,1], c=rgba_colors)
#                     rgba_colors = np.zeros((N,4))
                    if plot_kmeans:
                        rgba_tmp = np.ones((sum(levelcats==cl),3))
                        for k_i in range(n_subs):
                            rgba_tmp[subs['labels']==k_i,:] = self.colors[k_i]
                        rgba_colors[levelcats==cl,:3] = rgba_tmp
                    else:
                        rgba_colors[:,:3] = self.colors[cl]
                    rgba_colors[:,3] = clus_probs
                    plt.scatter(latent_top[:,0], latent_top[:,1], c=rgba_colors)
                    if plot_kmeans:
                        plt.title('Analysing cluster %i\n(initial clustering by %s)'%(cl+1, init_cluster))
                    else:
                        plt.title('Analysing cluster %i'%(cl+1))
                    if savefigs:
                        plt.savefig('init_cl%i_lvl%i_%s.png'%(cl, lvl, samplingmethod))
                    plt.show()
                n_subc_clus = []
                
                
                
                

                if n_subs == 1:
                    # If cluster doesnt contain more subclusters, just copy it over to the next level
                    clus_latent = self.latent[-1]
                    print('Cluster ', cl+1, ' doesnt contain any more subclusters')
                    new_probs = clus_probs[np.newaxis].T
                    lvl_latents += clus_latent*clus_probs[np.newaxis].T
                    mask = clus_probs>vis_threshold
                    dead_end[-1].append(n_subs==1)
                    
                    # And plot if chosen so
                    if plotting:
                        rgba_colors = np.zeros((N,4))
                        rgba_colors[mask,:3] = self.colors[count]
                        rgba_colors[:,3] = clus_probs
                        fig = plt.figure(figsize=(6,6))
                        if M==2:
                            ax = fig.add_subplot(111)
                            ax.scatter(clus_latent[mask,0], clus_latent[mask,1], c=rgba_colors[mask,:], zorder=0)
                            cc_x, cc_y = np.average(clus_latent, weights=clus_probs, axis=0)
                            ax.scatter(cc_x, cc_y, c='black', s=500, zorder=10)
                            ax.text(cc_x, cc_y, str(i_c),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
                        if M>2:
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(clus_latent[mask,self.dimx], clus_latent[mask,self.dimy], clus_latent[mask,self.dimz], c=rgba_colors[mask,:], zorder=0)
                            cc_x, cc_y, cc_z = np.average(clus_latent, weights=clus_probs, axis=0)[self.dim_mask]
                            ax.scatter(cc_x, cc_y, cc_z, c='black', s=500, zorder=10)
                            ax.text(cc_x, cc_y, cc_z, str(i_c),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
                        ax.set_title('Latent data of subcluster '+str(cl+1)+' (copied over from higher level)')
                        if savefigs:
                            plt.savefig('latent_cl%i_lvl%i_%s.png'%(cl, lvl, samplingmethod))
                        plt.show()
                    count+=1
                    i_c+=1
                else:
                    # If cluster contains more subclusters, initiate MoPPCAs
                    print('First guess: cluster %i contains %i subclusters (out of a maximum of %i)'%(cl+1, n_subs,k_max))
#                     try:

#                     if samplingmethod == 'VB':
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
                            
                            starttime = time.time()
                            
                            if samplingmethod == 'VB':
                                fit = moppcas_weighted.vb(data=moppcas_dat, init=[init_dic])
                                df = pd.read_csv(fit['args']['sample_file'].decode('ascii'), comment='#').dropna()
                                dfmean = df.mean()

                                dfclus = dfmean[dfmean.index.str.startswith('R.')]
                                rawprobs = np.array(dfclus).reshape(n_subs,N).T
                            elif samplingmethod == 'NUTS':
                                fit = moppcas_weighted.sampling(data=moppcas_dat, init=[init_dic], iter=its, chains=1)
                                fitreturn = fit.extract()
                                rawprobs = np.mean(fitreturn['R'], axis=0)
                            
                            self.times.append(time.time()-starttime)
                            
                            moppcas_cats = np.argmax(rawprobs,axis=1)
                            moppcas_cats_set = set(moppcas_cats)
                            self.knots_tried.append(n_subs)
                            n_subs2 = len(moppcas_cats_set)
                            self.knots_found.append(n_subs2)
                            print('Found MoPPCAs fit with %i clusters.'%n_subs2)
                            if n_subs2 == n_subs:
                                tries = n_try

                            tries +=1
                            if samplingmethod == 'VB':
                                df_tries.append(dfmean.copy())
                            elif samplingmethod == 'NUTS':
                                df_tries.append(fitreturn.copy())
                            n_subs_found.append(n_subs2)
                            if tries>=n_try:
                                if samplingmethod == 'VB':
                                    dfmean = df_tries[np.argmax(n_subs_found)]
                                    dfclus = dfmean[dfmean.index.str.startswith('R.')]
                                    found_W = dfmean[dfmean.index.str.startswith('W.')]
                                    found_z = dfmean[dfmean.index.str.startswith('z.')]
                                    rawprobs = np.array(dfclus).reshape(n_subs,N).T
                                elif samplingmethod == 'NUTS':
                                    fitreturn = df_tries[np.argmax(n_subs_found)]
                                    rawprobs = np.mean(fitreturn['R'], axis=0)
                                    found_W = np.mean(fitreturn['W'], axis=0)
                                    found_z = np.mean(fitreturn['z'], axis=0)
                                moppcas_cats = np.argmax(rawprobs,axis=1)
                                moppcas_cats_set = set(moppcas_cats)
                                n_subs2 = len(moppcas_cats_set)
                                break
                            print('Trying again for a better fit.')

                        if n_subs2==n_subs:
                            if samplingmethod == 'VB':
                                dfz = dfmean[dfmean.index.str.startswith('z.')]
                                newfound_latent = np.reshape(np.array(dfz),(M,N)).T
                            elif samplingmethod == 'NUTS':
                                newfound_latent = np.mean(fitreturn['z'], axis=0)
                            n_subc_clus.append(n_subs)
                            break
                        else:
                            if n_subs2==1:
                                print('MoPPCAS was looking for %i clusters, but no more subclusters were found.'%n_subs)
                                rawprobs = np.ones((N,1))
                                if samplingmethod == 'VB':
                                    dfz = dfmean[dfmean.index.str.startswith('z.')]
                                    newfound_latent = np.reshape(np.array(dfz),(M,N)).T
                                elif samplingmethod == 'NUTS':
                                    newfound_latent = np.mean(fitreturn['z'], axis=0)
                                n_subs = 1
                                n_subc_clus.append(n_subs)
    #                             dfz = dfmean[dfmean.index.str.startswith('z.')]
    #                             newfound_latent = np.reshape(np.array(dfz),(M,N)).T
                                break
                            else:
                                print('MoPPCAS was looking for %i clusters, but found only %i clusters.'%(n_subs, n_subs2))
                                n_subs2_mask = np.array([i in moppcas_cats_set for i in range(np.shape(rawprobs)[1])])
                                if samplingmethod == 'VB':
                                    W_found = np.reshape(np.array(dfmean[dfmean.index.str.startswith('W.')]),(M,D,n_subs)).T[n_subs2_mask]
                                    theta_found = np.array(dfmean[dfmean.index.str.startswith('theta.')])[n_subs2_mask]
                                    newfound_latent = np.reshape(np.array(dfmean[dfmean.index.str.startswith('z.')]),(M,N)).T
                                    mu_found_r = np.reshape(np.array(dfmean[dfmean.index.str.startswith('raw_mu.')]),(D,n_subs)).T[n_subs2_mask]
                                    sigma_found_r = np.array(dfmean[dfmean.index.str.startswith('raw_sigma.')])[n_subs2_mask]
                                    mu_found = np.reshape(np.array(dfmean[dfmean.index.str.startswith('mu.')]),(D,n_subs)).T[n_subs2_mask]
                                    sigma_found = np.array(dfmean[dfmean.index.str.startswith('sigma.')])[n_subs2_mask]
                                elif samplingmethod == 'NUTS':
                                    W_found = np.mean(fitreturn['W'], axis=0)
                                    theta_found = np.mean(fitreturn['theta'], axis=0)
                                    newfound_latent = np.mean(fitreturn['z'], axis=0)
                                    mu_found_r = np.mean(fitreturn['raw_mu'], axis=0)
                                    sigma_found_r = np.mean(fitreturn['raw_sigma'], axis=0)
                                    mu_found = np.mean(fitreturn['mu'], axis=0)
                                    sigma_found = np.mean(fitreturn['sigma'], axis=0)
                                    
                                rawprobs = rawprobs[:,n_subs2_mask]/np.sum(rawprobs,axis=1)[np.newaxis].T
                                theta_found = theta_found/np.sum(theta_found)
                                sigma_min_found = np.array([np.min(np.std(x[moppcas_cats==k_i],axis=0)) for k_i in moppcas_cats_set])
                                sigma_max_found = np.array([np.max(np.std(x[moppcas_cats==k_i],axis=0)) for k_i in moppcas_cats_set])
                                sigma_std_found = np.array([np.std(np.std(x[moppcas_cats==k_i],axis=0)) for k_i in moppcas_cats_set])
                                mu_max_found = np.array([np.max(x[moppcas_cats==k_i],axis=0) for k_i in moppcas_cats_set])
                                mu_min_found = np.array([np.min(x[moppcas_cats==k_i],axis=0) for k_i in moppcas_cats_set])
                                mu_std_found = np.array([np.std(x[moppcas_cats==k_i],axis=0) for k_i in moppcas_cats_set])

                                init_dic = {'raw_mu':mu_found_r, 'theta':theta_found, 'raw_sigma':sigma_found_r, 'R':rawprobs, 'W':W_found, 'z':newfound_latent}
#                                     n_subs, subs = est_k(x[levelcats==cl], k_max = n_subs2, k_min=n_subs2, gmm = gmm, refs = est_ref, weights=clus_probs)    
                                n_subs = n_subs2
                                n_subc_clus.append(n_subs)
                                moppcas_dat = {'N': N, 'M': M, 'K': n_subs2, 'D':D, 'y':x, 'weights':clus_probs, 'mean_mu':mu_found, 'std_mu':mu_std_found, 'mean_sigma':sigma_found, 'std_sigma':sigma_std_found, 'lim_sigma_up':1.25*sigma_max_found, 'lim_sigma_low':0.75*sigma_min_found,  'lim_mu_up':mu_max_found,'lim_mu_low':mu_min_found, 'found_theta':theta_found, 'found_R':rawprobs}
                                break

                        print('Accepted MoPPCAs fit with %i clusters.'%n_subs)


#                     elif samplingmethod == 'NUTS':
#                         moppcas_dat = {'N':N, 'M':M,'K':n_subs, 'D':D, 'y':x, 'weights':clus_probs}
#                         fit = moppcas_weighted.sampling(data=moppcas_dat, chains=1, iter=its, init=[{'mu':subs['mu'],
#                                                                                                      'z':[np.zeros((M,N)) for i in range(n_subs)]}])
#                         fit_ext_molv1 = fit.extract()
#                         best_molv1 = np.where(fit_ext_molv1['lp__']==max(fit_ext_molv1['lp__']))[0][0]
#                         newfound_latent = fit_ext_molv1['z'][best_molv1]
#                         rawprobs = np.mean(fit_ext_molv1['clusters'],axis=0).T
#                     else:
#                         print("Please use 'NUTS' or 'VB' as samplingmethod!")
#                         return 1

                    lvl_latents += newfound_latent*clus_probs[np.newaxis].T
                    new_probs = rawprobs*clus_probs[np.newaxis].T
                    for s in range(n_subs):
                        dead_end[-1].append(n_subs==1)

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
                                ax.text(cc_x, cc_y, str(k_i+i_c),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
                            if M>2:
                                ax.scatter(self.latent[-1][new_probs[:,k_i]>vis_threshold,self.dimx],self.latent[-1][new_probs[:,k_i]>vis_threshold,self.dimy], self.latent[-1][new_probs[:,k_i]>vis_threshold,self.dimz], c=rgba_colors[mask,:],  zorder=1)
                                cc_x, cc_y, cc_z = np.average(self.latent[-1], weights=new_probs[:,k_i], axis=0)[self.dim_mask]
                                
                                ax.scatter(cc_x, cc_y, cc_z, c='black', s=500, zorder=1)
                                ax.text(cc_x, cc_y, cc_z, str(k_i+i_c),fontweight= 'bold', size=14, c = 'white', zorder=100,horizontalalignment='center',verticalalignment='center')
                                
                        ax.set_title('Latent data of cluster %i with found clusters'%(cl+1))
                        if savefigs:
                            plt.savefig('latent_cl%i_lvl%i_%s.png'%(cl, lvl, samplingmethod))
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
                            ax.set_title('Subcluster '+str(i_c))
                            i_c+=1
                            no_plotje+=1
                        plt.suptitle('Latent data of subclusters')
                        if savefigs:
                            plt.savefig('latent_subcs_cl%i_lvl%i_%s.png'%(cl, lvl, samplingmethod))
                        plt.show()
                
                probs_round = np.hstack((probs_round, new_probs))
            
            cats = np.argmax(probs_round,axis=1)
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
                        cc_x, cc_y, cc_z = np.average(latent_top, weights=probs_round[:,c], axis=0)[self.dim_mask]
                        ax.scatter(cc_x, cc_y, cc_z, c='black', s=500, zorder=1)
                        ax.text(cc_x, cc_y, cc_z, str(c+1),fontweight= 'bold', size=14, c = 'white', zorder=10,horizontalalignment='center',verticalalignment='center')
                ax.set_title('Clusters after level '+str(lvl+1))
                if savefigs:
                    plt.savefig('clusters_lvl%i_%s.png'%(cl, lvl, samplingmethod))
                plt.show()

                
            self.probs.append(probs_round.copy())
            next_clus = np.shape(probs_round)[1]
            self.cats_per_lvl.append(cats.copy())
            self.latent.append(lvl_latents.copy())
            
            # Stop if all clusters are fully analyzed
            if more_depth == False:
                print('All clusters are fully analyzed!')
                return self.latent, self.cats_per_lvl, self.probs, self.times, self.knots_tried, self.knots_found
            
        print('Maximum depth has been reached!')
        return self.latent, self.cats_per_lvl, self.probs, self.times, self.knots_tried, self.knots_found
    
    def visualize_tree(self,categories = None, vis_threshold=0.05, labelnames=None, savefigs=False, plotlegend=False):
        # plot the subdivision of clusters in hierarchical order
        givencats = 'given'
        if np.all(categories)==None:
            categories = self.cats_per_lvl[-1]
            givencats = 'found'
        if np.all(labelnames)==None:
            plotlegend = False
            labelnames = ['unknown group' for i in categories]
        else:
            plotlegend = True
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
                if self.M==2:
                    ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus+1)
                if self.M>2:
                    ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus, projection='3d')
                for k_i in range(n_clus):
                    rgba_colors[:,3] = prob_cur[:,clus]
                    vis_mask = np.array(prob_cur[:,clus]>vis_threshold)*np.array(categories==k_i)
                    if self.M==2:
                        ax.scatter(self.latent[lvl][vis_mask,0],self.latent[lvl][vis_mask,1],c = rgba_colors[vis_mask,:], label=labelnames[k_i])
                    if self.M>2:
                        ax.scatter(self.latent[lvl][vis_mask,self.dimx],self.latent[lvl][vis_mask,self.dimy],self.latent[lvl][vis_mask,self.dimz],c = rgba_colors[vis_mask,:], label=labelnames[k_i])
            plt.suptitle('Clusters on level '+str(lvl))
            if plotlegend:
                plt.legend()
            if savefigs:
                plt.savefig('tree_%s _lvl%i_%s.png'%(givencats, lvl, self.samplingmethod))
            plt.show()
        return
            
    def visualize_end(self,categories = None, vis_threshold=0.05, labelnames=None, savefigs=False, plotlegend=False):
        # plot the top-level latent data with the end-result of the clustering as colouring
        if np.all(categories)==None:
            categories = self.cats_per_lvl[-1]
            givencats = 'found'
            title = 'Top-level latent data coloured by guessed clusters'
        else:
            title = 'Top-level latent data coloured by given clusters'
            givencats = 'given'
        if np.all(labelnames)==None:
            plotlegend = False
            labelnames = ['unknown group' for i in categories]
        else:
            plotlegend = True
            
        rgba_colors = np.ones((self.N, 4))
        for k_i in range(int(max(categories))):
            rgba_colors[categories==k_i,:3] = self.colors[k_i]
#             rgba_colors[categories==k_i,3] = self.probs[-1][categories==k_i,k_i]
        fig = plt.figure(figsize=(6,6))
        if self.M==2:
#             print('Plotting 2-dimensional latent data with final cluster colouring.')
            ax = fig.add_subplot(111)
            for k_i in range(int(max(categories))):
                ax.scatter(self.latent[0][categories==k_i,0],self.latent[0][categories==k_i,1],c = rgba_colors[categories==k_i,:], label=labelnames[k_i])
        if self.M>2:
#             print('Plotting 3-dimensional latent data with final cluster colouring.')
            ax = fig.add_subplot(111, projection='3d')
            for k_i in range(int(max(categories))):
                ax.scatter(self.latent[0][categories==k_i,self.dimx],self.latent[0][categories==k_i,self.dimy],self.latent[0][categories==k_i,self.dimz],c = rgba_colors, label=labelnames[k_i])
        ax.set_title(title)
        if plotlegend:
            ax.legend()
        if savefigs:
            plt.savefig('end_%s_%s.png'%(givencats, self.samplingmethod))
        plt.show()
        return
    
    def visualize_latent_final(self, categories=None, vis_threshold=0.05, labelnames=None, savefigs=False, plotlegend=False):
        # plot the latent spaces of all found subclusters
        if np.all(categories)==None:
            title = 'Latent data coloured by guessed clusters'
            categories = self.cats_per_lvl[-1]
            givencats = 'found'
        else:
            title = 'Latent data coloured by given clusters'
            givencats = 'given'
        if np.all(labelnames)==None:
            labelnames = ['unknown group' for i in set(categories)]
        n_clus = len(set(categories))
        fig = plt.figure(figsize=(min(n_clus*6, 24), (int((n_clus-1)/4)+1)*6))
        for clus in range(n_clus):
            rgba_colors = np.zeros((self.N, 4))
            for k_i in range(int(max(categories))):
                rgba_colors[categories==k_i,:3] = self.colors[k_i]
            prob_cur = self.probs[-1]
            if self.M==2:
                ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus+1)
            if self.M>2:
                ax = fig.add_subplot(int((n_clus-1)/4)+1, min(n_clus, 4), clus, projection='3d')
            for k_i in range(n_clus):
                rgba_colors[categories==k_i,3] = prob_cur[categories==k_i,k_i]
                vis_mask = np.array(prob_cur[:,clus]>vis_threshold)*np.array(categories==k_i)
                if self.M==2:
                    ax.scatter(self.latent[-1][vis_mask,0],self.latent[-1][vis_mask,1],c = rgba_colors[vis_mask,:], label=labelnames[k_i])
                if self.M>2:
                    ax.scatter(self.latent[-1][vis_mask,self.dimx],self.latent[-1][vis_mask,self.dimy],self.latent[-1][vis_mask,self.dimz],c = rgba_colors[vis_mask,:], label=labelnames[k_i])
        plt.suptitle(title)
        if savefigs:
            plt.savefig('end_%s _lvl%i_%s.png'%(givencats, lvl, self.samplingmethod))
        plt.show()
        return
    
    def ari_per_level(self, ind):
        # Get ARI for each level
        lvls = len(self.cats_per_lvl)
        return [adjusted_rand_score(self.cats_per_lvl[lvl], ind) for lvl in range(lvls)]
    
    def visual_score(self, ind, plot_hmppca = True, plot_hmppca_logres = False, plot_real = True, plot_logreg = True, vis_threshold=0.1, labelnames=[], savefigs=False, plotlegend= False):
        # Plot result and give logistic regression scores
        acc_scores =[]
        
        if len(labelnames)==0:
            labelnames = ['unknown group' for i in set(ind)]
        for lvl in range(len(self.latent)):

            lvl_i = min(len(self.latent)-1, lvl+1)
            nclus = len(set(self.cats_per_lvl[lvl]))
            print('level ', lvl)
            if plot_hmppca:
                
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                for clus in range(nclus):
                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>vis_threshold
                    rgba_cols = np.zeros((self.N,4))
                    for k_i in set(self.cats_per_lvl[lvl_i][np.argmax(self.probs[lvl],axis=1)==clus]):
                        rgba_cols[self.cats_per_lvl[lvl_i]==k_i,:3] = self.colors[k_i]
                        rgba_cols[self.cats_per_lvl[lvl_i]==k_i,3] = self.probs[lvl][self.cats_per_lvl[lvl_i]==k_i,clus]
                        if lvl<len(self.latent)-1:
                            cc_x, cc_y = np.average(self.latent[lvl], axis=0, weights = self.probs[lvl_i][:,k_i])
                            plt.scatter(cc_x, cc_y, s = 500, c = 'black', zorder=9)
                            plt.text(cc_x, cc_y, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=10,horizontalalignment='center',
                verticalalignment='center')
                    ax.scatter(self.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:], zorder=1)
                    ax.set_title('subcluster '+str(clus+1))
                plt.suptitle('HmPPCA clusters')
                if savefigs:
                    plt.savefig('vscore_hmppca_lvl%i_%s.png'%(lvl, self.samplingmethod))
                plt.show()


            if plot_hmppca_logres:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                c_i = 0
                for clus in range(nclus):

                    classifier = sklearn.linear_self.SGDClassifier(loss='log')
                    classifier.fit(self.latent[lvl], self.cats_per_lvl[lvl_i], sample_weight=self.probs[lvl][:,clus])
                    preds = classifier.predict(self.latent[lvl])
                    preds = np.unique(preds, return_inverse=True)[1]

                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>vis_threshold
                    rgba_cols = np.ones((self.N,4))

                    for k_i in range(len(set(preds))):
                        rgba_cols[preds==k_i,:3] = self.colors[c_i]
                        c_i+=1

                    ax.scatter(self.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:])
                plt.suptitle('log.reg. clusters (based on hmppca)')
                if savefigs:
                    plt.savefig('vscore_lr_hmppca_lvl%i_%s.png'%(lvl, self.samplingmethod))
                plt.show()

            if plot_real:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                handle_list = []
                labellist= []
                for clus in range(nclus):
                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    rgba_cols = np.ones((self.N,4))
                    for k_i in set(ind):
                        mask = np.array(self.probs[lvl][:,clus]>0.1)*np.array(ind==k_i)
                        rgba_cols[ind==k_i,:3] = self.colors[k_i]
                        ax.scatter(self.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:], label= labelnames[k_i])
                        handles, labels = ax.get_legend_handles_labels()
                        for line in range(len(labels)):
                            if labels[line] not in labellist:
                                handle_list.append(handles[line])
                                labellist.append(labels[line])
                plt.suptitle('Real clusters') 
                if plotlegend:
                    fig.legend(handle_list, labellist, loc='center right')
    #                     plt.legend()
                if savefigs:
                    plt.savefig('vscore_real_lvl%i_%s.png'%(lvl, self.samplingmethod))
                plt.show()

            if plot_logreg:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                c_i = 0
                w_ARI = 0
                w_ACC = 0
                for clus in range(nclus):
                    weights_logres = self.probs[lvl][:,clus].copy()
                    preds, acc, ari = logres_scores(self.latent[lvl], ind, weights_logres, K=5)
#                     classifier = sklearn.linear_model.SGDClassifier(loss='log')
                    
#                     for k_i in set(ind):
#                         weights_logres[ind==k_i] /= (sum(ind==k_i)/self.N)
#                     classifier.fit(self.latent[lvl], ind, sample_weight=weights_logres)
#                     preds = classifier.predict(self.latent[lvl])

                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    rgba_cols = np.ones((self.N,4))

                    for k_i in set(preds):
                        rgba_cols[preds==k_i,:3] = self.colors[int(k_i)]
                        mask = np.array(weights_logres>0.1)*np.array(preds==k_i)
                        c_i+=1
                        ax.scatter(self.latent[lvl][mask,0],self.latent[lvl][mask,1], c=rgba_cols[mask,:], label=labelnames[int(k_i)])
#                     ari = adjusted_rand_score(preds[mask], ind[mask])
#                     acc = accuracy_score(preds, ind, sample_weight = self.probs[lvl][:,clus])
#                     ax.set_title('Cluster %i - ARI: %.3f, ACC: %.3f'%(clus+1,ari,acc))
                    ax.set_title('Cluster %i - ACC: %.3f'%(clus+1,acc))
                    w_ARI+= ari*(sum(weights_logres))/self.N
                    w_ACC+= acc*(sum(weights_logres))/self.N
    #             plt.suptitle('log.reg. - w. ARI: %.3f, w. ACC: %.3f'%(w_ARI, w_ACC))
                plt.suptitle('log.reg. - w. ACC: %.3f'%(w_ACC))
                acc_scores.append(w_ACC)
                if plotlegend:
                    fig.legend(handle_list, labellist, loc='center right')
                if savefigs:
                    plt.savefig('vscore_lr_real_lvl%i_%s.png'%(lvl, self.samplingmethod))
                plt.show()
            
        return acc_scores

            
def weighted_Accuracy(true, weights):
    # Compute weighted accuracy
    total = 0
    correct = 0
    for i in range(len(true)):
        total+=1
        correct+=weights[i,int(true[i])]
        
    return correct/total