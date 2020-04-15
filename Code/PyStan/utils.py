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
    if k_min>=k_max:
        print('k_min must be smaller than k_max!')
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
        
        self.latent = [[]]
        self.mus = [[]]
        self.cats_per_lvl = []
    
    def fit(self,x, M=2, max_depth=5, k_max = 2, plotting=True, min_clus_size=10, vis_threshold=0.05, gmm = False, its=300, samplingmethod='vb'):
        
        # initialize the algorithm
        self.M = M
        
        if M>3 and plotting:
            print('Latent dimensions greater than 3, plotting only the three most important dimensions.')
        moppcas_weighted = loadStan('moppcas_weighted')
        ppca_weighted = loadStan('ppca_weighted')

        N,D = np.shape(x)
        self.N = N
        self.probs = [np.ones(self.N)[np.newaxis].T]
        self.colors = [np.random.uniform(size=3) for k in range(k_max**max_depth)]
        
        if gmm:
            est_ref = 1
        else:
            est_ref = 2

        # top-level latent data
        print('Latent data on top level:')
        ppca_dat = {'N':N, 'M':M, 'D':D, 'x':x, 'weights': self.probs[-1][:,0]}
        if samplingmethod == 'vb':
            fit = ppca_weighted.vb(data=ppca_dat)
            df = pd.read_csv(fit['args']['sample_file'].decode('ascii'), comment='#').dropna()
            dfmean = df.mean()
            dfz = dfmean[dfmean.index.str.startswith('z.')]
            latent_top = np.array(dfz).reshape(self.N, M).T
        elif samplingmethod == 'NUTS':
            fit_top = ppca_weighted.sampling(data=ppca_dat, iter=its, chains=1)
            fitreturn_top = fit_top.extract()
            best_ind_top = np.where(fitreturn_top['lp__']==max(fitreturn_top['lp__']))[0][0]
            latent_top = fitreturn_top['z'][best_ind_top]
        else:
            print("Please use 'NUTS' or 'vb' as samplingmethod!")
            return 1
        
        self.latent[-1].append(latent_top)

        # top-level cluster determination
        K_1, clusters_1 = est_k(x, k_max = k_max, gmm = gmm, refs = est_ref)
        self.mus[-1].append(clusters_1['mu'])
        print('Estimated number of clusters (level 0): ', K_1)
        
        # in case M>3, find out which three dimensions to plot in 3D
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
            rgba_colors = np.zeros((N,4))
#             for k_i in range(K_1):
            rgba_colors[:,:3] = self.colors[0]
            rgba_colors[:,3] = 1.0
            fig = plt.figure(figsize=(6,6))
            if M==2:
                ax = fig.add_subplot(111)
                plt.scatter(self.latent[-1][0][0,:],self.latent[-1][0][1,:], c = rgba_colors)
            if M>2:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(self.latent[-1][0][self.dimx,:],self.latent[-1][0][self.dimy,:],self.latent[-1][0][self.dimz,:], c = rgba_colors)
            ax.set_title('Top-level latent data')
            plt.show()

        cats = np.argmax(self.probs[-1],axis=1)
        self.cats_per_lvl.append(cats.copy())

        for lvl in range(max_depth):
            # repeat cluster detemination and MoPPCAs until max_depth is reached or until all clusters fully analyzed
            more_depth = False
            print('level ', lvl+1, ':')
            cur_probs = self.probs[-1].copy()
            # probability matrix of clusters is sometimes transposed
            if np.shape(cur_probs)[1]==N:
                cur_probs = cur_probs.T
            n_clus = np.shape(cur_probs)[1]
            count = 0
            self.latent.append([])
            
            # analyze all subclusters as found in the last level
            for cl in range(n_clus):
                
                clus_probs = cur_probs[:,cl]
                
                # Dont divide clusters further if they are too small
                if sum(cats==cl)>k_max:
                    n_subs, subs = est_k(x[cats==cl], k_max = k_max, gmm = gmm, refs = est_ref, weights=cur_probs)
                    while np.any([sum(subs['labels']==k_i)<min_clus_size for k_i in range(n_subs)]):
                        if n_subs <= 2:
                            n_subs = 1
                            break
                        n_subs, subs = est_k(x[cats==cl], k_max = n_subs-1, gmm = gmm, refs = est_ref, weights=cur_probs)
                else:
                    n_subs = 1
                
                if n_subs == 1:
                    # If cluster doesnt contain more subclusters, just copy it over to the next level
                    print('Cluster ', cl+1, ' doesnt contain any more subclusters')
                    new_probs = clus_probs[np.newaxis].T
                    cur_latent = self.latent[-2][cl]
                    self.latent[-1].append(cur_latent)
                    mask = clus_probs>vis_threshold
                    # And plot if chosen so
                    if plotting:
                        rgba_colors = np.zeros((sum(mask),4))
                        rgba_colors[:,:3] = self.colors[count]
                        rgba_colors[:,3] = new_probs[mask,0]
                        fig = plt.figure(figsize=(6,6))
                        if M==2:
                            ax = fig.add_subplot(111)
                            ax.scatter(cur_latent[0,mask], cur_latent[1,mask], c=rgba_colors)
                        if M>2:
                            ax = fig.add_subplot(111, projection='3d')
                            ax.scatter(cur_latent[self.dimx,mask], cur_latent[self.dimy,mask], cur_latent[self.dimz,mask], c=rgba_colors)
                        ax.set_title('Latent data of subcluster '+str(cl+1)+' (copied over from higher level)')
                        plt.show()
                    count+=1
                else:
                    # If cluster contains more subclusters, initiate MoPPCAs
                    more_depth = True
                    print('Cluster ', cl+1, ' contains ',n_subs,' subclusters')
#                     try:
                    moppcas_dat = {'N':N, 'M':M,'K':n_subs, 'D':D, 'y':x, 'weights':clus_probs}
                    if samplingmethod == 'vb':
#                         if lvl==0:
#                             fit = pickle.load(open('smallsplat_vb.pkl', 'rb'))
#                         else:
                        fit = moppcas_weighted.vb(data=moppcas_dat, init=[{'mu':subs['mu'], 'theta':subs['theta'], 'z':[np.zeros((M,self.N)) for i in range(n_subs)]}])
                        df = pd.read_csv(fit['args']['sample_file'].decode('ascii'), comment='#').dropna()
                        dfmean = df.mean()
                        dfz = dfmean[dfmean.index.str.startswith('z.')]
                        cur_latent = np.array(dfz).reshape(self.N, M, n_subs).T
                        dfclus = dfmean[dfmean.index.str.startswith('clusters.')]
                        rawprobs = np.array(dfclus).reshape(n_subs,self.N).T
                    elif samplingmethod == 'NUTS':
                        fit = moppcas_weighted.sampling(data=moppcas_dat, chains=1, iter=its, init=[{'mu':subs['mu'],
                                                                                                     'z':[np.zeros((M,self.N)) for i in range(n_subs)]}])
                        fit_ext_molv1 = fit.extract()
                        best_molv1 = np.where(fit_ext_molv1['lp__']==max(fit_ext_molv1['lp__']))[0][0]
                        cur_latent = fit_ext_molv1['z'][best_molv1]
                        rawprobs = np.mean(fit_ext_molv1['clusters'],axis=0).T
                    else:
                        print("Please use 'NUTS' or 'vb' as samplingmethod!")
                        return 1
                    
                        
                    new_probs = (rawprobs*clus_probs[np.newaxis].T).T
                    plotcats = np.argmax(new_probs, axis=1)
                    
                    # and plot latent data of all newfound subclusters if chosen so
                    for i,l in enumerate(cur_latent):
                        plotprobs = new_probs[i,:]
                        mask = plotprobs>vis_threshold
                        self.latent[-1].append(l)
                        if plotting:
                            rgba_colors = np.zeros((sum(mask),4))
#                             n_subs, subs = est_k(x[mask], k_max = k_max)
#                             for k_i in range(n_subs):
                            rgba_colors[:,:3] = self.colors[count]
                            count+=1
                            rgba_colors[:,3] = plotprobs[mask]
                            fig = plt.figure(figsize=(6,6))
                            if M==2:
                                ax = fig.add_subplot(111)
                                ax.scatter(l[0,mask],l[1,mask], c=rgba_colors)
                            if M>2:
                                ax = fig.add_subplot(111, projection='3d')
                                ax.scatter(l[self.dimx,mask],l[self.dimy,mask], l[self.dimz,mask], c=rgba_colors)
                            ax.set_title('Latent data of subcluster '+str(i+1))
                            plt.show()
                if cl==0:
                    probs_round = new_probs
                else:
                    probs_round = np.hstack((probs_round, new_probs))
            
            
            if np.shape(probs_round)[0]==self.N:
                probs_round = probs_round.T
#             print(lvl, np.shape(probs_round))
            cats = np.argmax(probs_round,axis=1)
            
            # Plotting Top-level latent data with new cluster-colouring
            fig = plt.figure(figsize=(6,6))
            if M==2:
                ax = fig.add_subplot(111)
            if M>2:
                ax = fig.add_subplot(111, projection='3d')
            for c in range(np.shape(probs_round)[0]):
                mask = probs_round[c,:]>vis_threshold
                rgba_colors = np.zeros((sum(mask),4))
                rgba_colors[:,:3] = self.colors[c]
                rgba_colors[:,3] = probs_round[c,:][mask]
                if M==2:
                    ax.scatter(self.latent[0][0][0,mask],self.latent[0][0][1,mask], c = rgba_colors)
                if M>2:
                    ax.scatter(self.latent[0][0][self.dimx,mask],self.latent[0][0][self.dimy,mask],self.latent[0][0][self.dimz,mask], c = rgba_colors)
            ax.set_title('Clusters after level '+str(lvl+1))
            plt.show()

            # Stop if all clusters are fully analyzed
            if more_depth == False:
                print('All clusters are fully analyzed!')
                self.latent = self.latent[:-1]
                return self.latent[-1], self.cats_per_lvl[-1], self.probs[-1]
            
            self.probs.append(probs_round.copy())
            self.cats_per_lvl.append(cats.copy())
            
        return self.latent[-1], self.cats_per_lvl[-1], self.probs[-1]
    
    def visualize_tree(self,categories = None, vis_threshold=0.05):
        # plot the subdivision of clusters in hierarchical order
        if np.all(categories)==None:
            categories = self.cats_per_lvl[-1]
        for lvl in range(len(self.latent)):
            print('Level ', lvl)
            n_lat = len(self.latent[lvl])
            fig = plt.figure(figsize=(min(n_lat*6, 24), (int((n_lat-1)/4)+1)*6))
            for lat in range(n_lat):
                rgba_colors = np.zeros((self.N, 4))
                for k_i in range(int(max(categories))):
                    rgba_colors[categories==k_i,:3] = self.colors[k_i]
                if np.shape(self.probs[lvl])[0]==self.N:
                    n_cat = np.shape(self.probs[lvl])[1]
                    prob_cur = self.probs[lvl]
                else:
                    n_cat = np.shape(self.probs[lvl])[0]
                    prob_cur = self.probs[lvl].T
                for k_i in range(n_cat):
                    rgba_colors[:,3] = prob_cur[:,lat]
                vis_mask = np.array(prob_cur[:,lat]>vis_threshold)
                if self.M==2:
                    ax = fig.add_subplot(int((n_lat-1)/4)+1, min(n_lat, 4), lat+1)
                    ax.scatter(self.latent[lvl][lat][0,vis_mask],self.latent[lvl][lat][1,vis_mask],c = rgba_colors[vis_mask,:])
                if self.M>2:
                    ax = fig.add_subplot(int((n_lat-1)/4)+1, min(n_lat, 4), lat, projection='3d')
                    ax.scatter(self.latent[lvl][lat][self.dimx,vis_mask],self.latent[lvl][lat][self.dimy,vis_mask],self.latent[lvl][lat][self.dimz,vis_mask],c = rgba_colors[vis_mask,:])
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
            ax.scatter(self.latent[0][0][0,:],self.latent[0][0][1,:],c = rgba_colors)
        if self.M>2:
#             print('Plotting 3-dimensional latent data with final cluster colouring.')
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.latent[0][0][self.dimx,:],self.latent[0][0][self.dimy,:],self.latent[0][0][self.dimz,:],c = rgba_colors)
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
        n_lat = len(self.latent[-1])
        fig = plt.figure(figsize=(min(n_lat*6, 24), (int((n_lat-1)/4)+1)*6))
        for lat in range(n_lat):
            rgba_colors = np.zeros((self.N, 4))
            for k_i in range(int(max(categories))):
                rgba_colors[categories==k_i,:3] = self.colors[k_i]
            if np.shape(self.probs[-1])[0]==self.N:
                n_cat = np.shape(self.probs[-1])[1]
                prob_cur = self.probs[-1]
            else:
                n_cat = np.shape(self.probs[-1])[0]
                prob_cur = self.probs[-1].T
            for k_i in range(n_cat):
                rgba_colors[:,3] = prob_cur[:,lat]
            vis_mask = np.array(prob_cur[:,lat]>vis_threshold)
            if self.M==2:
                ax = fig.add_subplot(int((n_lat-1)/4)+1, min(n_lat, 4), lat+1)
                ax.scatter(self.latent[-1][lat][0,vis_mask],self.latent[-1][lat][1,vis_mask],c = rgba_colors[vis_mask,:])
            if self.M>2:
                ax = fig.add_subplot(int((n_lat-1)/4)+1, min(n_lat, 4), lat, projection='3d')
                ax.scatter(self.latent[-1][lat][self.dimx,vis_mask],self.latent[-1][lat][self.dimy,vis_mask],self.latent[-1][lat][self.dimz,vis_mask],c = rgba_colors[vis_mask,:])
        plt.suptitle(title)
        plt.show()
        return
    
    def ari_per_level(self, ind):
        lvls = len(self.cats_per_lvl)
        return [adjusted_rand_score(self.cats_per_lvl[lvl], ind) for lvl in range(lvls)]
    
    def visual_score(self, ind, plot_hmppca = True, plot_hmppca_logres = False, plot_real = True, plot_logreg = True):
        for lvl in range(len(self.latent)):

            lvl_i = min(len(self.latent)-1, lvl+1)

            if plot_hmppca:

                nclus = len(self.latent[lvl])
                print('level ', lvl)
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                for clus in range(nclus):
                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>0.1
                    rgba_cols = np.zeros((N,4))
                    for k_i in set(self.cats_per_lvl[lvl_i][np.argmax(self.probs[lvl],axis=1)==clus]):
                        rgba_cols[self.cats_per_lvl[lvl_i]==k_i,:3] = self.colors[k_i]
                        rgba_cols[self.cats_per_lvl[lvl_i]==k_i,3] = self.probs[lvl][self.cats_per_lvl[lvl_i]==k_i,clus]
                        if lvl<len(self.latent)-1:
                            cc_x, cc_y = np.average(self.latent[lvl][clus], axis=1, weights = self.probs[lvl_i][:,k_i])
                            plt.scatter(cc_x, cc_y, s = 500, c = 'black', zorder=9)
                            plt.text(cc_x, cc_y, str(k_i+1),fontweight= 'bold', size=14, c = 'white', zorder=10,horizontalalignment='center',
                verticalalignment='center')
                    ax.scatter(self.latent[lvl][clus][0,:][mask],self.latent[lvl][clus][1,:][mask], c=rgba_cols[mask,:], zorder=1)
                    ax.set_title('subcluster '+str(clus+1))
                plt.suptitle('HmPPCA clusters')
                plt.show()


            if plot_hmppca_logres:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                c_i = 0
                for clus in range(len(self.latent[lvl])):

                    classifier = sklearn.linear_model.SGDClassifier(loss='log')
                    classifier.fit(self.latent[lvl][clus].T, self.cats_per_lvl[lvl_i], sample_weight=self.probs[lvl][:,clus])
                    preds = classifier.predict(self.latent[lvl][clus].T)
                    preds = np.unique(preds, return_inverse=True)[1]

                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>0.1
                    rgba_cols = np.ones((N,4))

                    for k_i in range(len(set(preds))):
                        rgba_cols[preds==k_i,:3] = self.colors[c_i]
                        c_i+=1

                    ax.scatter(self.latent[lvl][clus][0,:][mask],self.latent[lvl][clus][1,:][mask], c=rgba_cols[mask,:])
                plt.suptitle('log.reg. clusters (based on hmppca)')
                plt.show()

            if plot_real:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                for clus in range(len(self.latent[lvl])):
                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>0.1
                    rgba_cols = np.ones((N,4))
                    for k_i in range(len(set(ind))):
                        rgba_cols[ind==k_i,:] = self.colors[k_i]
                    ax.scatter(self.latent[lvl][clus][0,:][mask],self.latent[lvl][clus][1,:][mask], c=rgba_cols[mask,:])
                plt.suptitle('Real clusters')
                plt.show()

            if plot_logreg:
                fig = plt.figure(figsize=(min(nclus*4, 24),5*(int(nclus/6)+1)))
                c_i = 0
                w_ARI = 0
                w_ACC = 0
                for clus in range(len(self.latent[lvl])):

                    classifier = sklearn.linear_model.SGDClassifier(loss='log')
                    classifier.fit(self.latent[lvl][clus].T, ind, sample_weight=self.probs[lvl][:,clus])
                    preds = classifier.predict(self.latent[lvl][clus].T)

                    ax = fig.add_subplot(int(nclus/6)+1,min(nclus,6),clus+1)
                    mask = self.probs[lvl][:,clus]>0.1
                    rgba_cols = np.zeros((N,4))

                    for k_i in set(preds):
                        rgba_cols[preds==k_i,:] = cols[int(k_i)]
                        c_i+=1
                    rgba_cols[:,3] = np.ones(N)
                    ax.scatter(model.latent[lvl][clus][0,:][mask],self.latent[lvl][clus][1,:][mask], c=rgba_cols[mask,:])
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