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
from scipy.special import logsumexp
from sklearn.cluster import SpectralClustering
from scipy.stats import norm,multivariate_normal
from sklearn.metrics import adjusted_rand_score
sns.set()


def loadStan(file, recompile=False, automatic_pickle = True):
    if recompile:
        try:
            model = pystan.StanModel(file = 'StanModels/'+file+'.stan')
            print('Model compiled succesfully.')
            if automatic_pickle:
                with open('pickled_models/'+file+'.pkl', 'wb') as f:
                    pickle.dump(model, f)
                print('Model saved succesfully to cache.')
        except FileNotFoundError:
            print(file+'.stan not found!')
        except ValueError:
            print('Could not compile! Error in code maybe!')
    else:
        try:
            model = pickle.load(open('pickled_models/'+file+'.pkl', 'rb'))
            print('Model loaded succesfully from cache.')
        except:
            try:
                model = pystan.StanModel(file = 'StanModels/'+file+'.stan')
                print('Model compiled succesfully.')
                if automatic_pickle:
                    with open('pickled_models/'+file+'.pkl', 'wb') as f:
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

def est_k(points, k_max = 2, refs = 3, method='aic', verbose=False):
    clus = []
    aics = []
    bics = []
    mods = []
    for k in range(1,k_max+1):
        for ref in range(refs):
            model = KMeans(k).fit(points)
            mods.append(model)
            clus.append(k)
            a, b =kmeans_AIC(points,model, verbose=verbose)
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
    
def kmeans_AIC(points, model, verbose=False):
    theta = np.array([sum(model.labels_==k) for k in range(model.n_clusters)])/np.shape(points)[0]
    probs = np.zeros((np.shape(points)[0],model.n_clusters))
    for k in range(model.n_clusters):
        probs[:,k] = multivariate_normal.logpdf(points, mean=model.cluster_centers_[k], cov=np.std(points[model.labels_==k].T)) 
    probs+=np.log(theta)
    llh = np.sum(logsumexp(probs,axis=1))
    
    aic = -2*llh + 2*np.shape(points)[1]*model.n_clusters
    bic = -2*llh + np.log(np.shape(points)[0])*np.shape(points)[1]*model.n_clusters
    if verbose:
        print('n_clusters: ',model.n_clusters, 'AIC: ',aic, 'BIC: ',bic)
    return aic, bic

class hierarchical_model:
    
    def __init__(self):
        
        self.latent = [[]]
        self.mus = [[]]
        self.cats_per_lvl = []
    
    def fit(self,x, M=2, max_depth=5, k_max = 2, plotting=True, min_clus_size=10, vis_threshold=0.05):
        
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

        # top-level latent data
        print('Latent data on top level:')
        ppca_dat = {'N':N, 'M':M, 'D':D, 'x':x, 'weights': self.probs[-1][:,0]}
        fit_top = ppca_weighted.sampling(data=ppca_dat, iter=200, chains=1)
        fitreturn_top = fit_top.extract()
        best_ind_top = np.where(fitreturn_top['lp__']==max(fitreturn_top['lp__']))[0][0]
        latent_top = fitreturn_top['z'][best_ind_top]
        self.latent[-1].append(latent_top)

        # top-level cluster determination
        K_1, clusters_1 = est_k(x)
        self.mus[-1].append(clusters_1.cluster_centers_)
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
                # Dont divide clusters further if they are too small
                if sum(cats==cl)>k_max:
                    n_subs, subs = est_k(x[cats==cl], k_max = k_max)
                    while np.any([sum(subs.labels_==k_i)<min_clus_size for k_i in range(n_subs)]):
                        if n_subs <= 2:
                            n_subs = 1
                            break
                        n_subs, subs = est_k(x[cats==cl], k_max = n_subs-1)
                else:
                    n_subs = 1
                clus_probs = cur_probs[:,cl]
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
                    moppcas_dat = {'N':N, 'M':M,'K':n_subs, 'D':D, 'y':x, 'weights':clus_probs}
                    fit = moppcas_weighted.sampling(data=moppcas_dat, chains=1, iter=100, init=[{'mu':subs.cluster_centers_}])
                    fit_ext_molv1 = fit.extract()
                    best_molv1 = np.where(fit_ext_molv1['lp__']==max(fit_ext_molv1['lp__']))[0][0]
                    cur_latent = fit_ext_molv1['z'][best_molv1]
                    new_probs = (np.mean(fit_ext_molv1['clusters'],axis=0).T*clus_probs).T
                    plotcats = np.argmax(new_probs, axis=1)
                    
                    # and plot latent data of all newfound subclusters if chosen so
                    for i,l in enumerate(cur_latent):
                        plotprobs = new_probs[:,i]
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

            cats = np.argmax(probs_round,axis=1)
            
            # Plotting Top-level latent data with new cluster-colouring
            fig = plt.figure(figsize=(6,6))
            if M==2:
                ax = fig.add_subplot(111)
            if M>2:
                ax = fig.add_subplot(111, projection='3d')
            for c in range(np.shape(probs_round)[1]):
                mask = probs_round[:,c]>vis_threshold
                rgba_colors = np.zeros((sum(mask),4))
                rgba_colors[:,:3] = self.colors[c]
                rgba_colors[:,3] = probs_round[:,c][mask]
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
                if np.shape(self.probs[lvl])[0]==N:
                    n_cat = np.shape(self.probs[lvl])[1]
                    prob_cur = self.probs[lvl]
                else:
                    n_cat = np.shape(self.probs[lvl])[0]
                    prob_cur = self.probs[lvl].T
                for k_i in range(n_cat):
                    rgba_colors[:,3] = prob_cur[:,lat]
                vis_mask = np.array(prob_cur[:,lat]>vis_threshold)
                if M==2:
                    ax = fig.add_subplot(int((n_lat-1)/4)+1, min(n_lat, 4), lat+1)
                    ax.scatter(self.latent[lvl][lat][0,vis_mask],self.latent[lvl][lat][1,vis_mask],c = rgba_colors[vis_mask,:])
                if M>2:
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
            vis_mask = np.array(prob_cur[:,lat]>self.vis_threshold)
            if self.M==2:
                ax = fig.add_subplot(int((n_lat-1)/4)+1, min(n_lat, 4), lat+1)
                ax.scatter(self.latent[-1][lat][0,vis_mask],self.latent[-1][lat][1,vis_mask],c = rgba_colors[vis_mask,:])
            if self.M>2:
                ax = fig.add_subplot(int((n_lat-1)/4)+1, min(n_lat, 4), lat, projection='3d')
                ax.scatter(self.latent[-1][lat][self.dimx,vis_mask],self.latent[-1][lat][self.dimy,vis_mask],self.latent[-1][lat][self.dimz,vis_mask],c = rgba_colors[vis_mask,:])
        plt.suptitle(title)
        plt.show()
        return