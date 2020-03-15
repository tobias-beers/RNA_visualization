import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pystan
import numpy as np
import seaborn as sns
from scipy.stats import norm
import itertools
from sklearn.cluster import KMeans
from sklearn import datasets
import os
import pickle
import scipy


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