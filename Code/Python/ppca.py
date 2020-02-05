import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class PPCA:
    
    def __init__(self):
        
        self.N = None
        self.D = None
        self.M = None
        
        self.mu_ML = None
        self.sigma_ML_closed = None
        self.W_ML_closed = None
        self.sigma_ML_em = None
        self.W_ML_em = None
        
        self.fitted = False
        self.x_sim_closed = None
    
    def fit(self, data, M=2, form='closed', em_iterations = 25, em_calc_loglikelihood = False):
        
        self.data = data.T
        
        self.M = M
        self.D, self.N = np.shape(self.data)
        
        self.mu_ML = self.mu_ML = np.mean(self.data,axis=1)[np.newaxis].T
        
        S = np.cov(self.data)
        
        if form == 'closed':
            

            eigenvalues, eigenvectors = np.linalg.eig(S)

            # We wish to use the largest eigenvalues, so we sort the eigenvalues and their corresponsing eigenvectors
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[order]

            self.sigma_ML_closed = 0.0
            for i in range(self.M,self.D):
                self.sigma_ML_closed += eigenvalues[i]
            self.sigma_ML_closed = self.sigma_ML_closed/(self.D-self.M)
            
            U = eigenvectors[:self.M].T
            L = np.diag(eigenvalues[:self.M])
            R = np.eye(self.M)
            self.W_ML_closed = np.matmul(np.matmul(U,np.sqrt((L-self.sigma_ML_closed*np.eye(self.M)))),R)
            
            self.fitted = True
            
        elif form == 'em':
            self.sigma_ML_em = 1.0
            self.W_ML_em = np.ones((self.D,self.M))
            
            if em_calc_loglikelihood == True:
                log_likes = []
            
            for iteration in range(em_iterations):
                if em_calc_loglikelihood == True:
                    # E-step
                    inv_mat = np.linalg.inv(self.sigma_ML_em*np.eye(self.M)+np.matmul(self.W_ML_em.T,self.W_ML_em))
                    E_z = np.matmul(np.matmul(inv_mat,self.W_ML_em.T),(self.data-self.mu))
                    E_zzT = self.sigma_ML_em*inv_mat+np.matmul(E_z,E_z.T)

                    sums = 0
                    for n in range(self.N):
                        sums += -0.5*np.trace(E_zzT)
                        sums -= np.trace(np.matmul((self.data-self.mu),(self.data-self.mu).T)-2*np.matmul(np.matmul((self.data-self.mu),E_z.T),self.W_ML_em.T)+np.matmul(np.matmul(self.W_ML_em,E_zzT),self.W_ML_em.T))/(2*self.sigma_ML_em)

                    Log_likelihood = -0.5*self.D*np.log(self.sigma_ML_em) - sums
                    log_likes.append(Log_likelihood)


                # M-step
                M_mat = self.sigma_ML_em*np.eye(self.M) + np.matmul(self.W_ML_em.T,self.W_ML_em)   # note that M_mat refers to a matrix and M to the number of latent dimensions

                W_ML_new = np.matmul(np.matmul(S,self.W_ML_em),np.linalg.inv(self.sigma_ML_em*np.eye(self.M)+np.matmul(np.matmul(np.matmul(np.linalg.inv(M_mat),self.W_ML_em.T),S),self.W_ML_em)))
                self.sigma_ML_em = np.trace(S-np.matmul(np.matmul(np.matmul(S,self.W_ML_em),np.linalg.inv(M_mat)),W_ML_new.T))/self.D

                self.W_ML_em = W_ML_new.copy()
                
            self.fitted = True
            
        
    def parameters(self, solution = 'closed'):
        if self.fitted == False:
            print("Let's fit our model first! Use 'PPCA.fit(observed_data)'!")
            return 1
        else:
            if solution == 'closed':
                return self.mu_ML, self.sigma_ML_closed, self.W_ML_closed
            if solution == 'em':
                return self.mu_ML, self.sigma_ML_em, self.W_ML_em
   
                  
    def predict(self, latent_data, solution = 'closed'):
        if np.shape(latent_data) == (self.M, self.N):
            self.x_sim_closed = np.random.normal(np.matmul(self.W_ML_closed,latent_data)+self.mu_ML,self.sigma_ML_closed)
            return self.x_sim_closed
        else:
            print('Make sure to input latent data of shape (M,N)!')
            return 1
        
                  
    def plot_simulation(self, plot_input_data=True, plot_simulated_data=True, solution = 'closed', save_location = None):
        if plot_input_data==False and plot_simulated_data==False:
            print('Nothing to plot!')
            return 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if plot_input_data == True:
            ax.scatter(self.data[0,:],self.data[1,:],self.data[2,:], alpha=0.2, label='Real observed data')
        if plot_simulated_data == True:
            if self.solution == 'closed':
                if self.x_sim_closed == None:
                    print('Generate simulated points first!')
                else:
                    ax.scatter(self.x_sim_closed[0,:],self.x_sim_closed[1,:],self.x_sim_closed[2,:], alpha=0.2, label='Simulated observed data')
            elif self.solution == 'em':
                if self.x_sim_em == None:
                    print('Generate simulated points first!')
                else:
                    ax.scatter(self.x_sim_em[0,:],self.x_sim_em[1,:],self.x_sim_em[2,:], alpha=0.2, label='Simulated observed data')
            else:
                print("I don't know what to plot!")
                return 1
        title = ax.set_title("Observed data")
        plt.setp(title, color='black') 
        ax.set_xlabel('Gene 1')
        ax.set_ylabel('Gene 2')
        ax.set_zlabel('Gene 3')
        plt.legend()
        if save_location != None:
            plt.savefig(save)
        plt.show()
            
            
            
            
            
            
            
            
            
            
            