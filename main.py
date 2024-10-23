from DP_BSS import ebreg
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import time
from abess.linear import LinearRegression
from scipy.linalg import toeplitz
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
        
# Data Generation
np.random.seed(1)

data_type = 'Gaussian' # Uniform or ar1 or Gaussian
n = 900
p = 2000
s = 4
B = 0.5
rho = 0.3
signal = 'weak'

if data_type == 'Uniform':
    X = np.random.uniform(-1, 1, n*p).reshape(n,p)
    e = np.random.uniform(-0.1, 0.1, size = n)
if data_type == 'ar1':
    Sigma = toeplitz([rho**i for i in range(p)])
    X = np.random.multivariate_normal(mean = np.zeros(p), cov = Sigma, size = n)
    scaler1 = StandardScaler(with_mean=True, with_std= True).fit(X)
    X = normalize(scaler1.transform(X), axis=0)
    e = np.random.uniform(-0.1, 0.1, size = n)
if data_type == 'Gaussian':
    X = np.random.multivariate_normal(mean = np.zeros(p), cov = np.identity(p), size = n)
    # scaler1 = StandardScaler(with_mean=True, with_std= True).fit(X)
    # X = normalize(scaler1.transform(X), axis=0)
    # X = normalize(X, axis=0)
    X = normalize(X, norm='max', axis=1) # normalize by max by row
    e = np.random.uniform(-0.1, 0.1, size = n)
beta = np.zeros(p)
if signal == 'weak': beta[0:s] = 2.0 * np.sqrt(1 * np.log(p)/n)
else: beta[0:s] = 2.0 * np.sqrt(s * np.log(p)/n)

y = X @ beta + e
y1 = y
X1 = X

# np.savetxt("data_X.csv", X, delimiter=",", fmt="%f")
# np.savetxt("data_y.csv", y, delimiter=",", fmt="%f")
# np.savetxt("data_beta.csv", beta, delimiter=",", fmt="%f")

Du = (B * 1 + np.abs(y1).max())**2
print(f"Du: {Du:.4f}")
print(f"Max abs(y): {np.abs(y).max():.4f}, Sum of Beta: {beta.sum():.4f}")


# run DP-BSS
tuning_parameters = {'epsilon': 0.5,
                        'sensitivity_scaling': Du,
                        's'      : s,
                        'B'      : B,   
                        'max_iter': 1e5,
                    'initialization': 'Random',
                    'standardization': False}


eps_list = [0.5, 1, 2.5, 3, 5, 10]
for eps in eps_list:
    tuning_parameters['epsilon'] = eps
    model = ebreg(tuning_parameters)

    num_chains = 10

    def fit_MCMC(i):
        model.fit(X1,y1)
        # print(str(i)+'th chain fitting complete')
        S_hat = np.where(model.S_list[-1]>0)[0]
        S = np.where(beta!=0)[0]
        prec = len(np.intersect1d(S_hat, S))/max(1,len(S_hat))
        recall = len(np.intersect1d(S_hat, S))/len(S)
        F1 = 0
        if prec + recall>0: F1 = 2 * (prec * recall)/(prec + recall)
        return {'RSS': model.RSS, 'F1': F1}

    start = time.time()
    results = Parallel(n_jobs= num_chains)(delayed(fit_MCMC)(i) for i in range(num_chains))
    end = time.time()
    print(f"epsilon_{eps}_K_{B} time usage: {end-start:.4f}")
    model_abess = LinearRegression(support_size = s)
    model_abess.fit(X1, y1)
    S_abess = np.where(model_abess.coef_!=0)[0]
    S = np.where(beta!=0)[0]
    prec = len(np.intersect1d(S_abess, S))/max(1,len(S_abess))
    recall = len(np.intersect1d(S_abess, S))/len(S)
    F1_abess = 2/(1/prec + 1/recall)
    print(f"F1 score ABESS in signal {signal}: {F1_abess:.4f}")

    RSS_abess = np.linalg.norm(y1 - X1@model_abess.coef_)**2/np.linalg.norm(y1)**2
    RSS_true = np.linalg.norm(y1 - X1@beta)**2/np.linalg.norm(y1)**2

    plt.axhline(y = 1- RSS_true, color = 'b', linestyle = '--', label = '$R_{\gamma^*}$')
    plt.axhline(y = 1- RSS_abess, color = 'r', linestyle = '--', label = '$R_{\gamma_{BSS}}(non-private)$')

    for i in range(num_chains):
        RSS = results[i]['RSS']
        if i==0: plt.plot(range(len(RSS)), (1-RSS), color = 'grey', alpha = 0.3, label = '$R_{\gamma_{BSS}}(\epsilon = $' + 
                          str(tuning_parameters['epsilon']) + ')')
        else: plt.plot(range(len(RSS)), (1-RSS), color = 'grey', alpha = 0.3)
    
    F1 = np.mean([results[i]['F1'] for i in range(num_chains)])
    plt.text(0.05, 0.85, 'Mean F1  = ' + str(F1) + '\n K = ' + str(B), size = 13, bbox = {'fc': 'wheat', 'alpha':0.8}, 
             transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left')
    # plt.text(1, 0.8, 'Mean F1  = ' + str(F1) + '\n K = ' + str(B), size = 13, bbox = {'fc': 'wheat', 'alpha':0.8})
    # plt.ylim(ymax= 1)
    plt.xlabel('iterations', size = 13)
    #plt.xlim(xmax = 1e5)
    plt.ylabel('$R_\gamma$', size = 13)
    #plt.title()
    #plt.yscale('log')  
    plt.xscale('log')
    plt.legend(loc = 'center left',fontsize = 13)
    #plt.show()
    name = str(data_type)+'_design_multiple_MH_chain_'+str(num_chains)+'_epsilon_'+str(tuning_parameters['epsilon'])+'_B_'+ str(B)+ '_s_'+str(s)+'_p_'+str(p)+'_n_'+str(n)+'.pdf'
    fig_dir = "figs/"

    if signal == 'weak': plt.savefig(fig_dir + 'weak_'+name)
    else: plt.savefig(fig_dir + name)
    plt.clf()
