import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
from abess.linear import LinearRegression

def get_data(eps, iter, signal, num_chains, data_type, K, s) :
    dir_name = data_type + "/eps_" + str(eps) + "_iter_" + str(iter) + "_" + signal + "_K_" + K + "_s_" + str(s)
    X = pd.read_csv(dir_name + "/data_X.csv", header=None)
    y = pd.read_csv(dir_name + "/data_y.csv", header=None)
    beta = pd.read_csv(dir_name + "/data_beta.csv", header=None)

    temp = list() # initalize empty dataframe
    f1_temp = list()

    for i in range(num_chains) :
        file_name = "/chain_" + str(i+1) + "_eps_" + str(eps) + "_iter_" + str(iter) + ".csv"
        result = pd.read_csv(dir_name + file_name, header=None, index_col=None)
        temp.append(result)

        f1_file_name = "/F1_chain_" + str(i+1) + "_eps_" + str(eps) + "_iter_" + str(iter) + ".csv"
        f1 = pd.read_csv(dir_name + f1_file_name, header=None, index_col=None)
        f1_temp.append(f1)

    RSS = pd.concat(temp, axis=1, ignore_index=True)
    F1_scores = pd.concat(f1_temp, axis=1, ignore_index=True)
    F1 = F1_scores.iloc[:-1]
    sens_scale = F1_scores.iloc[-1][0]

    return {'RSS': RSS, 'F1': F1, 'Sensitivity_scale': sens_scale, 'X': X, 'y': y, 'beta': beta}

def get_graph(eps, iter, signal, num_chains, data_type, K, s, text_x=0.05, text_y=0.85) :
    data = get_data(eps, iter, signal, num_chains, data_type, K, s)
    RSS_all = data['RSS']
    F1 = data['F1']
    Sens_scale = data['Sensitivity_scale']
    X = data['X']
    y = data['y']
    beta = data['beta']

    # parameter of data
    n, p = X.shape

    # ABESS
    model_abess = LinearRegression(support_size = s)
    model_abess.fit(X, y)
    S_abess = np.where(model_abess.coef_!=0)[0]
    S = np.where(beta!=0)[0]
    prec = len(np.intersect1d(S_abess, S))/max(1,len(S_abess))
    recall = len(np.intersect1d(S_abess, S))/len(S)
    F1_abess = 2/(1/prec + 1/recall)

    RSS_abess = np.linalg.norm(y - X@model_abess.coef_.reshape((-1, 1)))**2/np.linalg.norm(y)**2
    RSS_true = np.linalg.norm(y - X@beta)**2/np.linalg.norm(y)**2

    print('R_gamma^* :', 1-RSS_true)
    print('R_gamma_BSS :', 1-RSS_abess)

    plt.axhline(y = 1- RSS_true, color = 'b', linestyle = '--', label = '$R_{\gamma^*}$')
    plt.axhline(y = 1- RSS_abess, color = 'r', linestyle = '--', label = '$R_{\gamma_{BSS}}(non-private)$')

    for i in range(num_chains):
        RSS = RSS_all.loc[:,i]
        if i==0: plt.plot(range(len(RSS)), (1-RSS), color = 'grey', alpha = 0.3, label = '$R_{\gamma_{BSS}}(\epsilon = $' + 
                        str(eps) + ')')
        else: plt.plot(range(len(RSS)), (1-RSS), color = 'grey', alpha = 0.3)

    plt.xlabel('iterations', size = 15)
    plt.ylabel('$R_\gamma$', size = 15) 
    plt.xscale('log')
    plt.legend(loc = 'center left',fontsize = 10)
    plt.text(text_x, text_y, 'Mean F1  = ' + str(F1) + '\n K = ' + K, size = 13, bbox = {'fc': 'wheat', 'alpha':0.8}, 
             transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left')
    
    name = 'cpp_' + str(signal) + '_' + str(data_type)+'_design_multiple_MH_chain_' +\
        str(num_chains)+'_eps_' + str(eps) + '_B_'+K+'_s_'+str(s)+'_p_'+str(p)+'_n_'+str(n)+'.pdf'

    print("eps_" + str(eps) + "_iter_" + str(iter) + "_" + signal + "_K_" + K)
    print("Mean F1 = %.4f" % F1.iloc[-1].mean())

    plt.savefig("cpp_figs/" + name, dpi=1000)


if (len(sys.argv) < 8):
    # default parameter
    print("Using default parameter: eps=3, iter=1000000, signal=strong, num_chain=10, data_type=Uniform, K=3.5, s=4")
    eps = 3
    iter = 1000000
    signal = "strong"
    num_chains = 10
    data_type = "Uniform"
    K = 3.5,
    sparse = 4
else:
    eps = float(sys.argv[1])
    iter = int(sys.argv[2])
    signal = str(sys.argv[3])
    num_chains = int(sys.argv[4])
    data_type = str(sys.argv[5])
    K = str(sys.argv[6])
    sparse = int(sys.argv[7])

get_graph(eps, iter, signal, num_chains, data_type, K, sparse)