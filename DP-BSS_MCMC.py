#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 01:00:27 2023

@author: roysapta
"""
#%%
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import binom
import matplotlib.pyplot as plt
#%%
'''
S <- stored as p-dim np array of 0,1 indicating which variables active

'''

tuning_parameters = {'epsilon': 1.0,
                     'sensitivity_scaling': 10,
                     's'      : 10,
                     'burn_in': 10000,
                     'thinning_size': 3,
                     'sample_size': 500}


def compare_to_rows(SS, S):
    # S <- p-dim array, SS <- (m,p) array
    # returns -1 if not present otherwise row number
    subtracted = SS - S
    sub_norm = np.linalg.norm(subtracted, axis=1)
    if len(np.where(sub_norm == 0)[0]) > 0:
        return np.where(sub_norm == 0)[0][0]
    else:
        return -1


class ebreg:

    def __init__(self, tuning_parameters):

        #tuning parameters <- dictionary
        self.s = tuning_parameters['s']
        self.epsilon = tuning_parameters['epsilon']
        self.sensitivity_scaling = tuning_parameters['sensitivity_scaling']

        # MCMC parameters
        self.burn_in = tuning_parameters['burn_in']
        self.thinning_size = tuning_parameters['thinning_size']
        self.sample_size = tuning_parameters['sample_size']

        # some options
        self.standardize = True

    def fit(self, X, y):
        self.n, self.p = X.shape
        if self.n >= 100: self.burn_in = 500
        # standardize X
        if self.standardize is True:
            scaler1 = StandardScaler(with_mean=True, with_std=False).fit(X)
            X = scaler1.transform(X)

            y = y - y.mean()

        self.X = X
        self.y = y
        
        self.xmax = np.max(np.abs(X))
        self.ymax = np.max(np.abs(y))

        self.MCMC()

   

    def draw_q(self, S):
        S1 = S.copy()
        s = S.sum()
        idx_0 = np.where(S1 == 0)[0]
        idx_1 = np.where(S1 == 1)[0]
        S1[np.random.choice(idx_1, 1)] = 0
        S1[np.random.choice(idx_0, 1)] = 1

        return S1

    def OLS_pred_and_pi_n(self, S):
        X_S = self.X[:, S == 1]
        y = self.y

        reg = LR(fit_intercept=False).fit(X_S, y)  # what happens when singular
        Y_S = reg.predict(X_S)

        epsilon = self.epsilon
        sigma2 = self.sigma2_hat
        # gamma = self.gamma
        Du = self.sensitivity_scaling #2 * self.ymax**2 + 2 * self.s * (self.xmax**2) *  self.sensitivity_scaling * sigma2
        # s = S.sum()
        log_pi_n =  - epsilon * np.linalg.norm(y - Y_S) ** 2/(
            2*Du)  # np.log(gamma + alpha/sigma2)
        return Y_S, log_pi_n

    def initialize(self):
        S = np.zeros(self.p)
        #S[np.random.choice(self.p, self.s, replace= False)] = 1
        S[0:s] = 1
        S[np.random.choice(self.s, 2, replace= False)] = 0
        S[self.s] = 1; S[self.s+1] = 1
        # use lasso to get S and estimate of sigma^2
        
        
        reg = LassoCV(n_alphas=500, fit_intercept=False,
                      cv=5, max_iter=2000).fit(self.X, self.y)
        # S = 1 * (reg.coef_ != 0)
        # if S.sum()==0: S[np.random.choice(self.p, 1)] = 1
        # if S.sum()>=self.n:
        #     num = np.min([S.sum()/2, self.n/2])
        #     S[np.random.choice(self.p, int(num), replace = False)] = 0
        y_hat = reg.predict(self.X)
        
        self.sigma2_hat = ((self.y- y_hat)**2).mean()
        self.S = S
        
        
        return S

    def MCMC(self):
        burn_in = self.burn_in
        thinning_size = self.thinning_size
        m = self.sample_size  # number of samples
        max_iter = burn_in + m * thinning_size

        # initialize
        S = self.initialize()
        self.S_list = np.array([self.S])
        Y_S, log_pi_n = self.OLS_pred_and_pi_n(self.S)
        self.Y_S_list = np.array([Y_S])
        self.log_pi_n_list = [log_pi_n]

        # final collection objects
        S_samples = np.array([self.S])  # finally (m+1)Xp
        

        S = self.S
        y = self.y

        iter1 = 0
        no_acceptances = 0

        while (iter1 < max_iter):

            # proposal draw
            S_new = self.draw_q(S)

            # check if this S has already been seen
            idx = compare_to_rows(self.S_list, S_new)
            if idx == -1:
                Y_S_new, log_pi_n_new = self.OLS_pred_and_pi_n(S_new)
                self.S_list = np.vstack((self.S_list, S_new))
                self.Y_S_list = np.vstack((self.Y_S_list, Y_S_new))
                self.log_pi_n_list.append(log_pi_n_new)
            else:
                Y_S_new = self.Y_S_list[idx]
                log_pi_n_new = self.log_pi_n_list[idx]

            # compute hastings ratio
            R = np.min([1, np.exp(log_pi_n_new - log_pi_n)])
            if stats.uniform.rvs() <= R:
                # accept
                S = S_new
                log_pi_n = log_pi_n_new
                no_acceptances += 1

            if (iter1 >= burn_in) and ((iter1-burn_in) % thinning_size == 0):
                S_samples = np.vstack((S_samples, S))
                X_S = self.X[:, S == 1]
                # reg = LR(fit_intercept=False).fit(X_S, y)
                # beta = np.zeros(self.p)
                # beta[np.where(S == 1)[0]] = reg.coef_
                # beta_samples = np.vstack((beta_samples, beta))

            iter1 += 1

        self.S_samples = S_samples
        # self.beta_samples = beta_samples
        print('acceptance rate is', no_acceptances/iter1)
        self.acceptance = no_acceptances 


#%%

# Example
n = 900
p = 2000
s = 10
rho = 0.0
Sigma = np.eye(p)*(1-rho) + rho * np.ones((p, p))
X = np.random.multivariate_normal(np.zeros(p), Sigma, n)
beta = np.zeros(p)
beta[0:s] = 10.0


y = X @ beta + np.random.normal(0, 1, n)
#%%

model = ebreg(tuning_parameters)
model.fit(X, y)

# Y_hat_mat = X @ model.beta_samples.T
# Y_hat_mean = Y_hat_mat.mean(axis=1)
# print('mean prediction error=', np.linalg.norm(Y_hat_mean - y)**2 / n)

# # lasso prediction
# beta_lasso = model.beta_samples[0]
# y_hat_lasso = X @ beta_lasso
# print('lasso pred=', np.linalg.norm(y_hat_lasso - y)**2 / n)

S_size = model.S_samples.sum(axis=1)
plt.plot(range(len(S_size)), S_size)

# prediciton errors
# m = model.beta_samples.shape[0]
# err = np.zeros(m)
# for i in range(m):
#     beta_ = model.beta_samples[i]
#     y_hat_ = X @ beta_
#     err[i] = np.linalg.norm(y_hat_ - y)**2 / n

#plt.plot(range(m), err)