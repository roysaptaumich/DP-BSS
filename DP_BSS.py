import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler
from scipy import stats
import cvxpy as cp



'''
S <- stored as p-dim np array of 0,1 indicating which variables active

'''



def compare_to_rows(SS, S):
    # S <- p-dim array, SS <- (m,p) array
    # returns -1 if not present otherwise row number
    logic = np.any(np.all(SS == S, axis =1))
    if logic == True:
        return np.where(logic == True)[0][0]
    else:
        return -1

class ebreg:

    def __init__(self, tuning_parameters):

        #tuning parameters <- dictionary
        self.s = tuning_parameters['s']
        self.B = tuning_parameters['B']
        self.epsilon = tuning_parameters['epsilon']
        self.sensitivity_scaling = tuning_parameters['sensitivity_scaling']

        # MCMC parameters
        self.max_iter = tuning_parameters['max_iter']
        
        # some options
        self.standardize = tuning_parameters["standardization"]
        self.initialization = tuning_parameters["initialization"]

    def fit(self, X, y):
        self.n, self.p = X.shape
        #if self.n >= 100: self.burn_in = 500
        # standardize X
        if self.standardize is True:
            scaler1 = StandardScaler(with_mean=True, with_std=False).fit(X)
            X = scaler1.transform(X)

            y = y - y.mean()

        self.X = X
        self.y = y
        

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
        Du = self.sensitivity_scaling #2 * self.ymax**2 + 2 * self.s * (self.xmax**2) *  self.sensitivity_scaling * sigma2
        # s = S.sum()
        log_pi_n =  - epsilon * np.linalg.norm(y - Y_S)**2/(Du)  # np.log(gamma + alpha/sigma2)
        return Y_S, log_pi_n
                    
    def regOLS_pred_and_pi_n(self, S):
        X_S = self.X[:, S == 1]
        y = self.y
        
        b = cp.Variable(shape = X_S.shape[1])
        constraints = [cp.norm1(b) <= self.B]
        
        obj = cp.Minimize(cp.sum_squares(X_S@b - y))

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve() 
        Y_S = X_S @ b.value
        Du = self.sensitivity_scaling #2 * self.ymax**2 + 2 * self.s * (self.xmax**2) *  self.sensitivity_scaling * sigma2
        # s = S.sum()
        log_pi_n =  - self.epsilon * np.linalg.norm(y - Y_S)**2/(Du)
        
        return Y_S, log_pi_n
    
    
    def initialize(self):
        
        S = np.zeros(self.p)
        
        if self.initialization == "Lasso":
            reg = LassoCV(n_alphas=100, fit_intercept=False,
                          cv=5, max_iter=2000).fit(self.X, self.y)
            scores = np.abs(reg.coef_)
            s_max_scores_id = np.argsort(scores)[::-1][:s]
            S[s_max_scores_id] = 1
        
        if self.initialization == "MS":
            c = X.T@y/self.n
            c1 = np.abs(c)
            c2 = np.argsort(c1)[::-1][:s]
            S[c2] = 1
        
        else: S[np.random.choice(self.p, self.s, replace= False)] = 1 # random
        
        self.initial_state = S
        
        self.S = S
        
        
        return S

    def MCMC(self):
        max_iter = self.max_iter
        
        # initialize
        S = self.initialize()
        self.S_list = [self.S]
        Y_S, log_pi_n = self.regOLS_pred_and_pi_n(self.S)
        self.Y_S_old = Y_S
        self.log_pi_n_list = [log_pi_n]
        self.RSS = np.array([np.linalg.norm(self.y - Y_S)**2/np.linalg.norm(self.y)**2])
        #self.F1 = [0]

        

        S = self.S
        y = self.y

        iter1 = 0
        no_acceptances = 0

        while (iter1 < max_iter):

            # proposal draw
            S_new = self.draw_q(S)
            Y_S_new, log_pi_n_new = self.regOLS_pred_and_pi_n(S_new)
            

            # compute hastings ratio
            try: HR = np.exp(log_pi_n_new - log_pi_n)
            except ValueError: print('Hastings ratio uncomputable')
            R = np.min([1, HR])
            if stats.uniform.rvs() <= R:
                # accept
                self.RSS = np.vstack((self.RSS, np.linalg.norm(self.y - Y_S_new)**2/np.linalg.norm(self.y)**2))
                self.S_list.pop()
                self.S_list.append(S_new)
                self.Y_S_old = Y_S_new
                S = S_new
                log_pi_n = log_pi_n_new
                no_acceptances += 1
            else:
                self.RSS = np.vstack((self.RSS, np.linalg.norm(self.y - self.Y_S_old)**2/np.linalg.norm(self.y)**2))

        

            iter1 += 1

        
        self.acceptance = no_acceptances 