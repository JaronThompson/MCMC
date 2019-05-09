import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import linregress
import matplotlib.pyplot as plt
from MCMC import MCMC, TMCMC, Proposal, Prior
plt.style.use('ggplot')

# import data
features = np.genfromtxt('features.csv')
targets = np.genfromtxt('targets.csv')

#%% make sure that features and targets are standardized
features = (features - np.mean(features, 0)) / np.std(features, 0)
targets = (targets - np.mean(targets)) / np.std(targets)

NS, NF = features.shape
X_train = features[:int(.8*NS), :]
X_test = features[int(.8*NS):, :]
Y_train = targets[:int(.8*NS)]
Y_test = targets[int(.8*NS):]

# define our model
def model(params, input, *args):
    return np.dot(input, params)

# input mean, sampling covariance, and covariance for evaluating pdf
mu = np.zeros(NF)
#mu = np.append(np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train), 0)
Sigma = np.eye(NF)
prior = Prior(mu, .5*Sigma)
proposal = Proposal(mu, .001*np.eye(NF))

# we should be able to use this model to fit to the training data

# instantiate MC class object
mc = MCMC(model, prior, proposal)

# fit model to data using K samples from the proposal distribution
XI = mc.fit([X_train, Y_train], K=7000, Kreject=2000)
Y_pred, err = mc.predict(X_test, return_stats=False)

#%%
slope, intercept, r_value, p_value, std_err = linregress(Y_test, Y_pred)
plt.errorbar(Y_test, Y_pred, capsize=3, linestyle='none', marker='o', yerr = err, label='R = {:.3f}'.format(r_value))
plt.legend()
plt.show()
