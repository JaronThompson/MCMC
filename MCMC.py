import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import linregress
import matplotlib.pyplot as plt

class MC:

    def __init__(self, model, prior, proposal, *args):
        # define the model (must be function of parameters, input )
        self.model = model
        self.model_args = args
        # the prior defines the mean and covariance of model parameters
        self.prior = prior
        # the proposal pdf is the pdf used to generate the Markov chain
        self.proposal = proposal
        # optimize the error of the estimate (precision beta)
        # self.b = b

    def fit(self, data, K):
        return self.MCMC(data, K)

    def predict(self, X):
        # use direct MCS to determine E(y) = int [y(w, x)*p(w)] dw
        H = np.zeros([len(self.XI), X.shape[0]])
        for i, xi in enumerate(self.XI):
            H[i, :] = self.model(xi, X, self.model_args)
        return np.mean(H, 0), np.std(H, 0)

    def likelihood(self, xi, X, y):
        # unpack data
        N = len(y)

        # evaluate log of likelihood fxn
        beta = 1
        lnL = -beta/2 * np.sum((y - self.model(xi, X, self.model_args))**2) + N/2*np.log(beta) - N/2 * np.log(2*np.pi)

        return np.exp(lnL)

    def MCMC(self, data, K):
        # metropolis hastings algorithm to generate posterior distributions
        X, y = data

        # define target distribution function
        def pi(xi):
            return self.likelihood(xi, X, y)*self.prior.pdf(xi)

        xi = self.prior.rvs()
        Xi = [xi]

        # using a Gaussian centered at xi(k) as proposal density for now...
        for i in range(K):
            # sample from proposal density
            xi = self.proposal.sample(Xi[-1])
            # calculate acceptance ratio
            r = min(1, pi(xi)*self.proposal.pdf(Xi[-1], xi) / (pi(Xi[-1])*self.proposal.pdf(xi, Xi[-1])))
            # simulate u from Uniform
            u = np.random.uniform(0, 1)
            # evaluate acceptance criteria
            if r >= u:
                Xi.append(xi)
            else:
                Xi.append(Xi[-1])

        XI = np.zeros([K, len(xi)])
        for i in range(K):
            XI[i, :] = Xi[i]

        self.XI = XI

        return self.XI

#%%

# import data
features = np.genfromtxt('features.csv')
targets = np.genfromtxt('targets.csv')

# make sure that features and targets are standardized
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

# define class structure for proposal pdf
class proposal:

    def __init__(self, mu, a, p):
        # set up hyper-parameters for the proposal pdf here
        self.Sigma = a*np.eye(len(mu))
        self.p = p

    def sample(self, mu):
        mu = mvn(mu, self.Sigma).rvs()
        return mu

    def pdf(self, mu1, mu2):
        # return conditional probability of mu1 given mu2
        return mvn(mu2, self.p*self.Sigma).pdf(mu1)

P = proposal(np.zeros(NF), .05, 1)

# we should be able to use this model to fit to the training data
# mu = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
mu = np.zeros(NF)
Sigma = np.eye(NF)*.05
prior = mvn(mu, Sigma)

# instantiate MC class object
mc = MC(model, prior, P)

# fit model to data using K samples from the proposal distribution
XI = mc.fit([X_train, Y_train], K=5000)
Y_pred, err = mc.predict(X_test)

slope, intercept, r_value, p_value, std_err = linregress(Y_test, Y_pred)
plt.errorbar(Y_test, Y_pred, capsize=3, linestyle='none', marker='o', yerr = err, label='R = {:.3f}'.format(r_value))
plt.legend()
plt.show()

# %% what if we used MCMC to optimize model hyperparameters?

def meta_model(hyper_params, input, args):

    a, p = hyper_params
    X_train, Y_train = args

    P = proposal(np.zeros(NF), a, p)

    # we should be able to use this model to fit to the training data
    # mu = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
    mu = np.zeros(NF)
    Sigma = np.eye(NF)*a
    prior = mvn(mu, Sigma)

    # instantiate MC class object
    mc = MC(model, prior, P)

    # fit model to data using K samples from the proposal distribution
    XI = mc.fit([X_train, Y_train], K=1000)
    Y_pred, err = mc.predict(X_train)

    return Y_pred

# fit model to data using K samples from the proposal distribution
mu = np.array([.05, 1])
Sigma = np.eye(len(mu))*.001
prior = mvn(mu, Sigma)

class proposal2:

    def __init__(self, mu, ap):
        # set up hyper-parameters for the proposal pdf here
        self.Sigma = ap*np.eye(len(mu))

    def sample(self, mu):
        mu = mvn(mu, self.Sigma).rvs()
        return np.abs(mu)

    def pdf(self, mu1, mu2):
        # return conditional probability of mu1 given mu2
        return mvn(mu2, self.Sigma).pdf(mu1)

P = proposal2([.05, 1], .001)

mc = MC(meta_model, prior, P, X_train, Y_train)
XI = mc.fit([X_train, Y_train], K=100)
a, p = np.mean(XI, 0)

#%%
P = proposal(np.zeros(NF), a, p)

# we should be able to use this model to fit to the training data
# mu = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
mu = np.zeros(NF)
Sigma = np.eye(NF)*a
prior = mvn(mu, Sigma)

# instantiate MC class object
mc = MC(model, prior, P)

# fit model to data using K samples from the proposal distribution
XI = mc.fit([X_train, Y_train], K=5000)
Y_pred, err = mc.predict(X_test)

slope, intercept, r_value, p_value, std_err = linregress(Y_test, Y_pred)
plt.errorbar(Y_test, Y_pred, capsize=3, linestyle='none', marker='o', yerr = err, label='R = {:.3f}'.format(r_value))
plt.legend()
plt.show()
