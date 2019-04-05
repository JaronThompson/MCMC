import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import linregress
import matplotlib.pyplot as plt

class MC:

    def __init__(self, model, prior):
        # define the model (must be function of parameters, input )
        self.model = model
        # the prior defines the mean and covariance
        self.mu0, self.Sigma = prior

        def proposal_sample(mu):
            return mvn(mu, self.Sigma).rvs()

        def proposal_pdf(mu1, mu2):
            # return conditional probability of mu1 given mu2
            return mvn(mu2, self.Sigma).pdf(mu1)

        self.proposal_sample = proposal_sample
        self.proposal_pdf = proposal_pdf

    def fit(self, data, K):
        return self.MCMC(data, K)

    def predict(self, X, K):

        posterior = mvn(np.mean(self.XI, 0), np.cov(self.XI.T))
        NS_test, NF = X.shape
        Y_test = np.zeros([K, NS_test])

        for i in range(K):
            Y_test[i, :] = self.model(posterior.rvs(), X)

        return np.mean(Y_test, 0), np.std(Y_test, 0)

    def likelihood(self, xi, X, y):
        # unpack data
        N = len(y)

        # evaluate log of likelihood fxn
        beta = 1 #1 / sd**2
        lnL = -beta/2 * np.sum((y - self.model(xi, X))**2) + N/2*np.log(beta) - N/2 * np.log(2*np.pi)

        #mse = np.sum((y - self.model(xi, X))**2)
        return np.exp(lnL) #np.exp(mse)

    def MCMC(self, data, K):
        # metropolis hastings algorithm to generate posterior distributions
        X, y = data

        def prior(xi):
            return mvn(self.mu0, self.Sigma).pdf(xi)

        def pi(xi):
            return self.likelihood(xi, X, y)*prior(xi)

        Xi = [self.proposal_sample(self.mu0)]

        # using a Gaussian centered at xi(k) as proposal density for now...
        for i in range(K):
            # sample from proposal density
            xi = self.proposal_sample(Xi[-1])
            # calculate acceptance ratio
            r = min(1, pi(xi)*self.proposal_pdf(Xi[-1], xi) / (pi(Xi[-1])*self.proposal_pdf(xi, Xi[-1])))
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

        return np.mean(XI, 0), np.cov(XI.T)

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
def model(params, input):
    return np.dot(input, params)

# we should be able to use this model to fit to the training data
# mu = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
mu = np.zeros(NF)
Sigma = np.eye(NF)*.05
prior = [mu, Sigma]

# instantiate MC class object
mc = MC(model, prior)

# fit model to data using K samples from the proposal distribution
posterior = mc.fit([X_train, Y_train], K=5000)

Y_pred, err = mc.predict(X_test, K=100)

slope, intercept, r_value, p_value, std_err = linregress(Y_test, Y_pred)
plt.errorbar(Y_test, Y_pred, linestyle='none', marker='o', yerr = err, label='R = {:.3f}'.format(r_value))
plt.legend()
plt.show()
