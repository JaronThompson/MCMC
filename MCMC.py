import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import linregress
import matplotlib.pyplot as plt

class MC:

    def __init__(self, model, prior, a=1, b=1):
        # define the model (must be function of parameters, input )
        self.model = model
        # the prior defines the mean and covariance
        self.mu0, self.Sigma = prior
        # optimize the error of the estimate (precision beta)
        self.b = b
        # optmize the covariance of the prior (alpha)
        self.a = a
        # optimize the covariance of the proposal PDF
        self.p = a

        def proposal_sample(mu, b, a, p):
            mu = mvn(mu, p*self.Sigma).rvs()
            b = abs(mvn(b, .1).rvs())
            a = abs(mvn(a, .1).rvs())
            p = abs(mvn(p, .1).rvs())
            return mu, b, a, p

        def proposal_pdf(mu1, mu2, p):
            # return conditional probability of mu1 given mu2
            return mvn(mu2, p*self.Sigma).pdf(mu1)

        self.proposal_sample = proposal_sample
        self.proposal_pdf = proposal_pdf

    def fit(self, data, K):
        return self.MCMC(data, K)

    def predict(self, X):
        Y_pred = self.model(np.mean(self.XI, 0), X)
        Y_err = self.model(np.std(self.XI, 0), X)
        return Y_pred, Y_err

    def likelihood(self, xi, X, y, beta):
        # unpack data
        N = len(y)

        # evaluate log of likelihood fxn
        lnL = -beta/2 * np.sum((y - self.model(xi, X))**2) + N/2*np.log(beta) - N/2 * np.log(2*np.pi)

        return np.exp(lnL)

    def MCMC(self, data, K):
        # metropolis hastings algorithm to generate posterior distributions
        X, y = data

        def prior(xi, a):
            return mvn(self.mu0, a*self.Sigma).pdf(xi)

        def pi(xi, b, a):
            return self.likelihood(xi, X, y, b)*prior(xi, a)

        xi, b, a, p = self.proposal_sample(self.mu0, self.b, self.a, self.p)
        Xi = [xi]
        B = [b]
        A = [a]
        P = [p]

        # using a Gaussian centered at xi(k) as proposal density for now...
        for i in range(K):
            # sample from proposal density
            xi, b, a, p = self.proposal_sample(Xi[-1], B[-1], A[-1], P[-1])
            # calculate acceptance ratio
            r = min(1, pi(xi, b, a)*self.proposal_pdf(Xi[-1], xi, p) / (pi(Xi[-1], B[-1], A[-1])*self.proposal_pdf(xi, Xi[-1], P[-1])))
            # simulate u from Uniform
            u = np.random.uniform(0, 1)
            # evaluate acceptance criteria
            if r >= u:
                Xi.append(xi)
                B.append(b)
                A.append(a)
                P.append(p)
            else:
                Xi.append(Xi[-1])
                B.append(B[-1])
                A.append(A[-1])
                P.append(P[-1])

        XI = np.zeros([K, len(xi)])
        for i in range(K):
            XI[i, :] = Xi[i]

        self.XI = XI

        return np.mean(XI, 0), np.std(XI, 0), B, A, P

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
 #mu = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(Y_train)
mu = np.zeros(NF)
Sigma = np.eye(NF)*.05 #np.var(mu)
prior = [mu, Sigma]

# instantiate MC class object
mc = MC(model, prior, b=1, a=1)

# fit model to data using K samples from the proposal distribution
w, wsd, b, a, p = mc.fit([X_train, Y_train], K=5000)

Y_pred, err = mc.predict(X_test)

#%%
slope, intercept, r_value, p_value, std_err = linregress(Y_test, Y_pred)
plt.errorbar(Y_test, Y_pred, capsize=3, linestyle='none', marker='o', yerr = err, label='R = {:.3f}'.format(r_value))
plt.legend()
plt.show()
#%%
mse = (Y_test - Y_pred)**2
plt.scatter(mse, abs(err))
