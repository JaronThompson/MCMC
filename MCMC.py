import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import linregress
from scipy.optimize import fmin
import matplotlib.pyplot as plt

class MCMC:
    '''
    Class for implementing MCMC with Metropolis-Hastings algorithm
    '''

    def __init__(self, model, prior, proposal, *args):
        # define the model (must be function of parameters, input )
        self.model = model
        self.model_args = args
        # the prior defines the mean and covariance of model parameters
        self.prior = prior
        # the proposal pdf is the pdf used to generate the Markov chain
        self.proposal = proposal

    def fit(self, data, K, Kreject=1000, levels=None):
        return self.mcmc(data, K, Kreject)

    def predict(self, X, return_stats=False):
        # define dimensions
        K = self.XI.shape[0]    # number of samples from posterior distribution
        S = X.shape[0]          # len of g(xi) vector (number of estimates)

        # use direct MCS to determine E(y) = int [y(w, x)*p(w)] dw
        H = np.zeros([K, S])
        for i, xi in enumerate(self.XI):
            H[i, :] = self.model(xi, X, self.model_args) + mvn(0, self.var).rvs()

        H_est = np.mean(H, 0)
        H_std = np.std(H, 0)

        if return_stats:
            # calculate mean COV over all samples
            cov_MH = H_std / abs(H_est)

            # calculate correlation factor of lag k
            r = np.zeros([S, int(.05*K)])

            for k in range(int(.05*K)):
                aux = np.zeros(S)
                for l in range(K-k):
                    aux += (H[l, :] - H_est)*(H[l+k, :] - H_est)
                r[:, k] = 1/(K-k)*aux/np.var(H, 0)

            for i in range(r.shape[0]):
                plt.plot(r[i, :], label="sample {}".format(i+1))

            # calculate integrated autocorrelation time for each estimate
            tau = 1 + 2*np.sum(r, 1)  # summing over k

            # calculate the Keff using average integrated autocorrelation time
            Keff = K / tau;

            # calculate cov of the MHMC estimate
            cov_MHMC = cov_MH / np.sqrt(Keff)

            plt.ylabel('p(k)')
            plt.xlabel('k')
            plt.title("MCMC")
            #plt.savefig('figures/mcmc_autocorrelation.png', dpi=300)
            plt.show()

            return H_est, H_std, cov_MHMC, Keff, self.acc_rate

        else:
            return H_est, H_std

    def likelihood(self, xi, X, y):
        # unpack data
        N = len(y)

        # evaluate log of likelihood fxn
        beta = 1/np.mean((y - self.model(xi, X, self.model_args))**2)
        lnL = -beta/2 * np.sum((y - self.model(xi, X, self.model_args))**2) + N/2*np.log(beta) - N/2 * np.log(2*np.pi)

        return np.exp(lnL)

    def mcmc(self, data, K, Kreject):
        # metropolis hastings algorithm to generate posterior distributions
        X, y = data

        # define target distribution function
        def pi(xi):
            return self.likelihood(xi, X, y)*self.prior.pdf(xi)

        xi = self.prior.rvs()
        Xi = [xi]

        # keep track of acceptance rate
        acc = 0

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
                acc += 1
            else:
                Xi.append(Xi[-1])

        # record acceptance rate
        self.acc_rate = acc / K
        print(self.acc_rate)

        # reject first Kreject samples in chain
        XI = np.zeros([K-Kreject, len(xi)])
        for i, j in enumerate(range(Kreject, K)):
            XI[i, :] = Xi[j]

        self.XI = XI
        self.var = np.mean((y - self.model(np.mean(self.XI, 0), X, self.model_args))**2)
        return self.XI

class TMCMC:
    '''
    Class for implementing Transitional MCMC with Metropolis-Hastings algorithm
    '''

    def __init__(self, model, prior, proposal, *args):
        # define the model (must be function of parameters, input )
        self.model = model
        self.model_args = args
        # the prior defines the mean and covariance of model parameters
        self.prior = prior
        # the proposal pdf is the pdf used to generate the Markov chain
        self.proposal = proposal

    def fit(self, data, K, Kreject=1000, levels=10):
        return self.mcmc(data, K, Kreject, levels)

    def predict(self, X, return_stats=False):
        # define dimensions
        K = self.XI.shape[0]    # number of samples from posterior distribution
        S = X.shape[0]          # len of g(xi) vector (number of estimates)

        # use direct MCS to determine E(y) = int [y(w, x)*p(w)] dw
        H = np.zeros([K, S])
        for i, xi in enumerate(self.XI):
            H[i, :] = self.model(xi, X, self.model_args) + mvn(0, self.var).rvs()

        H_est = np.mean(H, 0)
        H_std = np.std(H, 0)

        if return_stats:
            # calculate mean COV over all samples
            cov_MH = H_std / abs(H_est)

            # calculate correlation factor of lag k
            r = np.zeros([S, int(.05*K)])

            for k in range(int(.05*K)):
                aux = np.zeros(S)
                for l in range(K-k):
                    aux += (H[l, :] - H_est)*(H[l+k, :] - H_est)
                r[:, k] = 1/(K-k)*aux/np.var(H, 0)

            for i in range(r.shape[0]):
                plt.plot(r[i, :], label="sample {}".format(i+1))

            # calculate integrated autocorrelation time for each estimate
            tau = 1 + 2*np.sum(r, 1)  # summing over k

            # calculate the Keff using average integrated autocorrelation time
            Keff = K / tau;

            # calculate cov of MHMC estimate
            cov_MHMC = cov_MH / np.sqrt(Keff)

            plt.ylabel('p(k)')
            plt.xlabel('k')
            plt.title("Transitional MCMC")
            #plt.savefig('figures/tmcmc_autocorrelation.png', dpi=300)
            plt.show()

            return H_est, H_std, cov_MHMC, Keff, self.acc_rate

        else:
            return H_est, H_std

    def likelihood(self, xi, X, y):
        # unpack data
        N = len(y)

        # evaluate log of likelihood fxn
        beta = 1 / np.mean((y - self.model(xi, X, self.model_args))**2)
        lnL = -beta/2 * np.sum((y - self.model(xi, X, self.model_args))**2) + N/2*np.log(beta) - N/2 * np.log(2*np.pi)

        return np.exp(lnL)

    def mcmc(self, data, K, Kreject, levels):
        # metropolis hastings algorithm to generate posterior distributions
        X, y = data

        # define target distribution function
        def pi(xi, beta):
            return self.likelihood(xi, X, y)**beta*self.prior.pdf(xi)

        for j in range(1, levels+1):
            print('\nlevel')
            print(j)
            # determine tampering factor
            beta = 10**((j-levels)/5)
            print('beta')
            print(beta)

            # metropolis algorithm
            # initialize chain
            if j == 1:
                Xi = [self.prior.rvs()]
            else:
                Xi = [np.mean(XI, 0)]
            # keep track of acceptance rate
            acc = 0

            for i in range(K):
                # sample from proposal density
                xi = self.proposal.sample(Xi[-1])
                # calculate acceptance ratio
                r = min(1, pi(xi, beta)*self.proposal.pdf(Xi[-1], xi) / (pi(Xi[-1], beta)*self.proposal.pdf(xi, Xi[-1])))
                # simulate u from Uniform
                u = np.random.uniform(0, 1)
                # evaluate acceptance criteria
                if r >= u:
                    Xi.append(xi)
                    acc += 1
                else:
                    Xi.append(Xi[-1])

            # record acceptance rate
            self.acc_rate = acc / K
            print('acceptance rate')
            print(self.acc_rate)

            # acceptance rate of 1 is bad news
            if self.acc_rate == 1:
                break

            # reject first Kreject samples in chain
            XI = np.zeros([K-Kreject, len(xi)])
            for i, j in enumerate(range(Kreject, K)):
                XI[i, :] = Xi[j]

            # update proposal pdf with sampling mean and covariance
            print('mean')
            print(np.mean(XI, 0))
            print('cov')
            print(np.diagonal(np.cov(XI.T)))
            self.proposal = Proposal(np.mean(XI, 0), np.cov(XI.T), local=False)

        self.XI = XI
        self.var = np.mean((y - self.model(np.mean(self.XI, 0), X, self.model_args))**2)
        return self.XI

class MAP:
    '''
    Class for implementing MAP estimates
    '''

    def __init__(self, model, prior, *args):
        # define the model (must be function of parameters, input )
        self.model = model
        self.model_args = args
        # the prior defines the mean and covariance of model parameters
        self.prior = prior

    def loglikelihood(self, xi, X, y):
        # unpack data
        N = len(y)

        # evaluate log of likelihood fxn
        beta = 1 / xi[-1]**2
        lnL = -beta/2 * np.sum((y - self.model(xi, X, self.model_args))**2) + N/2*np.log(beta) - N/2 * np.log(2*np.pi)

        return lnL

    def fit(self, data):
        X, y = data

        def obj(xi, X, y):
            return -(self.loglikelihood(xi, X, y) + np.log(self.prior.pdf(xi)))

        self.xi_map = fmin(obj, self.prior.rvs(), args=(X, y,), xtol=1e-5, maxfun=1e5)
        return self.xi_map

    def predict(self, X_test):
        y = X_test.dot(self.xi_map[:-1].T)
        return y

# define class structure for proposal pdf
class Proposal:

    def __init__(self, mu, Sigma, local=True):
        # set up hyper-parameters for the proposal pdf here
        self.mu = mu
        self.Sigma = Sigma
        self.local = local

    def sample(self, mu):
        if self.local:
            # if local random walk
            s = mvn(mu, self.Sigma, allow_singular=True).rvs()
        else:
            # if global random walk
            s = mvn(self.mu, self.Sigma, allow_singular=True).rvs()
        return s

    def pdf(self, mu1, mu2):
        # return conditional probability of mu1 given mu2
        return mvn(mu2, self.Sigma, allow_singular=True).pdf(mu1)

# define class structure for proposal pdf
class Prior:

    def __init__(self, mu, Sigma, a0=1):
        # set up hyper-parameters for the proposal pdf here
        self.mu = mu
        self.Sigma = Sigma
        self.a0 = a0

    def rvs(self):
        s = mvn(self.mu, self.a0*self.Sigma).rvs()
        return s

    def pdf(self, xi):
        # return pdf of prior
        return mvn(self.mu, self.a0*self.Sigma).pdf(xi)
