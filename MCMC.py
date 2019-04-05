import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt

class MC:

    def __init__(self, pi, prior):
        # pi is the target distribution
        self.pi = pi
        # the prior defines the mean and covariance
        self.mu0, self.Sigma = prior

        def proposal_sample(mu):
            return mvn(mu, self.Sigma).rvs()

        def proposal_pdf(mu1, mu2):
            # return conditional probability of mu1 given mu2
            return mvn(mu2, self.Sigma).pdf(mu1)

        self.proposal_sample = proposal_sample
        self.proposal_pdf = proposal_pdf

    def MCMC(self, K):
        # metropolis hastings algorithm to generate posterior distributions

        Xi = [self.proposal_sample(self.mu0)]
        # using a Gaussian centered at xi(k) as proposal density for now...
        for i in range(K):
            # sample from proposal density
            xi = self.proposal_sample(Xi[-1])
            # calculate acceptance ratio
            r = min(1, self.pi.pdf(xi)*self.proposal_pdf(Xi[-1], xi) / (pi.pdf(Xi[-1])*self.proposal_pdf(xi, Xi[-1])))
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

        return XI

if __name__ == "__main__":
    #%%
    # to prove that the MCMC algorithm works, lets try to estimate a known parameter
    # distribution with two variables

    pi = mvn([1.5, 1.5], [[1, 0.5], [0.5, 1]])

    # for the sake of experiment, let's say that X is hard to directly sample from
    # in future endeavors, this will be the posterior distribution of parameters for
    # a model

    # define the mean and standard deviation of the proposal density
    mu0 = [1, 1]
    Sigma = [[0.5, 0], [0, 0.5]]
    prior = [mu0, Sigma]

    # give it a whirl!
    mc = MC(pi, prior)
    K = 1000
    MCMCparams = mc.MCMC(K)

    plt.scatter(pi.rvs(K)[:, 0], pi.rvs(K)[:, 1], alpha=0.5, label='True')
    plt.scatter(MCMCparams[:, 0], MCMCparams[:, 1], alpha=0.5, label='MCMC')
    plt.legend()
    plt.show()
    #%%
