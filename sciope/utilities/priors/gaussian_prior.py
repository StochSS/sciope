from sciope.utilities.priors.prior_base import PriorBase
from toolz import partition_all
import numpy as np
import scipy

######### OBS ######
# Gausssin class taken from:
# @inproceedings{NIPS2016_6aca9700,
# author = {Papamakarios, George and Murray, Iain},
# booktitle = {Advances in Neural Information Processing Systems},
# editor = {D. Lee and M. Sugiyama and U. Luxburg and I. Guyon and R. Garnett},
# pages = {},
# publisher = {Curran Associates, Inc.},
# title = {Fast \epsilon -free Inference of Simulation Models with Bayesian Conditional Density Estimation},
# url = {https://proceedings.neurips.cc/paper/2016/file/6aca97005c68f1206823815f66102863-Paper.pdf},
# volume = {29},
# year = {2016}



class GaussianPrior(PriorBase):
    def __init__(self, m=None, U=None, S=None, P=None, Pm=None):
        if m is not None:
            m = np.asarray(m)
            self.m = m
            self.ndim = m.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.Pm = np.dot(self.P, m)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.Pm = np.dot(self.P, m)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')

        elif Pm is not None:
            Pm = np.asarray(Pm)
            self.Pm = Pm
            self.ndim = Pm.size

            if P is not None:
                P = np.asarray(P)
                L = np.linalg.cholesky(P)
                self.P = P
                self.C = np.linalg.inv(L)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(L)))

            elif U is not None:
                U = np.asarray(U)
                self.P = np.dot(U.T, U)
                self.C = np.linalg.inv(U.T)
                self.S = np.dot(self.C.T, self.C)
                self.m = np.linalg.solve(self.P, Pm)
                self.logdetP = 2.0 * np.sum(np.log(np.diagonal(U)))

            elif S is not None:
                S = np.asarray(S)
                self.P = np.linalg.inv(S)
                self.C = np.linalg.cholesky(S).T
                self.S = S
                self.m = np.dot(S, Pm)
                self.logdetP = -2.0 * np.sum(np.log(np.diagonal(self.C)))

            else:
                raise ValueError('Precision information missing.')

        else:
            raise ValueError('Mean information missing.')
    def draw(self, n=1, chunk_size=1):
        """Returns independent samples from the gaussian."""

        z = np.random.randn(n, self.ndim)
        samples = np.dot(z, self.C) + self.m
        samples = list(partition_all(chunk_size, samples))

        return samples
    
    def pdf(self, x, ii=None, log=True):
        """
        Evaluates the gaussian pdf.
        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """

        if ii is None:
            xm = x - self.m
            lp = -np.sum(np.dot(xm, self.P) * xm, axis=1)
            lp += self.logdetP - self.ndim * np.log(2.0 * np.pi)
            lp *= 0.5

        else:
            m = self.m[ii]
            S = self.S[ii][:, ii]
            lp = scipy.stats.multivariate_normal.logpdf(x, m, S)
            lp = np.array([lp]) if x.shape[0] == 1 else lp

        res = lp if log else np.exp(lp)
        return res
    
    def __mul__(self, other):
        """Multiply with another gaussian."""

        assert isinstance(other, GaussianPrior)

        P = self.P + other.P
        Pm = self.Pm + other.Pm

        return Gaussian(P=P, Pm=Pm)

    def __imul__(self, other):
        """Incrementally multiply with another gaussian."""

        assert isinstance(other, GaussianPrior)

        res = self * other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __div__(self, other):
        """Divide by another gaussian. Note that the resulting gaussian might be improper."""

        assert isinstance(other, GaussianPrior)

        P = self.P - other.P
        Pm = self.Pm - other.Pm

        return GaussianPrior(P=P, Pm=Pm)

    def __idiv__(self, other):
        """Incrementally divide by another gaussian. Note that the resulting gaussian might be improper."""

        assert isinstance(other, GaussianPrior)

        res = self / other

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def __pow__(self, power, modulo=None):
        """Raise gaussian to a power and get another gaussian."""

        P = power * self.P
        Pm = power * self.Pm

        return GaussianPrior(P=P, Pm=Pm)

    def __ipow__(self, power):
        """Incrementally raise gaussian to a power."""

        res = self ** power

        self.m = res.m
        self.P = res.P
        self.C = res.C
        self.S = res.S
        self.Pm = res.Pm
        self.logdetP = res.logdetP

        return res

    def kl(self, other):
        """Calculates the kl divergence from this to another gaussian, i.e. KL(this | other)."""

        assert isinstance(other, GaussianPrior)
        assert self.ndim == other.ndim

        t1 = np.sum(other.P * self.S)

        m = other.m - self.m
        t2 = np.dot(m, np.dot(other.P, m))

        t3 = self.logdetP - other.logdetP

        t = 0.5 * (t1 + t2 + t3 - self.ndim)

        return t