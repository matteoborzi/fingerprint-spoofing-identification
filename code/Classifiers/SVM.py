from . import svm_kernels
import numpy
import mlpr
from scipy.optimize import fmin_l_bfgs_b


class SVM():
    def __init__(self, kernel, C, K, pi, incorporate_bias=False) -> None:        
        self.C = C
        self.K = K
        self.kernel = kernel
        self.pi = pi
        self.incorporate_bias = incorporate_bias


    def __LDual(self, alpha):
        Ha = numpy.dot(self.H, mlpr.vcol(alpha))
        return 0.5 * numpy.dot(mlpr.vrow(alpha),Ha) - alpha.sum(), Ha - 1


    def __JPrimal(self, w):
        S = numpy.dot(mlpr.vrow(w), self.DTR)
        loss = numpy.maximum(numpy.zeros(S.shape), 1 - self.ZTR*S).sum()
        return 0.5 * (numpy.linalg.norm(w) ** 2) + self.C * loss


    def train(self, DTR: numpy.ndarray, LTR: numpy.ndarray):
        n = DTR.shape[1]
        self.DTR = numpy.vstack([DTR, self.K+numpy.zeros((1, n))]) if self.incorporate_bias else DTR
        self.ZTR = LTR * 2.0 - 1.0        
        self.nt = self.DTR[:, LTR == 1].shape[1]
        self.nf = self.DTR.shape[1] - self.nt
        self.H = mlpr.vcol(self.ZTR) * mlpr.vrow(self.ZTR) * self.kernel(self.DTR, self.DTR, self.K)

        x0 = numpy.zeros(n)
        rebalanced_costs = numpy.vstack([self.C*(LTR==1)*(self.pi/self.nt), self.C *(LTR==0)*((1-self.pi)/self.nf)]).sum(0)
        bounds = [(0, c) for c in rebalanced_costs] 
        self.alphastar, _, _ = fmin_l_bfgs_b(self.__LDual, x0=x0, bounds=bounds, factr=1.0)


    def primal_loss(self):
        assert self.kernel == svm_kernels.dot
        wstar = numpy.dot(self.DTR, mlpr.vcol(self.alphastar * self.ZTR))
        return wstar, self.__JPrimal(wstar)


    def compute_scores(self, DTE: numpy.ndarray) -> numpy.ndarray:
        DTE = numpy.vstack([DTE, self.K+numpy.zeros((1, DTE.shape[1]))]) if self.incorporate_bias else DTE
        scores = numpy.sum(numpy.dot(self.alphastar * mlpr.vrow(self.ZTR), self.kernel(self.DTR, DTE, self.K)), axis=0)
        return scores 