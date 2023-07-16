import numpy
import mlpr
from Classifiers.Classifier import Classifier
from scipy.special import logsumexp 


####################################################
# TIED COVARIANCE NAIVE BAYES GAUSSIAN CLASSIFIER
####################################################

class TiedNaiveBayesMVG(Classifier):
    def __init__(self):
        super().__init__()


    def train(self, DTR: numpy.ndarray, LTR: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        mus = []
        sigmas = []
        
        for i in range(2):
            mus.append(mlpr.compute_mean(DTR[:, LTR == i]))
            sigmas.append(mlpr.covariance_matrix(DTR[:,LTR==i]) * DTR[:, LTR==i].shape[1])
        self.mu = numpy.array(mus)
        self.C = numpy.array(sigmas).sum(axis=0) / DTR.shape[1]
        self.C *= numpy.eye(self.C.shape[1])
        self.priors = mlpr.vcol(numpy.array([DTR[:, LTR==1].shape[1]/DTR.shape[1], DTR[:, LTR==0].shape[1]/DTR.shape[1]]))


    def compute_scores(self, DTE: numpy.ndarray) -> numpy.ndarray:
        S = numpy.zeros((2, DTE.shape[1]))

        for i in range(2):
            S[i] = mlpr.logpdf_GAU_ND_fast(DTE, self.mu[i], self.C)
        
        SJoint = S + numpy.log(self.priors)
        logpost = SJoint - mlpr.vrow(logsumexp(SJoint, axis=0))
        scores = logpost[1] - logpost[0]
        return scores

