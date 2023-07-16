import mlpr
import numpy
from Classifiers.Classifier import Classifier
from scipy.optimize import fmin_l_bfgs_b


class LogReg(Classifier):
    def __init__(self, l, weighted=0, quadratic=False):
        self.l = l
        self.weighted = weighted
        self.quadratic = quadratic
        if weighted:
            self.pi = weighted

    
    def __logreg_obj(self, v):      
        w, b = mlpr.vcol(v[0:self.M]), v[-1]
        scores = numpy.dot(w.T,self.DTR) + b
        logistic_loss = numpy.logaddexp(0,-self.ZTR*scores).mean()
        loss = 0.5 * self.l * numpy.linalg.norm(w)**2 + logistic_loss
        return loss
    

    def __logreg_obj_weighted(self, v):
        w, b = mlpr.vcol(v[0:self.M]), v[-1]
        regulation = 0.5 * self.l * numpy.linalg.norm(w)**2
        scores = (numpy.dot(w.T,self.DTR) + b).ravel()
        logloss0 = numpy.logaddexp(0,-self.ZTR[self.LTR == 0]*scores[self.LTR == 0]).sum()
        logloss1 = numpy.logaddexp(0,-self.ZTR[self.LTR == 1]*scores[self.LTR == 1]).sum()
        loss = regulation + (self.pi / self.nt) * logloss1 + ((1-self.pi) / self.nf ) * logloss0
        return loss


    def __vec(self, x):
        return numpy.ravel(x, order='F')


    def expand(self, X):
        expanded = []
        for i in range(X.shape[1]):
            expanded.append(numpy.vstack([mlpr.vcol(self.__vec(numpy.dot(X[:,i],X[:,i].T))), mlpr.vcol(X[:,i])]))
        return numpy.hstack(expanded)


    def train(self, DTR: numpy.ndarray, LTR: numpy.ndarray):
        self.DTR = self.expand(DTR) if self.quadratic else DTR
        self.LTR = LTR
        self.ZTR = LTR * 2.0 - 1.0
        self.M = self.DTR.shape[0]
        self.nt = self.DTR[:, self.LTR == 1].shape[1]
        self.nf = self.DTR.shape[1] - self.nt

        x0 = numpy.zeros(self.DTR.shape[0]+1)
        logreg_obj = self.__logreg_obj_weighted if self.weighted else self.__logreg_obj 
        x0pt, _, _= fmin_l_bfgs_b(logreg_obj, x0=x0, approx_grad=True)
        self.w, self.b = x0pt[:-1], x0pt[-1]


    def compute_scores(self, DTE: numpy.ndarray) -> numpy.ndarray:
        if self.quadratic:
            DTE = self.expand(DTE)

        scores = numpy.dot(self.w.T, DTE) + self.b
        return scores 