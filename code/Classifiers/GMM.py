import numpy
import mlpr
import scipy.special


def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for g in range(len(gmm)):
        w, mu, C = gmm[g]
        S[g,:] = mlpr.logpdf_GAU_ND_fast(X,mu,C) + numpy.log(w)
    logdens = mlpr.vrow(scipy.special.logsumexp(S, axis=0))
    return S, logdens


def GMM_EM_full(X, gmm, delta_error, psi=1e-2, thresholding=False):
    G, N = len(gmm), X.shape[1]
    ll_new, ll_old = None, None

    while ll_old is None or ll_new - ll_old > delta_error:
        ll_old = ll_new
        SJ, SM = logpdf_GMM(X,gmm)
        ll_new = SM.sum() / N
        P = numpy.exp(SJ-SM)
        gmm_new = []

        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (mlpr.mrow(gamma)*X).sum(axis=1)
            S = numpy.dot(X, (mlpr.mrow(gamma)*X).T)
            w = Z/N
            mu = mlpr.mcol(F/Z)
            sigma = S/Z - numpy.dot(mu,mu.T)

            if thresholding:
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, mlpr.mcol(s)*U.T)

            gmm_new.append((w,mu,sigma))
        gmm = gmm_new
    return gmm


def GMM_EM_diagonal(X, gmm, delta_error, psi=1e-2, thresholding=False):
    G, N = len(gmm), X.shape[1]
    ll_new, ll_old = None, None

    while ll_old is None or ll_new - ll_old > delta_error:
        ll_old = ll_new
        SJ, SM = logpdf_GMM(X, gmm)
        ll_new = SM.sum() / N
        P = numpy.exp(SJ-SM)
        gmm_new = []

        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (mlpr.mrow(gamma)*X).sum(axis=1)
            S = numpy.dot(X, (mlpr.mrow(gamma)*X).T)
            w = Z/N
            mu = mlpr.mcol(F/Z)
            sigma = (S/Z - numpy.dot(mu,mu.T)) * numpy.eye(S.shape[0])

            if thresholding:
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, mlpr.mcol(s)*U.T) * numpy.eye(S.shape[0])

            gmm_new.append((w,mu,sigma))
        gmm = gmm_new
    return gmm


def GMM_EM_tied(X, gmm, delta_error, psi=1e-2, thresholding=False):
    G, N = len(gmm), X.shape[1]
    ll_new, ll_old = None, None
    
    while ll_old is None or ll_new - ll_old > delta_error:
        ll_old = ll_new
        SJ, SM = logpdf_GMM(X, gmm)
        ll_new = SM.sum() / N
        P = numpy.exp(SJ-SM)
        gmm_new = []
        sigma_tied = numpy.zeros((X.shape[0], X.shape[0]))
        
        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (mlpr.mrow(gamma)*X).sum(axis=1)
            S = numpy.dot(X, (mlpr.mrow(gamma)*X).T)
            w = Z/N
            mu = mlpr.mcol(F/Z)
            sigma = (S/Z - numpy.dot(mu,mu.T))
            sigma_tied += Z*sigma
            gmm_new.append((w,mu))
        
        sigma_tied /= N

        if thresholding:
            U, s, _ = numpy.linalg.svd(sigma_tied)
            s[s<psi] = psi
            sigma_tied = numpy.dot(U, mlpr.mcol(s)*U.T)

        gmm = [(w, mu, sigma_tied) for w, mu in gmm_new]
    return gmm


def GMM_EM_tied_diagonal(X, gmm, delta_error, psi=1e-2, thresholding=False):
    G, N = len(gmm), X.shape[1]
    ll_new, ll_old = None, None
    
    while ll_old is None or ll_new - ll_old > delta_error:
        ll_old = ll_new
        SJ, SM = logpdf_GMM(X, gmm)
        ll_new = SM.sum() / N
        P = numpy.exp(SJ-SM)
        sigma_tied = numpy.zeros((X.shape[0], X.shape[0]))
        gmm_new = []
        
        for g in range(G):
            gamma = P[g,:]
            Z = gamma.sum()
            F = (mlpr.mrow(gamma)*X).sum(axis=1)
            S = numpy.dot(X, (mlpr.mrow(gamma)*X).T)
            w = Z/N
            mu = mlpr.mcol(F/Z)
            sigma = (S/Z - numpy.dot(mu,mu.T))
            sigma_tied += Z*sigma
            gmm_new.append((w,mu))

        sigma_tied = (sigma_tied/N) * numpy.eye(sigma_tied.shape[0])

        if thresholding:
            U, s, _ = numpy.linalg.svd(sigma_tied)
            s[s<psi] = psi
            sigma_tied = numpy.dot(U, mlpr.mcol(s)*U.T) * numpy.eye(sigma_tied.shape[0])

        gmm = [(w, mu, sigma_tied) for w, mu in gmm_new]
    return gmm


def GMM_EM(X, gmm, delta_error, gmm_type='full', thresholding=False):
    match gmm_type:
        case 'tied':
            return GMM_EM_tied(X, gmm, delta_error, thresholding)
        case 'diagonal':
            return GMM_EM_diagonal(X, gmm, delta_error, thresholding)
        case 'tied_diagonal':
            return GMM_EM_tied_diagonal(X, gmm, delta_error, thresholding)
        case _:
            return GMM_EM_full(X, gmm, delta_error, thresholding)


def LBG(X, G, alpha=0.1, psi=1e-2, delta_error=1e-6, gmm_type='full', thresholding=False):
    sigma = mlpr.covariance_matrix(X)
    if thresholding:
        U, s, _ = numpy.linalg.svd(X)
        s[s<psi] = psi
        sigma = numpy.dot(U, mlpr.mcol(s)*U.T)
    
    gmm = [(1.0, mlpr.compute_mean(X), sigma)]

    while len(gmm) <= G:
        new_gmm = []

        for g in gmm:
            w, mu, sigma = g
            U, s, _ = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            new_gmm.extend([(w/2, mu-d, sigma), (w/2, mu+d, sigma)])
        gmm = GMM_EM(X, new_gmm, delta_error, gmm_type, thresholding)

    return gmm
    

class GMM():

    def __init__(self, gmm_type='full', TK=1, NTK=2, alpha=0.1, psi=0.01, delta_error=1e-6, thresholding=False):
        self.gmm_type = gmm_type
        self.TK = TK
        self.NTK = NTK
        self.alpha = alpha
        self.psi = psi
        self.delta_error = delta_error
        self.thresholding = thresholding

    def train(self, DTR, LTR):
        self.gmm0 = LBG(DTR[:, LTR==0], self.NTK, self.alpha, self.psi, self.delta_error, self.gmm_type, self.thresholding)
        self.gmm1 = LBG(DTR[:, LTR==1], self.TK, self.alpha, self.psi, self.delta_error, self.gmm_type, self.thresholding)


    def compute_scores(self, DTE: numpy.ndarray):
        ll0 = logpdf_GMM(DTE, self.gmm0)[1]
        ll1 = logpdf_GMM(DTE, self.gmm1)[1]
        return (ll1-ll0).ravel()