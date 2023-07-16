import numpy
import Classifiers
import const


####################################################
# DATASET PREPARATION
####################################################

def split_dataset(D: numpy.ndarray, L: numpy.ndarray, perc: int =0.5, seed: int =0):
    nTrain = int(D.shape[1]*perc)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


def z_normalization(D: numpy.ndarray):
    _, DC = center_dataset(D)
    return DC / vcol(D.std(1))


def shuffle(D: numpy.ndarray, L: numpy.ndarray, seed: int = 0):
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    return (D[:,idx], L[idx])


def kfold(classifier_name: str, D: numpy.ndarray, L: numpy.ndarray, k: int =5, pca: int=0, **kwargs):
    DTR_splits = numpy.array_split(D, k, axis=1)
    LTR_splits = numpy.array_split(L, k)
    DVAL = []
    scores = []

    classifier = getattr(getattr(Classifiers, classifier_name), classifier_name)(**kwargs['init'])

    for i in range(k):
        if i == 0:
            DTR = numpy.hstack(DTR_splits[i+1:])
            LTR = numpy.hstack(LTR_splits[i+1:])
        elif i == k-1:
            DTR = numpy.hstack(DTR_splits[0:i])
            LTR = numpy.hstack(LTR_splits[0:i])
        else:
            DTR = numpy.hstack([numpy.hstack(DTR_splits[0:i]), numpy.hstack(DTR_splits[i+1:])])
            LTR = numpy.hstack([numpy.hstack(LTR_splits[0:i]), numpy.hstack(LTR_splits[i+1:])])

        if pca:
            DTR, P, _ = PCA(DTR, pca)
            DVAL = numpy.dot(P.T, DTR_splits[i])
        else:
            DVAL = DTR_splits[i]

        classifier.train(DTR, LTR)
        scores.extend(classifier.compute_scores(DVAL).tolist())
    return scores


####################################################
# ARRAY TRANSFORMATIONS
####################################################

def mcol(arr_1d: numpy.ndarray) -> numpy.ndarray:
    """ This function returns a column vector with shape (n,1)
        where n is the length of the 1-D array given as parameter
    """
    return arr_1d.reshape((arr_1d.size, 1))


def mrow(arr_1d: numpy.ndarray) -> numpy.ndarray:
    """ This function returns a row vector with shape (1,n)
        where n is the length of the 1-D array given as parameter
    """
    return arr_1d.reshape((1, arr_1d.size))


def vcol(row_arr: numpy.ndarray) -> numpy.ndarray:
    """ This function reshapes a row vector row_arr with shape (1,n)
        into a column vector with shape (n,1)
    """
    return row_arr.reshape((row_arr.shape[0], 1))


def vrow(col_arr: numpy.ndarray) -> numpy.ndarray:
    """ This function reshapes a column vector col_arr with
        shape (n,1) into a row vector with shape (1,n)
    """
    return col_arr.reshape((1, col_arr.shape[0]))


####################################################
# DIMENSIONALITY REDUCTION
####################################################

def PCA(D: numpy.ndarray, m: int) -> numpy.ndarray:
    """Compute Principal Component Analysis on array D 
        with dimensionality m through the decomposition
        of the covariance matrix C.
        Returns DP, the projected dataset on P
    """
    _, DC = center_dataset(D)
    C = numpy.dot(DC, DC.T) / DC.shape[1] 
    w, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, D)
    frac_var = w[::-1][0:m].sum() / w.sum()
    return DP, P, frac_var


def PCA_SVD(D: numpy.ndarray, m: int) -> numpy.ndarray:
    """Compute Principal Component Analysis on array D 
        with dimensionality m through the SVD decomposition
        of the covariance matrix C and returns the basis P
        and the projected dataset on P
    """
    _, DC = center_dataset(D)
    C = numpy.dot(DC, DC.T) / DC.shape[1] 
    U, _, _ = numpy.linalg.svd(C)
    P = U[:, 0:m]
    DP = numpy.dot(P.T, D)    
    return P, DP


def center_dataset(D: numpy.ndarray) -> tuple:
    """ This function computes the mean of the rows of a
        2d array D, then subtracts it to D.
        Returns the mean and the centered dataset
    """
    m = compute_mean(D,1)
    return m, D - m


def compute_mean(D: numpy.ndarray, axis=1) -> numpy.ndarray:
    return vcol(D.mean(axis))


def covariance_matrix(D: numpy.ndarray) -> numpy.ndarray:
    """ This function computes the covariance matrix of a 
        given dataset D by centering it and then applying 
        the formula C = 1/N * D_C * D_C^T
    """
    _, DC = center_dataset(D)
    C = numpy.dot(DC, DC.T) / DC.shape[1] 
    return C


def compute_SB_SW(D: numpy.ndarray, L: numpy.ndarray) -> tuple:
    """ This function serves as a tool for between class covariance matrix SB
        and within class covariance matrix SW calculation.
        SB is calculated using the formula as is, while SW is calculated the 
        covariance matrix for each class set.
    """
    SB = 0
    SW = 0
    mu = compute_mean(D,1)

    for i in range(L.max()+1):
        mu_c_mu = compute_mean(D[:,L==i], 1) - mu
        SB += sum(L==i) * numpy.dot(mu_c_mu, mu_c_mu.T) 
        SW += sum(L==i) * covariance_matrix(D[:,L==i])

    SB = SB / D.shape[1]
    SW = SW / D.shape[1] 
    return (SB, SW)


def LDA(D: numpy.ndarray, L: numpy.ndarray, m: int, sdp=False) -> numpy.ndarray:
    SB, SW = compute_SB_SW(D,L)
    if sdp:
        return LDA_SDP(SB, SW, m)
    else:
        return LDA_JD(SB, SW, m)


def LDA_SDP(SB: numpy.ndarray, SW: numpy.ndarray, m: int) -> numpy.ndarray:
    from scipy import linalg
    _, U = linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    return W


def LDA_JD(SB: numpy.ndarray, SW: numpy.ndarray, m: int) -> numpy.ndarray:
    U, s, _ = numpy.linalg.svd(SW)
    P1 = numpy.dot(U * vrow(1.0/(s**0.5)), U.T)
    SBT = numpy.dot(P1, numpy.dot(SB, P1.T))
    U, _, _ = numpy.linalg.svd(SBT)
    P2 = U[:, 0:m]
    W = numpy.dot(P1.T, P2)
    return W


####################################################
# PROBABILITY DISTRIBUTION
####################################################

def logpdf_GAU_ND(X: numpy.ndarray, mu: numpy.ndarray, C: numpy. ndarray) -> numpy.ndarray:
    """ Compute the log-multivariate gaussian probability density function
        for a given matrix X with shape (N, M) given the following parameters:
            * mean mu with shape (M, 1)
            * covariance matrix C with shape (M, M)
        Returns a 1-D array of log-densities Y with shape (N, 1)
    """
    Y = []
    M = X.shape[0]
    cnst = -0.5 * M * numpy.log(2*numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    XC = X - mu
    for i in range(X.shape[1]):
        exp_i = numpy.dot(numpy.dot(XC[:,i].T,numpy.linalg.inv(C)),XC[:,i])
        Y.append(cnst - 0.5*logdet - 0.5*exp_i)
    return numpy.array(Y).ravel()


def logpdf_GAU_ND_fast(X: numpy.ndarray, mu: numpy.ndarray, C: numpy.ndarray) -> numpy.ndarray:
    """ Fast computation for the log-multivariate gaussian probability density function
        for a given matrix X with shape (N, M) given the following parameters:
            * mean mu with shape (M, 1)
            * covariance matrix C with shape (M, M)

        Returns a 1-D array of log-densities Y with shape (N, 1)
        
        When estimating GMMs with many components, the model may overfit: too many
        components may bring some components to be (soft) assigned to a single point,
        resulting in a variance -> 0, thus in a singular covariance matrix.
        To prevent errors in computation, numpy.linalg.pinv is used.
    """
    M = X.shape[0]
    XC = X - mu
    cnst = -0.5 * M * numpy.log(2*numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.pinv(C)
    v = (XC*numpy.dot(L,XC)).sum(0)
    return cnst - 0.5*logdet - 0.5*v


def loglikelihood(X: numpy.ndarray, m_ML: numpy.ndarray, C_ML: numpy.ndarray) -> float:
    """ Given an array X, returns the log-likelihood of the estimates of 
        its mean m_ML and its covariance matrix C_ML
    """
    return numpy.sum(logpdf_GAU_ND(X, m_ML, C_ML))

