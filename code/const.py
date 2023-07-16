from itertools import product
from model_evaluation import compute_effective_prior
from Classifiers import svm_kernels
from matplotlib.cm import get_cmap

APP = (0.5,1,10)
APPS = [APP, (0.2,1,10), (0.4,1,10)]

TK = [1,2,4]
NTK = [2**i for i in range(6)]

cmap = get_cmap('tab10')

generatives = [
    {
        'desc': 'MVG',
        'model_name': 'MVG',
        'init':{},
    }, {
        'desc': 'Tied MVG',
        'model_name': 'TiedMVG',
        'init': {}, 
    }, {
        'desc': 'Naive Bayes MVG',
        'model_name': 'NaiveBayesMVG',
        'init':{}, 
    }, {
        'desc': 'Tied Naive Bayes MVG',
        'model_name': 'TiedNaiveBayesMVG',
        'init':{},
    }
]


logregs = [
    {
        'desc': f'{"Weighted" if w else ""} Linear LR',
        'model_name': 'LogReg',
        'init': {
            'l': 10**l,
            'weighted': w,
            'quadratic':False,
            }
    } for w,l in product([compute_effective_prior(APP)[0]], range(-4,5,1))
]


qlogregs = [
    {
        'desc': f'{"Weighted" if w else ""} Quadratic LogReg',
        'model_name': 'LogReg',
        'init': {
            'l': 10**l,
            'weighted': w,
            'quadratic': True,
            }
    } for w,l in product([compute_effective_prior(APP)[0]], range(-4,5,1))
]


linearsvms = [
    {
        'desc': f'Linear SVM',
        'model_name': 'SVM',
        'init': {
            'kernel': svm_kernels.dot,
            'C': 10**C,
            'K': 10**K,
            'pi': compute_effective_prior(APP)[0],
            'incorporate_bias':True
        }
    } for K,C in product([0, 1], range(-2,7))
]


poly2svms = [
    {
        'desc': f'SVM Poly(2)',
        'model_name': 'SVM',
        'c': c,
        'init': {
            'kernel': svm_kernels.poly(c,2),
            'C': 10**C,
            'K': 10**K,
            'pi': compute_effective_prior(APP)[0],
            'incorporate_bias':False
        }
    } for c,K,C in product([1], [0, 1], range(-2,7))
]


poly3svms = [
    {
        'desc': f'SVM - Poly(3)',
        'model_name': 'SVM',
        'c': c,
        'init': {
            'kernel': svm_kernels.poly(c,3),
            'C': 10**C,
            'K': 10**K,
            'pi': compute_effective_prior(APP)[0],
            'incorporate_bias':False
        }
    } for c,C,K in product([1], range(-1,2), [0, 1])
]


rbfsvms = [
    {
        'desc': 'RBF SVM',
        'model_name': 'SVM',
        'gamma': gamma,
        'init': {
            'kernel': svm_kernels.rbf(gamma),
            'C': 10**C,
            'K': 10**K,
            'pi': compute_effective_prior(APP)[0],
            'incorporate_bias':False
        }
    } for gamma,C,K in product([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1], range(-2,6), [1])
]


gmms = {
    gt: [
        {
            'desc': f'{gt.capitalize().replace("_"," ")} GMM',
            'model_name': 'GMM',
            'init': {
                'gmm_type': gt,
                'TK': t,
                'NTK': nt,
                'thresholding':True,
            }
        } for t, nt in product(TK, NTK)
    ] for gt in ['full','tied','diagonal','tied_diagonal']
}


selected_models = {
    'mvg': {
        'desc': 'Full Covariance MVG',
        'model_name': 'MVG',
        'calibrate': False,
        'pca': 0,
        'init':{},
    },
    'qlogreg': {
        'desc': 'Weighted Q-LogReg',
        'model_name': 'LogReg',
        'calibrate': True,
        'pca': 9,
        'init': {
            'l': 1e-2,
            'weighted': compute_effective_prior(APP)[0],
            'quadratic': True,
            }
    },
    'poly2': {
        'desc': 'Poly(2) SVM',
        'model_name': 'SVM',
        'calibrate': True,
        'pca': 0,
        'init': {
            'kernel': svm_kernels.poly(1,2),
            'C': 1,
            'K': 10,
            'pi': compute_effective_prior(APP)[0],
            'incorporate_bias':False
        }
    },
    'rbf': {
        'desc': 'RBF SVM',
        'model_name': 'SVM',
        'calibrate': True,
        'pca': 0,
        'init': {
            'kernel': svm_kernels.rbf(0.002),
            'C': 10**4,
            'K': 10,
            'pi': compute_effective_prior(APP)[0],
            'incorporate_bias':False
        }
    },
    'gmm': {
        'desc': 'Tied Diagonal GMM',
        'model_name': 'GMM',
        'calibrate': False,
        'pca': 0,
        'init': {
            'gmm_type': 'tied_diagonal',
            'TK': 4,
            'NTK': 4,
            'thresholding':True,
        }
    }
}


calibration_model = {
    'desc': 'Calibration LogReg',
    'model_name': 'LogReg',
    'init': {
        'l': 0,
        'weighted': compute_effective_prior(APP)[0],
        'quadratic':False,
        }
}


best3_models = {
    'mvg': selected_models['mvg'],
    'rbf': selected_models['rbf'],
    'gmm': selected_models['gmm'],
}


best1_model_eval = {
    'gmm': {
        'desc': 'Tied GMM',
        'model_name': 'GMM',
        'pca': 0,
        'init': {
            'gmm_type': 'tied',
            'TK': 4,
            'NTK': 4,
            'thresholding':True,
        }
    }
}