import numpy


def FNR(cm: numpy.ndarray):
    """Compute the False Negative Ratio as follows:
        FNR = FN / (FN+TP)
    """
    return cm[0,1] / cm[:,1].sum()


def FPR(cm: numpy.ndarray):
    """Compute the False Postive Ratio as follows:
        FPR = FP / (FP+TN)
    """
    return cm[1,0] / cm[:,0].sum()


def compute_effective_prior(application: tuple) -> tuple:
    pi, cfn, cfp = application
    pi_t = (pi*cfn) / (pi*cfn + (1-pi)*cfp)
    return (pi_t, 1, 1)


def compute_confusion_matrix(predicted, actual, num_classes: int):
    cm = numpy.zeros((num_classes, num_classes))
    for pred, truth in zip(predicted, actual):
        cm[pred, truth] += 1
    return cm


def optimal_bayes_decision(llr: numpy.ndarray, truth: numpy.ndarray, working_point: tuple[float, int, int], t_default: float = None):
    pi, Cfn, Cfp = working_point
    t = - numpy.log((pi * Cfn) / ((1-pi)*Cfp)) if t_default is None else t_default
    labels = numpy.array([l > t for l in llr], dtype=numpy.int32)
    return compute_confusion_matrix(labels, truth, 2)


def DCF(llr: numpy.ndarray, truth: numpy.ndarray, application: tuple[float,int,int], t_default: float = None) -> float:
    pi_1, Cfn, Cfp = application
    cm = optimal_bayes_decision(llr, truth, application, t_default)
    return pi_1*Cfn*FNR(cm) + (1-pi_1)*Cfp*FPR(cm)


def normalized_DCF(llr: numpy.ndarray, truth: numpy.ndarray, application: tuple[float,int,int], t_default: float = None) -> float:
    pi_1, Cfn, Cfp = application
    dcf = DCF(llr, truth, application, t_default)
    bdummy = min([pi_1*Cfn, (1-pi_1)*Cfp])
    return dcf / bdummy


def minDCF(llr: numpy.ndarray, truth: numpy.ndarray, application: tuple[float,int,int]) -> float:
    thresholds = numpy.hstack([-numpy.inf, numpy.sort(llr), numpy.inf])
    dcfs = numpy.array([normalized_DCF(llr, truth, application, t_default=t) for t in thresholds])
    return numpy.min(dcfs)


def compute_bayes_error(llr: numpy.ndarray, truth: numpy.ndarray, effPriorLogOdds: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    pis = 1 / (1 + numpy.exp(-effPriorLogOdds))
    eff_wps = [(p, 1, 1) for p in pis]
    dcf = [normalized_DCF(llr, truth, e) for e in eff_wps]
    mindcf = [minDCF(llr, truth, e) for e in eff_wps]
    return dcf, mindcf


def apply_metric(scores: numpy.ndarray, labels: numpy.ndarray, evaluator: callable, apps):
    eval_metrics = []
    for app in apps:
        eval_metrics.append(evaluator(scores, labels, app))
    return eval_metrics

actDCF = normalized_DCF