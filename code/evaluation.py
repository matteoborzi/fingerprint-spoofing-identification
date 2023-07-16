import Classifiers, const, utils, mlpr, plot
from prettytable import PrettyTable
import model_evaluation as meval
from itertools import product
import numpy


def train_for_evaluation(model: str, DTR: numpy.ndarray, LTR: numpy.ndarray, **kwargs):
    classifier = getattr(getattr(Classifiers, model), model)(**kwargs['init'])
    classifier.train(DTR, LTR)
    return classifier


def evaluate_gaussian_models(DTR, LTR, DTE, LTE):
    table = PrettyTable(['Model', 'PCA'] + [f'minDCF{wp}' for wp in const.APPS])
    pcas = [0] + [DTR.shape[0]-i for i in range(4) if DTR.shape[0]-i > 0]
    dcfs = []

    utils.log_test('MVG: ')
    for v,p in product(const.generatives, pcas):
        if p:
            DTR, P, _ = mlpr.PCA(DTR, p)
            DTE = numpy.dot(P.T, DTE)

        classifier = train_for_evaluation(v['model_name'], DTR, LTR, init=v['init'])
        scores = classifier.compute_scores(DTE)
        metrics = meval.apply_metric(scores, LTE, meval.minDCF, const.APPS)
        dcfs.append(metrics)
        table.add_row([v['desc'], '-' if p==0 else str(p)] + ['%.3f' % m for m in metrics])
        utils.log_test('x')
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}EVAL_generatives')
    utils.save_table(table, filename=f'{utils.test}EVAL_generatives')


def evaluate_linear_logreg_models(DTR, LTR, DTE, LTE, pca=0, prefix = ""):
    hyp = numpy.arange(-4,5,1)
    dcfs = {w['init']['weighted']:[] for w in const.logregs}
    dcfs_val = utils.json_load(filename=f'{prefix}dcfs_linear_lr{"_pca" if pca else ""}')

    if pca:
        utils.log_test('Linear LR (PCA): ')
        DTR, P, _ = mlpr.PCA(DTR, pca)
        DTE = numpy.dot(P.T, DTE)
    else:
        utils.log_test('Linear LR: ')

    for v in const.logregs:
        classifier = train_for_evaluation(v['model_name'], DTR, LTR, init=v['init'])
        scores = classifier.compute_scores(DTE)
        metrics = meval.apply_metric(scores, LTE, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['weighted']
        dcfs[w].append(metrics)
    utils.log_test('\n')
    utils.json_dump(dcfs, filename=f'EVAL_{utils.test}{prefix}dcfs_linear_lr')

    for w in dcfs.keys():
        test_values = numpy.array(dcfs[w]).T
        val_values = numpy.array(dcfs_val[str(w)]).T
        plot.plot_evaluation(
            hyp, test_values,
            label=f'{"Unw" if str(w) == "0" else "W"}eighted LogReg',
            savefig=f'{utils.test}EVAL_{prefix}dcf_linear_lr_{"unweighted" if w==0  else "weighted"}{"_pca" if pca else ""}',
            xlabel='log(λ)',
            ylabel='minDCF',
            plot_val= val_values
        )


def evaluate_quadratic_logreg_models(DTR, LTR, DTE, LTE, pca=0, prefix = ""):
    hyp = numpy.arange(-4,5,1)
    dcfs = {w['init']['weighted']:[] for w in const.qlogregs}
    dcfs_val = utils.json_load(filename=f'{prefix}dcfs_quadratic_lr{"_pca" if pca else ""}')

    if pca:
        utils.log_test('Q-LogReg (PCA): ')
        DTR, P, _ = mlpr.PCA(DTR, pca)
        DTE = numpy.dot(P.T, DTE)
    else:
        utils.log_test('Q-LogReg: ')

    for v in const.qlogregs:
        classifier = train_for_evaluation(v['model_name'], DTR, LTR, init=v['init'])
        scores = classifier.compute_scores(DTE)
        metrics = meval.apply_metric(scores, LTE, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['weighted']
        dcfs[w].append(metrics)
    utils.log_test('\n')
    
    utils.json_dump(dcfs, filename=f'EVAL_{utils.test}{prefix}dcfs_quadratic_lr')
    
    for w in dcfs.keys():
        test_values = numpy.array(dcfs[w]).T
        val_values = numpy.array(dcfs_val[str(w)]).T
        plot.plot_evaluation(
            hyp, test_values,
            label=f'Q-LogReg',
            savefig=f'{utils.test}EVAL_{prefix}dcf_quadratic_lr_{"unweighted" if w==0  else "weighted"}{"_pca" if pca else ""}',
            xlabel='log(λ)',
            ylabel='minDCF',
            plot_val= val_values
        )


def evaluate_linear_SVM(DTR, LTR, DTE, LTE, K=10, pca=0, prefix=""):
    table = PrettyTable(['Model', 'PCA', 'K', 'C'] + [f'minDCF{wp}' for wp in const.APPS])
    Cs = numpy.arange(-2,7)

    dcfs = {K:[]}
    dcfs_val = utils.json_load(filename=f'{prefix}dcfs_linear_SVMs{"_pca" if pca else ""}')

    if pca:
        utils.log_test('Linear SVM (PCA): ')
        DTR, P, _ = mlpr.PCA(DTR, pca)
        DTE = numpy.dot(P.T, DTE)
    else:
        utils.log_test('Linear SVM: ')

    for v in const.linearsvms:
        w = v['init']['K']
        if w != K:
            continue 
        classifier = train_for_evaluation(v['model_name'], DTR, LTR, init=v['init'])
        scores = classifier.compute_scores(DTE)
        metrics = meval.apply_metric(scores, LTE, meval.minDCF, const.APPS)
        utils.log_test('x')
        dcfs[w].append(metrics)
        table.add_row([v['desc'], pca if pca else '-', v['init']['K'], '%.0e' % v['init']['C']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'EVAL_{utils.test}{prefix}dcfs_linear_SVMs{"_pca" if pca else ""}')
    utils.save_table(table, filename=f'EVAL_{utils.test}{prefix}dcfs_linear_SVMs')

    for k in dcfs.keys():
        values = numpy.array(dcfs[k]).T
        val_values = numpy.array(dcfs_val[str(k)]).T
        plot.plot_evaluation(
            Cs, values,
            label=f'Linear SVM (K={k})',
            savefig=f'EVAL_{utils.test}{prefix}dcf_linear_SVM',
            xlabel='log(C)',
            ylabel='minDCF',
            plot_val=val_values
        )


def evaluate_poly2_SVM(DTR, LTR, DTE, LTE, K=10, pca=0, prefix=""):
    table = PrettyTable(['Model', 'PCA', 'K', 'C'] + [f'minDCF{wp}' for wp in const.APPS])
    Cs = numpy.arange(-2,7)

    dcfs = {K:[]}
    dcfs_val = utils.json_load(filename=f'{prefix}dcfs_poly2_SVMs{"_pca" if pca else ""}')

    if pca:
        utils.log_test('Poly(2) SVM (PCA): ')
        DTR, P, _ = mlpr.PCA(DTR, pca)
        DTE = numpy.dot(P.T, DTE)
    else:
        utils.log_test('Poly(2) SVM: ')

    for v in const.poly2svms:
        w = v['init']['K']
        if w != K:
            continue
        classifier = train_for_evaluation(v['model_name'], DTR, LTR, init=v['init'])
        scores = classifier.compute_scores(DTE)
        metrics = meval.apply_metric(scores, LTE, meval.minDCF, const.APPS)
        utils.log_test('x')
        dcfs[w].append(metrics)
        table.add_row([v['desc'], '-', v['init']['K'], '%.0e' % v['init']['C']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'EVAL_{utils.test}{prefix}dcfs_poly2_SVMs')
    utils.save_table(table, filename=f'EVAL_{utils.test}{prefix}dcfs_poly2_SVMs')

    for k in dcfs.keys():
        values = numpy.array(dcfs[k]).T
        val_values = numpy.array(dcfs_val[str(k)]).T
        plot.plot_evaluation(
            Cs, values,
            label=f'Poly(2) SVM (K={k})',
            savefig=f'EVAL_{utils.test}{prefix}dcf_poly2_SVM',
            xlabel='log(C)',
            ylabel='minDCF',
            plot_val=val_values
        )


def evaluate_rbf_SVM(DTR, LTR, DTE, LTE, gamma=0.002, pca=0, prefix=""):
    table = PrettyTable(['Model', 'gamma', 'C', 'K'] + [f'minDCF{wp}' for wp in const.APPS])

    gammas = list({m['gamma'] for m in const.rbfsvms})
    Cs = numpy.arange(-2,6,1)

    dcfs = {g:[] for g in gammas}
    dcfs_val = utils.json_load(filename=f'{prefix}dcfs_rbf_SVMs{"_pca" if pca else ""}')
    try:
        dcfs = utils.json_load(f'EVAL_{prefix}dcfs_rbf_SVMs{"_pca" if pca else ""}')
    except:
        if pca:
            utils.log_test('RBF (PCA): ')
            DTR, P, _ = mlpr.PCA(DTR, pca)
            DTE = numpy.dot(P.T, DTE)
        else:
            utils.log_test('RBF SVM: ')

        for v in const.rbfsvms:
            g = v['gamma']
            if g != gamma:
                continue
            classifier = train_for_evaluation(v['model_name'], DTR, LTR, init=v['init'])
            scores = classifier.compute_scores(DTE)
            metrics = meval.apply_metric(scores, LTE, meval.minDCF, const.APPS)
            utils.log_test('x', end='', flush=True)
            dcfs[g].append(metrics)
            table.add_row([v['desc'], v['gamma'], v['init']['C'], v['init']['K']] + ['%.3f' % m for m in  metrics])
        utils.log_test('\n')

        utils.json_dump(dcfs, filename=f'EVAL_{utils.test}{prefix}dcfs_rbf_svms')
        utils.save_table(table, filename=f'EVAL_{utils.test}{prefix}dcfs_rbf_svms')

    val_values = numpy.array(dcfs_val[str(gamma)]).T
    values = numpy.array(dcfs[str(gamma)]).T

    plot.plot_evaluation(
        Cs, values,
        label=f'RBF SVM - γ={gamma}',
        savefig=f'EVAL_{utils.test}{prefix}dcf_rbf_SVM',
        plot_val=val_values
    )


def evaluate_gmm(DTR, LTR, DTE, LTE, gmm_type='full', pca=0, prefix=''):
    table = PrettyTable(['Model', 'K-T', 'K-NT'] + [f'minDCF{wp}' for wp in const.APPS])
    dcfs = {k:[] for k in const.TK if k <= 16}
    dcfs_val = utils.json_load(f'{prefix}dcfs_gmm_{gmm_type}{f"_pca{pca}" if pca else ""}')

    try:
        dcfs = utils.json_load(f'EVAL_{prefix}dcfs_gmm_{gmm_type}')
    except:
        utils.log_test(f'{gmm_type.capitalize().replace("_"," ")} GMM: ')
        for v in const.gmms[gmm_type]:
            classifier = train_for_evaluation(v['model_name'], DTR, LTR, init=v['init'], pca=pca)
            scores = classifier.compute_scores(DTE)
            metrics = meval.apply_metric(scores, LTE, meval.minDCF, const.APPS)
            utils.log_test('x')
            tk = v['init']['TK']
            dcfs[tk].append(metrics)
            table.add_row([v['desc'], v['init']['TK'], v['init']['NTK']] + ['%.3f' % m for m in metrics])
        utils.log_test('\n')

        utils.json_dump(dcfs, filename=f'EVAL_{utils.test}{prefix}dcfs_gmm_{gmm_type}{f"_pca{pca}" if pca else ""}')
        utils.save_table(table, filename=f'EVAL_{utils.test}{prefix}dcfs_gmm_{gmm_type}{f"_pca{pca}" if pca else ""}')
    plot.plot_gmm_evaluation(dcfs, dcfs_val, savefig=f'EVAL_{utils.test}{prefix}dcf_gmm_{gmm_type}{f"_pca{pca}" if pca else ""}')
