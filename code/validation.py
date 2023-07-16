from prettytable import PrettyTable
from itertools import product
import numpy
import mlpr, plot, utils, const
import model_evaluation as meval


def validate_generative_models(D, L):
    table = PrettyTable(['Model', 'PCA'] + [f'minDCF{wp}' for wp in const.APPS])
    pcas = [0] + [D.shape[0]-i for i in range(4) if D.shape[0]-i > 0]
    dcfs = []

    utils.log_test('MVG: ')
    for v,p in product(const.generatives, pcas):
        utils.log_test('x')
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'], pca=p)
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        dcfs.append(metrics)
        table.add_row([v['desc'], '-' if p==0 else str(p)] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}generatives')
    utils.save_table(table, filename=f'{utils.test}generatives')
    


def validate_linear_logreg_models(D, L, prefix = ""):
    table = PrettyTable(['Model', 'PCA', 'Weighted', 'λ'] + [f'minDCF{wp}' for wp in const.APPS])
    hyp = numpy.arange(-4,5,1)
    dcfs = {w['init']['weighted']:[] for w in const.logregs}
    dcfs_pca = {w['init']['weighted']:[] for w in const.logregs}

    utils.log_test('Linear LR: ')
    for v in const.logregs:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'])
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['weighted']
        dcfs[w].append(metrics)
        table.add_row([v['desc'], '-', v['init']['weighted'] if v['init']['weighted'] else '-', '%.0e' % v['init']['l']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')
    
    utils.log_test('Linear LR (PCA): ')
    for v in const.logregs:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'], pca=9)
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['weighted']
        dcfs_pca[w].append(metrics)
        table.add_row([v['desc'], 9, v['init']['weighted'] if v['init']['weighted'] else '-', v['init']['l']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')
    
    utils.json_dump(dcfs, filename=f'{utils.test}{prefix}dcfs_linear_lr')
    utils.json_dump(dcfs_pca, filename=f'{utils.test}{prefix}dcfs_linear_lr_pca')
    utils.save_table(table, filename=f'{utils.test}{prefix}linear_lr')
    
    
    for w in dcfs.keys():
        values = numpy.array(dcfs[w]).T
        values_pca = numpy.array(dcfs_pca[w]).T
        plot.plot_validation(
            hyp, values,
            label=f'{"Unw" if str(w) == "0" else "W"}eighted LogReg',
            savefig=f'{utils.test}{prefix}dcf_linear_lr_{"unweighted" if w==0  else "weighted"}',
            xlabel='log(λ)',
            ylabel='minDCF',
            plot_pca= values_pca 
        )


def validate_quadratic_logreg_models(D, L, prefix=""):
    table = PrettyTable(['Model', 'PCA', 'Weighted', 'λ'] + [f'minDCF{wp}' for wp in const.APPS])
    hyp = numpy.arange(-4,5,1)
    dcfs = {w['init']['weighted']:[] for w in const.qlogregs}
    dcfs_pca = {w['init']['weighted']:[] for w in const.qlogregs}

    utils.log_test('Q-LogReg: ')
    for v in const.qlogregs:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'])
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['weighted']
        dcfs[w].append(metrics)
        table.add_row([v['desc'], '-', v['init']['weighted'] if v['init']['weighted'] else '-', '%.0e' % v['init']['l']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.log_test('Q-LogReg (PCA): ')
    for v in const.qlogregs:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'], pca=9)
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['weighted']
        dcfs_pca[w].append(metrics)
        table.add_row([v['desc'], 9, v['init']['weighted'] if v['init']['weighted'] else '-', v['init']['l']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}{prefix}dcfs_quadratic_lr')
    utils.json_dump(dcfs_pca, filename=f'{utils.test}{prefix}dcfs_quadratic_lr_pca')
    utils.save_table(table, filename=f'{utils.test}{prefix}quadratic_lr')
        
    
    for w in dcfs.keys():
        values = numpy.array(dcfs[w]).T
        values_pca = numpy.array(dcfs_pca[w]).T
        plot.plot_validation(
            hyp, values,
            label=f'{"Unw" if str(w) == "0" else "W"}eighted Q-LogReg',
            savefig=f'{utils.test}{prefix}dcf_quadratic_lr_{"unweighted" if w==0  else "weighted"}',
            xlabel='log(λ)',
            ylabel='minDCF',
            plot_pca= values_pca 
        )


def validate_linear_SVM(D, L, prefix=""):
    table = PrettyTable(['Model', 'PCA', 'K', 'C'] + [f'minDCF{wp}' for wp in const.APPS])
    Cs = numpy.arange(-2,7)
    dcfs = {w['init']['K']:[] for w in const.linearsvms}
    dcfs_pca = {w['init']['K']:[] for w in const.linearsvms}

    utils.log_test('Linear SVM: ')
    for v in const.linearsvms:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'])
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['K']
        dcfs[w].append(metrics)
        table.add_row([v['desc'], '-', v['init']['K'], '%.0e' % v['init']['C']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.log_test('Linear SVM (PCA): ')
    for v in const.linearsvms:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'], pca=9)
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['K']
        dcfs_pca[w].append(metrics)
        table.add_row([v['desc'], 9, v['init']['K'], '%.0e' % v['init']['C']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}{prefix}dcfs_linear_SVMs')
    utils.json_dump(dcfs_pca, filename=f'{utils.test}{prefix}dcfs_linear_SVMs_pca')
    utils.save_table(table, filename=f'{utils.test}{prefix}dcfs_linear_SVMs')

    for k in dcfs.keys():
        values = numpy.array(dcfs[k]).T
        values_pca = numpy.array(dcfs_pca[k]).T
        plot.plot_validation(
            Cs, values,
            label=f'Linear SVM (K={k})',
            savefig=f'{utils.test}{prefix}dcf_linear_SVM_k{k}',
            xlabel='log(C)',
            ylabel='minDCF',
            plot_pca=values_pca
        )


def validate_poly2_SVM(D, L, prefix=""):
    table = PrettyTable(['Model', 'PCA', 'K', 'C'] + [f'minDCF{wp}' for wp in const.APPS])
    Cs = numpy.arange(-2,7)
    dcfs = {w['init']['K']:[] for w in const.poly2svms}
    dcfs_pca = {w['init']['K']:[] for w in const.poly2svms}

    utils.log_test('Poly2 SVM: ')
    for v in const.poly2svms:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'])
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['K']
        dcfs[w].append(metrics)
        table.add_row([v['desc'], '-', v['init']['K'], '%.0e' % v['init']['C']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.log_test('Poly2 SVM (PCA): ')
    for v in const.poly2svms:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'], pca=9)
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['K']
        dcfs_pca[w].append(metrics)
        table.add_row([v['desc'], 9, v['init']['K'], '%.0e' % v['init']['C']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}{prefix}dcfs_poly2_SVMs')
    utils.json_dump(dcfs_pca, filename=f'{utils.test}{prefix}dcfs_poly2_SVMs_pca')
    utils.save_table(table, filename=f'{utils.test}{prefix}dcfs_poly2_SVMs')
        

    for k in dcfs.keys():
        values = numpy.array(dcfs[k]).T
        values_pca = numpy.array(dcfs_pca[k]).T
        plot.plot_validation(
            Cs, values,
            label=f'Poly(2) SVM (K={k})',
            savefig=f'{utils.test}{prefix}dcf_poly2_SVM_k{k}',
            xlabel='log(C)',
            ylabel='minDCF',
            plot_pca=values_pca
        )


def validate_poly3_SVM(D, L, prefix=""):
    table = PrettyTable(['Model', 'K', 'C'] + [f'minDCF{wp}' for wp in const.APPS])
    Cs = numpy.arange(-1,2)
    dcfs = {w['init']['K']:[] for w in const.poly3svms}

    utils.log_test('Poly3 SVM: ')
    for v in const.poly3svms:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'])
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        w = v['init']['K']
        dcfs[w].append(metrics)
        table.add_row([v['desc'], v['init']['K'], '%.0e' % v['init']['C']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}{prefix}dcfs_poly3_SVMs')
    utils.save_table(table, filename=f'{utils.test}{prefix}dcfs_poly3_SVMs')

    values = numpy.array(dcfs[str(1)]).T
    values_k = {10:numpy.array(dcfs[str(10)]).T}
    plot.plot_validation(
        Cs, values,
        label=f'Poly(3) SVM',
        savefig=f'{utils.test}{prefix}dcf_poly3_SVM',
        xlabel='log(C)',
        ylabel='minDCF',
        plot_k_svm=values_k
    )


def validate_rbf_SVM(D, L, prefix=""):
    table = PrettyTable(['Model', 'gamma', 'C', 'K'] + [f'minDCF{wp}' for wp in const.APPS])
    gammas = list({m['gamma'] for m in const.rbfsvms})
    Cs = numpy.arange(-2,6,1)
    dcfs = {g:[] for g in gammas}

    utils.log_test('RBF SVM: ')
    for v in const.rbfsvms:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'])
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x', end='', flush=True)
        dcfs[v['gamma']].append(metrics)
        table.add_row([v['desc'], v['gamma'], v['init']['C'], v['init']['K']] + ['%.3f' % m for m in  metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}{prefix}dcfs_rbf_SVMs')
    utils.save_table(table, filename=f'{utils.test}{prefix}dcfs_rbf_SVMs')
    plot.plot_validation_rbf(Cs, dcfs, savefig=f'{utils.test}{prefix}dcf_rbf_SVM')


def validate_gmm(D, L, gmm_type='full', prefix="", pca=None):
    table = PrettyTable(['Model', 'K-T', 'K-NT'] + [f'minDCF{wp}' for wp in const.APPS])
    dcfs = {k:[] for k in const.TK}

    utils.log_test(f'{gmm_type.capitalize().replace("_"," ")} GMM: ')
    for v in const.gmms[gmm_type]:
        scores = mlpr.kfold(v['model_name'], D, L, 5, init=v['init'], pca=pca)
        metrics = meval.apply_metric(scores, L, meval.minDCF, const.APPS)
        utils.log_test('x')
        tk = v['init']['TK']
        dcfs[tk].append(metrics)
        table.add_row([v['desc'], v['init']['TK'], v['init']['NTK']] + ['%.3f' % m for m in metrics])
    utils.log_test('\n')

    utils.json_dump(dcfs, filename=f'{utils.test}{prefix}dcfs_gmm_{gmm_type}{f"_pca{pca}" if pca else ""}')
    utils.save_table(table, filename=f'{utils.test}{prefix}dcfs_gmm_{gmm_type}{f"_pca{pca}" if pca else ""}')
    plot.plot_gmm(dcfs, savefig=f'{utils.test}{prefix}dcf_gmm_{gmm_type}{f"_pca{pca}" if pca else ""}')
