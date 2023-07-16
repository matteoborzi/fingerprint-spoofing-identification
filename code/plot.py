from itertools import product
import model_evaluation as meval
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy
import mlpr, utils, const

def plot_hist(D: numpy.ndarray, L: numpy.ndarray, labels_dict: dict, savefig='', label='', bins=60, title='') -> None:
    plt.figure()
    plt.xlabel(label)
    plt.title(title)

    for l, name in labels_dict.items():
        D_i = D[L==l]
        plt.hist(D_i, bins=bins, density=True, alpha=0.4, label=name)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(f'{utils.IMAGE_PATH}/{savefig}')


def plot_scatter(D1: numpy.ndarray, D2: numpy.ndarray, L: numpy.ndarray, labels_dict: dict, savefig='', xlabel='', ylabel='') -> None:
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    for l, name in labels_dict.items():
        D1m = D1[L==l]
        D2m = D2[L==l]
        plt.scatter(D1m, D2m, label=name, alpha=0.2, s=1)
    
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(f'{utils.IMAGE_PATH}/{savefig}')


def plot_hist_features(D,L, labels_dict, bins=60, title=''):
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    fig, axs = plt.subplots(D.shape[0]//2, 2)

    for i in range(D.shape[0]):
        ax = axs[i % 5, i//5]
        #ax.set_box_aspect(1/2)
        for l, name in labels_dict.items():
            D_i = D[i][L==l]
            ax.hist(D_i, bins=bins, density=True, alpha=0.4, label=name)
            ax.legend(loc=1, prop=dict(size=6))

    fig.set_size_inches(9,20)
    fig.subplots_adjust(top=0.97, bottom=0.03, left=0.03, right=0.97, hspace=.1, wspace=.1)
    fig.savefig(f'{utils.IMAGE_PATH}/{title + "_" if title else ""}features_hist_{bins}.png', bbox_inches='tight', dpi=300)


def plot_features(D, L, labels_dict, savefig='', bins=60, **kwargs):
    plt.rcParams.update(matplotlib.rcParamsDefault)
    
    if 'all_features' in kwargs.keys():
        plt.rc('xtick', labelsize=4)
        plt.rc('ytick', labelsize=4)
        plt.rcParams['xtick.major.pad'] = 0.8
        plt.rcParams['ytick.major.pad'] = 1
        plt.rcParams['axes.linewidth'] = 0.8
    else:
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)

    fig, axs = plt.subplots(D.shape[0], D.shape[0])
        
    for i,j in product(range(D.shape[0]), range(D.shape[0])):
        if i == j:
            for l, name in labels_dict.items():
                D_i = D[i][L==l]
                axs[i,j].hist(D_i, bins=bins, density=True, alpha=0.4, label=name)
        else:
            for l, name in labels_dict.items():
                D1m = D[i][L==l]
                D2m = D[j][L==l]
                axs[i,j].scatter(D1m, D2m, label=name, alpha=0.2, s=1)

    if 'all_features' in kwargs.keys():
        fig.set_size_inches(12,8)
        fig.subplots_adjust(top=0.97, bottom=0.03, left=0.03, right=0.97, hspace=.35, wspace=.3)
    else:
        fig.tight_layout()

    if savefig:
        fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_correlations(DTR, cmap="Greys", savefig=''):
    plt.rcParams.update(matplotlib.rcParamsDefault)

    # Fast correlation matrix computation:
    #   corr = D^-1 * sigma * (D^-1).T
    # where
    #   D = sqrt(diag(sigma))
    # There is no need to transpose D^-1 since it's diagonal
    corr = mlpr.covariance_matrix(DTR)
    dinv = numpy.diag(1 / numpy.sqrt(numpy.diag(corr)))
    corr = numpy.dot(numpy.dot(dinv, corr), dinv)
    
    plt.figure()
    heatmap = sns.heatmap(numpy.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=True)
    fig = heatmap.get_figure()

    if savefig:
        fig.savefig(f"{utils.IMAGE_PATH}/{savefig}.png", dpi=300)


def plot_PCA_variance(X, Y, savefig=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('PCA dimensions')
    ax.set_ylabel('Fraction of explained variance')
    ax.set_xticks(X)
    ax.set_yticks(numpy.arange(0.0,1.1,0.1))
    ax.set_yticks(numpy.arange(0.0,1.1,0.05), minor=True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 0)
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.7)

    ax.plot(X,Y, linewidth=1.2)

    if savefig:
        fig.savefig(f"{utils.IMAGE_PATH}/{savefig}.png", dpi=300)


def plot_validation(x: numpy.ndarray, rows: list, label='', savefig='tmp', xlabel='', ylabel='', plot_pca=None, plot_k_svm=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_yticks([i/10 for i in range(1,11)])

    for j, row in enumerate(rows):
        ax.plot(x, row, label=label + f' (π={const.APPS[j][0]}){", K=1" if bool(plot_k_svm) else ""}')

    if bool(plot_pca):
        plt.gca().set_prop_cycle(None)
        for j, row in enumerate(plot_pca):
            ax.plot(x, row, linestyle='dashed', label=label + f' (π={const.APPS[j][0]}), PCA=9')

    elif bool(plot_k_svm):
        k = list(plot_k_svm.keys())[0]
        plt.gca().set_prop_cycle(None)
        for j, row in enumerate(plot_k_svm.values()):
            ax.plot(x, row, linestyle='dashed', label=label + f' (π={const.APPS[j][0]}), K={k}')

    if bool(plot_k_svm):
        ax.legend(loc='best', prop=dict(size=10))
    else:
        ax.legend(loc=2, prop=dict(size=10))
    plt.xlim([numpy.min(x),numpy.max(x)])
    plt.ylim([0, 1.1])
    plt.grid()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_validation_rbf(x: numpy.ndarray, dcfs: list, savefig='tmp'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('log(C)')
    ax.set_ylabel('minDCF')
    ax.set_xticks(x)
    ax.set_yticks([i/10 for i in range(1,11)])

    for gamma, row in dcfs.items():
        row = numpy.array(row).T[0]
        ax.plot(x, row, label=f'RBF SVM - γ={gamma}')
    
    plt.legend(loc='best', prop=dict(size=10))
    plt.xlim([numpy.min(x),numpy.max(x)])
    plt.ylim([0, 1.1])
    plt.grid()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_gmm(dcfs: dict, savefig='tmp'):
    labels = list(dcfs.keys())
    scores = numpy.array([list(numpy.around(numpy.array(v)[:,0].ravel(), decimals=3)) for k,v in dcfs.items()]).T

    KNT = len(scores)
    x = numpy.arange(len(labels))
    shift = [i-KNT//2 for i in range(KNT)]
    width = 1 / (KNT+2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', plt.cm.brg(numpy.linspace(0,0.6,KNT)))
    ax.set_ylim([0.0, 1.1])
    ax.set_xlabel('Target K')
    ax.set_ylabel('minDCF')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines[['right', 'top']].set_visible(False)

    for i in range(scores.shape[0]):
        bar = ax.bar(x + (shift[i] * width) + width/2, scores[i], width, label=f'NT-K={2**i}', alpha=0.9, edgecolor='black', linewidth=0.3)
        ax.bar_label(bar, rotation='vertical', padding=3)

    ax.legend(bbox_to_anchor=(1, 1.05), ncols=3)
    fig.tight_layout()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_bayes_error(scores, labels, model_name, savefig='tmp', color='tab:blue'):
    effPriorLogOdds = numpy.linspace(-3, 3, 30)
    actdcf, mindcf = meval.compute_bayes_error(scores, labels, effPriorLogOdds)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(effPriorLogOdds, mindcf, label=f'{model_name} minDCF', color=color)
    ax.plot(effPriorLogOdds, actdcf, label=f'{model_name} actDCF', color=color, linestyle='dotted')

    ax.set_xlabel('log(π/1-π)')
    ax.set_ylabel('DCF')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.grid()
    plt.legend()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_ROC_curve(scores: numpy.ndarray, labels: numpy.ndarray, savefig='tmp', **kwargs):
    thresholds = numpy.hstack([-numpy.inf, numpy.sort(scores), numpy.inf])
    plot_points = []

    for t in thresholds:
        cm = meval.compute_confusion_matrix(numpy.array(scores > t,  dtype=numpy.int32), labels, 2)
        plot_points.append((meval.FPR(cm),1-meval.FNR(cm)))

    plot_points = numpy.array(plot_points).T
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(plot_points[0], plot_points[1], color=kwargs['color'], label=kwargs['label'])
    plt.xlim((-0.05,1.05))
    plt.ylim((-0.05,1.05))
    plt.xticks([i/10 for i in range(11)])
    plt.yticks([i/10 for i in range(11)])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_DET_curve(scores_list: dict, labels: numpy.ndarray, savefig='tmp'):
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for model_desc, scores in scores_list.items():
        thresholds = numpy.hstack([-numpy.inf, numpy.sort(scores), numpy.inf])
        plot_points = []

        for t in thresholds:
            cm = meval.compute_confusion_matrix(numpy.array(scores > t,  dtype=numpy.int32), labels, 2)
            plot_points.append((meval.FPR(cm),meval.FNR(cm)))

        plot_points = numpy.array(plot_points).T

        ax.plot(plot_points[0], plot_points[1], label=model_desc)
    
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_evaluation(x: numpy.ndarray, rows: list, label='', savefig='tmp', xlabel='', ylabel='', plot_val=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_yticks([i/10 for i in range(1,11)])

    for j, row in enumerate(rows):
        ax.plot(x, row, label=label + f' (π={const.APPS[j][0]}) [Eval]')

    if bool(plot_val.any()):
        plt.gca().set_prop_cycle(None)
        for j, row in enumerate(plot_val):
            ax.plot(x, row, linestyle='dashed', label=label + f' (π={const.APPS[j][0]}) [Val]')

    ax.legend(loc='best', prop=dict(size=10))
    plt.xlim([numpy.min(x),numpy.max(x)])
    plt.ylim([0, 1.1])
    plt.grid()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)


def plot_gmm_evaluation(dcfs: dict, dcfs_val: dict, savefig='tmp'):
    labels = list(dcfs.keys())
    scores = numpy.array([list(numpy.around(numpy.array(v)[:,0].ravel(), decimals=3)) for v in dcfs.values()]).T
    scores_val = numpy.array([list(numpy.around(numpy.array(v)[:-1,0].ravel(), decimals=3)) for v in dcfs_val.values()]).T

    KNT = len(scores)
    x = numpy.arange(len(labels))
    shift = [i-KNT for i in range(0,2*KNT,2)]
    width = 1 / (2*(KNT+2))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color', plt.cm.brg(numpy.linspace(0,0.6,KNT)))
    ax.set_ylim([0.0, 1.1])
    ax.set_xlabel('Target K')
    ax.set_ylabel('minDCF')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines[['right', 'top']].set_visible(False)

    for i in range(scores.shape[0]):
        bar = ax.bar(x + (shift[i] * width) + width/2, scores_val[i], width, label=f'NT-K={2**i} [Val]', alpha=0.9, edgecolor='black', linewidth=0.3)
        ax.bar_label(bar, rotation='vertical', padding=3)
    for i in range(scores_val.shape[0]):
        vbar = ax.bar(x + ((shift[i]+1) * width) + width/2, scores[i], width, label=f'NT-K={2**i} [Eval]', alpha=0.9, edgecolor='black', hatch='\\\\', linewidth=0.3)
        ax.bar_label(vbar, rotation='vertical', padding=3)

    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(f'{utils.IMAGE_PATH}/{savefig}.png', bbox_inches='tight', dpi=300)

