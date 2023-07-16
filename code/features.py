import matplotlib.pyplot as plt
import numpy
import mlpr
import plot


def dataset_overview(D,L, dataset_name='Training'):
    empirical_prior = numpy.array([D[:, L==0].shape[1]/D.shape[1], D[:, L==1].shape[1]/D.shape[1]]).ravel().tolist()
    print(f'{dataset_name} set size: {D.shape}')
    print(f'\tAuthentic fingerprints: {D[:,L==1].shape[1]}')
    print(f'\tSpoofed fingerprints: {D[:,L==0].shape[1]}')
    print('\tEmpirical prior: [%.3f, %.3f]' % (empirical_prior[0], empirical_prior[1]))
    print()


def feature_analysis(D: numpy.ndarray, DZ: numpy.ndarray, L: numpy.ndarray):
    M = D.shape[0] # Number of features
    labels_dict = {0:'Spoofed', 1:'Authentic'}
    
    # Plot features histograms (for the report)
    plot.plot_hist_features(D, L, labels_dict)

    # All features scatter and histogram plots
    plot.plot_features(D,L, labels_dict, savefig='features', all_features=True) # Features plot'
    plot.plot_features(DZ, L, labels_dict, savefig='features_znorm', all_features=True) #Z-Norm features plot

    # Pearson correlation hitmaps
    plot.plot_correlations(D, savefig='correlations_all')
    plot.plot_correlations(D[:, L==1], cmap='Reds', savefig='correlations_true')
    plot.plot_correlations(D[:, L==0], cmap='Blues', savefig='correlations_false')

    # PCA analysis
    pca_dims = numpy.arange(M+1)
    pca_vars = numpy.zeros(M+1)

    for m in pca_dims:
        D_PCA, _, pca_vars[m] = mlpr.PCA(D,m=m)
        if m == 10:
            plot.plot_features(D_PCA, L, labels_dict, savefig=f'PCA_m{m}', all_features=True)

    plot.plot_PCA_variance(pca_dims,pca_vars, savefig='PCA_explained_variance')

    # LDA analysis
    W = mlpr.LDA(D,L,m=2)
    DP = numpy.dot(W.T, D)
    plot.plot_hist(DP[0],L,labels_dict, savefig='LDA')

    WZ = mlpr.LDA(DZ, L, m=2)
    DZP = numpy.dot(WZ.T, DZ)
    plot.plot_hist(DZP[0], L, labels_dict, savefig='LDA_znorm')

