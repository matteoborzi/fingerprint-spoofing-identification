import features, validation, calibration, evaluation, utils, mlpr


if __name__ == '__main__':
    print(f"---------------- Fingerprint Spoofing Detection ---------------- ")

    D, L = utils.load('Train.txt')
    D, L = mlpr.shuffle(D,L)
    DZ = mlpr.z_normalization(D)
    DTE, LTE = utils.load(f'Test.txt')
    DZTE = mlpr.z_normalization(DTE)

    
    #############################################################
    # Dataset overview and feature analysis
    #############################################################
    features.dataset_overview(D, L, 'Training')
    features.dataset_overview(DTE, LTE, 'Test')
    features.feature_analysis(D, DZ, L)


    #############################################################
    # Model selection and validation
    #############################################################
    print('--- Model selection ---')
    
    validation.validate_generative_models(D, L)
    validation.validate_linear_logreg_models(D,L, prefix='RAW_')
    validation.validate_linear_logreg_models(DZ,L,'ZNorm_')
    validation.validate_quadratic_logreg_models(D,L, prefix='RAW_')
    validation.validate_quadratic_logreg_models(DZ,L,'ZNorm_')
    validation.validate_linear_SVM(D,L,'RAW_')
    validation.validate_linear_SVM(DZ,L,'ZNorm_')
    validation.validate_poly2_SVM(D,L,'RAW_')
    validation.validate_poly2_SVM(D,L,'ZNorm_')
    validation.validate_poly3_SVM(D,L,'RAW_')
    validation.validate_rbf_SVM(D,L, 'RAW_')
    validation.validate_gmm(D,L, gmm_type='full', prefix='RAW_')
    validation.validate_gmm(D,L, gmm_type='tied', prefix='RAW_')
    validation.validate_gmm(D,L, gmm_type='diagonal', prefix='RAW_')
    validation.validate_gmm(D,L, gmm_type='tied_diagonal', prefix='RAW_')


    #############################################################
    # Models calibration and fusion
    #############################################################
    print('--- Model calibration ---')

    calibration.best_models_bayes_error(D,L)
    calibration.best_models_roc_curve(D,L)
    calibration.best_models_calibration(D, L, roc=True, bayes=True, table=True)
    calibration.model_fusion(D,L, det=True) #bayes=True, roc=True, table=True

    
    #############################################################
    # Models evaluation
    #############################################################
    print('--- Evaluation ---')

    evaluation.evaluate_gaussian_models(D, L, DTE, LTE)
    evaluation.evaluate_linear_logreg_models(D, L, DTE, LTE, prefix='RAW_')
    evaluation.evaluate_linear_logreg_models(D, L, DTE, LTE, pca=9, prefix='RAW_')
    evaluation.evaluate_quadratic_logreg_models(D, L, DTE, LTE, prefix='RAW_')
    evaluation.evaluate_quadratic_logreg_models(D, L, DTE, LTE, pca=9, prefix='RAW_')
    evaluation.evaluate_linear_logreg_models(DZ, L, DZTE, LTE, prefix='ZNorm_')
    evaluation.evaluate_linear_logreg_models(DZ, L, DZTE, LTE, pca=9, prefix='ZNorm_')
    evaluation.evaluate_quadratic_logreg_models(DZ, L, DZTE, LTE, prefix='ZNorm_')
    evaluation.evaluate_quadratic_logreg_models(DZ, L, DZTE, LTE, pca=9, prefix='ZNorm_')
    evaluation.evaluate_linear_SVM(D, L, DTE, LTE, K=10, prefix='RAW_')
    evaluation.evaluate_linear_SVM(DZ, L, DZTE, LTE, K=10, prefix='ZNorm_')
    evaluation.evaluate_poly2_SVM(D, L, DTE, LTE, K=10, prefix='RAW_')
    evaluation.evaluate_rbf_SVM(D, L, DTE, LTE, gamma=0.002, prefix='RAW_')
    evaluation.evaluate_gmm(D, L, DTE, LTE, gmm_type='full', prefix='RAW_')
    evaluation.evaluate_gmm(D, L, DTE, LTE, gmm_type='tied', prefix='RAW_')
    evaluation.evaluate_gmm(D, L, DTE, LTE, gmm_type='diagonal', prefix='RAW_')
    evaluation.evaluate_gmm(D, L, DTE, LTE, gmm_type='tied_diagonal', prefix='RAW_')
    calibration.model_fusion_evaluation(D, L, DTE, LTE, bayes=False, roc=False, table=True, det=True)
    calibration.best1_bayes_error(D, L, DTE, LTE)
    calibration.best1_calibration(D, L, DTE, LTE, True, True, True)
    