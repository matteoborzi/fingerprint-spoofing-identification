from const import APP, APPS, selected_models, best3_models, calibration_model, best1_model_eval, cmap
from prettytable import PrettyTable 
import utils, const, plot, mlpr
from evaluation import train_for_evaluation 
import model_evaluation as meval
import numpy



def best_models_bayes_error(D, L):
    for i, model in enumerate(selected_models.values()):
        utils.log_test(model['desc'], end='\n')
        filename = f'scores_{utils.clean_filename(model["desc"])}'
        color = const.cmap(float(i)/ len(selected_models))

        all_scores = mlpr.kfold(model['model_name'], D, L, pca=model['pca'], init=model['init'])
        utils.json_dump(all_scores,filename)

        plot.plot_bayes_error(
            all_scores,
            L,
            model['desc'],
            savefig=f'bayes_error_{utils.clean_filename(model["desc"])}',
            color=color)
        

def best_models_roc_curve(D, L):
    for i, model in enumerate(selected_models.values()):
        utils.log_test(model['desc'], end='\n')
        filename = f'scores_{utils.clean_filename(model["desc"])}'
        color = cmap(float(i)/ len(selected_models))

        all_scores = mlpr.kfold(model['model_name'], D, L, pca=model['pca'], init=model['init'])
        utils.json_dump(all_scores,filename)

        plot.plot_ROC_curve(
            all_scores,
            L,
            savefig=f'roc_curve_{utils.clean_filename(model["desc"])}',
            color=color,
            label=model['desc'])

    
def model_calibration_plots(calibrated_scores, L, model_desc, color_index=0, **kwargs):
    color = cmap(float(color_index)/ len(selected_models))

    if kwargs['bayes']:
        plot.plot_bayes_error(
            calibrated_scores, L,
            model_name=model_desc,
            savefig=f'bayes_error_{utils.clean_filename(model_desc)}_calibrated',
            color= color
        )

    if kwargs['roc']:
        plot.plot_ROC_curve(
                calibrated_scores,
                L,
                savefig=f'roc_curve_{utils.clean_filename(model_desc)}_calibrated',
                color=color,
                label=model_desc
        )


def compute_calibrated_scores(S, L):
    S = mlpr.mrow(S) if S.ndim == 1 else S
    pi = meval.compute_effective_prior(APP)[0]
    calibrated_scores = mlpr.kfold('LogReg', S, L, k=5, init=calibration_model['init'])
    return calibrated_scores - numpy.log(pi / (1-pi))


def model_calibration(uncalibrated_scores, L, model):
    filename = f'scores_{utils.clean_filename(model["desc"])}'
    utils.log_test(f'Calibrating {model["desc"]}', end='\n')

    uncalibrated_scores = numpy.array(uncalibrated_scores)     
    calibrated_scores = compute_calibrated_scores(uncalibrated_scores, L)
    utils.json_dump(calibrated_scores.tolist(), filename=f'{filename}_calibrated')

    return calibrated_scores


def best_models_calibration(D, L, bayes=False, roc=False, table=False):
    table = PrettyTable(['Model'] + [f'Uncalibrated actDCF{app}' for app in APPS] + [f'Calibrated actDCF{app}' for app in APPS])

    for i, model in enumerate(selected_models.values()):
        filename = f'scores_{utils.clean_filename(model["desc"])}'

        uncal = mlpr.kfold(model['model_name'], D, L, pca=model['pca'], init=model['init'])
        utils.json_dump(uncal, filename)

        if model['calibrate']:
            cal = model_calibration(uncal, L, model)
            model_calibration_plots(cal, L, model["desc"], color_index=i, bayes=bayes, roc=roc)
            if table:
                old_dcfs = meval.apply_metric(uncal, L, meval.normalized_DCF, APPS)
                cal_dcfs = meval.apply_metric(cal, L, meval.normalized_DCF, APPS)
                table.add_row([model['desc']] + ['%.3f' % m for m in old_dcfs] + ['%.3f' % m for m in cal_dcfs])
        elif table:
            old_dcfs = meval.apply_metric(uncal, L, meval.normalized_DCF, APPS)
            table.add_row([model['desc']] + ['%.3f' % m for m in old_dcfs] + ['-' for _ in range(3)])

    if table:
        utils.save_table(table, filename='best_models_calibrated')


def model_fusion(D, L, bayes=False, roc=False, table=False, det=False):
    table = PrettyTable(['Model'] + [f'minDCF{app}' for app in APPS] + [f'actDCF{app}' for app in APPS])
    all_models = dict()

    for fusion in utils.powerset(best3_models.keys(), start=1):
        if len(fusion) == 1:
            model = best3_models[str(fusion[0])]
            filename = f'scores_{utils.clean_filename(model["desc"])}{"_calibrated" if model["calibrate"] else ""}'
            all_models[model["desc"]] = utils.json_load(filename)
        else:
            fusion_filename = f'fusion_{"_".join(fusion)}'
            fusion_desc = '+'.join(fusion).upper()
            utils.log_test(f'Fusion {fusion_desc}', end='\n')

            scores = []
            for modelname in fusion:
                model = best3_models[str(modelname)]
                filename = f'scores_{utils.clean_filename(model["desc"])}{"_calibrated" if model["calibrate"] else ""}'
                scores.append(utils.json_load(filename))
            scores = numpy.array(scores)

            calibrated_scores = compute_calibrated_scores(scores, L)
            utils.json_dump(calibrated_scores.tolist(), filename=fusion_filename)

            all_models[fusion_desc] = calibrated_scores
            if table:
                mindcfs = meval.apply_metric(calibrated_scores, L, meval.minDCF, APPS)
                actdcfs = meval.apply_metric(calibrated_scores, L, meval.normalized_DCF, APPS)
                table.add_row([fusion_desc] + ['%.3f' % m for m in mindcfs] + ['%.3f' % m for m in actdcfs])
            
            model_calibration_plots(calibrated_scores, L, 'fusion_'+fusion_desc, bayes=bayes, roc=roc)
    
    if det:
        plot.plot_DET_curve(
            all_models, L,
            savefig=f'det_curve_all_models'
        )

    if table:
        utils.save_table(table, filename='fusions')


def model_calibration_for_evaluation(DTR, LTR, DTE, LTE, model):
    filename = f'scores_eval_{utils.clean_filename(model["desc"])}'
    utils.log_test(f'Calibrating {model["desc"]}', end='\n')

    classifier = train_for_evaluation(model['model_name'], DTR, LTR, pca=model['pca'], init=model['init'])
    uncalibrated_scores = classifier.compute_scores(DTE)
    utils.json_dump(uncalibrated_scores, filename)

    uncalibrated_scores = numpy.array(uncalibrated_scores)     
    calibrated_scores = compute_calibrated_scores(uncalibrated_scores, LTE)
    utils.json_dump(calibrated_scores.tolist(), filename=f'{filename}_calibrated')

    return calibrated_scores


def best1_bayes_error(DTR, LTR, DTE, LTE):
    model = const.best1_model_eval['gmm']
    utils.log_test(model['desc'], end='\n')
    filename = f'scores_eval_{utils.clean_filename(model["desc"])}'

    classifier = train_for_evaluation(model['model_name'], DTR, LTR, init=model['init'])
    all_scores = classifier.compute_scores(DTE).tolist()
    utils.json_dump(all_scores,filename)

    plot.plot_bayes_error(
        all_scores,
        LTE,
        model['desc'],
        savefig=f'eval_bayes_error_{utils.clean_filename(model["desc"])}',
    )
    

def best1_calibration(DTR, LTR, DTE, LTE, bayes=False, roc=False, table=False):
    table = PrettyTable(['Model'] + [f'Uncalibrated actDCF{app}' for app in APPS] + [f'Calibrated actDCF{app}' for app in APPS])

    for i, model in enumerate(best1_model_eval.values()):
        classifier = train_for_evaluation(model['model_name'], DTR, LTR, init=model['init'])
        uncal = classifier.compute_scores(DTE)
        old_dcfs = meval.apply_metric(uncal, LTE, meval.normalized_DCF, APPS)

        cal = model_calibration(uncal, LTE, model)
        model_calibration_plots(cal, LTE, model["desc"], color_index=i, bayes=bayes, roc=roc)
        if table:
            old_dcfs = meval.apply_metric(uncal, LTE, meval.normalized_DCF, APPS)
            cal_dcfs = meval.apply_metric(cal, LTE, meval.normalized_DCF, APPS)
            table.add_row([model['desc']] + ['%.3f' % m for m in old_dcfs] + ['%.3f' % m for m in cal_dcfs])

    if table:
        utils.save_table(table, filename='best1_calibrated')


def model_fusion_evaluation(DTR, LTR, DTE, LTE, bayes=False, roc=False, table=False, det=False):
    table = PrettyTable(['Model'] + [f'minDCF{app}' for app in APPS] + [f'actDCF{app}' for app in APPS])
    all_models = dict()

    for fusion in utils.powerset(best3_models.keys(), start=2):
        fusion_filename = f'eval_fusion_{"_".join(fusion)}'
        fusion_desc = '+'.join(fusion).upper()
        utils.log_test(f'Fusion {fusion_desc}', end='\n')

        try:
            calibrated_scores = utils.load(fusion_filename)
        except:
            scores = []
            for modelname in fusion:
                model = best3_models[str(modelname)]
                classifier = train_for_evaluation(model['model_name'], DTR, LTR, init=model['init'])
                s = classifier.compute_scores(DTE).tolist()
                scores.append(s)
            scores = numpy.array(scores)
            calibrated_scores = compute_calibrated_scores(scores, LTE)

            utils.json_dump(calibrated_scores.tolist(), filename=fusion_filename)


        all_models[fusion_desc] = calibrated_scores
        if table:
            mindcfs = meval.apply_metric(calibrated_scores, LTE, meval.minDCF, APPS)
            actdcfs = meval.apply_metric(calibrated_scores, LTE, meval.normalized_DCF, APPS)
            table.add_row([fusion_desc] + ['%.3f' % m for m in mindcfs] + ['%.3f' % m for m in actdcfs])
        
        model_calibration_plots(calibrated_scores, LTE, 'eval_fusion_'+fusion_desc, bayes=bayes, roc=roc)

    if det:
        plot.plot_DET_curve(
            all_models, LTE,
            savefig=f'eval_det_curve_all_models'
        )
    
    if table:
        utils.save_table(table, fusion_filename)