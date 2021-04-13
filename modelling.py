import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate
from skopt import BayesSearchCV
import pandas as pd


def plot_estimators(pipes, estimators, n_splits, data, target, metrics):
    '''
    Evaluating each estimator and comparing their metrics (mean absolute
    error)
    '''
    scorers = []
    labels = []
    for pipe_name in pipes.keys():
        if pipe_name in estimators:
            print('CV on', pipe_name)
            pipe = pipes[pipe_name]
            labels.append(pipe_name)
            kf = KFold(n_splits)
            model_score = cross_validate(pipe,
                                         data,
                                         target,
                                         scoring=metrics,
                                         cv=kf,
                                         verbose=True,
                                         n_jobs=-1)
            scorers.append(model_score)

    score_lists = {}
    for metric in metrics:
        score_lists[metric] = [score['test_' + metric] for score in scorers]

    for i, (title, _list) in enumerate(score_lists.items()):
        plt.figure(i)
        plot = sns.boxplot(data=_list).set_xticklabels(labels, rotation=45)
        plt.title(title)


def tune_param(model, pipes, param_grid, refit, data, target, cv=5, n_iter=6):
    '''
    Tuning parameters with bayesian search
    '''

    param_grid = {
        model + '__' + key: param_grid[key]
        for key in param_grid.keys()
    }

    xgbcv = BayesSearchCV(pipes[model],
                          param_grid,
                          scoring="neg_mean_absolute_error",
                          n_iter=n_iter,
                          refit=refit,
                          n_jobs=-1,
                          verbose=True,
                          cv=cv)
    xgbcv.fit(data, target)

    print('best score: ' + str(xgbcv.best_score_))
    print('best params: ' + str(xgbcv.best_params_))
    results = pd.DataFrame(xgbcv.cv_results_)

    return xgbcv, results

