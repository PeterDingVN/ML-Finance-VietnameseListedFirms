from .libs import *
from .ml_hyperpars import *
from panelsplit.cross_validation import PanelSplit
from sklearn.model_selection import GridSearchCV


class InputData:
    def __init__(self, df: pd.DataFrame, id_col: str, time_col: str, target: str, reg: bool):
        self.df = df
        self.id_col = id_col
        self.time_col = time_col
        self.target = target
        self.reg = reg

    def optimal_param(self, n_splits: int, test_size: int) -> pd.DataFrame:
        '''

        :param n_splits: number of folds to loop through
        :param test_size: limit test set size, max number == n_samples // (n_splits + 1)
        :return: a table summarizing the score of all hyper-parameters from best -> worst

        '''
        idx = [self.id_col, self.time_col]
        df_copy = self.df.copy()
        df_copy.set_index(idx, inplace=True)

        # setup
        X = df_copy.drop([self.target], axis=1)
        y = df_copy[self.target]
        periods = df_copy.index.get_level_values(level=1)
        cv_strat = PanelSplit(periods = periods, test_size=test_size, n_splits=n_splits)

        result_fin = []

        # if it's regression
        if self.reg:
            for name, (algo, hyperpar) in algorithm.items():
                print(f'Processing {name} ...')
                grid = GridSearchCV(algo,
                                    scoring=score_reg,
                                    param_grid = hyperpar,
                                    cv=cv_strat,
                                    refit='r2')
                grid_fit = grid.fit(X, y)
                results = pd.DataFrame(grid_fit.cv_results_)
                results['algo_used'] = f'{name}'
                result_fin.append(results[['algo_used', 'params',
                                           'mean_test_r2', 'mean_test_mape', 'mean_test_rmse']])

            eval_output = pd.concat(result_fin, ignore_index=True)
            eval_output.sort_values(by=['mean_test_r2', 'mean_test_rmse', 'mean_test_mape'],
                                    ascending=False,
                                    inplace=True)

        # if classification task
        else:
            for name, (algo, hyperpar) in algorithm.items():
                print(f'Processing {name} ...')
                grid = GridSearchCV(algo,
                                    scoring=score_class,
                                    param_grid = hyperpar,
                                    cv=cv_strat,
                                    refit='accuracy')
                grid_fit = grid.fit(X, y)
                results = pd.DataFrame(grid_fit.cv_results_)
                results['algo_used'] = f'{name}'
                result_fin.append(results[['algo_used', 'params',
                                           'mean_test_accuracy',
                                           'mean_test_roc_auc', 'mean_test_precision',
                                           'mean_test_recall']])

            eval_output = pd.concat(result_fin, ignore_index=True)
            eval_output.sort_values(by=['mean_test_accuracy',
                                        'mean_test_roc_auc',
                                        'mean_test_precision',
                                        'mean_test_recall'],
                                    ascending=False,
                                    inplace=True)

        return eval_output






