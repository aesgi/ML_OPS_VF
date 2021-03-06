import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score
from lightgbm.sklearn import LGBMClassifier
from hyperopt import hp, tpe, fmin
from sklearn.base import BaseEstimator
from typing import Callable, Tuple, Any, Dict
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import f1_score

MODELS = [
    {
        "name": "LightGBM",
        "class": xgb,
        "params": {
            "objective": "reg:linear",
            "random_state":42
        },
    }
]


def train_model(
    instance: BaseEstimator,
    training_set: Tuple[np.ndarray, np.ndarray],
    params: Dict = {},
) -> BaseEstimator:
    """
    Trains a new instance of model with supplied training set and hyper-parameters.
    """
    override_schemas = list(filter(lambda x: x["class"] == instance, MODELS))[0][
        "override_schemas"
    ]
    for p in params:
        if p in override_schemas:
            params[p] = override_schemas[p](params[p])
    model = instance(**params)
    model.fit(*training_set)
    return model

def optimize_hyp(
    instance: BaseEstimator,
    dataset: Tuple[np.ndarray, np.ndarray],
    search_space: Dict,
    metric: Callable[[Any, Any], float],
    max_evals: int = 40,
) -> BaseEstimator:
    """
    Trains model's instances on hyper-parameters search space and returns most accurate
    hyper-parameters based on eval set.
    """
    X, y = dataset

    def objective(params):
        rep_kfold = RepeatedKFold(n_splits=4, n_repeats=1)
        scores_test = []
        for train_I, test_I in rep_kfold.split(X):
            X_fold_train = X.iloc[train_I, :]
            y_fold_train = y.iloc[train_I].values.flatten()
            X_fold_test = X.iloc[test_I, :]
            y_fold_test = y.iloc[test_I].values.flatten()
            # On entra??ne une instance du mod??le avec les param??tres params
            model = train_model(
                instance=instance,
                training_set=(X_fold_train, y_fold_train),
                params=params
            )
            # On calcule le score du mod??le sur le test
            scores_test.append(
                metric(y_fold_test, model.predict(X_fold_test))
            )

        return np.mean(scores_test)

    return fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals)


def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 40
) -> BaseEstimator:
    """
    Runs training of multiple model instances and select the most accurated based on objective function.
    """
    X = pd.concat((X_train, X_test))
    y = pd.concat((y_train, y_test))

    opt_models = []
    for model_specs in MODELS:
        # Finding best hyper-parameters with bayesian optimization
        optimum_params = optimize_hyp(
            model_specs["class"],
            dataset=(X, y),
            search_space=model_specs["params"],
            metric=lambda x, y: -f1_score(x, y),
            max_evals=max_evals
        )
        print("done")
        # Training the supposed best model with found hyper-parameters
        model = train_model(
            model_specs["class"],
            training_set=(X_train, y_train),
            params=optimum_params,
        )
        opt_models.append(
            {
                "model": model,
                "name": model_specs["name"],
                "params": optimum_params,
                "score": f1_score(y_test, model.predict(X_test)),
            }
        )

    # In case we have multiple models
    best_model = max(opt_models, key=lambda x: x["score"])
    return dict(model=best_model)






