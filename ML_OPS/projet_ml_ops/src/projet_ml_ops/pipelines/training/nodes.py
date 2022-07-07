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


MODELS = [
    {
        "name": "LightGBM",
        "class": xgb,
        "params": {
            "objective": "reg:linear",
            "random_state":42
        },
        "override_schemas": {
            "num_leaves": int,
            "min_child_samples": int,
            "max_depth": int,
            "num_iterations": int,
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







