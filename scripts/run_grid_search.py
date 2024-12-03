from modules.functions import define_grid_search_params
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from modules.Dataset import DavisDataset
from modules.params import get_params
from sklearn.cluster import KMeans
from scipy.spatial import distance
from pandas import DataFrame
from os.path import join
from numpy import (
    array,
    bincount,
    sum,
    log,
    pi,
    where,
)
from typing import (
    Callable,
    Tuple,
)


def BIC_Gaussian(
    estimator: Callable,
    X: array,
) -> array:
    """
    Callable to pass to GridSearchCV that will use the BIC score.
    """
    return estimator.bic(X)


def silhouette_score_grid_search(
    estimator: Callable,
    X: array,
) -> array:
    cluster = estimator.predict(X)
    score = silhouette_score(
        X,
        cluster,
    )
    score = estimator.score(
        X,
    )
    return -score


def get_grid_search_inputs(
    model: str,
) -> Tuple[
    Callable,
    dict,
    Callable,
]:
    param_grid: dict = None
    model_class: Callable = None
    score: Callable = None
    if model == "Gaussian":
        model_class = GaussianMixture()
        param_grid = {
            "n_components": range(
                2,
                7,
            ),
            "covariance_type": [
                "spherical",
                "tied",
                "diag",
                "full",
            ],
        }
        score = BIC_Gaussian
    if model == "KMeans":
        model_class = KMeans()
        param_grid = {
            "n_clusters": range(
                2,
                9,
            ),
            "algorithm": [
                "lloyd",
                "elkan",
            ]
        }
        score = silhouette_score_grid_search
    return model_class, param_grid, score


params = get_params()
args = define_grid_search_params()
model, param_grid, score = get_grid_search_inputs(
    args.model,
)
dataset = DavisDataset(
    params,
    args.month,
)
data = dataset.get_data()
data_input = data.to_numpy()
grid_search = GridSearchCV(
    model,
    param_grid=param_grid,
    scoring=score,
    verbose=2,
)
grid_search.fit(
    data_input,
)
results = DataFrame(
    grid_search.cv_results_,
)
filename = f"{args.model}_grid_search.csv"
filename = join(
    params["results_path"],
    filename,
)
results.to_csv(
    filename,
)
