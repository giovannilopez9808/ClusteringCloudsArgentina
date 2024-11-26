from modules.functions import define_grid_search_params
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from modules.Dataset import DavisDataset
from modules.Model import ClusterModel
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


def BIC_Kmeans(
    estimator: Callable,
    X: array,
) -> array:
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = estimator.cluster_centers_
    # labels = estimator.labels_
    labels = estimator.predict(
        X,
    )
    # number of clusters
    m = estimator.n_clusters
    # size of the clusters
    n = bincount(labels)
    # size of data set
    N, d = X.shape
    # labels = labels[:N]

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum(
        [
            sum(
                distance.cdist(
                    X[where(labels == i)],
                    [centers[i]],
                    'euclidean'
                )**2
            ) for i in range(m)
        ])

    const_term = 0.5 * m * log(N) * (d+1)

    BIC = sum(
        [
            n[i] * log(n[i]) -
            n[i] * log(N) -
            ((n[i] * d) / 2) * log(2*pi*cl_var) -
            ((n[i] - 1) * d / 2) for i in range(m)
        ]
    ) - const_term

    return -BIC


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
                9,
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
        score = BIC_Kmeans
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
