from modules.params import get_params
from matplotlib import pyplot
from pandas import (
    DataFrame,
    read_csv,
)


def read(
    model: str,
) -> DataFrame:
    filename = f"../results/{model}_grid_search.csv"
    data = read_csv(
        filename,
        index_col=0,
    )
    columns = list()
    if model == "Gaussian":
        columns = [
            "param_covariance_type",
            "param_n_components",
            "mean_test_score",
        ]
    if model == "KMeans":
        columns = [
            "param_algorithm",
            "param_n_clusters",
            "mean_test_score",
        ]
    data = data[columns]
    return data


params = get_params()
kmeans = read(
    "KMeans",
)
gaussian = read(
    "Gaussian"
)
fig, ax = pyplot.subplots()
covariance_types = sorted(
    list(
        set(
            gaussian["param_covariance_type"],
        )
    )
)
for i, covariance_type in enumerate(covariance_types):
    _data = gaussian[
        gaussian["param_covariance_type"] == covariance_type
    ]
    i -= len(covariance_types)//2
    print(_data)
    ax.bar(
        _data["param_n_components"]+0.2*i,
        _data["mean_test_score"],
        label=covariance_type,
        width=0.2
    )
pyplot.ylim(
    0,
    350000,
)
pyplot.legend(
    ncols=4,
)
pyplot.savefig(
    "gaussian.png"
)
pyplot.clf()
fig, ax = pyplot.subplots()
algorithm_types = sorted(
    list(
        set(
            kmeans["param_algorithm"],
        )
    )
)
for i, algorithm_type in enumerate(algorithm_types):
    _data = kmeans[
        kmeans["param_algorithm"] == algorithm_type
    ]
    i -= len(algorithm_types)//2
    print(_data)
    ax.bar(
        _data["param_n_clusters"]+0.2*i,
        _data["mean_test_score"],
        label=algorithm_type,
        width=0.2
    )
pyplot.ylim(
    0,
    180000,
)
pyplot.legend(
    ncols=2,
)
pyplot.savefig(
    "kmeans.png"
)
