from modules.functions import define_model_params
from modules.Model import ClusterModel
from modules.params import get_params
from matplotlib import pyplot
from modules.Dataset import (
    DavisDataset,
)
from numpy import min, max, cos, sin, pi

args = define_model_params()
params = get_params()
dataset = DavisDataset(
    params,
    args.month,
)
davis_data = dataset.get_data()
print(davis_data.mean())
# davis_data = davis_data.resample("D").mean()
# davis_data = davis_data.dropna()
data_input = davis_data.to_numpy()
# mean_ = min(
# data_input,
# axis=0,
# )
# std_ = max(
# data_input,
# axis=0,
# )
# data_input = (data_input-mean_)/(std_-mean_)
model = ClusterModel(
    args.month,
)
model.load()
prediction = model.run(
    data_input,
)
davis_data["Cluster"] = prediction

for cluster in range(
    model.n_clusters,
):
    _data = davis_data[
        davis_data["Cluster"] == cluster
    ]
    _data = _data.dropna()
    _x = _data["WindSpeed"]*cos(_data["WindDir"]*pi/180)
    _y = _data["WindSpeed"]*sin(_data["WindDir"]*pi/180)
    print(_x.max())
    pyplot.scatter(
        _x,
        _y,
        label=cluster,
    )
pyplot.xlim(
    -25,
    35,
)
pyplot.xticks(
    range(
        -25,
        40,
        5,
    )
)
pyplot.ylim(
    -30,
    25,
)
pyplot.yticks(
    range(
        -30,
        30,
        5,
    )
)
pyplot.xlabel(
    "Componente X de la velocidad del viento (m/s)",
)
pyplot.ylabel(
    "Componente Y de la velocidad del viento (m/s)",
)
pyplot.legend(
    ncol=model.n_clusters,
    frameon=False,
)
pyplot.tight_layout()
pyplot.savefig(
    "test.png"
)