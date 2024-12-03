from modules.Model import ClusterModel
from modules.params import get_params
from modules.functions import define_model_params
from matplotlib import pyplot
from pandas import concat, to_datetime
from modules.Dataset import (
    DavisDatasetComplete,
    DavisDataset,
    FireDataset,
)
from numpy import min, max

args = define_model_params()
params = get_params()
dataset = DavisDataset(
    params,
    args.month,
)
dataset_complete = DavisDatasetComplete(
    params,
    args.month,
)
davis_data_complete = dataset_complete.get_data()
davis_data = dataset.get_data()
data_input = davis_data.to_numpy()
davis_data_complete["Date"] = to_datetime([
    date.date()
    for date in davis_data.index

])
fire_data = FireDataset()
fire_data = fire_data.get_data()
fire_data["Date"] = fire_data.index.date
fire_data = fire_data[
    (fire_data.index.month >= 7) &
    (fire_data.index.month <= 8)
]
model = ClusterModel(
    args.month
)
model.load()
prediction = model.run(
    data_input,
)
davis_data_complete["Cluster"] = prediction
data = davis_data_complete.join(
    fire_data,
    rsuffix="_b",
    on="Date",
)
statistics = data.groupby("Cluster").describe()
columns_to_check = [
    # "TempOut",
    "OutHum",
    "Rain",
    "NI",
]
for column in columns_to_check:
    if column != "Cluster":
        print("="*30)
        print(column)
        print(statistics[column])
print("="*30)
for cluster in range(
    model.n_clusters,
):
    _data = data[
        data["Cluster"] == cluster
    ]
    _data = _data.dropna()
    pyplot.boxplot(
        _data["NI"],
        positions=[cluster],
        # whis=(25, 75),
        patch_artist=True,
        showfliers=False,
    )
pyplot.show()
