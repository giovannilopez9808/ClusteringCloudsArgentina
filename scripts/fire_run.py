from modules.Model import ClusterModel
from modules.params import get_params
from modules.functions import define_model_params
from matplotlib import pyplot
from pandas import concat, to_datetime
from modules.Dataset import (
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
davis_data = dataset.get_data()
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
davis_data["Date"] = to_datetime([
    date.date()
    for date in davis_data.index

])
fire_data = FireDataset()
fire_data = fire_data.get_data()
fire_data["Date"] = fire_data.index.date
model = ClusterModel(
    args.month
)
model.load()
prediction = model.run(
    data_input,
)
davis_data["Cluster"] = prediction
data = davis_data.join(
    fire_data,
    rsuffix="_b",
    on="Date",
)
statistics = data.groupby("Cluster").describe()
columns_to_check = [
    "TempOut",
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
