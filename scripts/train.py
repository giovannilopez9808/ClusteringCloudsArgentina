from modules.functions import define_model_params
from modules.Dataset import DavisDataset
from modules.Model import ClusterModel
from modules.params import get_params

params = get_params()
args = define_model_params()
dataset = DavisDataset(
    params,
    args.month,
)
data = dataset.get_data()
# data = data.resample("D").mean()
# data = data.dropna()
data_input = data.to_numpy()
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
model.train(
    data_input,
)
model.save()
