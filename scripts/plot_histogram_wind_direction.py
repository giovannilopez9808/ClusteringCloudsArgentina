from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from modules.functions import define_model_params
from modules.Dataset import DavisDataset
from modules.params import get_params
from matplotlib import pyplot, cm
from numpy import linspace
from pandas import (
    to_timedelta,
    to_datetime,
    DataFrame,
    cut,
)
from numpy import (
    concatenate,
    array,
    pi
)


args = define_model_params()
params = get_params()
dataset = DavisDataset(
    params,
    args.month,
)
data = dataset.get_data()
data = data[
    data.index.date == to_datetime("2020-07-31").date()
]
fig, ax = pyplot.subplots(
    figsize=(
        10, 10
    ),
    subplot_kw=dict(
        projection='polar'
    ),
)
bins = linspace(
    -180,
    180,
    17,
)
results = DataFrame(
    index=bins[:-1]+180,
    columns=["Count"]
)
for inf, sup in zip(
    bins[:-1],
    bins[1:],
):
    _data = data[
        (data["WindDir"] >= inf+180) &
        (data["WindDir"] < sup+180)
    ]
    count = _data["WindDir"].count()
    results.loc[inf+180] = count
bins = bins[:-1]
ax.bar(
    (bins+180)*pi/180,
    results["Count"],
    # align="edge",
    color="#ff8fab",
    width=0.3,
)
ax.set_ylim(
    0,
    30,
)
ax.set_yticks(
    range(
        0,
        30,
        5,
    )
)
ax.set_xticks(
    (bins+180)*pi/180,
)
ax.set_xticklabels(
    dataset.direction_to_degree.keys()
)
ax.tick_params(
    labelsize=25,
    pad=30,
)
fig.tight_layout(
    pad=4,
)
pyplot.savefig(
    "test.png"
)
