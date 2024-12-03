from .functions import ls
from os.path import join
from typing import List
from pandas import (
    to_datetime,
    DataFrame,
    read_csv,
    concat,
)
from re import sub


class FireDataset:
    def __init__(
        self,
    ) -> None:
        self.data = self._read()

    def _read(
        self,
    ) -> DataFrame:
        data = read_csv(
            "../data/Fire/NI.csv",
            parse_dates=True,
            index_col=0,
        )
        return data

    def get_data(
        self,
    ) -> DataFrame:
        return self.data


class DavisDataset:
    def __init__(
        self,
        params: dict,
        month: int,
    ) -> None:
        self.data = self._read(
            params,
            month,
        )

    def _read(
        self,
        params: dict,
        month: int,
    ) -> DataFrame:
        folder = join(
            params["data_path"],
            "Davis",
        )
        files = ls(
            folder,
        )
        data = DataFrame()
        for file in files:
            filename = join(
                folder,
                file,
            )
            month_dataset = _MonthDavisDataset(
                filename,
            )
            self.direction_to_degree = month_dataset.direction_to_degree
            month_data = month_dataset.get_data()
            data = concat([
                data,
                month_data,
            ])
        if month == 0:
            data = data[
                (data.index.month >= 7) &
                (data.index.month <= 8)
            ]
            return data
        data = data[
            data.index.month == month
        ]
        return data

    def get_data(
        self,
    ) -> DataFrame:
        return self.data


class _MonthDavisDataset:
    def __init__(
        self,
        filename: str,
    ) -> None:
        self.direction_to_degree: dict = None
        self.data = self._read(
            filename,
        )

    def _read(
        self,
        filename: str,
    ) -> DataFrame:
        data = read_csv(
            filename,
            header=[0, 1],
            delimiter="\t"
        )
        columns = list(
            self._clean_header(
                header,
            )
            for header in data.columns
        )
        data.columns = columns
        data = self._format_index(
            data,
        )
        data = self._select_columns_to_use(
            data,
        )
        data = self._wind_direction_to_numerical(
            data,
        )
        data = self._remove_nan_values(
            data,
        )
        return data

    def _clean_header(
        self,
        header: List[str]
    ) -> str:
        header = " ".join(
            header,
        )
        header = sub(
            "Unnamed: [0-9]+_level_0",
            "",
            header,
        )
        header = sub(
            " +",
            "",
            header,
        )
        return header

    def _format_index(
        self,
        data: DataFrame,
    ) -> DataFrame:
        data["datetime"] = data.apply(
            lambda row:
            row["Date"]+" "+row["Time"],
            axis=1,
        )
        data.index = to_datetime(
            data["datetime"],
            format="%d/%m/%y %H:%M",
        )
        data = data.drop(
            columns=[
                "Date",
                "Time",
                "datetime",
            ]
        )
        return data

    def _select_columns_to_use(
        self,
        data: DataFrame,
    ) -> DataFrame:
        columns_to_use = [
            "WindSpeed",
            "WindDir",
            # "TempOut",
            "OutHum",
            "Rain",
            # "Bar",
        ]
        data = data[columns_to_use]
        return data

    def _wind_direction_to_numerical(
        self,
        data: DataFrame,
    ) -> DataFrame:
        self.direction_to_degree = {
            # "---": -180,
            "E": 0,
            "ENE": 22.5,
            "NE": 45,
            "NNE": 67.5,
            "N": 90,
            "NNW": 112.5,
            "NW": 135,
            "WNW": 157.5,
            "W": 180,
            "WSW": 202.5,
            "SW": 225,
            "SSW": 247.5,
            "S": 270,
            "SSE": 292.5,
            "SE": 315,
            "ESE": 337.5,
        }
        data = data[
            data["WindDir"] != "---"
        ]
        data.loc[:, "WindDir"] = data["WindDir"].map(
            lambda direction:
            self.direction_to_degree[direction]
        )
        return data

    def _remove_nan_values(
        self,
        data: DataFrame,
    ) -> DataFrame:
        data = data[
            data["OutHum"] != "---"
        ]
        data = data.astype(
            float,
        )
        return data

    def get_data(
        self,
    ) -> DataFrame:
        return self.data


class DavisDatasetComplete(
    DavisDataset
):
    def __init__(
        self,
        params: dict,
        month: int,
    ) -> None:
        super().__init__(
            params,
            month,
        )

    def _read(
        self,
        params: dict,
        month: int,
    ) -> DataFrame:
        folder = join(
            params["data_path"],
            "Davis",
        )
        files = ls(
            folder,
        )
        data = DataFrame()
        for file in files:
            filename = join(
                folder,
                file,
            )
            month_dataset = _MonthDavisDatasetComplete(
                filename,
            )
            self.direction_to_degree = month_dataset.direction_to_degree
            month_data = month_dataset.get_data()
            data = concat([
                data,
                month_data,
            ])
        if month == 0:
            data = data[
                (data.index.month >= 7) &
                (data.index.month <= 8)
            ]
            return data
        data = data[
            data.index.month == month
        ]
        return data


class _MonthDavisDatasetComplete(
    _MonthDavisDataset
):
    def __init__(
        self,
        filename: str,
    ) -> None:
        super().__init__(
            filename,
        )

    def _select_columns_to_use(
        self,
        data: DataFrame,
    ) -> DataFrame:
        columns_to_use = [
            "WindSpeed",
            "WindDir",
            # "TempOut",
            "OutHum",
            "Rain",
            # "Bar",
        ]
        data = data[columns_to_use]
        return data
