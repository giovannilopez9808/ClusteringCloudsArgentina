from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from numpy import array
from pickle import (
    dump,
    load,
)


class ClusterModel:
    def __init__(
        self,
        month: int,
    ) -> None:
        self.model: KMeans = None
        self.n_clusters = 8
        self.month = month

    def _define_model(
        self,
    ) -> KMeans:
        model = KMeans(
            n_clusters=self.n_clusters,
        )
        return model

    def train(
        self,
        data: array
    ) -> KMeans:
        model = self._define_model()
        self.model = model.fit(
            data,
        )

    def save(
        self,
    ) -> None:
        filename = self._get_path()
        file = open(
            filename,
            "wb",
        )
        dump(
            self.model,
            file,
        )
        file.close()

    def load(
        self,
    ) -> KMeans:
        filename = self._get_path()
        file = open(
            filename,
            "rb"
        )
        self.model = load(
            file,
        )
        file.close()

    def run(
        self,
        data: array,
    ) -> array:
        if self.model is None:
            text = "="*30
            text += "\nModelo no cargado\n"
            text += "="*30
            raise RuntimeError(text)
        predict = self.model.predict(
            data,
        )
        return predict

    def _get_path(
        self,
    ) -> str:
        filename = str(
            self.month
        )
        filename = filename.zfill(
            2
        )
        filename = f"../model/{filename}.pkl"
        return filename
