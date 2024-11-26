from argparse import (
    ArgumentParser,
    Namespace,
)
from typing import List
from os import listdir


def define_model_params(
) -> Namespace:
    args = ArgumentParser()
    args.add_argument(
        "--month",
        required=True,
        type=int,
    )
    args = args.parse_args()
    return args


def define_grid_search_params(
) -> Namespace:
    args = ArgumentParser()
    args.add_argument(
        "--month",
        required=True,
        type=int,
    )
    args.add_argument(
        "--model",
        required=True,
        type=str,
    )
    args = args.parse_args()
    return args


def ls(
    path: str,
) -> List[str]:
    files = listdir(
        path,
    )
    files = sorted(
        files,
    )
    return files
