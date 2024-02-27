import logging
import os
import random
import typing

import mrcfile
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import zoom
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from . import settings
from .vis import format, plot_affinity_matrix, plot_classes_distribution

np.random.seed(42)
from typing import Any, Literal, overload

from caked.dataloader import DiskDataLoader


@overload
def load_data(
    datapath: str,
    datatype: str,
    eval: Literal[True],
    lim: int | None = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    affinity: str | None = None,
    classes: str | None = None,
    gaussian_blur: bool = False,
    normalise: bool = False,
    shift_min: bool = False,
    rescale: bool | None = None,
) -> tuple[DataLoader, int]:
    ...


@overload
def load_data(
    datapath: str,
    datatype: str,
    eval: Literal[False],
    lim: int | None = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    affinity: str | None = None,
    classes: str | None = None,
    gaussian_blur: bool = False,
    normalise: bool = False,
    shift_min: bool = False,
    rescale: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, int]:
    ...


def load_data(
    datapath: str,
    datatype: str,
    eval: bool,
    lim: int | None = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    affinity: str | None = None,
    classes: str | None = None,
    gaussian_blur: bool = False,
    normalise: bool = False,
    shift_min: bool = False,
    rescale: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, int] | tuple[
    DataLoader, int
]:
    """Loads all data needed for training, testing and evaluation. Loads MRC files from a given path, selects subset of
    classes if requested, splits it into train / val  and test in batch sets, loads affinity matrix. Returns train,
    validation and test data as DataLoader objects.

    Parameters
    ----------
    datapath: str
        Path to the data directory.
    datatype: str
        data file formats : mrc, npy
    lim: int
        Limit the number of samples to load.
    splt: int
        Percentage of data to be used for validation.
    batch_s: int
        Batch size.
    no_val_drop: bool
        If True, the last batch of validation data will not be dropped if it is smaller than batch size.
    eval: bool
        If True, the data will be loaded only for evaluation.
    affinity: str
        Path to the affinity matrix.
    classes: list
        List of classes to be selected from the data.
    gaussian_blur: bool
        if True, Gaussian bluring is applied to the input before being passed to the model.
        This is added as a way to remove noise from the input data.
    normalise:
        In True, the input data is normalised before being passed to the model.
    shift_min: bool
        If True, the minimum value of the input data is shifted to 0 and maximum to 1.


    Returns
    -------
    train_data: DataLoader
        Train data, returned only if eval is False.
    val_data: DataLoader
        Validation data, returned only if eval is False.
    test_data: DataLoader
        Test data, if eval is True only test data is returned.
    lookup: pd.DataFrame
        Affinity matrix, returned only if eval is False.
    """
    if classes is not None:
        classes_list = pd.read_csv(classes).columns.tolist()
    else:
        classes_list = []

    if not eval:
        if affinity is not None:

            affinity = get_affinity_matrix(affinity, classes_list)

        transformations = []
        if gaussian_blur:
            transformations.append('gaussian_blur')
        if normalise:
            transformations.append('normalise')
        if shift_min:
            transformations.append('shift_min')
        if rescale is not None:
            transformations.append(f'rescale={rescale}')

        loader = DiskDataLoader(
            pipeline="disk",
            classes=classes_list,
            dataset_size=lim,
            training=True,
            transformations=(
                None if len(transformations) == 0 else transformations
            ),
        )

        loader.load(datapath=datapath, datatype=datatype)

        data = loader.data

        trains, vals = loader.get_loader(batch_size=batch_s, split=splt)

        # ################# Visualising class distribution ###################

        train_y: list = []
        val_y: list = []

        if settings.VIS_HIS:
            plot_classes_distribution(train_y, "train")
            plot_classes_distribution(val_y, "validation")

        logging.info("############################################### DATA")
        logging.info("Data size: {}".format(len(data)))
        logging.info("Class list: {}".format(data.classes))
        logging.info(
            "Train / val split: {}, {}".format(len(train_y), len(val_y))
        )
        logging.info(
            "Train / val batches: {}, {}\n".format(len(trains), len(vals))
        )

    if eval or ("test" in os.listdir(datapath)):
        if "test" in os.listdir(datapath):
            datapath = os.path.join(datapath, "test")

        test_loader = DiskDataLoader(
            pipeline="disk",
            classes=[],
            dataset_size=lim,
            training=False,
            training_path=transformations,
        )

        test_loader.load(datapath=datapath, datatype=datatype)

        data = test_loader.data
        tests = test_loader.get_loader(batch_size=batch_s, split=splt)

        logging.info("############################################### EVAL")
        logging.info("Eval data size: {}".format(len(data)))
        logging.info("Eval batches: {}\n".format(len(tests)))

    if eval:
        return tests, data.dim()
    else:
        return trains, vals, tests, affinity, data.dim()  # , dsize


def get_affinity_matrix(
    affinity_path: str | None, classes: list = []
) -> pd.DataFrame:
    """Loads affinity matrix from a given path, subsets it given selected classes and returns it as a pandas DataFrame.

    Parameters
    ----------
    affinity: str
        Path to the affinity matrix.
    classes: list
        List of classes to be selected from the data.

    Returns
    -------
    affinity: pd.DataFrame
        Affinity matrix.
    """
    if affinity_path is not None:
        # load affinity matrix
        affinity = pd.read_csv(affinity_path, header=0)
    else:
        affinity = None

    if affinity is not None:
        class_check = np.in1d(classes, affinity.columns)
        if not np.all(class_check):
            raise RuntimeError(
                "Not all classes in the training set are present in the "
                "affinity matrix. Missing classes: {}".format(
                    np.asarray(classes)[~class_check]
                )
            )
        if settings.VIS_AFF:
            plot_affinity_matrix(
                lookup=affinity,
                all_classes=affinity.columns.tolist(),
                selected_classes=classes,
            )

        # subset affinity matrix with only the relevant classes
        index = [affinity.columns.get_loc(f"{columns}") for columns in classes]
        sub_affinity = affinity.iloc[index, index]

    return sub_affinity.to_numpy(dtype=np.float32)
