import logging
import os
from pathlib import Path
from typing import Literal, overload

import lightning as lt
import numpy as np
import pandas as pd
from caked.dataloader import DiskDataLoader, DiskDataset
from torch.utils.data import DataLoader

from . import settings
from .vis import format, plot_affinity_matrix, plot_classes_distribution

lt.pytorch.seed_everything(42)


# As the function load_data can return different types of data depending on the value of the eval parameter, it uses
# function overloading to define multiple signatures for the function. This allows the function to have different return
# types and behaviors based on the input parameters.
@overload
def load_data(
    datapath: str,
    datatype: str,
    eval: Literal[True],
    fabric: lt.fabric.fabric,
    lim: int | None = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    affinity_path: str | None = None,
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
    fabric: lt.fabric.fabric,
    lim: int | None = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    affinity_path: str | None = None,
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
    fabric: lt.fabric.fabric,
    lim: int | None = None,
    splt: int = 20,
    batch_s: int = 64,
    no_val_drop: bool = False,
    affinity_path: str | None = None,
    classes: str | None = None,
    gaussian_blur: bool = False,
    normalise: bool = False,
    shift_min: bool = False,
    rescale: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, int] | tuple[
    DataLoader, int
]:
    """This function a wrapper around the DiskDataLoader class from the caked library. It loads data from a given path, selects a subset of classes if requested, splits it into train / val and test in batch sets, and loads an affinity matrix.
    The function is overloaded to return different types of data depending on the value of the eval parameter. If eval is True, the function returns only the test data and the dimension of the data. If eval is False, the function returns train, validation, and test data, the affinity matrix, and the dimension of the data.

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
    affinity_path: str
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
    rescale: int
        If not None, the input data is rescaled to the given value.


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

    # read the class list, if not provided all classes in the dataset will be used as default
    if classes is not None:
        classes_list = pd.read_csv(classes).columns.tolist()
    else:
        classes_list = []

    # get transformations into a list
    transformations = []
    if gaussian_blur:
        transformations.append('gaussianblur')
    if normalise:
        transformations.append('normalise')
    if shift_min:
        transformations.append('shiftmin')
    if rescale is not None:
        transformations.append(f'rescale={rescale}')

    if not eval:

        # configure the DiskDataLoader from the caked library with the data parameters
        # the caked library is used to load data from the given path, split it into train and validation, apply transformations, and get torch dataloaders
        # the caked implementation used to be a part of the avae library, but it was moved to the caked library to make it more modular and reusable
        # you can find the caked library here: https://github.com/alan-turing-institute/caked/
        loader = DiskDataLoader(
            pipeline="disk",
            classes=classes_list,
            dataset_size=lim,
            training=True,
            transformations=(
                None if len(transformations) == 0 else transformations
            ),
        )
        # using the caked library we load data from the given path
        loader.load(datapath=datapath, datatype=datatype)

        # if classes are not provided, use all classes in the dataset as obtained by the dataloader
        if len(classes_list) == 0:
            classes_list = loader.classes

        # for training, we need to load the affinity matrix
        if affinity_path is not None:
            affinity = get_affinity_matrix(affinity_path, classes_list)

        # assign the affinity matrix to the dataset (small modification from the caked DiskDataset, which only returns data and labels, and we need the affinity matrix indexes for training).
        loader.dataset = AffinityDiskDataset(
            dataset=loader.dataset, affinity=affinity, classes=classes_list
        )

        # using caked, we split the data into train and validation and get torch dataloaders
        trains, vals = loader.get_loader(
            batch_size=batch_s, split_size=splt, no_val_drop=no_val_drop
        )

        trains = fabric.setup_dataloaders(trains)
        vals = fabric.setup_dataloaders(vals)
        # ################# Visualising class distribution ###################

        # getting labels from dataloaders
        train_y = list(sum([y[1] for _, y in enumerate(trains)], ()))
        val_y = list(sum([y[1] for _, y in enumerate(vals)], ()))

        if settings.VIS_HIS:
            plot_classes_distribution(train_y, "train")
            plot_classes_distribution(val_y, "validation")

        logging.info("############################################### DATA")
        logging.info("Data size: {}".format(len(loader.dataset)))
        logging.info("Class list: {}".format(classes_list))
        logging.info(
            "Train / val split: {}, {}".format(len(train_y), len(val_y))
        )
        logging.info(
            "Train / val batches: {}, {}\n".format(len(trains), len(vals))
        )

    tests = []
    if eval or ("test" in os.listdir(datapath)):
        if "test" in os.listdir(datapath):
            datapath = os.path.join(datapath, "test")

        # configure the caked library dataloader with the given parameters for test or evaluation
        test_loader = DiskDataLoader(
            pipeline="disk",
            classes=[],
            dataset_size=lim,
            training=False,
            transformations=(
                None if len(transformations) == 0 else transformations
            ),
        )

        # load data from the given path
        test_loader.load(datapath=datapath, datatype=datatype)

        # assign the affinity matrix of None to the test dataset (this is only for test or evaluation)
        test_loader.dataset = AffinityDiskDataset(
            dataset=test_loader.dataset,
            classes=test_loader.classes,
            affinity=None,
        )

        # get torch dataloader
        tests = test_loader.get_loader(batch_size=batch_s)
        tests = fabric.setup_dataloaders(tests)

        logging.info("############################################### EVAL")
        logging.info("Eval data size: {}".format(len(test_loader.dataset)))
        logging.info("Eval batches: {}\n".format(len(tests)))

    if eval:
        return tests, test_loader.dataset.dim()
    else:
        return (
            trains,
            vals,
            tests,
            affinity.to_numpy(dtype=np.float32),
            loader.dataset.dim(),
        )  # , dsize


def get_affinity_matrix(
    affinity_path: str, classes: list = []
) -> pd.DataFrame:
    """Loads affinity matrix from a given path, subsets it given selected classes and returns it as a pandas DataFrame.

    Parameters
    ----------
    affinity: str | None
        Path to the affinity matrix if provided .
    classes: list
        List of classes to be selected from the data.

    Returns
    -------
    affinity: pd.DataFrame or None if no affinity matrix path is provided
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

        return sub_affinity

    else:
        return None


class AffinityDiskDataset(DiskDataset):

    """Modified version of the caked DiskDataset to include the affinity matrix and data metadata that is needed for the
    affinity pipeline"""

    def __init__(
        self,
        dataset: DiskDataset,
        classes: list,
        affinity: pd.DataFrame | None = None,
    ):
        super().__init__(
            paths=dataset.paths,
            datatype=dataset.datatype,
            rescale=dataset.rescale,
            shiftmin=dataset.shiftmin,
            normalise=dataset.normalise,
            gaussianblur=dataset.gaussianblur,
        )
        self.affinity = affinity
        self.classes = classes

    def __getitem__(self, index):

        # read data from path
        data = np.array(self.read(self.paths[index]))
        x = self.transformation(data)

        # get file basename
        filename = Path(self.paths[index]).name
        # ground truth
        y = Path(filename).name.split("_")[0]

        # similarity column / vector
        if self.affinity is not None:
            aff = self.affinity.columns.get_loc(f"{y}")
        else:
            # in evaluation mode - test set
            aff = 0  # cannot be None, but not used anywhere during evaluation

        # file info and metadata
        meta = "_".join(filename.split(".")[0].split("_")[1:])
        avg = np.around(np.average(x), decimals=4)
        img = format(x, len(data.shape))  # used for dynamic preview in Altair
        meta = {
            "filename": filename,
            "id": y,
            "meta": meta,
            "avg": avg,
            "image": img,
        }
        return x, y, aff, meta
