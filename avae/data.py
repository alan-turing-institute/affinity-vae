import logging
import os
import random
import typing

import mrcfile   # mrc files are how the proteins are stored in, we won't use
import numpy as np
import numpy.typing as npt # type annotation to enforce types
import pandas as pd
from scipy.ndimage import zoom # use for proteins, zooming on images
from torch import Tensor # main ML library
from torch.utils.data import DataLoader, Dataset, Subset # torch specific data objects, e.g. 'train' is a torch Dataloader
from torchvision import transforms # for rescaling, centering

from avae import settings # custom, e.g. interval of epochs to do a visualisation
from avae.vis import format, plot_affinity_matrix, plot_classes_distribution # plots, class dist., aff. matrix

np.random.seed(42)
from typing import Any, Literal, overload # type annotation, function overloading


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
    datapath: str, # required
    datatype: str, # required, it would be 'npy'
    eval: bool, # required
    lim: int | None = None,
    splt: int = 20,
    batch_s: int = 64, # usually 16,32,64
    no_val_drop: bool = False, # if data is not a multiple of batch size
    affinity: str | None = None,
    classes: str | None = None, # list of classes to be selected from the data (e.g. exclude a class as a test class)
                                # class names are pre-fixed to the sample file names, with underscore
                                # must match the affinity matrix column / row names = classes
    gaussian_blur: bool = False, # no required
    normalise: bool = False, # normalise to between 0 and 1? (check in code)
    shift_min: bool = False, # min / max normalisation
    rescale: bool | None = None, # another transformation - look up what required
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
## note: we could change the DataLoader object for omics

    if not eval:
        if affinity is not None: # only load the affinity matrix if no eval
            # load affinity matrix
            lookup = pd.read_csv(affinity, header=0)
        else:
            lookup = None

        # create ProteinDataset
        data = Dataset_reader(
            datapath,
            amatrix=lookup,
            classes=classes,
            gaussian_blur=gaussian_blur,
            normalise=normalise,
            shift_min=shift_min,
            rescale=rescale,
            lim=lim,
            datatype=datatype,
        )

        # ################# Visualising affinity matrix ###################
        if affinity is not None and settings.VIS_AFF:
            plot_affinity_matrix(
                lookup=lookup,
                all_classes=lookup.columns.tolist(),
                selected_classes=data.final_classes,
            )

        # updating affinity matrix with the final classes
        lookup = data.amatrix                                   # data has the affinity matrix for the classes we want

        # split into train / val sets
        idx = np.random.permutation(len(data))                     # create index for train / val. Could stratify the index...
        s = int(np.ceil(len(data) * int(splt) / 100))
        if s < 2:
            raise RuntimeError(
                "Train and validation sets must be larger than 1 sample, "
                "train: {}, val: {}.".format(len(idx[:-s]), len(idx[-s:]))
            )
        train_data = Subset(data, indices=idx[:-s])                 # subset based on train / val idx
        val_data = Subset(data, indices=idx[-s:])

        # ################# Visualising class distribution ###################

        train_y = [y[1] for _, y in enumerate(train_data)]             # get the labels of the training data
        val_y = [y[1] for _, y in enumerate(val_data)]                  # labels for the val set

        if settings.VIS_HIS:
            plot_classes_distribution(train_y, "train")
            plot_classes_distribution(val_y, "validation")

        # split into batches                                            # creating the torch data loader object in torch, batched
        trains = DataLoader(
            train_data,
            batch_size=batch_s,
            num_workers=0,
            shuffle=True,
            drop_last=True,             # drop data left over after last batch
        )
        vals = DataLoader(
            val_data,
            batch_size=batch_s,
            num_workers=0,              # for parallelisation, doesn't work on mac
            shuffle=True,
            drop_last=(not no_val_drop), # can leave end data in, as just evaluating
        )
        tests = []
        if len(vals) < 1 or len(trains) < 1:
            # ensure the batch size is not smaller than validation set
            raise RuntimeError(
                "Validation or train set is too small for the current batch "
                "size. Please edit either split percent '-sp/--split' or batch"
                " size '-ba/--batch' or set '-nd/--no_val_drop flag' (only if "
                "val is too small). Batch: {}, train: {}, val: {}, "
                "split: {}%.".format(
                    batch_s, len(train_data), len(val_data), splt
                )
            )
        logging.info("############################################### DATA")
        logging.info("Data size: {}".format(len(data)))
        logging.info("Class list: {}".format(data.final_classes))
        logging.info(
            "Train / val split: {}, {}".format(len(train_data), len(val_data))
        )
        logging.info(
            "Train / val batches: {}, {}\n".format(len(trains), len(vals))
        )

        if affinity is not None:                                # make affinity matrix a np file
            lookup = lookup.to_numpy(dtype=np.float32)
        else:
            lookup = None

    if eval or ("test" in os.listdir(datapath)):                # legacy, replaced by evaluation script
        if "test" in os.listdir(datapath):
            datapath = os.path.join(datapath, "test")
        data = Dataset_reader(
            datapath,
            gaussian_blur=gaussian_blur,
            normalise=normalise,
            shift_min=shift_min,
            rescale=rescale,
            lim=lim,
            datatype=datatype,
        )

        logging.info("############################################### EVAL")
        logging.info("Eval data size: {}".format(len(data)))
        tests = DataLoader(
            data, batch_size=batch_s, num_workers=0, shuffle=True
        )
        logging.info("Eval batches: {}\n".format(len(tests)))

    if eval:                                                                # defining what is returned
        return tests, data.dim()
    else:
        return trains, vals, tests, lookup, data.dim()  # , dsize


class Dataset_reader(Dataset):
    def __init__(
        self,
        root_dir: str,
        amatrix: npt.NDArray | None = None,
        classes: str | None = None,
        transform: typing.Any = None,
        gaussian_blur: bool = False,
        normalise: bool = False,
        shift_min: bool = False,
        rescale: bool | None = None,
        lim: int | None = None,
        datatype: str = "mrc",
    ):
        super().__init__()
        self.datatype = datatype
        self.shift_min = shift_min
        self.normalise = normalise
        self.gaussian_blur = gaussian_blur
        self.rescale = rescale
        self.transform = transform
        self.amatrix = amatrix
        self.root_dir = root_dir

        self.paths = [
            f for f in os.listdir(root_dir) if "." + self.datatype in f   # creat list of all the paths in the dir
        ]

        random.shuffle(self.paths)                                  # randomize for the split
        ids = np.unique([f.split("_")[0] for f in self.paths])      # split out the classes
        self.final_classes = ids
        if classes is not None:                                      # only use the selected classes
            classes_list = pd.read_csv(classes).columns.tolist()
            self.final_classes = classes_list

        if self.amatrix is not None:
            class_check = np.in1d(self.final_classes, self.amatrix.columns)     # subset the classes in the aff matrix
            if not np.all(class_check):
                raise RuntimeError(
                    "Not all classes in the training set are present in the "
                    "affinity matrix. Missing classes: {}".format(
                        np.asarray(ids)[~class_check]
                    )
                )

            # subset affinity matrix with only the relevant classes
            index = [
                self.amatrix.columns.get_loc(f"{columns}")
                for columns in self.final_classes
            ]
            self.amatrix = self.amatrix.iloc[index, index]

        self.paths = [                                              # update the paths to the relevant classes
            p
            for p in self.paths
            for c in self.final_classes
            if c in p.split("_")[0]
        ]

        self.paths = self.paths[:lim]                               # only load the paths up to the limit

    def __len__(self):
        return len(self.paths)

    def dim(self):
        return len(np.array(self.read(self.paths[0])).shape)

    def __getitem__(self, item):
        filename = self.paths[item]

        data = np.array(self.read(filename))
        x = self.voxel_transformation(data)          # x is the numpy array

        # ground truth
        y = filename.split("_")[0]                  # get the ground truth class

        # similarity column / vector
        if self.amatrix is not None:
            aff = self.amatrix.columns.get_loc(f"{y}")          # get the column for this class in the aff matrix
        else:
            # in evaluation mode - test set
            aff = 0  # cannot be None, but not used anywhere during evaluation

        # file info and metadata
        meta = "_".join(filename.split(".")[0].split("_")[1:])              # creating a dataframe to downstream analysis
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

    def read(self, filename):                           # read function based on file type

        if self.datatype == "npy":
            return np.load(os.path.join(self.root_dir, filename))

        elif self.datatype == "mrc":
            with mrcfile.open(os.path.join(self.root_dir, filename)) as f:
                return np.array(f.data)

    def voxel_transformation(self, x):                      # define which scaling method we want

        if self.rescale:
            x = np.asarray(x, dtype=np.float32)
            sh = tuple([self.rescale / s for s in x.shape])
            x = zoom(x, sh)

        # convert numpy to torch tensor
        x = Tensor(x)

        # unsqueeze adds a dimension for batch processing the data
        x = x.unsqueeze(0)

        if self.shift_min:
            x = (x - x.min()) / (x.max() - x.min())

        if self.gaussian_blur:
            T = transforms.GaussianBlur(3, sigma=(0.08, 10.0))
            x = T(x)

        if self.normalise:
            T = transforms.Normalize(0, 1, inplace=False)
            x = T(x)

        if self.transform:
            x = self.transform(x)
        return x




