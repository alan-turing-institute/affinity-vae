import logging
import os.path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import yaml
from pydantic import (
    BaseModel,
    FilePath,
    NonNegativeInt,
    PositiveInt,
    conbool,
    confloat,
    constr,
)
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

from . import config


def accuracy(x_train, y_train, x_val, y_val, classifier="NN"):
    """Computes the accuracy using a given classifier. Currently only supports
    neural network, K-nearest neighbors and logistic regression. A grid search on the
    hyperparameters is performed.

    Parameters
    ----------
    x_train: np.array
        Training data.
    y_train: np.array
        Training labels.
    x_val: np.array
        Validation data.
    y_val: np.array
        Validation labels.
    classifier: str
        Classifier to use. Either 'NN' for neural network, 'KNN' for K-nearest neighbors or LR for logistic regression.


    Returns
    -------
    train_acc: float
        Training accuracy.
    val_acc: float
        Validation accuracy.
    val_acc_selected: float
        Validation accuracy calculated only for labels existing in the training set (this is useful in evaluation).
    y_pred_train: np.array
        Predicted training labels.
    y_pred_val: np.array
        Predicted validation labels.

    """
    logging.info(
        "############################################### Computing accuracy..."
    )
    labs = np.unique(np.concatenate((y_train, y_val)))
    le = preprocessing.LabelEncoder()
    le.fit(labs)

    classes_list_training = np.unique(y_train)
    if np.setdiff1d(classes_list_training, np.unique(y_val)).size > 0:
        logging.info(
            f"Class {np.setdiff1d(classes_list_training, np.unique(y_val))}  was unseen in training data. Computing accuracy for sets of seen and unseen data"
        )

    index = np.argwhere(np.isin(y_val, classes_list_training)).ravel()

    y_train = le.transform(y_train)
    y_val = le.transform(y_val)

    if classifier == "NN":

        parameters = {
            "hidden_layer_sizes": [
                (250, 150, 30),
                (100, 50, 15),
                (50, 20, 10),
                (20, 10, 5),
                (200,),
                (50,),
            ],
        }
        method = MLPClassifier(
            max_iter=500,
            activation="relu",
            solver="adam",
            random_state=1,
            alpha=1,
        )

    elif classifier == "LR":
        method = LogisticRegression(
            random_state=0, multi_class="multinomial", max_iter=1000
        )

        parameters = [{"penalty": ["l1", "l2"]}, {"C": [1, 10, 100, 1000]}]

    elif classifier == "KNN":
        parameters = dict(n_neighbors=list(range(1, 500, 100)))
        method = KNeighborsClassifier()
    else:
        raise ValueError("Invalid classifier type must be NN, KNN or LR")

    clf_cv = GridSearchCV(
        estimator=method,
        param_grid=parameters,
        scoring="f1_macro",
        cv=2,
        verbose=0,
    )
    clf = make_pipeline(preprocessing.StandardScaler(), clf_cv)
    clf.fit(x_train, y_train)
    logging.info(
        f"Best parameters found for {classifier}: {clf_cv.best_params_}"
    )

    y_pred_train = clf.predict(x_train)
    y_pred_val = clf.predict(x_val)

    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    val_acc = metrics.accuracy_score(y_val, y_pred_val)
    val_acc_selected = metrics.accuracy_score(
        np.array(y_val)[index].tolist(), np.array(y_pred_val)[index].tolist()
    )

    y_pred_train = le.inverse_transform(y_pred_train)
    y_pred_val = le.inverse_transform(y_pred_val)

    return train_acc, val_acc, val_acc_selected, y_pred_train, y_pred_val


def create_grid_for_plotting(rows, columns, dsize, padding=0):

    # define the dimensions for the napari grid

    if len(dsize) == 3:
        grid_for_napari = np.zeros(
            (
                rows * dsize[0],
                dsize[1] * columns + padding * columns,
                dsize[2],
            ),
            dtype=np.float32,
        )

    elif len(dsize) == 2:
        grid_for_napari = np.zeros(
            (
                rows * dsize[0],
                dsize[1] * columns + padding * columns,
            ),
            dtype=np.float32,
        )

    return grid_for_napari


def fill_grid_for_plottting(rows, columns, grid, dsize, array, padding=0):

    if len(dsize) == 3:
        for j in range(columns):
            for i in range(rows):
                grid[
                    i * dsize[0] : (i + 1) * dsize[0],
                    j * (dsize[1] + padding) : (j + 1) * dsize[1]
                    + padding * j,
                    :,
                ] = array[i, j, :, :, :]

    elif len(dsize) == 2:
        for j in range(columns):
            for i in range(rows):
                grid[
                    i * dsize[0] : (i + 1) * dsize[0],
                    j * (dsize[1] + padding) : (j + 1) * dsize[1]
                    + padding * j,
                ] = array[i, j, :, :]
    return grid


def save_imshow_png(
    fname,
    array,
    cmap=None,
    min=None,
    max=None,
    writer=None,
    figname=None,
    epoch=0,
):
    if not os.path.exists("plots"):
        os.mkdir("plots")

    fig, _ = plt.subplots(figsize=(10, 10))
    plt.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last

    plt.savefig("plots/" + fname)

    if writer:
        writer.add_figure(figname, fig, epoch)

    plt.close()


def save_mrc_file(fname, array):
    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/" + fname, overwrite=True) as mrc:
        mrc.set_data(array)


def colour_per_class(classes: list):
    # Define the number of colors you want
    num_colors = len(classes)

    # Choose colormaps for combining
    cmap_1 = plt.get_cmap("tab20")
    cmap_2 = plt.get_cmap("Accent")
    cmap_3 = plt.get_cmap("Pastel1")
    cmap_4 = plt.get_cmap("Set1")

    # Combine the four colormaps
    combined_cmap = [cmap_1(i % 20) for i in range(20)]
    combined_cmap.extend([cmap_2(i % 8) for i in range(8)])
    combined_cmap.extend([cmap_3(i % 8) for i in range(8)])
    combined_cmap.extend([cmap_4(i % 8) for i in range(8)])

    # Create the colormap object
    custom_cmap = plt.cm.colors.ListedColormap(
        combined_cmap, name="custom_cmap"
    )

    # Generate a list of colors based on the modulo operation of i with respect to the number of colors in the combined colormap
    colours = [custom_cmap(i % len(combined_cmap)) for i in range(num_colors)]
    return colours
