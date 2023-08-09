import os.path

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import yaml
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
    y_pred_train: np.array
        Predicted training labels.
    y_pred_val: np.array
        Predicted validation labels.

    """
    labs = np.unique(np.concatenate((y_train, y_val)))
    le = preprocessing.LabelEncoder()
    le.fit(labs)

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
                (10,),
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
        parameters = dict(n_neighbors=list(range(1, 500, 10)))
        method = KNeighborsClassifier()
    else:
        raise ValueError("Invalid classifier type must be NN, KNN or LR")

    clf_cv = GridSearchCV(
        estimator=method,
        param_grid=parameters,
        scoring="f1_macro",
        cv=5,
        verbose=0,
    )
    clf = make_pipeline(preprocessing.StandardScaler(), clf_cv)
    clf.fit(x_train, y_train)
    print(f"Best parameters found for: {classifier}\n", clf_cv.best_params_)

    y_pred_train = clf.predict(x_train)
    y_pred_val = clf.predict(x_val)
    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    val_acc = metrics.accuracy_score(y_val, y_pred_val)

    y_pred_train = le.inverse_transform(y_pred_train)
    y_pred_val = le.inverse_transform(y_pred_val)

    return train_acc, val_acc, y_pred_train, y_pred_val


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


def save_imshow_png(fname, array, cmap=None, min=None, max=None):
    if not os.path.exists("plots"):
        os.mkdir("plots")

    plt.subplots(figsize=(10, 10))
    plt.imshow(array, cmap=cmap, vmin=min, vmax=max)  # channels last

    plt.savefig("plots/" + fname)
    plt.close()


def save_mrc_file(fname, array):
    if not os.path.exists("plots"):
        os.mkdir("plots")
    with mrcfile.new("plots/" + fname, overwrite=True) as mrc:
        mrc.set_data(array)


def load_config_params(config_file, local_vars, logging):

    if config_file is not None:
        with open(config_file, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # returns JSON object as

        for key, val in local_vars.items():
            if (
                val is not None
                and isinstance(val, (int, float, bool, str))
                or data.get(key) is None
            ):
                logging.warning(
                    "Command line argument "
                    + key
                    + " is overwriting config file value to: "
                    + str(val)
                )
                data[key] = val
            else:
                logging.info(
                    "Setting "
                    + key
                    + " to config file value: "
                    + str(data[key])
                )
    else:
        # if no config file is provided, use command line arguments
        data = local_vars

        # Check for missing values and set to default values
    for key, val in data.items():
        if (val is None or val == "None") and key != "config_file":
            #  make sure data variables are provided
            if key == "data_path":
                logging.error(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
                raise ValueError(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
            elif key == "affinity" or key == "classes":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Setting to default value."
                )
                filename_default = os.path.join(data["datapath"], key + ".csv")

                if os.path.isfile(filename_default):
                    data[key] = filename_default
                else:
                    data[key] = None

                logging.info(
                    "Setting up "
                    + key
                    + " in config file to "
                    + str(data[key])
                )

            elif key == "state":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest state if in evaluation mode."
                )
            elif key == "meta":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest meta if in evaluation mode."
                )
            else:
                # set missing variables to default value
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Setting to default value."
                )
                data[key] = config.DEFAULT_RUN_CONFIGS[key]
                logging.info(
                    "Setting " + key + " to default value: " + str(data[key])
                )

    return data
