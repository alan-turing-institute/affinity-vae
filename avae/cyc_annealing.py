import math

import numpy as np


def configure_annealing(
    epochs: int,
    value_max: float,
    value_min: float,
    cyc_method: str,
    n_cycle: int,
    ratio: float,
    cycle_load: str | None = None,
):
    """
    This function is used to configure the annealing of the beta and gamma values.
    It creates an array of values that oscillate between a maximum and minimum value
    for a defined number of cycles. This is used for gamma and beta in the loss term.
    The function also allows for the loading of a pre-existing array of beta or gamma values.

    Parameters
    ----------
    epochs: int
        Number of epochs in training
    value_max: float
        Maximum value of the beta or gamma
    value_min: float
        Minimum value of the beta or gamma
    cyc_method : str
        The method for constructing the cyclical mixing weight
                    - Flat : regular beta-vae
                    - Linear
                    - Sigmoid
                    - Cosine
                    - ramp
                    - delta
    n_cycle: int
        Number of cycles of the variable to oscillate between min and max
                during the epochs
    ratio: float
        Ratio of increase during ramping
    cycle_load: str | None
        Path to a file containing the beta or gamma array

    Returns
    -------
    cycle_arr: np.ndarray
        Array of beta or gamma values

    """
    if value_max == 0 and cyc_method != "flat" and cycle_load is not None:
        raise RuntimeError(
            "The maximum value for beta is set to 0, it is not possible to"
            "oscillate between a maximum and minimum. Please choose the flat method for"
            "cyc_method_beta"
        )

    if cycle_load is None:
        # If a path for loading the beta array is not provided,
        # create it given the input
        cycle_arr = (
            cyc_annealing(
                epochs,
                cyc_method,
                n_cycle=n_cycle,
                ratio=ratio,
            ).var
            * (value_max - value_min)
            + value_min
        )
    else:
        cycle_arr = np.load(cycle_load)
        if len(cycle_arr) != epochs:
            raise RuntimeError(
                f"The length of the beta array loaded from file is {len(cycle_arr)} but the number of Epochs specified in the input are {epochs}.\n"
                "These two values should be the same."
            )

    return cycle_arr


class cyc_annealing:

    """
    This class presents an array which will have a value changing between a minimum
    and maximum for a defined number of cycles. This is used for gamma and beta in loss term.

    Parameters
    ----------
    n_epochs   : number of epochs in training
    cyc_method : The method for constructing the cyclical mixing weight
                    - Flat : regular beta-vae
                    - Linear
                    - Sigmoid
                    - Cosine
                    - ramp
                    - delta
    start       :  The starting point (min)
    stop        :  The starting point (min)
    n_cycle     : Number of cycles of the variable to oscillate between min and max
                during the epochs
    ratio       : ratio of increase during ramping
    """

    def __init__(
        self,
        n_epoch: int,
        cyc_method: str = "flat",
        n_cycle: int = 4,
        ratio: float = 0.5,
    ):

        self.n_epoch = n_epoch

        # The start and stop control where in each the cycle, the sigmoid function starts and stops
        # This is for the moment hard coded in, as the feature does not contribute to the outcome of the model for our purpose
        self.start = 0
        self.stop = 1
        self.n_cycle = n_cycle
        self.ratio = ratio

        if cyc_method == "flat":
            self.var = self._frange_flat()

        elif cyc_method == "cycle_linear":
            self.var = self._frange_cycle_linear()

        elif cyc_method == "cycle_sigmoid":
            self.var = self._frange_cycle_sigmoid()

        elif cyc_method == "cycle_cosine":
            self.var = self._frange_cycle_cosine()

        elif cyc_method == "ramp":
            self.var = self._frange_ramp()

        elif cyc_method == "delta":
            self.var = self._frange_delta()

        elif cyc_method == "mixed":
            self.var = self._frange_mixed()

        else:
            raise RuntimeError(
                "Select a valid method for cyclical method for your variable. Available options are : "
                "flat, cycle_linear, cycle_sigmoid, cycle_cosine, ramp, delta, mixed"
            )

    def _frange_flat(self):
        L = np.ones(self.n_epoch)
        return L

    def _frange_cycle_linear(self):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / self.n_cycle
        step = (self.stop - self.start) / (
            period * self.ratio
        )  # linear schedule

        for c in range(self.n_cycle):

            v, i = self.start, 0
            while v <= self.stop and (int(i + c * period) < self.n_epoch):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L

    def _frange_cycle_sigmoid(self):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / self.n_cycle
        step = (self.stop - self.start) / (
            period * self.ratio
        )  # step is in [0,1]

        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(self.n_cycle):

            v, i = self.start, 0
            while v <= self.stop:
                L[int(i + c * period)] = 1.0 / (
                    1.0 + np.exp(-(v * 12.0 - 6.0))
                )
                v += step
                i += 1
        return L

    #  function  = 1 − cos(a), where a scans from 0 to pi/2

    def _frange_cycle_cosine(self):
        L = np.ones(self.n_epoch)
        period = self.n_epoch / self.n_cycle
        step = (self.stop - self.start) / (
            period * self.ratio
        )  # step is in [0,1]

        # transform into [0, pi] for plots:

        for c in range(self.n_cycle):

            v, i = self.start, 0
            while v <= self.stop:
                L[int(i + c * period)] = 0.5 - 0.5 * math.cos(v * math.pi)
                v += step
                i += 1
        return L

    def _frange_ramp(self):
        L = np.ones(self.n_epoch)
        v, i = self.start, 0
        period = self.n_epoch / self.n_cycle

        step = (self.stop - self.start) / (period * self.ratio)
        while v <= self.stop:
            L[i] = v
            v += step
            i += 1
        return L

    def _frange_delta(self):
        L = np.zeros(self.n_epoch)
        period = self.n_epoch / (self.n_cycle + 1)

        for n in range(self.n_cycle + 1):
            if n % 2 == 1:
                L[int(period) * n : int(period) * (n + 1)] = self.stop
            else:
                L[int(period) * n : int(period) * (n + 1)] = self.start
        return L

    def _frange_mixed(self):

        L = np.ones(self.n_epoch)
        on = 300
        off = 600
        L[0:on] = 0

        period = (off - on) / self.n_cycle
        step = (self.stop - self.start) / (
            period * self.ratio
        )  # step is in [0,1]

        # transform into [-6, 6] for plots: v*12.-6.

        for c in range(self.n_cycle):
            v, i = self.start, on
            while v <= self.stop:
                L[int(i + c * period)] = 1.0 / (
                    1.0 + np.exp(-(v * 12.0 - 6.0))
                )
                v += step
                i += 1
        return L
