import logging
import os
import types
import typing
import warnings
from pathlib import Path

import click

from avae.config import (
    AffinityConfig,
    load_config_params,
    setup_visualisation_config,
    write_config_file,
)
from avae.evaluate import evaluate
from avae.train import train

# Short flags for CLI options, keyed by AffinityConfig field name.
_CLI_SHORT_FLAGS = {
    "datapath": "-d",
    "datatype": "-dtype",
    "debug": "-dbg",
    "restart": "-res",
    "state": "-st",
    "meta": "-mt",
    "limit": "-lm",
    "split": "-sp",
    "new_out": "-newo",
    "no_val_drop": "-nd",
    "affinity": "-af",
    "classes": "-cl",
    "classifier": "-clf",
    "epochs": "-ep",
    "batch": "-ba",
    "depth": "-de",
    "channels": "-ch",
    "filters": "-fl",
    "latent_dims": "-ld",
    "pose_dims": "-pd",
    "bnorm_encoder": "-bn_enc",
    "bnorm_decoder": "-bn_dec",
    "gsd_conv_layers": "-gsdcl",
    "n_splats": "-spl",
    "klreduction": "-kr",
    "beta": "-be",
    "beta_load": "-bl",
    "gamma": "-g",
    "gamma_load": "-gl",
    "learning": "-lr",
    "loss_fn": "-lf",
    "beta_min": "-bs",
    "beta_cycle": "-bc",
    "beta_ratio": "-br",
    "cyc_method_beta": "-cycmb",
    "gamma_min": "-gs",
    "gamma_cycle": "-gc",
    "gamma_ratio": "-gr",
    "cyc_method_gamma": "-cycmg",
    "gpu": "-gpu",
    "gpu_devices": "-gdev",
    "eval": "-ev",
    "dynamic": "-dn",
    "model": "-m",
    "vis_los": "-vl",
    "vis_acc": "-vac",
    "vis_rec": "-vr",
    "vis_emb": "-ve",
    "vis_int": "-vi",
    "vis_dis": "-vt",
    "vis_pos": "-vps",
    "vis_pose_class": "-vpsc",
    "vis_z_n_int": "-vzni",
    "vis_cyc": "-vc",
    "vis_aff": "-vaf",
    "vis_his": "-his",
    "vis_sim": "-similarity",
    "vis_all": "-vall",
    "vis_format": "-vf",
    "freq_eval": "-fev",
    "freq_sta": "-fs",
    "freq_acc": "-fac",
    "freq_rec": "-fr",
    "freq_emb": "-fe",
    "freq_int": "-fi",
    "freq_dis": "-ft",
    "freq_pos": "-fp",
    "freq_sim": "-fsim",
    "freq_all": "-fa",
    "opt_method": "-opt",
    "gaussian_blur": "-gb",
    "normalise": "-nrm",
    "shift_min": "-sftm",
    "rescale": "-rsc",
    "tensorboard": "-tb",
    "strategy": "-str",
}

# Click kwargs the model type cannot express (path-valued flags), by field name.
_CLI_OVERRIDES = {
    "beta_load": {"is_flag": True},
    "gamma_load": {"is_flag": True},
}

_CLICK_TYPE_MAP = {int: int, float: float, str: str, Path: str, list: str}


def _resolve_click_type(annotation):
    """Unwrap Optional/Union and Annotated to the underlying scalar type."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        annotation = next(
            a for a in typing.get_args(annotation) if a is not type(None)
        )
    if hasattr(annotation, "__metadata__"):  # typing.Annotated[...]
        annotation = annotation.__origin__
    return annotation


def model_cli_options(model_cls):
    """Generate one click option per model field, using the model as the
    single source of truth for names, help and existence. All options default
    to None so precedence stays: model defaults < config file < CLI."""

    def decorator(fn):
        for name, field in reversed(model_cls.model_fields.items()):
            if name == "config_file":
                continue  # declared explicitly with an existence check
            base = _resolve_click_type(field.annotation)
            kwargs = {"default": None, "help": field.description or ""}
            if base is bool:
                kwargs["is_flag"] = True
                kwargs["type"] = bool
            else:
                kwargs["type"] = _CLICK_TYPE_MAP.get(base, str)
            kwargs.update(_CLI_OVERRIDES.get(name, {}))
            names = [f"--{name}"]
            if name in _CLI_SHORT_FLAGS:
                names.append(_CLI_SHORT_FLAGS[name])
            fn = click.option(*names, **kwargs)(fn)
        return fn

    return decorator


@click.command(name="Affinity Trainer")
@click.option("--config_file", type=click.Path(exists=True))
@model_cli_options(AffinityConfig)
def run(**kwargs):

    warnings.simplefilter("ignore", FutureWarning)

    config_file = kwargs.get("config_file")

    # read config file and command line arguments and assign to local variables that are used in the rest of the code
    logging.info("Reading submission configuration file %s", config_file)
    local_vars = dict(kwargs)
    data = load_config_params(config_file, local_vars)

    if data["debug"]:
        logging.info("Debug mode enabled")
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("matplotlib.font_manager").disabled = True

    # visualisation global settings defined from config file
    data = setup_visualisation_config(data)

    if data["new_out"]:
        dir_name = f'results_{data["date_time_run"]}_model_{data["model"]}_lat{data["latent_dims"]}_pose{data["pose_dims"]}_lr{data["learning"]}_beta{data["beta"]}_gamma{data["gamma"]}'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            logging.info(f"Directory {dir_name} already exists")
        os.chdir(dir_name)

    if not os.path.exists("logs"):
        os.mkdir("logs")

    # setup logger inside the  directory where we are running the code
    fileh = logging.FileHandler(
        "logs/avae_run_log_" + data["date_time_run"] + ".log", "a"
    )
    logging.getLogger().addHandler(fileh)

    logging.info(
        "Saving final submission config file to: "
        + "avae_final_config"
        + data["date_time_run"]
        + ".yaml"
    )

    write_config_file(data)

    try:
        run_pipeline(data)

    except Exception as e:
        logging.exception("An exception was thrown: %s", e)
        raise


def run_pipeline(data):

    if not data["eval"]:
        train(
            datapath=data["datapath"],
            datatype=data["datatype"],
            restart=data["restart"],
            state=data["state"],
            lim=data["limit"],
            splt=data["split"],
            batch_s=data["batch"],
            no_val_drop=data["no_val_drop"],
            affinity=data["affinity"],
            classes=data["classes"],
            epochs=data["epochs"],
            channels=data["channels"],
            depth=data["depth"],
            filters=data["filters"],
            lat_dims=data["latent_dims"],
            pose_dims=data["pose_dims"],
            bnorm_encoder=data["bnorm_encoder"],
            bnorm_decoder=data["bnorm_decoder"],
            gsd_conv_layers=data["gsd_conv_layers"],
            n_splats=data["n_splats"],
            klred=data["klreduction"],
            learning=data["learning"],
            beta_load=data["beta_load"],
            beta_min=data["beta_min"],
            beta_max=data["beta"],
            beta_cycle=data["beta_cycle"],
            beta_ratio=data["beta_ratio"],
            cyc_method_beta=data["cyc_method_beta"],
            gamma_load=data["gamma_load"],
            gamma_min=data["gamma_min"],
            gamma_max=data["gamma"],
            gamma_cycle=data["gamma_cycle"],
            gamma_ratio=data["gamma_ratio"],
            cyc_method_gamma=data["cyc_method_gamma"],
            recon_fn=data["loss_fn"],
            use_gpu=data["gpu"],
            gpu_devices=data["gpu_devices"],
            model=data["model"],
            opt_method=data["opt_method"],
            gaussian_blur=data["gaussian_blur"],
            normalise=data["normalise"],
            shift_min=data["shift_min"],
            rescale=data["rescale"],
            tensorboard=data["tensorboard"],
            classifier=data["classifier"],
            strategy=data["strategy"],
        )
    else:
        evaluate(
            datapath=data["datapath"],
            datatype=data["datatype"],
            state=data["state"],
            meta=data["meta"],
            lim=data["limit"],
            splt=data["split"],
            batch_s=data["batch"],
            classes=data["classes"],
            use_gpu=data["gpu"],
            gaussian_blur=data["gaussian_blur"],
            normalise=data["normalise"],
            shift_min=data["shift_min"],
            rescale=data["rescale"],
            classifier=data["classifier"],
        )


if __name__ == "__main__":
    run()
