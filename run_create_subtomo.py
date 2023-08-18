import logging
import os
import warnings
from datetime import datetime

import click
import yaml

from tools.create_subtomo import create_subtomo

if not os.path.exists("logs"):
    os.mkdir("logs")
dt_name = datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/avae_run_log_" + dt_name + ".log"),
        logging.StreamHandler(),
    ],
)


@click.command(name="Affinity Trainer")
@click.option("--config_file", type=click.Path(exists=True))
@click.option(
    "--input_path",
    "-ip",
    type=str,
    default=None,
    help="Path to the folder containing the full tomogram/image.",
)
@click.option(
    "--output_path",
    "-op",
    type=str,
    default=None,
    help="Path to the folder for output subtomograms",
)
@click.option(
    "--annot_path",
    "-op",
    type=str,
    default=None,
    help="path to the folder containing the name of the particles and their x,y,z coordinates",
)
@click.option(
    "--datatype",
    "-dtype",
    type=str,
    default=None,
    help="Type of the data: mrc, npy",
)
@click.option(
    "--vox_size",
    "-vs",
    type=(int, int, int),
    default=None,
    help="size of each subtomogram voxel given as a list where vox_size: [x,y,x]",
)
@click.option(
    "--vox_size",
    "-vs",
    type=list,
    default=[],
    help=" size of each subtomogram voxel given as a list where vox_size: [x,y,x]",
)
@click.option(
    "--gaussian_blur",
    "-gb",
    type=bool,
    default=None,
    is_flag=True,
    help="Applying gaussian bluring to the image data which should help removing noise. The minimum and maximum for this is hardcoded.",
)
@click.option(
    "--normalise",
    "-nrm",
    type=bool,
    default=None,
    is_flag=True,
    help="Normalise data",
)
@click.option(
    "--bandpass",
    "-bp",
    type=bool,
    default=None,
    is_flag=True,
    help="Apply band pass",
)
@click.option(
    "--low_freq",
    "-lf",
    type=(float, float),
    default=None,
    help="Lower  frequency threshold for the band pass filter",
)
@click.option(
    "--high_freq",
    "-hf",
    type=(float, float),
    default=None,
    help="higher frequency threshold for the band pass filter",
)
def run(
    config_file,
    input_path,
    output_path,
    datatype,
    annot_path,
    vox_size,
    bandpass,
    low_freq=0,
    high_freq=15,
    gaussian_blur=False,
    add_noise=False,
    noise_int=0,
    padding=False,
    padded_size=[32, 32, 32],
    augment=False,
    aug_num=5,
    aug_th_min=-45,
    aug_th_max=45,
):

    warnings.simplefilter("ignore", FutureWarning)
    # read config file and command line arguments and assign to local variables that are used in the rest of the code
    local_vars = locals().copy()
    print(local_vars)

    if config_file is not None:
        with open(config_file, "r") as f:
            logging.info("Reading submission configuration file" + config_file)
            data = yaml.load(f, Loader=yaml.FullLoader)
        # returns JSON object as
        print(data.get("gaussian_blur"))

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

        logging.info(
            "Saving final submission config file to: "
            + "avae_final_config"
            + dt_name
            + ".yaml"
        )
        file = open("avae_final_config" + dt_name + ".yaml", "w")
        yaml.dump(data, file)
        file.close()
        logging.info("YAML File saved!")

    create_subtomo(
        input_path=data["input_path"],
        output_path=data["output_path"],
        bandpass=data["bandpass"],
        low_freq=data["low_freq"],
        high_freq=data["high_freq"],
        add_noise=data["add_noise"],
        noise_int=data["noise_intensity"],
        padding=data["padding"],
        augment=data["augment"],
        aug_th_min=data["aug_th_min"],
        aug_th_max=data["aug_th_max"],
    )


if __name__ == "__main__":
    run()
