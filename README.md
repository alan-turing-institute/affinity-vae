[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/alan-turing-institute/affinity-vae/actions/workflows/tests.yml/badge.svg)](https://github.com/alan-turing-institute/affinity-vae/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

> Note: This a development version of the code. The code in the `main` branch is a more stable version of the code.

# Affinity-VAE


**Affinity-VAE for disentanglement, clustering and classification of objects in multidimensional image data**  
Mirecka J, Famili M, Kotanska A, Jurashcko N, Costa-Gomes B, Palmer CM, Thiyagalingam J, Burnley T, Basham M & Lowe AR  
[![doi:10.48550/arXiv.2209.04517](https://img.shields.io/badge/doi-10.48550/arXiv.2209.04517-blue)](https://doi.org/10.48550/arXiv.2209.04517)

## Installation

### Installing with pip + virtual environments

> Note: This has been tested in the `refactor` branch.

You can install the libraries needed for this package on a [fresh virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) with the following:

```
python3 -m venv env
source env/bin/activate
pip install -e .
```
> Note: This is the preffered option for running on Turing macOS laptops.

> Warning: M1 macOS can not do [pytorch paralelisation](https://github.com/pytorch/pytorch/issues/70344). A temporary solution for this is to modify the code on the DataLoaders in data.py to `num_workers=0` in order to run the code. Otherwise you will get the error: `AttributeError: Can't pickle local object 'ProteinDataset.__init__.<locals>.<lambda>'`.

### Installing with conda in Baskerville

The following is the recommended way of installing all libraries in Baskervile.

```
conda create --name affinity_env
conda activate affinity_env

conda install --yes python=3.10
conda install --yes numpy
conda install --yes requests
conda install -c anaconda pandas
conda install -c anaconda scikit-image
conda install -c anaconda scikit-learn
conda install -c anaconda scipy
conda install -c anaconda pillow
conda install -c conda-forge mrcfile
conda install -c conda-forge altair
conda install -c conda-forge umap-learn
conda install -c conda-forge matplotlib
conda install -c anaconda click
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

if the follwoing error  occurs:
```
:ImportError: libtiff.so.5: cannot open shared object file: No such file or directory
```
you can resolve it via:

```
conda install -c anaconda libtiff==4.4.0
```

### Quick start

Affinity-vae has a running script (`run.py`)that allows you to configure and run the code. You can look at the avaible configuration options by running:

```
python run.py --help
```

which will give you:

```
Usage: run.py [OPTIONS]

Options:
  -d, --datapath TEXT         Path to training data.  [required]
  -lm, --limit INTEGER        Limit the number of samples loaded (default
                              None).
  -sp, --split INTEGER        Train/val split in %.
  -nd, --no_val_drop          Do not drop last validate batch if if it is
                              smaller than batch_size.
  -af, --affinity TEXT        Path to affinity matrix for training.
  -cl, --classes TEXT         Path to a CSV file containing a list of classes
                              for training.
  -ep, --epochs INTEGER       Number of epochs (default 100).
  -ba, --batch INTEGER        Batch size (default 128).
  -de, --depth INTEGER        Depth of the convolutional layers (default 3).
  -ch, --channels INTEGER     First layer channels (default 64).
  -ld, --latent_dims INTEGER  Latent space dimension (default 10).
  -pd, --pose_dims INTEGER    If pose on, number of pose dimensions. If 0 and
                              gamma=0 it becomesa standard beta-VAE.
  -b, --beta FLOAT            Variational beta (default 1).
  -g, --gamma FLOAT           Scale factor for the loss component
                              corresponding to shape similarity (default 1).
                              If 0 and pd=0 it becomes a standardbeta-VAE.
  -lr, --learning FLOAT       Learning rate (default 1e-4).
  -lf, --loss_fn TEXT         Loss type: 'MSE' or 'BCE' (default 'MSE').
  -fev, --freq_eval INTEGER   Frequency at which to evaluate test set (default
                              every 10 epochs).
  -fs, --freq_sta INTEGER     Frequency at which to save state (default every
                              10 epochs).
  -fe, --freq_emb INTEGER     Frequency at which to visualise the latent space
                              embedding (default every 10 epochs).
  -fr, --freq_rec INTEGER     Frequency at which to visualise reconstructions
                              (default every 10 epochs).
  -fi, --freq_int INTEGER     Frequency at which to visualise latent
                              spaceinterpolations (default every 10 epochs).
  -ft, --freq_dis INTEGER     Frequency at which to visualise single
                              transversals (default every 10 epochs).
  -fp, --freq_pos INTEGER     Frequency at which to visualise pose (default
                              every 10 epochs).
  -fac, --freq_acc INTEGER    Frequency at which to visualise confusion
                              matrix.
  -fa, --freq_all INTEGER     Frequency at which to visualise all plots except
                              loss (default every 10 epochs).
  -ve, --vis_emb              Visualise latent space embedding.
  -vr, --vis_rec              Visualise reconstructions.
  -vl, --vis_los              Visualise loss.
  -vi, --vis_int              Visualise interpolations.
  -vt, --vis_dis              Visualise single transversals.
  -vps, --vis_pos             Visualise pose interpolations in the first 2
                              dimensions
  -vac, --vis_acc             Visualise confusion matrix.
  -va, --vis_all              Visualise all above.
  -g, --gpu                   Use GPU for training.
  -ev, --eval                 Evaluate test data.
  -dn, --dynamic              Enable collecting meta and dynamic latent space
                              plots.
  --help                      Show this message and exit.

```

Note that setting ```-g/--gamma``` to ```0``` and ```-pd/--pose_dims``` to ```0``` will run a vanilla beta-VAE.

### Quickstart

#### Configuring from the command line


You can run on example data with the following command:

```
python affinity-vae/run.py -d data/subtomo_files --split 20 --epochs 10 -ba 128 -lr 0.001 -de 4 -ch 64 -ld 8 -pd 3 --beta 1 --gamma 2 --limit 1000 --freq_all 5 --vis_all --dynamic
```
where the **subtomo_files** is a directory with a number of `.mcr` proteine image files named with the protein keyword such as (`1BXN_m0_156_Th0.mrc`,`5MRC_m8_1347_Th0.mrc`, etc). The **subtomo_files** directory should also have be a `classes.csv` file with a list of the protein names and keywords to be considered (`1BXN`, `5MRC`, etc.) and a `affinity_scores.csv` matrix with the initial values for the proteins named in the `classes.csv`.

#### Using a config submission file

You can also run the code using a submission config file (you can find an example with default values
on `configs/avae-test-config.yml`). For example, you can run the following command:

```
python affinity-vae/run.py --config_file affinity-vae/configs/avae-test-config.yml
```

You can also use a mix of config file and command line arguments. For example, you can run the following command:

```
python affinity-vae/run.py --config_file affinity-vae/configs/avae-test-config.yml --epochs 10 --affinity /path/to/different_affinity.csv
```

this will rewrite the values for the epochs and affinity path in the config file.

At the end of the run, the code will save the final config file used for the run in the working directory. This will
account for any changes made to the config file from the command line. Running the code again with that config file will reproduce the results.

#### tools
In the tools folder you can find notebooks which will assist you in creating the input files for Affinity-VAE or analyse the output of the model.

#### Considerations

##### Test folder : If test folder is present, the program will read the test files regardless of the eval flag

##### Evaluation: To run evaluation on a trained model you can turn the `eval` flag to True. This will load the last model present on the `states` directory (within the working directory path where you run the code) and run the evaluation on data set by the `datapath` flag. The evaluation will be saved in the `plots` and `latents` directory with the `eval` suffix on the names.

##### The name of the state file consist of avae_date_time_Epoch_latent_pose.pt
