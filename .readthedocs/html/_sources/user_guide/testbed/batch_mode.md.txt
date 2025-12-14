(testbed-batch-mode)=

# Batch Mode

The batch mode of mlrl-testbed allows to run multiple experiments at once by defining the datasets and algorithmic parameters to be used in the different runs via a YAML configuration file. For example, this can be useful for measuring the performance of a machine learning algorithm across several datasets or for studying the effects of different parameter settings. As seen below, the batch mode can be activated by passing the argument `--mode batch` to the command line API:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode batch \
       --config path/to/config.yaml \
       --save-all true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode batch \
       --config path/to/config.yaml \
       --save-all true
   ```
````

Most of the arguments that can be used for controlling a single experiment (see {ref}`here <arguments>` for an overview) can also be passed to the command line API in batch mode. They apply to all experiments run in a batch. In the previous example, the argument `--save-all` specifies that experimental results for each experiment should be written to output files.

## Configuration File

Arguments that should vary from experiment to experiment must be defined in a YAML file specified via the argument `--config`. An exemplary file is shown below:

```yaml
datasets:
  - directory: path/to/datasets/
    names:
      - first-dataset
      - second-dataset
  - directory: path/to/other/datasets/
    names: third-dataset
default_arguments:
  - name: --first-default-argument
    value: first-value
  - name: --second-default-argument
    value: second-value
parameters:
  - name: --first-parameter
    values:
      - 50
      - 100
  - name: --second-parameter
    values:
      - false
      - value: true
        additional_arguments:
          --third-parameter true
```

The YAML file must contain the definition of at least one dataset (a schema file can be found {repo-file}`here <python/subprojects/testbed-sklearn/mlrl/testbed_sklearn/batch_config.schema.yml>`), consisting of the path to the directory where the dataset is located, as well as the name of the dataset (see {ref}`testbed-datasets`). In addition, several parameters and corresponding values can be specified. Each combination of the values given for different parameters will be used in experiments conducted on each available dataset. If two or more parameter values should be set jointly, they can be listed under `additional_arguments`. Finally, it is also possible to define arbitrary default arguments under the optional section `default_arguments`. Default arguments are the same for all experiments.

```{tip}
By default, when using a cross validation, a separate experiment is run for each cross validation fold. If a single experiment should perform all folds, the default behavior can be disabled via the argument `--separate-folds false`. 
```

## Listing Commands

Before any experiments are started, the given command line arguments and the configuration file are validated. If a problem is found with either of them, you receive immediate feedback in the form of an error message. Nevertheless, we recommend to take a look at the commands that will be run as part of a batch. For this purpose, the flag `--list` can be specified:

```text
mlrl-testbed custom_runnable.py \
     --mode batch \
     --config path/to/config.yaml \
     --save-all true \
     --list
```

The previous command prints all commands that will be executed for running individual experiments. In the following, we show an example of how a single command might look:

```text
mlrl-testbed mlrl.boosting \
    --base-dir experiments/2025-08-10_02-44
    --result-dir first-parameter_50/second-parameter_true/third-parameter_true/dataset_first-dataset/results \
    --model-save-dir first-parameter_50/second-parameter_true/third-parameter_true/dataset_first-dataset/models \
    --parameter-save-dir first-parameter_50/second-parameter_true/third-parameter_true/dataset_first-dataset/parameters \
    --data-dir path/to/datasets/ \
    --dataset first-dataset \
    --first-default-argument first-value \
    --second-default-argument second-value \
    --first-parameter 50 \
    --second-parameter true \
    --third-parameter true \
    --save-meta-data false
```

We can see that the argument `--save-all true`, which has been given as a command line argument, is used for this particular experiment since these arguments are global. In contrast, the arguments `--instance-sampling none` and `--loss logistic-decomposable` are one particular combination of the parameters defined in the configuration file to be used in this experiment. Hence, their values are different in each experiment. Similarly, the arguments `--data-dir data` and `--dataset bibtex` correspond to one particular dataset defined in the configuration file. Finally, the paths specified via the arguments `--result-dir`, `--model-save-dir` and `--parameter-save-dir` are set automatically to ensure that output data produced by an individual experiment will end up in a distinct directory, independent of other experiments.

(testbed-slurm)=

## Slurm Support

```{note}
Slurm support is provided by the optional package [mlrl-testbed-slurm](https://pypi.org/project/mlrl-testbed-slurm/). As shown below, it must be installed manually.
```

### Installation

By installing the optional package [mlrl-testbed-slurm](https://pypi.org/project/mlrl-testbed-slurm/), support for the [Slurm Workload Manager](https://wikipedia.org/wiki/Slurm_Workload_Manager) is added to mlrl-testbed's batch mode. This enables to run experiments on a network of different compute nodes. The package can be installed as follows:

```text
pip install mlrl-testbed-slurm
```

### Usage

```{important}
The Slurm support of mlrl-testbed requires the command `sbatch` to be available, as it is usually the case for local installations of Slurm or login nodes of computing clusters. 
```

After the installation, experiments can be registered at the Slurm scheduler by passing argument `--runner slurm` to mlrl-testbed's command line API.

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true
   ```
````

### Inspecting Slurm Scripts

When using the argument `--runner slurm`, shell scripts are generated and passed to Slurm's [`sbatch`](https://slurm.schedmd.com/sbatch.html) command. Most importantly, these scripts contain the command to be executed for running an experiment. In addition, they may also contain different arguments to be interpreted by the `sbatch` command (an overview can be found [here](https://slurm.schedmd.com/sbatch.html#SECTION_OPTIONS)).

By using the argument `--print-slurm-scripts`, the content of the scripts can be printed:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true \
       --print-slurm-scripts true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true \
       --print-slurm-scripts true
   ```
````

Alternatively, the arguments `--save-slurm-scripts` and `--slurm-save-dir` can be used to save the generated scripts to a specific directory:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true \
       --save-slurm-scripts true \
       --slurm-save-dir path/to/output/
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true \
       --save-slurm-scripts true \
       --slurm-save-dir path/to/output/
   ```
````

```{note}
When printing the content of Slurm scripts or saving them to output files, they are not passed to the `sbatch` command. 
```

### Customizing Slurm Scripts

To provide maximum flexibility, it is possible to customize the contents of Slurm scripts generated by mlrl-testbed:

1. Arguments to be interpreted by the `sbatch` command, e.g., for providing job meta-data, controlling the scheduling of the job, or setting resource requests (a full list can be found [here](https://slurm.schedmd.com/sbatch.html#SECTION_OPTIONS)). In the generated scripts, these arguments start with `#SBATCH`.
2. Shell commands to be executed before or after an experiment, e.g., for implementing the experimental setup and archiving experiment results afterward.

Customization of these aspects is possible by specifying a configuration file in the YAML format via the argument `--slurm-config`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true \
       --slurm-config path/to/slurm-config.yaml
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode batch \
       --config path/to/config.yaml \
       --runner slurm \
       --save-all true \
       --slurm-config path/to/slurm-config.yaml
   ```
````

For reference, we provide an exemplary configuration below:

```yaml
sbatch-arguments:
  - --time=2:30:00
  - --cpus-per-tasks=1
before-script: |
  python -m venv .venv
  . .venv/bin/activate
  python -m pip install mlrl-testbed-sklearn
after-script: |
  deactivate
```

The individual arguments listed under `sbatch-arguments` will be added to generated scripts prefixed with `#SBATCH`. Shell commands to be executed before or after an experiment must be specified under `before-script` or `after-script`, respectively. In this example, a virtual Python environment is created and activated before the experiment and deactivated afterward. All of these customization options are optional (a schema file can be found {repo-file}`here <python/subprojects/testbed-slurm/mlrl/testbed_slurm/slurm_config.schema.yml>`).
