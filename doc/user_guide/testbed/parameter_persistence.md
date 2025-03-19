(parameter-persistence)=

# Saving and Loading Parameters

To remember the parameters that have been used for training a model, it might be useful to save them to disk. Similar to {ref}`saving models<model-persistence>`, keeping the resulting files allows to load a previously used configuration and reuse it at a later point in time.

This requires to specify a directory via the command line argument `--parameter-save-dir`, where parameters set via the command line API (see {ref}`setting-algorithmic-parameters`) should be saved. For example, the following command sets a custom value for a parameter, which is stored in an output file:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-save-dir /path/to/parameters \
       --shrinkage 0.5
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-save-dir /path/to/parameters \
       --heuristic precision
   ```
````

If {ref}`train-test splits<train-test-split>` are used for splitting the available data into training and test sets, a single model is trained and its configuration is saved to a file:

- `parameters_overall.csv`

If a {ref}`cross validation<cross-validation>` is performed instead, one model is trained per cross validation fold and the configurations of all of these models are stored in the specified directory. For example, a 5-fold cross validation results in the following files:

- `parameters_fold-1.csv`
- `parameters_fold-2.csv`
- `parameters_fold-3.csv`
- `parameters_fold-4.csv`
- `parameters_fold-5.csv`

```{note}
Only parameters with custom values are included in the output files. Parameters for which the default value is used are not included.
```

If you want to print all custom parameters that are used by a learning algorithm on the console, you can specify the argument `--print-parameters true`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-parameters true \
       --shrinkage 0.5
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-parameters true \
       --heuristic precision
   ```
````

Once parameters have been saved to a directory, they can be loaded in subsequent experiments by specifying the command line argument `--parameter-load-dir`. This allows to omit the respective parameters from the command line. If a parameter is included in both, the loaded file and the command line arguments, the latter takes precedence.

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-load-dir /path/to/parameters \
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-load-dir /path/to/parameters \
   ```
````

```{note}
The paths of the directories that are specified via the arguments `--parameter-save-dir` and `--parameter-load-dir` can be either absolute or relative to the working directory.
```
