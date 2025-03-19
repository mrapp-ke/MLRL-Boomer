(parameter-persistence)=

# Saving and Loading Parameters

To remember the parameters that have been used for training a model, it might be useful to save them to disk. Similar to {ref}`saving models<model-persistence>`, keeping the resulting files allows to load a previously used configuration and reuse it at a later point in time.

On the one hand, this requires to specify a directory where parameter settings should be saved via the command line argument `--parameter-save-dir`. On the other hand, the argument `--store-parameters true` instructs the program to save custom parameters that are set via command line arguments (see {ref}`setting-algorithmic-parameters`). For example, the following command sets a custom value for a parameter, which is stored in an output file:

````{tab} BOOMER
   ```text
   testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-save-dir /path/to/parameters \
       --store-parameters true \
       --shrinkage 0.5
   ```
````

````{tab} SeCo
   ```text
   testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-save-dir /path/to/parameters \
       --store-parameters true \
       --heuristic precision
   ```
````

```{note}
The path of the directory, where parameter settings should be saved, can be either absolute or relative to the working directory.
```

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

When executing the previously mentioned command again, the program restores the parameter settings from the files that are found in the specified directory. This allows to omit the respective parameters from the command line. If a parameter is included in both, the loaded file and the command line arguments, the latter takes precedence.

If you want to print all custom parameters that are used by a learning algorithm on the console, you can specify the argument `--print-parameters true`:

````{tab} BOOMER
   ```text
   testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-parameters true \
       --shrinkage 0.5
   ```
````

````{tab} SeCo
   ```text
   testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-parameters true \
       --heuristic precision
   ```
````
