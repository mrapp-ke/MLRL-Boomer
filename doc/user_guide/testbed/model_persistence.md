(model-persistence)=

# Saving and Loading Models

Because the training of machine learning models can be time-consuming, they are usually trained once and then reused later for making predictions. For this purpose, the command line API provides means to store models on disk and load them from the created files later on. This requires to specify the path of a directory, where models should be saved, via the command line argument `--model-dir`:

````{tab} BOOMER
   ```text
   testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --model-dir /path/to/models
   ```
````

````{tab} SeCo
   ```text
   testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --model-dir /path/to/models
   ```
````

```{note}
The path of the directory, where models should be saved, can be either absolute or relative to the working directory.
```

If {ref}`train-test splits<train-test-split>` are used for evaluating the predictive performance of models, a single model is fit to the training data and stored in a file:

- `boomer.model`

If a {ref}`cross validation<cross-validation>` is performed instead, one model is trained per cross validation fold and all of these models are stored in the specified directory. For example, a 5-fold cross validation results in the following files:

- `boomer_fold-1.model`
- `boomer_fold-2.model`
- `boomer_fold-3.model`
- `boomer_fold-4.model`
- `boomer_fold-5.model`

When executing the aforementioned command again, the program recognizes the previously stored models in the specified directory. Instead of training them from scratch, the models are then loaded from the respective files, which should be much faster than training them again.
