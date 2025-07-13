(model-persistence)=

# Saving and Loading Models

Because the training of machine learning models can be time-consuming, they are usually trained once and then reused later for making predictions. For this purpose, the package mlrl-testbed provides means to store models on disk and load them from the created files later on. This requires to specify the argument `--save-models`. Optionally, the path to a directory where models should be saved, as well as a directory from which models should be loaded can be set via the command line arguments `--model-save-dir` and `--model-load-dir`. If not specified manually, the default `models` is used.

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --model-save-dir /path/to/models \
       --model-load-dir /path/to/models \
       --save-models true \
       --load-models true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --model-save-dir /path/to/models \
       --model-load-dir /path/to/models \
       --save-models true \
       --load-models true
   ```
````

```{note}
The paths of the directories specified via the arguments `--model-save-dir` and `--model-load-dir` can be either absolute or relative to the working directory. They must not refer to the same directory, which allows saving models to a different directory than the one they are loaded from. 
```

If {ref}`train-test splits<train-test-split>` are used for evaluating the predictive performance of models, a single model is fit to the training data and stored in a file:

- `model.pickle`

If a {ref}`cross validation<cross-validation>` is performed instead, one model is trained per cross validation fold and all of these models are stored in the specified directory. For example, a 5-fold cross validation results in the following files:

- `model_fold-1.pickle`
- `model_fold-2.pickle`
- `model_fold-3.pickle`
- `model_fold-4.pickle`
- `model_fold-5.pickle`

When executing the aforementioned command again, the program recognizes the previously stored models in the specified directory. Instead of training them from scratch, the models are then loaded from the respective files, which should be much faster than training them again.
