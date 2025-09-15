(testbed-run-mode)=

# Run Mode

MLRL-Testbed provides an alternative mode of operation in which it takes the `metadata.yml` file that has been saved by a previous experiment as an input. By reading the meta-data, the experiment can be run again.

```{note}
A `metadata.yml` file is saved by an experiment if any output data is saved or if the argument `--save-meta-data true` is specified explicitly (see {ref}`here <meta-data>` for more details).
```

Following command illustrates how the so-call "run mode" can be activated by using the argument `--mode run` and specifying the path to a directory that contains the aforementioned `metadata.yml` file via the argument `--input-dir`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode run \
       --input-dir path/to/experiment/
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode run \
       --input-dir path/to/experiment/
   ```
````

By default, the experiment is run in the same way as it was originally, i.e., using the exact same command line arguments. This can cause problems. For example, if input files have been moved to a different location. For additional flexibility, some arguments can be overridden when re-running an experiment. This includes arguments controlling how input and output files are handled. For example, this enables to obtain additional output data, not considered in the original experiment. Arguments that affect the behavior of algorithms are not allowed when re-running an experiment to keep experimental results reproducible. The following example illustrates how custom arguments (in this case `--print-predictions` and `--print-ground-truth`) can be supplied to an experiment in run mode:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode run \
       --input-dir path/to/experiment/ \
       --print-predictions true \
       --print-ground-truth true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode run \
       --input-dir path/to/experiment/ \
       --print-predictions true \
       --print-ground-truth true
   ```
````
