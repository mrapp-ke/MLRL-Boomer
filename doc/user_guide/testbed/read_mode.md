(testbed-read-mode)=

# Read Mode

MLRL-Testbed provides an alternative mode of operation in which it takes the `metadata.yml` file that has been saved by a previous experiment as an input. By reading the meta-data, the experiment can be run again.

```{note}
A `metadata.yml` file is saved by an experiment if any output data is saved or if the argument `--save-meta-data true` is specified explicitly.
```

Following command illustrates how the so-call "read mode" can be activated by using the argument `--mode read` and specifying the path to a directory that contains the aforementioned `metadata.yml` file via the argument `--input-dir`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode read \
       --input-dir path/to/experiment/
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode read \
       --input-dir path/to/experiment/
   ```
````
