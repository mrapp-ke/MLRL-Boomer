(testbed-read-mode)=

# Read Mode

When mlrl-testbed operates in so-called "read mode", it reads the output data produced by a previously run experiment and allows to print it on the console or write it to different output files. Similar to the {ref}`run mode <testbed-run-mode>`, this mode of operation is based on the `metadata.yml` file that has been saved by a previous experiment.

```{note}
A `metadata.yml` file is saved by an experiment if any output data is saved or if the argument `--save-meta-data true` is specified explicitly (see {ref}`here <meta-data>` for more details).
```

The command below shows how the read mode can be enabled via the argument `--mode read`. This requires specifying an absolute or relative path to a directory that contains the aforementioned `metadata.yml` file via the argument `--input-dir`. In the given example, the evaluation results of the previously run experiment are printed (`--print-evaluation true`) and saved to output files (`--save-evaluation true`) in a given `--base-dir`. Other output data can be printed or saved via the respective arguments described {ref}`here <testbed-outputs>`.

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --mode read \
       --input-dir path/to/experiment/
       --base-dir some/arbitrary/path/
       --print-evaluation true
       --save-evaluation true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --mode read \
       --input-dir path/to/experiment/
       --base-dir some/arbitrary/path/
       --print-evaluation true
       --save-evaluation true
   ```
````

Of course, only data that has been saved to output files when the experiment was run originally can be accessed in read mode. If you are interested in data that has not been saved previously, you can use {ref}`run mode <testbed-run-mode>` to run experiments again. For this reason, a few arguments or some of their options listed {ref}`here <testbed-outputs>` may not be available in read mode, as is noted under the mentioned link.

If the meta-data that is provided to the read mode consists of multiple experiments, as is the case if the experiment was run in {ref}`batch mode <testbed-batch-mode>`, the output data of all these experiments will be printed or saved.
