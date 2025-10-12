# Fixes

- Evaluation results, models and algorithmic parameters are now saved to output files if the argument `--save-all true` is used.
- The number of decimal places used for values written into ARFF files is now chosen more carefully.
- When writing sparse predictions to ARFF files via the command line argument `--save-predictions true`, their attribute definitions are not malformed anymore.
