"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for specifying the aspects of an experiment's state that should be taken into account for finding a
suitable source or sink for handling data.
"""
from dataclasses import dataclass


@dataclass
class Context:
    """
    Specifies the aspects of an experiment's state that should be taken into account for finding a suitable source or
    sink for handling data.

    Attributes:
        include_dataset_type:       True, if the type of the dataset should be taken into account, False otherwise
        include_prediction_scope:   True, if the scope of predictions should be taken into account, False otherwise
        include_fold:               True, if the cross validation fold should be taken into account, False otherwise
    """
    include_dataset_type: bool = True
    include_prediction_scope: bool = True
    include_fold: bool = True
