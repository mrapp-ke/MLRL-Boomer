"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from dataclasses import dataclass


@dataclass
class Dataset:
    """
    Provides the names of datasets that can be used in tests.

    Attributes:
        default:            The name of the dataset that should be used by default
        numerical:          The name of a dataset with numerical features
        numerical_sparse:   The name of a dataset with sparse numerical features
        binary:             The name of a dataset with binary features
        nominal:            The name of a dataset with nominal features
        ordinal:            The name of a dataset with ordinal features
        single_output:      The name of a dataset with a single target variable
        meka:               The name of a dataset in the MEKA format
    """

    EMOTIONS = 'emotions'

    EMOTIONS_NOMINAL = 'emotions-nominal'

    EMOTIONS_ORDINAL = 'emotions-ordinal'

    ENRON = 'enron'

    LANGLOG = 'langlog'

    YEAST = 'yeast'

    BREAST_CANCER = 'breast-cancer'

    MEKA = 'meka'

    ATP7D = 'atp7d'

    ATP7D_NUMERICAL_SPARSE = 'atp7d-numerical-sparse'

    ATP7D_NOMINAL = 'atp7d-nominal'

    ATP7D_BINARY = 'atp7d-binary'

    ATP7D_ORDINAL = 'atp7d-ordinal'

    ATP7D_MEKA = 'atp7d-meka'

    BODYFAT = 'bodyfat'

    HOUSING = 'housing'

    default: str = EMOTIONS
    numerical: str = EMOTIONS
    numerical_sparse: str = LANGLOG
    binary: str = ENRON
    nominal: str = EMOTIONS_NOMINAL
    ordinal: str = EMOTIONS_ORDINAL
    single_output: str = BREAST_CANCER
    meka: str = MEKA
    svm: str = YEAST
