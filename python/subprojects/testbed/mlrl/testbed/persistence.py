"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for saving/loading models to/from disk.
"""
import logging as log

from os import path

import _pickle as pickle

from mlrl.testbed.fold import Fold
from mlrl.testbed.util.io import get_file_name_per_fold


class ModelLoader:
    """
    Allows to load models from a directory.
    """

    def __init__(self, directory: str):
        """
        :param directory: The path to the directory from which models should be loaded
        """
        self.directory = directory

    def load_model(self, fold: Fold):
        """
        Loads a model from a file.

        :param fold:    The fold of the available data, the model corresponds to
        :return:        The loaded model
        """
        file_name = get_file_name_per_fold('model', 'pickle', fold.index)
        file_path = path.join(self.directory, file_name)
        log.debug('Loading model from file \"%s\"...', file_path)

        try:
            with open(file_path, mode='rb') as input_stream:
                model = pickle.load(input_stream)
                log.info('Successfully loaded model from file \"%s\"', file_path)
                return model
        except IOError:
            log.error('Failed to load model from file \"%s\"', file_path)
            return None
