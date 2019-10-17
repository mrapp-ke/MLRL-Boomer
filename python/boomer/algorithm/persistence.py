#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for saving/loading models to/from disk.
"""
import _pickle as pickle
import os.path as path
import logging as log


class ModelPersistence:
    """
    Allows to save a model in a file and load it later.
    """

    def __init__(self, model_dir: str, model_name: str):
        """
        :param model_dir:   The path of the directory where models should be saved
        :param model_name:  The name of the model
        """
        self.model_dir = model_dir
        self.model_name = model_name

    def save_model(self, model, file_name_suffix: str = None, fold: int = None):
        """
        Saves a model to a file.

        :param file_name_suffix:    A suffix to be added to the name of the file
        :param model:               The model to be persisted
        :param fold:                The fold the model corresponds to or None, if no cross validation is used
        """

        file_path = path.join(self.model_dir, self.__get_file_name(file_name_suffix, fold))
        log.debug('Saving model to file \"%s\"...', file_path)

        try:
            with open(file_path, mode='wb') as output_stream:
                pickle.dump(model, output_stream, -1)
                log.info('Successfully saved model to file \"%s\"', file_path)
        except IOError:
            log.error('Failed to save model to file \"%s\"', file_path)

    def load_model(self, file_name_suffix: str = None, fold: int = None):
        """
        Loads a model from a file.

        :param file_name_suffix:    A suffix to be added to the name of the file
        :param fold:        The fold the model corresponds to or None, if no cross validation is used
        :return:                    The loaded model
        """

        file_path = path.join(self.model_dir, self.__get_file_name(file_name_suffix, fold))
        log.debug("Loading model from file \"%s\"...", file_path)

        try:
            with open(file_path, mode='rb') as input_stream:
                model = pickle.load(input_stream)
                log.info('Successfully loaded model from file \"%s\"', file_path)
                return model
        except IOError:
            log.error('Failed to load model from file \"%s\"', file_path)
            return None

    def __get_file_name(self, file_name_suffix: str, fold: int):
        file_name = self.model_name

        if file_name_suffix is not None:
            file_name += ('_' + file_name_suffix)

        if fold is not None:
            file_name += ('_fold-' + str(fold + 1))

        return file_name + '.model'
