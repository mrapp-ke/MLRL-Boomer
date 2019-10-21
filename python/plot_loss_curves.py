#!/usr/bin/python

import argparse
import logging as log
import os.path as path

import matplotlib.pyplot as plt
from skmultilearn.base import MLClassifierBase

from boomer.algorithm.boomer import Boomer
from boomer.algorithm.model import Theory
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.prediction import LinearCombination
from boomer.algorithm.stats import Stats
from boomer.evaluation import ClassificationEvaluation, HAMMING_LOSS
from boomer.experiments import CrossValidation


class Plotter(CrossValidation, MLClassifierBase):
    """
    Plots the performance of a model at each iteration.
    """

    MEASURES = [HAMMING_LOSS]

    evaluation = ClassificationEvaluation()

    prediction = LinearCombination()

    def __init__(self, model_dir: str, output_dir: str, data_dir: str, data_set: str, folds: int = 1):
        super().__init__(data_dir, data_set, folds)
        self.output_dir = output_dir
        self.require_dense = [True, True]  # We need a dense representation of the training data
        self.persistence = ModelPersistence(model_dir=model_dir, model_name=data_set)

    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, current_fold: int, total_folds: int):
        # Create a dense representation of the training data
        train_x = self._ensure_input_format(train_x)
        train_y = self._ensure_input_format(train_y)
        test_x = self._ensure_input_format(test_x)
        test_y = self._ensure_input_format(test_y)

        theory: Theory = self.__load_theory(current_fold)
        current_model = []
        num_iterations = len(theory)

        for i in range(0, num_iterations):
            log.info("Evaluating model at iteration %s / %s...", i + 1, num_iterations)

            rule = theory.pop(0)
            current_model.append(rule)

            predictions = self.prediction.predict(Stats.create_stats(train_x, train_y), current_model, train_x)
            name = Plotter.__get_experiment_name(prefix='train', iteration=i)
            self.evaluation.evaluate(name, predictions, train_y, current_fold=current_fold, total_folds=total_folds)

            predictions = self.prediction.predict(Stats.create_stats(test_x, test_y), current_model, test_x)
            name = Plotter.__get_experiment_name(prefix='test', iteration=i)
            self.evaluation.evaluate(name, predictions, test_y, current_fold=current_fold, total_folds=total_folds)

        if total_folds < 1 or current_fold == total_folds - 1:
            self.__plot(num_iterations=num_iterations)

    def __load_theory(self, fold: int):
        theory = self.persistence.load_model(file_name_suffix=Boomer.PREFIX_RULES, fold=fold)

        if theory is None:
            raise IOError('Unable to load model')

        return theory

    def __plot(self, num_iterations: int):
        log.info('Creating plot...')

        for measure in Plotter.MEASURES:
            for prefix in ['train', 'test']:
                x = []
                y = []

                for i in range(0, num_iterations):
                    name = Plotter.__get_experiment_name(prefix=prefix, iteration=i)
                    evaluation_result = self.evaluation.results[name]
                    score, std_dev = evaluation_result.avg(measure)
                    x.append(i + 1)
                    y.append(score)

                plt.plot(x, y, label=prefix)

            plt.legend()
            output_file = path.join(self.output_dir, self.data_set + '.pdf')
            plt.savefig(output_file)

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    @staticmethod
    def __get_experiment_name(prefix: str, iteration: int) -> str:
        name = prefix + ('_' if len(prefix) > 0 else '')
        name += 'iteration-' + str(iteration)
        return name


if __name__ == '__main__':
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(description='An multi-label classification experiment using BOOMER')
    parser.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    parser.add_argument('--output-dir', type=str, help='The path of the directory into which plots should be written')
    parser.add_argument('--model-dir', type=str, help='The path of the directory where models should be saved')
    parser.add_argument('--dataset', type=str, help='The name of the data set to be used')
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    plotter = Plotter(model_dir=args.model_dir, output_dir=args.output_dir, data_dir=args.data_dir,
                      data_set=args.dataset, folds=10)
    plotter.run()
