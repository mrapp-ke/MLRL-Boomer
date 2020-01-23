#!/usr/bin/python

import argparse
import logging as log
import os.path as path

import matplotlib.pyplot as plt
import numpy as np
from skmultilearn.base import MLClassifierBase

from boomer.algorithm.model import Theory, DTYPE_FLOAT32, DTYPE_FLOAT64
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.rule_learners import Boomer
from boomer.evaluation import ClassificationEvaluation, HAMMING_LOSS, SUBSET_01_LOSS
from boomer.training import CrossValidation
from main import configure_argument_parser, create_learner


class Plotter(CrossValidation, MLClassifierBase):
    """
    Plots the performance of a model at each iteration.
    """

    MEASURES = [HAMMING_LOSS, SUBSET_01_LOSS]

    evaluation = ClassificationEvaluation()

    def __init__(self, model_dir: str, output_dir: str, data_dir: str, data_set: str, num_folds: int, current_fold: int,
                 learner_name: str, model_name: str):
        super().__init__(data_dir, data_set, num_folds, current_fold)
        self.output_dir = output_dir
        self.require_dense = [True, True]  # We need a dense representation of the training data
        self.persistence = ModelPersistence(model_dir=model_dir)
        self.learner_name = learner_name
        self.model_name = model_name
        self.data_set = data_set

    def _train_and_evaluate(self, train_x, train_y, test_x, test_y, first_fold: int, current_fold: int, last_fold: int,
                            num_folds: int):
        # Create a dense representation of the training data
        train_x = np.asfortranarray(self._ensure_input_format(train_x), dtype=DTYPE_FLOAT32)
        train_y = self._ensure_input_format(train_y)
        test_x = np.asfortranarray(self._ensure_input_format(test_x), dtype=DTYPE_FLOAT32)
        test_y = self._ensure_input_format(test_y)

        theory: Theory = self.__load_theory(current_fold)
        num_iterations = len(theory)

        train_predictions = np.asfortranarray(np.zeros((train_x.shape[0], train_y.shape[1]), dtype=DTYPE_FLOAT64))
        test_predictions = np.asfortranarray(np.zeros((test_x.shape[0], test_y.shape[1]), dtype=DTYPE_FLOAT64))

        for i in range(0, num_iterations):
            log.info("Evaluating model at iteration %s / %s...", i + 1, num_iterations)

            rule = theory.pop(0)

            rule.predict(train_x, train_predictions)
            name = Plotter.__get_experiment_name(prefix='train', iteration=i)
            self.evaluation.evaluate(name, np.where(train_predictions > 0, 1, 0), train_y, first_fold=first_fold,
                                     current_fold=current_fold, last_fold=last_fold, num_folds=num_folds)

            rule.predict(test_x, test_predictions)
            name = Plotter.__get_experiment_name(prefix='test', iteration=i)
            self.evaluation.evaluate(name, np.where(test_predictions > 0, 1, 0), test_y, first_fold=first_fold,
                                     current_fold=current_fold, last_fold=last_fold, num_folds=num_folds)

        if current_fold == last_fold:
            self.__plot(num_iterations=num_iterations)

    def __load_theory(self, fold: int):
        theory = self.persistence.load_model(model_name=self.model_name, file_name_suffix=Boomer.PREFIX_RULES,
                                             fold=fold)

        if theory is None:
            raise IOError('Unable to load model')

        return theory

    def __plot(self, num_iterations: int):
        log.info('Creating plots...')

        for measure in Plotter.MEASURES:
            plt.title(self.data_set)

            # Customize x axis
            plt.xlabel('# rules')
            plt.xlim(left=0, right=num_iterations)
            x_ticks = np.arange(0, num_iterations + 200, 200)
            plt.xticks(ticks=x_ticks)

            # Draw vertical lines
            prev_x = None
            for x in x_ticks:
                if prev_x is not None:
                    new_x = prev_x + ((x - prev_x) / 2)
                    plt.plot([new_x, new_x], [0, 1.0], color='0.5', linestyle='dotted', linewidth=1)
                if 0 < x < num_iterations:
                    plt.plot([x, x], [0, 1.0], color='0.5', linestyle='dotted', linewidth=1)
                prev_x = x

            # Customize y axis
            plt.ylim(bottom=0, top=1)
            y_labels = [str(i * 10) + '%' for i in range(11)]
            y_ticks = np.arange(0, 1.1, 0.1)
            plt.yticks(ticks=y_ticks, labels=y_labels)

            # Draw horizontal lines
            prev_y = None
            for y in y_ticks:
                if prev_y is not None:
                    new_y = prev_y + ((y - prev_y) / 2)
                    plt.plot([0, num_iterations], [new_y, new_y], color='0.5', linestyle='dotted', linewidth=1)
                if 0 < y < 1.0:
                    plt.plot([0, num_iterations], [y, y], color='0.5', linestyle='dotted', linewidth=1)
                prev_y = y

            # Draw curves
            for prefix in ['train', 'test']:
                x = []
                y = []

                for i in range(0, num_iterations):
                    name = Plotter.__get_experiment_name(prefix=prefix, iteration=i)
                    evaluation_result = self.evaluation.results[name]
                    score, std_dev = evaluation_result.avg(measure)
                    x.append(i + 1)
                    y.append(score)

                plt.plot(x, y, label=(measure + ' (' + prefix + ')'))

            plt.legend()
            file_name = self.learner_name + '.pdf'
            output_file = path.join(self.output_dir, file_name)
            log.info('Saving plot to file \'' + output_file + '\'...')
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

    parser = argparse.ArgumentParser(description='Plots the performance of a BOOMER model')
    configure_argument_parser(parser)
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    learner = create_learner(args)
    plotter = Plotter(model_dir=args.model_dir, output_dir=args.output_dir, data_dir=args.data_dir,
                      data_set=args.dataset, num_folds=args.folds, current_fold=args.current_fold,
                      model_name=learner.get_model_name(), learner_name=learner.get_name())
    plotter.run()
