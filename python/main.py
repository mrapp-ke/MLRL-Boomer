#!/usr/bin/python

import argparse
import logging as log

from boomer.algorithm.boomer import Boomer
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.prediction import LinearCombination
from boomer.evaluation import SquaredErrorLossEvaluation, LogOutput, CsvOutput
from boomer.experiments import Experiment


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(description='An multi-label classification experiment using BOOMER')
    parser.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    parser.add_argument('--output-dir', type=str, help='The path of the directory into which results should be written')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='The path of the directory where models should be saved')
    parser.add_argument('--dataset', type=str, help='The name of the data set to be used')
    parser.add_argument('--folds', type=int, default=1, help='Number of folds to be used by cross validation')
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--store-predictions', type=boolean_string, default=False,
                        help='True, if the predictions should be stored as CSV files, False otherwise')
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    experiment_name = args.dataset
    learner = Boomer(prediction=LinearCombination())
    learner.random_state = args.random_state

    if args.model_dir is not None:
        learner.persistence = ModelPersistence(model_dir=args.model_dir, model_name=args.dataset)

    evaluation = SquaredErrorLossEvaluation(LogOutput(), CsvOutput(output_dir=args.output_dir,
                                                                   output_predictions=args.store_predictions))

    experiment = Experiment(experiment_name, learner, evaluation, data_dir=args.data_dir, data_set=args.dataset,
                            folds=args.folds)
    experiment.run()
