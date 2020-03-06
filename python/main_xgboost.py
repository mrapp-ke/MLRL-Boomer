#!/usr/bin/python

import argparse
import logging as log

from args import optional_string, log_level, boolean_string, current_fold_string
from boomer.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from boomer.experiments import Experiment
from boomer.parameters import ParameterCsvInput
from boomer.baselines.xgboost_br import XGBoostBR


def create_learner(params) -> XGBoostBR:
    return XGBoostBR(model_dir=params.model_dir, learning_rate=params.learning_rate, reg_lambda=params.reg_lambda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An multi-label classification experiment using BR with XGBoost')
    parser.add_argument('--log-level', type=log_level, default='info', help='The log level to be used')
    parser.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    parser.add_argument('--output-dir', type=optional_string, default=None,
                        help='The path of the directory into which results should be written')
    parser.add_argument('--model-dir', type=optional_string, default=None,
                        help='The path of the directory where models should be saved')
    parser.add_argument('--dataset', type=str, help='The name of the data set to be used')
    parser.add_argument('--folds', type=int, default=1, help='Total number of folds to be used by cross validation')
    parser.add_argument('--current-fold', type=current_fold_string, default=-1,
                        help='The cross validation fold to be performed')
    parser.add_argument('--learning-rate', type=float, default=1.0, help='The learning rate to be used')
    parser.add_argument('--reg-lambda', type=float, default=0.0, help='The L2 regularization weight to be used')
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--store-predictions', type=boolean_string, default=False,
                        help='True, if the predictions should be stored as CSV files, False otherwise')
    parser.add_argument('--parameter-dir', type=optional_string, default=None,
                        help='The path of the directory, parameter settings should be loaded from')
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    parameter_input = None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)
    evaluation_outputs = [EvaluationLogOutput()]

    if args.output_dir is not None:
        evaluation_outputs.append(EvaluationCsvOutput(output_dir=args.output_dir,
                                                      output_predictions=args.store_predictions,
                                                      clear_dir=args.current_fold == -1))

    learner = create_learner(args)
    parameter_input = parameter_input
    evaluation = ClassificationEvaluation(EvaluationLogOutput(), *evaluation_outputs)
    experiment = Experiment(learner, evaluation, data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                            current_fold=args.current_fold, parameter_input=parameter_input)
    experiment.run()
