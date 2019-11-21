#!/usr/bin/python

import argparse
import logging as log

from boomer.algorithm._losses import SquaredErrorLoss
from boomer.algorithm._sub_sampling import Bagging, RandomFeatureSubsetSelection

from boomer.algorithm.boomer import Boomer
from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.prediction import Sign, LinearCombination
from boomer.algorithm.rule_induction import GradientBoosting
from boomer.evaluation import ClassificationEvaluation, LogOutput, CsvOutput
from boomer.experiments import Experiment


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def loss_string(s):
    if s == 'squared-error-loss':
        return SquaredErrorLoss()
    raise ValueError('Not a valid Loss string')


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
    parser.add_argument('--num-rules', type=int, default=100, help='The number of rules to be induced')
    parser.add_argument('--bagging', type=boolean_string, default=False,
                        help='True, if bagging should be used, False otherwise')
    parser.add_argument('--feature-sampling', type=boolean_string, default=False,
                        help='True, if random feature subset selection should be used, False otherwise')
    parser.add_argument('--loss', type=loss_string, default='squared-error-loss',
                        help='The name of the loss function to be used')
    parser.add_argument('--shrinkage', type=float, default=1, help='The shrinkage parameter to be used')
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    experiment_name = args.dataset

    instance_sub_sampling = Bagging() if args.bagging else None
    feature_sub_sampling = RandomFeatureSubsetSelection() if args.feature_sampling else None
    rule_induction = GradientBoosting(num_rules=args.num_rules, loss=args.loss,
                                      instance_sub_sampling=instance_sub_sampling,
                                      feature_sub_sampling=feature_sub_sampling, shrinkage=args.shrinkage)
    learner = Boomer(rule_induction=rule_induction, prediction=Sign(LinearCombination()))
    learner.random_state = args.random_state

    if args.model_dir is not None:
        learner.persistence = ModelPersistence(model_dir=args.model_dir, model_name=args.dataset)

    evaluation = ClassificationEvaluation(LogOutput(), CsvOutput(output_dir=args.output_dir,
                                                                 output_predictions=args.store_predictions))

    experiment = Experiment(experiment_name, learner, evaluation, data_dir=args.data_dir, data_set=args.dataset,
                            folds=args.folds)
    experiment.run()
