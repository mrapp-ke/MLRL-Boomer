#!/usr/bin/python

import argparse
import logging as log

from boomer.algorithm._head_refinement import SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._losses import SquaredErrorLoss
from boomer.algorithm._pruning import IREP
from boomer.algorithm._sub_sampling import Bagging, RandomFeatureSubsetSelection

from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.rule_learners import Boomer
from boomer.evaluation import ClassificationEvaluation, LogOutput, CsvOutput
from boomer.experiments import BatchExperiment


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def instance_sub_sampling_string(s):
    if s == 'bagging':
        return Bagging()
    return None


def feature_sub_sampling_string(s):
    if s == 'random-feature-selection':
        return RandomFeatureSubsetSelection()
    return None


def pruning_string(s):
    if s == 'irep':
        return IREP()
    return None


def loss_string(s):
    if s == 'squared-error-loss':
        return SquaredErrorLoss()
    raise ValueError('Not a valid Loss string')


def head_refinement_string(s):
    if s == 'single-label':
        return SingleLabelHeadRefinement()
    elif s == 'full':
        return FullHeadRefinement()
    return None


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
    parser.add_argument('--num-rules', type=int, default=100, help='The number of rules to be induced per iteration')
    parser.add_argument('--iterations', type=int, default=1, help='The number of iterations')
    parser.add_argument('--instance-sub-sampling', type=instance_sub_sampling_string, default=None,
                        help='The name of the strategy to be used for instance sub-sampling or None')
    parser.add_argument('--feature-sub-sampling', type=feature_sub_sampling_string, default=None,
                        help='The name of the strategy to be used for feature sub-sampling or None')
    parser.add_argument('--pruning', type=pruning_string, default=None,
                        help='The name of the strategy to be used for pruning or None')
    parser.add_argument('--loss', type=loss_string, default='squared-error-loss',
                        help='The name of the loss function to be used')
    parser.add_argument('--head-refinement', type=head_refinement_string, default=None)
    parser.add_argument('--shrinkage', type=float, default=1, help='The shrinkage parameter to be used')
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    experiment = None

    for i in range(1, args.iterations + 1):
        num_rules = args.num_rules * i
        model_dir = args.model_dir
        model_name = args.dataset + '_num_rules=' + str(num_rules)

        if experiment is None:
            default_learner = Boomer(num_rules=num_rules, loss=args.loss, pruning=args.pruning,
                                     instance_sub_sampling=args.instance_sub_sampling, shrinkage=args.shrinkage,
                                     feature_sub_sampling=args.feature_sub_sampling)
            default_learner.random_state = args.random_state

            if model_dir is not None:
                default_learner.persistence = ModelPersistence(model_dir=model_dir, model_name=model_name)

            evaluation = ClassificationEvaluation(LogOutput(), CsvOutput(output_dir=args.output_dir,
                                                                         output_predictions=args.store_predictions))
            experiment = BatchExperiment(default_learner, evaluation, data_dir=args.data_dir, data_set=args.dataset,
                                         folds=args.folds)
        else:
            persistence = None if model_dir is None else ModelPersistence(model_dir=model_dir, model_name=model_name)
            experiment.add_variant(num_rules=num_rules, persistence=persistence)

    experiment.run()
