#!/usr/bin/python

import argparse
import logging as log

from boomer.algorithm._head_refinement import SingleLabelHeadRefinement, FullHeadRefinement
from boomer.algorithm._losses import SquaredErrorLoss
from boomer.algorithm._pruning import IREP
from boomer.algorithm._sub_sampling import Bagging, RandomInstanceSubsetSelection, RandomFeatureSubsetSelection

from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.rule_learners import Boomer
from boomer.evaluation import ClassificationEvaluation, LogOutput, CsvOutput
from boomer.experiments import BatchExperiment


def __boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def __instance_sub_sampling_string(s):
    if s == 'bagging':
        return Bagging()
    elif s == 'random-instance-selection':
        return RandomInstanceSubsetSelection()
    return None


def __feature_sub_sampling_string(s):
    if s == 'random-feature-selection':
        return RandomFeatureSubsetSelection()
    return None


def __pruning_string(s):
    if s == 'irep':
        return IREP()
    return None


def __loss_string(s):
    if s == 'squared-error-loss':
        return SquaredErrorLoss()
    raise ValueError('Not a valid Loss string')


def __head_refinement_string(s):
    if s == 'single-label':
        return SingleLabelHeadRefinement()
    elif s == 'full':
        return FullHeadRefinement()
    return None


def configure_argument_parser(p: argparse.ArgumentParser):
    p.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    p.add_argument('--output-dir', type=str, help='The path of the directory into which results should be written')
    p.add_argument('--model-dir', type=str, default=None, help='The path of the directory where models should be saved')
    p.add_argument('--dataset', type=str, help='The name of the data set to be used')
    p.add_argument('--folds', type=int, default=1, help='Number of folds to be used by cross validation')
    p.add_argument('--num-rules', type=int, default=100, help='The number of rules to be induced per iteration')
    p.add_argument('--instance-sub-sampling', type=__instance_sub_sampling_string, default=None,
                   help='The name of the strategy to be used for instance sub-sampling or None')
    p.add_argument('--feature-sub-sampling', type=__feature_sub_sampling_string, default=None,
                   help='The name of the strategy to be used for feature sub-sampling or None')
    p.add_argument('--pruning', type=__pruning_string, default=None,
                   help='The name of the strategy to be used for pruning or None')
    p.add_argument('--loss', type=__loss_string, default='squared-error-loss',
                   help='The name of the loss function to be used')
    p.add_argument('--head-refinement', type=__head_refinement_string, default=None)
    p.add_argument('--shrinkage', type=float, default=1, help='The shrinkage parameter to be used')


def create_learner(rules: int, params) -> Boomer:
    return Boomer(num_rules=rules, loss=params.loss, pruning=params.pruning,
                  instance_sub_sampling=params.instance_sub_sampling, shrinkage=params.shrinkage,
                  feature_sub_sampling=params.feature_sub_sampling, head_refinement=params.head_refinement)


if __name__ == '__main__':
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(description='An multi-label classification experiment using BOOMER')
    configure_argument_parser(parser)
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--store-predictions', type=__boolean_string, default=False,
                        help='True, if the predictions should be stored as CSV files, False otherwise')
    parser.add_argument('--iterations', type=int, default=1, help='The number of iterations')
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    experiment = None
    persistence = None if args.model_dir is None else ModelPersistence(model_dir=args.model_dir)

    for i in range(1, args.iterations + 1):
        num_rules = args.num_rules * i

        if experiment is None:
            default_learner = create_learner(num_rules, args)
            default_learner.random_state = args.random_state
            default_learner.persistence = persistence
            evaluation = ClassificationEvaluation(LogOutput(), CsvOutput(output_dir=args.output_dir,
                                                                         output_predictions=args.store_predictions))
            experiment = BatchExperiment(default_learner, evaluation, data_dir=args.data_dir, data_set=args.dataset,
                                         folds=args.folds)
        else:
            experiment.add_variant(num_rules=num_rules, persistence=persistence)

    experiment.run()
