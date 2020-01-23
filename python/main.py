#!/usr/bin/python

import argparse
import logging as log

from boomer.algorithm.persistence import ModelPersistence
from boomer.algorithm.rule_learners import Boomer
from boomer.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from boomer.experiments import Experiment


def __boolean_string(s):
    if s.lower() == 'false':
        return False
    if s.lower() == 'true':
        return True
    raise ValueError('Invalid boolean argument given: ' + str(s))


def __optional_string(s):
    if s is None or s.lower() == 'none':
        return None
    return s


def __current_fold_string(s):
    n = int(s)
    if n > 0:
        return n - 1
    elif n == -1:
        return -1
    raise ValueError('Invalid argument given for parameter \'--current-fold\': ' + str(n))


def configure_argument_parser(p: argparse.ArgumentParser):
    p.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    p.add_argument('--output-dir', type=str, help='The path of the directory into which results should be written')
    p.add_argument('--model-dir', type=__optional_string, default=None,
                   help='The path of the directory where models should be saved')
    p.add_argument('--dataset', type=str, help='The name of the data set to be used')
    p.add_argument('--folds', type=int, default=1, help='Total number of folds to be used by cross validation')
    p.add_argument('--current-fold', type=__current_fold_string, default=-1,
                   help='The cross validation fold to be performed')
    p.add_argument('--num-rules', type=int, default=100, help='The number of rules to be induced')
    p.add_argument('--instance-sub-sampling', type=__optional_string, default=None,
                   help='The name of the strategy to be used for instance sub-sampling or None')
    p.add_argument('--feature-sub-sampling', type=__optional_string, default=None,
                   help='The name of the strategy to be used for feature sub-sampling or None')
    p.add_argument('--pruning', type=__optional_string, default=None,
                   help='The name of the strategy to be used for pruning or None')
    p.add_argument('--loss', type=str, default='squared-error-loss', help='The name of the loss function to be used')
    p.add_argument('--head-refinement', type=__optional_string, default=None,
                   help='The name of the strategy to be used for finding the heads of rules')
    p.add_argument('--shrinkage', type=float, default=1.0, help='The shrinkage parameter to be used')


def create_learner(params) -> Boomer:
    return Boomer(num_rules=params.num_rules, loss=params.loss, pruning=params.pruning,
                  instance_sub_sampling=params.instance_sub_sampling, shrinkage=params.shrinkage,
                  feature_sub_sampling=params.feature_sub_sampling, head_refinement=params.head_refinement)


if __name__ == '__main__':
    log.basicConfig(level=log.INFO)

    parser = argparse.ArgumentParser(description='An multi-label classification experiment using BOOMER')
    configure_argument_parser(parser)
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--store-predictions', type=__boolean_string, default=False,
                        help='True, if the predictions should be stored as CSV files, False otherwise')
    args = parser.parse_args()
    log.info('Configuration: %s', args)

    learner = create_learner(args)
    learner.random_state = args.random_state
    learner.persistence = None if args.model_dir is None else ModelPersistence(model_dir=args.model_dir)
    evaluation = ClassificationEvaluation(EvaluationLogOutput(),
                                          EvaluationCsvOutput(output_dir=args.output_dir,
                                                              output_predictions=args.store_predictions,
                                                              clear_dir=args.current_fold == -1))
    experiment = Experiment(learner, evaluation, data_dir=args.data_dir, data_set=args.dataset, num_folds=args.folds,
                            current_fold=args.current_fold)
    experiment.run()
