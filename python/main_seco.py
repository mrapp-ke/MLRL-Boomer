#!/usr/bin/python

import argparse
import logging as log

from args import current_fold_string
from args import optional_string, log_level, boolean_string
from boomer.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from boomer.experiments import Experiment
from boomer.parameters import ParameterCsvInput
from boomer.printing import RulePrinter, ModelPrinterLogOutput, ModelPrinterTxtOutput
from boomer.seco.seco_learners import SeparateAndConquerRuleLearner, AVERAGING_LABEL_WISE, HEURISTIC_PRECISION, \
    LIFT_FUNCTION_PEAK
from boomer.training import DataSet


def configure_argument_parser(p: argparse.ArgumentParser):
    p.add_argument('--log-level', type=log_level, default='info', help='The log level to be used')
    p.add_argument('--data-dir', type=str, help='The path of the directory where the data sets are located')
    p.add_argument('--output-dir', type=optional_string, default=None,
                   help='The path of the directory into which results should be written')
    p.add_argument('--model-dir', type=optional_string, default=None,
                   help='The path of the directory where models should be saved')
    p.add_argument('--dataset', type=str, help='The name of the data set to be used')
    p.add_argument('--one-hot-encoding', type=boolean_string, default=False,
                   help='True, if one-hot-encoding should be used, False otherwise')
    p.add_argument('--folds', type=int, default=1, help='Total number of folds to be used by cross validation')
    p.add_argument('--current-fold', type=current_fold_string, default=-1,
                   help='The cross validation fold to be performed')
    p.add_argument('--max-rules', type=int, default=500, help='The maximum number of rules to be induced or -1')
    p.add_argument('--time-limit', type=int, default=-1,
                   help='The duration in seconds after which the induction of rules should be canceled or -1')
    p.add_argument('--label-sub-sampling', type=optional_string, default=None,
                   help='The name of the strategy to be used for label sub-sampling or None')
    p.add_argument('--instance-sub-sampling', type=optional_string, default=None,
                   help='The name of the strategy to be used for instance sub-sampling or None')
    p.add_argument('--feature-sub-sampling', type=optional_string, default=None,
                   help='The name of the strategy to be used for feature sub-sampling or None')
    p.add_argument('--pruning', type=optional_string, default=None,
                   help='The name of the strategy to be used for pruning or None')
    p.add_argument('--loss', type=str, default=AVERAGING_LABEL_WISE, help='The name of the loss function to be used')
    p.add_argument('--heuristic', type=str, default=HEURISTIC_PRECISION, help='The name of the heuristic to be used')
    p.add_argument('--head-refinement', type=optional_string, default=None,
                   help='The name of the strategy to be used for finding the heads of rules')
    p.add_argument('--min-coverage', type=int, default=1,
                   help='The minimum number of training examples that must be covered by a rule')
    p.add_argument('--max-conditions', type=int, default=-1,
                   help='The maximum number of conditions to be included in a rule\'s body or -1')
    p.add_argument('--max-head-refinements', type=int, default=1,
                   help='The maximum number of times the head of a rule may be refined or -1')
    p.add_argument('--lift-function', type=optional_string, default=LIFT_FUNCTION_PEAK,
                   help='The lift function to be used')


def create_learner(params) -> SeparateAndConquerRuleLearner:
    return SeparateAndConquerRuleLearner(random_state=params.random_state, max_rules=params.max_rules,
                                         time_limit=params.time_limit, loss=params.loss, heuristic=params.heuristic,
                                         pruning=params.pruning, label_sub_sampling=params.label_sub_sampling,
                                         instance_sub_sampling=params.instance_sub_sampling,
                                         feature_sub_sampling=params.feature_sub_sampling,
                                         head_refinement=params.head_refinement, min_coverage=params.min_coverage,
                                         max_conditions=params.max_conditions, lift_function=params.lift_function,
                                         max_head_refinements=params.max_head_refinements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A multi-label classification experiment using Separate and Conquer')
    configure_argument_parser(parser)
    parser.add_argument('--random-state', type=int, default=1, help='The seed to be used by RNGs')
    parser.add_argument('--store-predictions', type=boolean_string, default=False,
                        help='True, if the predictions should be stored as CSV files, False otherwise')
    parser.add_argument('--print-rules', type=boolean_string, default=True,
                        help='True, if the induced rules should be printed on the console, False otherwise')
    parser.add_argument('--store-rules', type=boolean_string, default=False,
                        help='True, if the induced rules should be stored in TXT files, False otherwise')
    parser.add_argument('--parameter-dir', type=optional_string, default=None,
                        help='The path of the directory, parameter settings should be loaded from')
    parser.add_argument('--evaluate-training-data', type=boolean_string, default=False,
                        help='True, if the models should be evaluated on the training data, False otherwise')
    args = parser.parse_args()
    log.basicConfig(level=args.log_level)
    log.info('Configuration: %s', args)

    parameter_input = None if args.parameter_dir is None else ParameterCsvInput(input_dir=args.parameter_dir)
    evaluation_outputs = [EvaluationLogOutput()]
    model_printer_outputs = []
    output_dir = args.output_dir

    if args.print_rules:
        model_printer_outputs.append(ModelPrinterLogOutput())

    if output_dir is not None:
        evaluation_outputs.append(EvaluationCsvOutput(output_dir=output_dir, output_predictions=args.store_predictions,
                                                      clear_dir=args.current_fold == -1))

        if args.store_rules:
            model_printer_outputs.append(ModelPrinterTxtOutput(output_dir=output_dir, clear_dir=False))

    learner = create_learner(args)
    parameter_input = parameter_input
    model_printer = RulePrinter(*model_printer_outputs) if len(model_printer_outputs) > 0 else None
    train_evaluation = ClassificationEvaluation(*evaluation_outputs) if args.evaluate_training_data else None
    test_evaluation = ClassificationEvaluation(*evaluation_outputs)
    data_set = DataSet(data_dir=args.data_dir, data_set_name=args.dataset, use_one_hot_encoding=args.one_hot_encoding)
    experiment = Experiment(learner, test_evaluation=test_evaluation, train_evaluation=train_evaluation,
                            data_set=data_set, num_folds=args.folds, current_fold=args.current_fold,
                            parameter_input=parameter_input, model_printer=model_printer)
    experiment.run()
