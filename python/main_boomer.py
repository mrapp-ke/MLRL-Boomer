#!/usr/bin/python

import logging as log

from args import ArgumentParserBuilder
from boomer.boosting.boosting_learners import Boomer
from boomer.evaluation import ClassificationEvaluation, EvaluationLogOutput, EvaluationCsvOutput
from boomer.experiments import Experiment
from boomer.parameters import ParameterCsvInput
from boomer.persistence import ModelPersistence
from boomer.printing import RulePrinter, ModelPrinterLogOutput, ModelPrinterTxtOutput
from boomer.training import DataSet


def create_learner(params) -> Boomer:
    return Boomer(random_state=params.random_state, max_rules=params.max_rules, time_limit=params.time_limit,
                  loss=params.loss, pruning=params.pruning, label_sub_sampling=params.label_sub_sampling,
                  instance_sub_sampling=params.instance_sub_sampling, shrinkage=params.shrinkage,
                  feature_sub_sampling=params.feature_sub_sampling, head_refinement=params.head_refinement,
                  l2_regularization_weight=params.l2_regularization_weight, min_coverage=params.min_coverage,
                  max_conditions=params.max_conditions, max_head_refinements=params.max_head_refinements)


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment using BOOMER') \
        .add_boosting_learner_arguments() \
        .build()
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

    model_dir = args.model_dir
    persistence = None if model_dir is None else ModelPersistence(model_dir)
    learner = create_learner(args)
    parameter_input = parameter_input
    model_printer = RulePrinter(*model_printer_outputs) if len(model_printer_outputs) > 0 else None
    train_evaluation = ClassificationEvaluation(*evaluation_outputs) if args.evaluate_training_data else None
    test_evaluation = ClassificationEvaluation(*evaluation_outputs)
    data_set = DataSet(data_dir=args.data_dir, data_set_name=args.dataset, use_one_hot_encoding=args.one_hot_encoding)
    experiment = Experiment(learner, test_evaluation=test_evaluation, train_evaluation=train_evaluation,
                            data_set=data_set, num_folds=args.folds, current_fold=args.current_fold,
                            parameter_input=parameter_input, model_printer=model_printer, persistence=persistence)
    experiment.random_state = args.random_state
    experiment.run()
