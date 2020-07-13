import logging as log
from typing import List

import numpy as np
from sklearn import metrics

from args import ArgumentParserBuilder
from boomer.bbc_cv import BbcCvAdapter, BbcCv, DefaultBootstrapping, DefaultBbcCvObserver
from boomer.common.arrays import DTYPE_FLOAT64
from boomer.common.learners import Learner
from boomer.seco.seco_learners import SeparateAndConquerRuleLearner
from boomer.training import DataSet
from runnables import Runnable


class SecoBbcCbAdapter(BbcCvAdapter):

    def __init__(self, data_set: DataSet, num_folds: int, model_dir: str, min_rules: int, max_rules: int,
                 step_size_rules: int):
        super().__init__(data_set, num_folds, model_dir)
        self.min_rules = min_rules
        self.max_rules = max_rules
        self.step_size_rules = step_size_rules

    def _store_predictions(self, model: Learner, test_indices, test_x, train_y, num_total_examples: int,
                           num_labels: int,
                           predictions, configurations, current_fold, last_fold, num_folds):
        rules = model.model_.rules
        num_rules = len(rules)
        c = 0

        if len(predictions) > 0:
            current_predictions = predictions[0]
            current_config = configurations[0]
        else:
            current_predictions = np.zeros((num_total_examples, num_labels), dtype=DTYPE_FLOAT64, order='C')
            predictions.append(current_predictions)
            current_config = self.configuration.copy()
            configurations.append(current_config)

        # Store predictions...
        min_rules = self.min_rules
        min_rules = max(min_rules, 1) if min_rules != -1 else 1
        max_rules = self.max_rules
        max_rules = min(num_rules, max_rules) if max_rules != -1 else num_rules
        step_size = min(max(1, self.step_size_rules), max_rules)

        for n in range(max_rules):
            rule = rules.pop(0)

            if np.isnan(np.asarray(rule.head.scores)).any():
                log.error("There's something wrong with this rule")
                raise ArithmeticError()
            else:
                if test_indices is None:
                    rule.predict(test_x.toarray(), current_predictions)
                else:
                    masked_predictions = current_predictions[test_indices, :]
                    rule.predict(test_x, masked_predictions)
                    current_predictions[test_indices, :] = masked_predictions

            if min_rules <= (n + 1) <= max_rules - 1 and (n + 1) % step_size == 0:
                c += 1

                current_config['max_rules'] = (n + 1)

                if c < len(predictions):
                    old_predictions = current_predictions
                    current_predictions = predictions[c]

                    if test_indices is None:
                        current_predictions[:, :] = old_predictions[:, :]
                    else:
                        current_predictions[test_indices] = old_predictions[test_indices]

                    current_config = configurations[c]
                else:
                    current_predictions = current_predictions.copy()
                    predictions.append(current_predictions)
                    current_config = current_config.copy()
                    configurations.append(current_config)


def _create_learner(args):
    return SeparateAndConquerRuleLearner(random_state=args.random_state, max_rules=args.max_rules,
                                         time_limit=args.time_limit, loss=args.loss, heuristic=args.heuristic,
                                         pruning=args.pruning, label_sub_sampling=args.label_sub_sampling,
                                         instance_sub_sampling=args.instance_sub_sampling,
                                         feature_sub_sampling=args.feature_sub_sampling,
                                         head_refinement=args.head_refinement, min_coverage=args.min_coverage,
                                         max_conditions=args.max_conditions, lift_function=args.lift_function,
                                         max_head_refinements=args.max_head_refinements)


def _create_configurations() -> List[dict]:
    result: List[dict] = []

    configuration = {
        'head_refinement': 'partial',
        'heuristic': 'f-measure',
    }
    result.append(configuration)

    return result


class BbcCvRunnable(Runnable):

    def _run(self, args):
        configurations = _create_configurations()
        learner = _create_learner(args)
        data_set = DataSet(data_dir=args.data_dir, data_set_name=args.dataset,
                           use_one_hot_encoding=args.one_hot_encoding)
        adapter = SecoBbcCbAdapter(data_set=data_set, num_folds=args.folds, model_dir=args.model_dir,
                                   min_rules=5, max_rules=50,
                                   step_size_rules=1)
        bbc_cv = BbcCv(configurations, adapter, DefaultBootstrapping(5), learner)
        bbc_cv.random_state = args.random_state
        bbc_cv.store_predictions()

        bbc_cv.evaluate(observer=DefaultBbcCvObserver(output_dir=args.output_dir, target_measure=metrics.hamming_loss,
                                                      target_measure_is_loss=True))


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment using Separate and Conquer') \
        .add_seco_learner_arguments() \
        .build()
    runnable = BbcCvRunnable()
    runnable.run(parser)
