#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
from args import ArgumentParserBuilder
from mlrl.seco.seco_learners import SeparateAndConquerRuleLearner
from runnables import RuleLearnerRunnable


class SecoRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return SeparateAndConquerRuleLearner(random_state=args.random_state, feature_format=args.feature_format,
                                             label_format=args.label_format, max_rules=args.max_rules,
                                             time_limit=args.time_limit, loss=args.loss, heuristic=args.heuristic,
                                             pruning_heuristic=args.pruning_heuristic, pruning=args.pruning,
                                             label_sampling=args.label_sampling,
                                             instance_sampling=args.instance_sampling,
                                             feature_sampling=args.feature_sampling, holdout=args.holdout,
                                             feature_binning=args.feature_binning, head_type=args.head_type,
                                             min_coverage=args.min_coverage, max_conditions=args.max_conditions,
                                             lift_function=args.lift_function,
                                             max_head_refinements=args.max_head_refinements,
                                             num_threads_rule_refinement=args.num_threads_refinement,
                                             num_threads_statistic_update=args.num_threads_statistic_update,
                                             num_threads_prediction=args.num_threads_prediction)


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment using Separate and Conquer') \
        .add_seco_learner_arguments() \
        .build()
    runnable = SecoRunnable()
    runnable.run(parser)
