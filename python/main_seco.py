#!/usr/bin/python

from args import ArgumentParserBuilder
from boomer.seco.seco_learners import SeparateAndConquerRuleLearner
from runnables import RuleLearnerRunnable


class SecoRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return SeparateAndConquerRuleLearner(random_state=args.random_state, max_rules=args.max_rules,
                                             time_limit=args.time_limit, loss=args.loss, heuristic=args.heuristic,
                                             pruning=args.pruning, label_sub_sampling=args.label_sub_sampling,
                                             instance_sub_sampling=args.instance_sub_sampling,
                                             feature_sub_sampling=args.feature_sub_sampling,
                                             head_refinement=args.head_refinement, min_coverage=args.min_coverage,
                                             max_conditions=args.max_conditions, lift_function=args.lift_function,
                                             max_head_refinements=args.max_head_refinements,
                                             num_threads=args.num_threads)


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment using Separate and Conquer') \
        .add_seco_learner_arguments() \
        .build()
    runnable = SecoRunnable()
    runnable.run(parser)
