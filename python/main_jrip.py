#!/usr/bin/python

from mlrl.seco.ripper_learners import RipperRuleLearner
from runnables import RuleLearnerRunnable
from args import ArgumentParserBuilder

import sklweka.jvm as jvm


class RipperRunnable(RuleLearnerRunnable):

    def _create_learner(self, args):
        return RipperRuleLearner(random_state=args.random_state, max_rules=args.max_rules, ripper=args.ripper)


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A multi-label classification experiment various ripper implementations') \
        .add_jrip_learner_arguments() \
        .build()

    if parser.parse_args().ripper == 'weka':
        jvm.start(packages=True)

    runnable = RipperRunnable()
    runnable.run(parser)

    if parser.parse_args().ripper == 'weka':
        jvm.stop()

