#!/usr/bin/python

from argparse import ArgumentParser

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

from args import ArgumentParserBuilder

import weka.core.converters as converters
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
import weka.core.jvm as jvm

import wittgenstein as lw

from mlrl.testbed.data import load_data_set_and_meta_data
from mlrl.testbed.io import SUFFIX_ARFF, SUFFIX_XML, get_file_name
from mlrl.testbed.training import DataSet


def run_weka(pars: ArgumentParser):
    args = pars.parse_args()
    seed = args.random_state
    data_dir = args.data_dir
    data_set = args.dataset

    jvm.start()

    data = converters.load_any_file(data_dir + "/" + data_set + ".arff", class_index="last")
    jrip = Classifier(classname="weka.classifiers.rules.JRip", options=["-S", str(seed)])
    jrip.build_classifier(data)

    print("\n" + jrip.jwrapper.toString())

    evl = Evaluation(data)
    evl.crossvalidate_model(jrip, data, 10, Random(1))

    print("recall: {}".format(evl.weighted_recall))
    print("precision: {}".format(evl.weighted_precision))
    print("f-measure: {}".format(evl.weighted_f_measure))

    jvm.stop()


def run_wittgenstein(pars: ArgumentParser):
    args = pars.parse_args()
    max_rules = args.max_rules
    random_state = args.random_state
    data_dir = args.data_dir
    dataset = args.dataset

    data_set = DataSet(data_dir=data_dir, data_set_name=dataset, use_one_hot_encoding=args.one_hot_encoding)

    data_set_name = data_set.data_set_name
    x, y, _ = load_data_set_and_meta_data(data_set.data_dir, get_file_name(data_set_name, SUFFIX_ARFF),
                                          get_file_name(data_set_name, SUFFIX_XML))

    ripper_clf = lw.RIPPER(max_rules=max_rules, random_state=random_state)
    ripper_clf.fit(x.todense(), y.todense())

    print(ripper_clf.ruleset_.out_pretty())

    print("precision: {}".format(
          cross_val_score(ripper_clf, x.todense(), y.todense(), cv=10, scoring=make_scorer(precision_score)).mean()))
    print("recall: {}".format(
        cross_val_score(ripper_clf, x.todense(), y.todense(), cv=10, scoring=make_scorer(recall_score)).mean()))
    print("f-measure: {}".format(
        cross_val_score(ripper_clf, x.todense(), y.todense(), cv=10, scoring=make_scorer(f1_score)).mean()))


if __name__ == '__main__':
    parser = ArgumentParserBuilder(description='A binary classification experiment using JRip') \
        .add_jrip_learner_arguments() \
        .build()

    if parser.parse_args().ripper == "weka":
        run_weka(parser)
    else:
        run_wittgenstein(parser)
