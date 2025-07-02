"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import path
from typing import List, Optional

from .datasets import Dataset


class CmdBuilder:
    """
    A builder that allows to configure a command for running a rule learner.
    """

    RULE_PRUNING_NO = 'none'

    RULE_PRUNING_IREP = 'irep'

    RULE_INDUCTION_TOP_DOWN_GREEDY = 'top-down-greedy'

    RULE_INDUCTION_TOP_DOWN_BEAM_SEARCH = 'top-down-beam-search'

    INSTANCE_SAMPLING_NO = 'none'

    INSTANCE_SAMPLING_WITH_REPLACEMENT = 'with-replacement'

    INSTANCE_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

    FEATURE_SAMPLING_NO = 'none'

    FEATURE_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

    OUTPUT_SAMPLING_NO = 'none'

    OUTPUT_SAMPLING_WITHOUT_REPLACEMENT = 'without-replacement'

    OUTPUT_SAMPLING_ROUND_ROBIN = 'round-robin'

    HOLDOUT_NO = 'none'

    HOLDOUT_RANDOM = 'random'

    FEATURE_BINNING_EQUAL_WIDTH = 'equal-width'

    FEATURE_BINNING_EQUAL_FREQUENCY = 'equal-frequency'

    def __init__(self,
                 expected_output_dir: str,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 dataset: str = Dataset.EMOTIONS):
        """
        :param expected_output_dir:     The path to the directory that contains the file with the expected output
        :param runnable_module_name:    The fully qualified name of the runnable to be invoked by the program
                                        'mlrl-testbed'
        :param runnable_class_name:     The class name of the runnable to be invoked by the program 'mlrl-testbed'
        :param dataset:                 The name of the dataset
        """
        self.expected_output_dir = path.join('python', 'tests', 'res', 'out', expected_output_dir)
        self.runnable_module_name = runnable_module_name
        self.runnable_class_name = runnable_class_name
        self.show_help = False
        self.dataset = dataset
        self.parameter_load_dir = None
        self.parameter_save_dir = None
        self.model_dir = None
        self.num_folds = 0
        self.current_fold = None
        self.args = []

    @property
    def output_dir(self) -> str:
        """
        The path to the directory where output files should be stored.
        """
        return path.join('python', 'tests', 'res', 'tmp', 'results')

    def build(self) -> List[str]:
        """
        Returns the command that has been configured via the builder.

        :return: A list that contains the executable and arguments of the command
        """
        args = ['mlrl-testbed', self.runnable_module_name]

        if self.runnable_class_name:
            args.extend(['-r', self.runnable_class_name])

        if self.show_help:
            args.append('--help')
            return args

        args.extend(['--log-level', 'debug'])
        args.extend(['--data-dir', path.join('python', 'tests', 'res', 'data')])
        args.extend(['--dataset', self.dataset])
        args.extend(['--output-dir', self.output_dir])
        return args + self.args

    def set_show_help(self, show_help: bool = True):
        """
        Configures whether the program's help text should be shown or not.

        :param show_help:   True, if the program's help text should be shown, False otherwise
        :return:            The builder itself
        """
        self.show_help = show_help
        return self

    def set_model_dir(self, model_dir: Optional[str] = path.join('python', 'tests', 'res', 'tmp', 'models')):
        """
        Configures the rule learner to store models in a given directory or load them, if available.

        :param model_dir:   The path to the directory where models should be stored
        :return:            The builder itself
        """
        self.model_dir = model_dir

        if model_dir:
            self.args.append('--model-load-dir')
            self.args.append(model_dir)
            self.args.append('--model-save-dir')
            self.args.append(model_dir)
        return self

    def set_parameter_load_dir(self, parameter_dir: Optional[str] = path.join('python', 'tests', 'res', 'in')):
        """
        Configures the rule learner to load parameter settings from a given directory, if available.

        :param parameter_dir:   The path to the directory from which parameter settings should be loaded
        :return:                The builder itself
        """
        self.parameter_load_dir = parameter_dir

        if parameter_dir:
            self.args.append('--parameter-load-dir')
            self.args.append(parameter_dir)
        return self

    def set_parameter_save_dir(self,
                               parameter_dir: Optional[str] = path.join('python', 'tests', 'res', 'tmp', 'results')):
        """
        Configures the rule learner to save parameter settings to a given directory.

        :param parameter_dir:   The path to the directory to which parameter settings should be saved
        :return:                The builder itself
        """
        self.parameter_save_dir = parameter_dir
        self.args.append('--parameter-save-dir')
        self.args.append(parameter_dir)
        return self

    def no_data_split(self):
        """
        Configures the rule learner to not use separate training and test data.

        :return: The builder itself
        """
        self.num_folds = 0
        self.current_fold = None
        self.args.append('--data-split')
        self.args.append('none')
        return self

    def cross_validation(self, num_folds: int = 10, current_fold: Optional[int] = None):
        """
        Configures the rule learner to use a cross validation.

        :param num_folds:       The total number of folds
        :param current_fold:    The fold to be run (starting at 1) or None, if all folds should be run
        :return:                The builder itself
        """
        self.num_folds = num_folds
        self.current_fold = current_fold
        self.args.append('--data-split')
        value = 'cross-validation{num_folds=' + str(num_folds)

        if current_fold:
            value += ',first_fold=' + str(current_fold) + ',last_fold=' + str(current_fold)

        value += '}'
        self.args.append(value)
        return self

    def sparse_feature_value(self, sparse_feature_value: float = 0.0):
        """
        Configures the value that should be used for sparse elements in the feature matrix.

        :param sparse_feature_value:    The value that should be used for sparse elements in the feature matrix
        :return:                        The builder itself
        """
        self.args.append('--sparse-feature-value')
        self.args.append(str(sparse_feature_value))
        return self

    def predict_for_training_data(self, predict_for_training_data: bool = True):
        """
        Configures whether predictions should be obtained for the training data or not.

        :param predict_for_training_data:   True, if predictions should be obtained for the training data, False
                                            otherwise
        :return:                            The builder itself
        """
        self.args.append('--predict-for-training-data')
        self.args.append(str(predict_for_training_data).lower())
        return self

    def incremental_evaluation(self, incremental_evaluation: bool = True, step_size: int = 50):
        """
        Configures whether the model that is learned by the rule learner should be evaluated repeatedly, using only a
        subset of the rules with increasing size.

        :param incremental_evaluation:  True, if the rule learner should be evaluated incrementally, False otherwise
        :param step_size:               The number of additional rules to be evaluated at each repetition
        :return:                        The builder itself
        """
        self.args.append('--incremental-evaluation')
        value = str(incremental_evaluation).lower()

        if incremental_evaluation:
            value += '{step_size=' + str(step_size) + '}'

        self.args.append(value)
        return self

    def print_evaluation(self, print_evaluation: bool = True):
        """
        Configures whether the evaluation results should be printed on the console or not.

        :param print_evaluation:    True, if the evaluation results should be printed, False otherwise
        :return:                    The builder self
        """
        self.args.append('--print-evaluation')
        self.args.append(str(print_evaluation).lower())
        return self

    def store_evaluation(self, store_evaluation: bool = True):
        """
        Configures whether the evaluation results should be written into output files or not.

        :param store_evaluation:    True, if the evaluation results should be written into output files or not
        :return:                    The builder itself
        """
        self.args.append('--store-evaluation')
        self.args.append(str(store_evaluation).lower())
        return self

    def print_parameters(self, print_parameters: bool = True):
        """
        Configures whether the parameters should be printed on the console or not.

        :param print_parameters:    True, if the parameters should be printed, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--print-parameters')
        self.args.append(str(print_parameters).lower())
        return self

    def print_predictions(self, print_predictions: bool = True):
        """
        Configures whether the predictions should be printed on the console or not.

        :param print_predictions:   True, if the predictions should be printed, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--print-predictions')
        self.args.append(str(print_predictions).lower())
        return self

    def store_predictions(self, store_predictions: bool = True):
        """
        Configures whether the predictions should be written into output files or not.

        :param store_predictions:   True, if the predictions should be written into output files, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--store-predictions')
        self.args.append(str(store_predictions).lower())
        return self

    def print_ground_truth(self, print_ground_truth: bool = True):
        """
        Configures whether the ground truth should be printed on the console or not.

        :param print_ground_truth:  True, if the ground truth should be printed, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--print-ground-truth')
        self.args.append(str(print_ground_truth).lower())
        return self

    def store_ground_truth(self, store_ground_truth: bool = True):
        """
        Configures whether the ground truth should be written into output files or not.

        :param store_ground_truth:  True, if the ground truth should be written into output files, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--store-ground-truth')
        self.args.append(str(store_ground_truth).lower())
        return self

    def print_prediction_characteristics(self, print_prediction_characteristics: bool = True):
        """
        Configures whether the characteristics of predictions should be printed on the console or not.

        :param print_prediction_characteristics:    True, if the characteristics of predictions should be printed, False
                                                    otherwise
        :return:                                    The builder itself
        """
        self.args.append('--print-prediction-characteristics')
        self.args.append(str(print_prediction_characteristics).lower())
        return self

    def store_prediction_characteristics(self, store_prediction_characteristics: bool = True):
        """
        Configures whether the characteristics of predictions should be written into output files or not.

        :param store_prediction_characteristics:    True, if the characteristics of predictions should be written into
                                                    output files, False otherwise
        :return:                                    The builder itself
        """
        self.args.append('--store-prediction-characteristics')
        self.args.append(str(store_prediction_characteristics).lower())
        return self

    def print_data_characteristics(self, print_data_characteristics: bool = True):
        """
        Configures whether the characteristics of datasets should be printed on the console or not.

        :param print_data_characteristics:  True, if the characteristics of datasets should be printed, False otherwise
        :return:                            The builder itself
        """
        self.args.append('--print-data-characteristics')
        self.args.append(str(print_data_characteristics).lower())
        return self

    def store_data_characteristics(self, store_data_characteristics: bool = True):
        """
        Configures whether the characteristics of datasets should be written into output files or not.

        :param store_data_characteristics:  True, if the characteristics of datasets should be written into output
                                            files, False otherwise
        :return:                            The builder itself
        """
        self.args.append('--store-data-characteristics')
        self.args.append(str(store_data_characteristics).lower())
        return self

    def print_model_characteristics(self, print_model_characteristics: bool = True):
        """
        Configures whether the characteristics of models should be printed on the console or not.

        :param print_model_characteristics: True, if the characteristics of models should be printed, False otherwise
        :return:                            The builder itself
        """
        self.args.append('--print-model-characteristics')
        self.args.append(str(print_model_characteristics).lower())
        return self

    def store_model_characteristics(self, store_model_characteristics: bool = True):
        """
        Configures whether the characteristics of models should be written into output files or not.

        :param store_model_characteristics: True, if the characteristics of models should be written into output files,
                                            False otherwise
        :return:                            The builder itself
        """
        self.args.append('--store-model-characteristics')
        self.args.append(str(store_model_characteristics).lower())
        return self

    def print_rules(self, print_rules: bool = True):
        """
        Configures whether textual representations of the rules in a model should be printed on the console or not.

        :param print_rules: True, if textual representations of rules should be printed, False otherwise
        :return:            The builder itself
        """
        self.args.append('--print-rules')
        self.args.append(str(print_rules).lower())
        return self

    def store_rules(self, store_rules: bool = True):
        """
        Configures whether textual representations of the rules in a model should be written into output files or not.

        :param store_rules: True, if textual representations of rules should be written into output files, False
                            otherwise
        :return:            The builder itself
        """
        self.args.append('--store-rules')
        self.args.append(str(store_rules).lower())
        return self

    def sparse_feature_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent the feature values of training examples or
        not.

        :param sparse:  True, if sparse data structures should be used to represent the feature values of training
                        examples, False otherwise
        :return:        The builder itself
        """
        self.args.append('--feature-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def sparse_output_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent the labels of training examples or not.

        :param sparse:  True, if sparse data structures should be used to represent the labels of training examples,
                        False otherwise
        :return:        The builder itself
        """
        self.args.append('--output-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def sparse_prediction_format(self, sparse: bool = True):
        """
        Configures whether sparse data structures should be used to represent predictions or not.

        :param sparse:  True, if sparse data structures should be used to represent predictions, False otherwise
        :return:        The builder itself
        """
        self.args.append('--prediction-format')
        self.args.append('sparse' if sparse else 'dense')
        return self

    def instance_sampling(self, instance_sampling: str = INSTANCE_SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available training examples.

        :param instance_sampling:   The name of the sampling method that should be used
        :return:                    The builder itself
        """
        self.args.append('--instance-sampling')
        self.args.append(instance_sampling)
        return self

    def feature_sampling(self, feature_sampling: str = FEATURE_SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available features.

        :param feature_sampling:    The name of the sampling method that should be used
        :return:                    The builder itself
        """
        self.args.append('--feature-sampling')
        self.args.append(feature_sampling)
        return self

    def output_sampling(self, output_sampling: str = OUTPUT_SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available outputs.

        :param output_sampling: The name of the sampling method that should be used
        :return:                The builder itself
        """
        self.args.append('--output-sampling')
        self.args.append(output_sampling)
        return self

    def rule_pruning(self, rule_pruning: str = RULE_PRUNING_IREP):
        """
        Configures the rule learner to use a specific method for pruning individual rules.

        :param rule_pruning:    The name of the pruning method that should be used
        :return:                The builder itself
        """
        self.args.append('--rule-pruning')
        self.args.append(rule_pruning)
        return self

    def rule_induction(self, rule_induction=RULE_INDUCTION_TOP_DOWN_GREEDY):
        """
        Configures the rule learner to use a specific algorithm for the induction of individual rules.

        :param rule_induction:  The name of the algorithm that should be used
        :return:                The builder itself
        """
        self.args.append('--rule-induction')
        self.args.append(rule_induction)
        return self

    def sequential_post_optimization(self, sequential_post_optimization: bool = True):
        """
        Configures whether the algorithm should use sequential post-optimization or not.

        :param sequential_post_optimization:    True, if sequential post-optimization should be used, False otherwise
        :return:                                The builder itself
        """
        self.args.append('--sequential-post-optimization')
        self.args.append(str(sequential_post_optimization).lower())
        return self

    def holdout(self, holdout: str = HOLDOUT_RANDOM):
        """
        Configures the algorithm to use a holdout set.

        :param holdout: The name of the sampling method that should be used to create the holdout set
        :return:        The builder itself
        """
        self.args.append('--holdout')
        self.args.append(holdout)
        return self

    def feature_binning(self, feature_binning: str = FEATURE_BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for feature binning.

        :param feature_binning: The name of the method that should be used for feature binning
        :return:                The builder itself
        """
        self.args.append('--feature-binning')
        self.args.append(feature_binning)
        return self
