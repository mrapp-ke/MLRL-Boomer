"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path
from typing import List, Optional

from .datasets import Dataset

from mlrl.common.config.parameters import BINNING_EQUAL_WIDTH, SAMPLING_WITHOUT_REPLACEMENT, \
    PartitionSamplingParameter, PostOptimizationParameter, RuleInductionParameter, RulePruningParameter
from mlrl.common.learners import SparsePolicy

from mlrl.testbed_sklearn.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments

from mlrl.testbed.modes import Mode

from mlrl.util.options import Options


class CmdBuilder:
    """
    A builder that allows to configure a command for running a rule learner.
    """

    RESOURCE_DIR = Path('python', 'tests', 'res')

    CONFIG_DIR = RESOURCE_DIR / 'config'

    EXPECTED_OUTPUT_DIR = RESOURCE_DIR / 'out'

    RERUN_DIR = Path('rerun')

    def __init__(self,
                 expected_output_dir: Path,
                 batch_config: Path,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 dataset: str = Dataset.EMOTIONS):
        """
        :param expected_output_dir:     The path to the directory that contains the file with the expected output
        :param runnable_module_name:    The fully qualified name of the runnable to be invoked by the program
                                        'mlrl-testbed'
        :param batch_config:            The path to the config file that should be used in batch mode
        :param runnable_class_name:     The class name of the runnable to be invoked by the program 'mlrl-testbed'
        :param dataset:                 The name of the dataset
        """
        self.expected_output_dir = expected_output_dir
        self.batch_config = batch_config
        self.runnable_module_name = runnable_module_name
        self.runnable_class_name = runnable_class_name
        self.mode: Optional[str] = None
        self.runner: Optional[str] = None
        self.show_help = False
        self.dataset = dataset
        self.parameter_save_dir: Optional[Path] = None
        self.model_save_dir: Optional[Path] = None
        self.model_load_dir: Optional[Path] = None
        self.num_folds = 0
        self.current_fold = None
        self.args: List[str] = []
        self.save_evaluation(True)
        self.problem_type: Optional[str] = None

    @property
    def base_dir(self) -> Path:
        """
        The base directory.
        """
        return self.RESOURCE_DIR / 'tmp'

    @property
    def result_dir(self) -> Path:
        """
        The relative path to the directory where experimental results should be saved.
        """
        return Path('results')

    @property
    def resolved_result_dir(self) -> Path:
        """
        The path to the directory where experimental results should be saved, resolved against the base directory.
        """
        return self.base_dir / self.result_dir

    @property
    def model_dir(self) -> Path:
        """
        The relative path to the directory where models should be saved.
        """
        return Path('models')

    @property
    def resolved_model_dir(self) -> Optional[Path]:
        """
        The  path to the directory where models should be saved, resolved against the base directory.
        """
        model_save_dir = self.model_save_dir
        return self.base_dir / model_save_dir if model_save_dir else None

    @property
    def resolved_parameter_dir(self) -> Optional[Path]:
        """
        The path to the directory where models should be saved, resolved against the base directory.
        """
        parameter_save_dir = self.parameter_save_dir
        return self.base_dir / parameter_save_dir if parameter_save_dir else None

    # pylint: disable=too-many-branches
    def build(self) -> List[str]:
        """
        Returns the command that has been configured via the builder.

        :return: A list that contains the executable and arguments of the command
        """
        args = ['mlrl-testbed', self.runnable_module_name]

        if self.runnable_class_name:
            args.extend(['-r', self.runnable_class_name])

        if self.mode:
            args.extend(('--mode', self.mode))

        if self.show_help:
            args.append('--help')
            return args

        base_dir = self.base_dir / self.RERUN_DIR if self.mode == Mode.MODE_RUN else self.base_dir

        args.extend(('--log-level', 'debug'))
        args.extend(('--base-dir', str(base_dir)))

        if self.mode == Mode.MODE_BATCH:
            args.extend(('--config', str(self.batch_config)))

            if self.runner:
                args.extend(('--runner', self.runner))

                if self.runner == 'slurm':
                    args.extend(('--slurm-config', str(self.CONFIG_DIR / 'slurm_config.yml'), '--print-slurm-scripts',
                                 'true', '--save-slurm-scripts', 'true', '--slurm-save-dir', str(base_dir)))
        else:
            if self.mode == Mode.MODE_RUN:
                args.extend(('--input-dir', str(self.base_dir)))
            else:
                args.extend(('--data-dir', str(self.RESOURCE_DIR / 'data')))
                args.extend(('--dataset', self.dataset))

            args.extend(('--result-dir', str(self.result_dir)))

            if self.model_load_dir and self.mode != Mode.MODE_RUN:
                self.args.append('--load-models')
                self.args.append(str(True).lower())
                self.args.append('--model-load-dir')
                self.args.append(str(self.model_load_dir))

            if self.model_save_dir:
                self.args.append('--model-save-dir')
                self.args.append(str(self.model_save_dir))

            if self.parameter_save_dir:
                self.args.append('--parameter-save-dir')
                self.args.append(str(self.parameter_save_dir))

        if self.problem_type and self.mode != Mode.MODE_RUN:
            args.extend(('--problem-type', self.problem_type))

        return args + self.args

    def set_mode(self, mode: Optional[str], *extra_args: str):
        """
        Configures the mode of operation to be used.

        :param mode:        The mode of operation to be used
        :param extra_args:  Additional arguments to be added
        :return:            The builder itself
        """
        self.mode = mode
        self.args.extend(extra_args)
        return self

    def set_runner(self, runner: Optional[str] = 'sequential'):
        """
        Configures the runner to be used in batch mode.

        :param runner:  The name of the runner to be used
        :return:        The builder itself
        """
        self.runner = runner
        return self

    def set_show_help(self, show_help: bool = True):
        """
        Configures whether the program's help text should be shown or not.

        :param show_help:   True, if the program's help text should be shown, False otherwise
        :return:            The builder itself
        """
        self.show_help = show_help
        return self

    def load_models(self):
        """
        Configures the rule learner to load models from a directory, if available.

        :return: The builder itself
        """
        self.model_load_dir = Path('models')
        return self

    def save_models(self):
        """
        Configures the rule learner to store models in a directory.

        :return: The builder itself
        """
        self.model_save_dir = self.model_dir
        self.args.append('--save-models')
        self.args.append(str(True).lower())
        return self

    def load_parameters(self):
        """
        Configures the rule learner to load parameter settings from a directory, if available.

        :return: The builder itself
        """
        self.args.append('--load-parameters')
        self.args.append(str(True).lower())
        self.args.append('--parameter-load-dir')
        self.args.append(str(self.RESOURCE_DIR / 'in'))
        return self

    def save_parameters(self):
        """
        Configures the rule learner to save parameter settings to a directory.

        :return: The builder itself
        """
        self.parameter_save_dir = self.result_dir
        self.args.append('--save-parameters')
        self.args.append(str(True).lower())
        return self

    def data_split(self,
                   data_split: Optional[str] = DatasetSplitterArguments.VALUE_TRAIN_TEST,
                   options: Options = Options()):
        """
        Configures the rule learner to use a specific strategy for splitting datasets into training and test datasets.

        :param data_split:  The name of the strategy to be used
        :param options:     Options to be taken into account
        :return:            The builder itself
        """
        self.args.append('--data-split')
        num_folds = 0
        current_fold = None

        if data_split:
            if data_split == DatasetSplitterArguments.VALUE_CROSS_VALIDATION:
                num_folds = options.get_int(DatasetSplitterArguments.OPTION_NUM_FOLDS, 10)
                first_fold = options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, 0)

                if first_fold > 0 and first_fold == options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, 0):
                    current_fold = first_fold

            self.args.append(data_split + (str(options) if options else ''))

        self.num_folds = num_folds
        self.current_fold = current_fold
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

    def print_all(self, print_all: bool = True):
        """
        Configures whether all experimental results should be printed on the console or not.

        :param print_all:   True, if all experimental results should be printed, False otherwise
        :return:            The builder itself
        """
        self.args.append('--print-all')
        self.args.append(str(print_all).lower())
        return self

    def save_all(self, save_all: bool = True):
        """
        Configures whether all experimental results should be written to output files or not.

        :param save_all:    True, if the all experimental results should be written to output files, False otherwise
        :return:            The builder itself
        """
        self.parameter_save_dir = self.result_dir
        self.model_save_dir = self.model_dir
        self.args.append('--save-all')
        self.args.append(str(save_all).lower())
        return self

    def print_meta_data(self, print_meta_data: bool = True):
        """
        Configures whether the meta-data of the experiment should be printed on the console or not.

        :param print_meta_data: True, if the meta-data should be printed, False otherwise
        :return:                The builder itself
        """
        self.args.append('--print-meta-data')
        self.args.append(str(print_meta_data).lower())
        return self

    def save_meta_data(self, save_meta_data: bool = True):
        """
        Configures whether the meta-data of the experiment should be written to output files or not.

        :param save_meta_data:  True, if the meta-data should be written to output files, False otherwise
        :return:                The builder itself
        """
        self.args.append('--save-meta-data')
        self.args.append(str(save_meta_data).lower())
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

    def save_evaluation(self, save_evaluation: bool = True):
        """
        Configures whether the evaluation results should be written to output files or not.

        :param save_evaluation: True, if the evaluation results should be written to output files, False otherwise
        :return:                The builder itself
        """
        self.args.append('--save-evaluation')
        self.args.append(str(save_evaluation).lower())
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

    def save_predictions(self, save_predictions: bool = True):
        """
        Configures whether the predictions should be written to output files or not.

        :param save_predictions:    True, if the predictions should be written to output files, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--save-predictions')
        self.args.append(str(save_predictions).lower())
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

    def save_ground_truth(self, save_ground_truth: bool = True):
        """
        Configures whether the ground truth should be written to output files or not.

        :param save_ground_truth:   True, if the ground truth should be written to output files, False otherwise
        :return:                    The builder itself
        """
        self.args.append('--save-ground-truth')
        self.args.append(str(save_ground_truth).lower())
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

    def save_prediction_characteristics(self, save_prediction_characteristics: bool = True):
        """
        Configures whether the characteristics of predictions should be written to output files or not.

        :param save_prediction_characteristics: True, if the characteristics of predictions should be written to
                                                output files, False otherwise
        :return:                                The builder itself
        """
        self.args.append('--save-prediction-characteristics')
        self.args.append(str(save_prediction_characteristics).lower())
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

    def save_data_characteristics(self, save_data_characteristics: bool = True):
        """
        Configures whether the characteristics of datasets should be written to output files or not.

        :param save_data_characteristics:   True, if the characteristics of datasets should be written to output
                                            files, False otherwise
        :return:                            The builder itself
        """
        self.args.append('--save-data-characteristics')
        self.args.append(str(save_data_characteristics).lower())
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

    def save_model_characteristics(self, save_model_characteristics: bool = True):
        """
        Configures whether the characteristics of models should be written to output files or not.

        :param save_model_characteristics:  True, if the characteristics of models should be written to output files,
                                            False otherwise
        :return:                            The builder itself
        """
        self.args.append('--save-model-characteristics')
        self.args.append(str(save_model_characteristics).lower())
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

    def save_rules(self, save_rules: bool = True):
        """
        Configures whether textual representations of the rules in a model should be written to output files or not.

        :param save_rules:  True, if textual representations of rules should be written to output files, False
                            otherwise
        :return:            The builder itself
        """
        self.args.append('--save-rules')
        self.args.append(str(save_rules).lower())
        return self

    def feature_format(self, feature_format: Optional[str] = SparsePolicy.FORCE_SPARSE):
        """
        Configures the format to be used for the feature values of training examples.

        :param feature_format:  The format to be used
        :return:                The builder itself
        """
        if feature_format:
            self.args.append('--feature-format')
            self.args.append(feature_format)

        return self

    def output_format(self, output_format: Optional[str] = SparsePolicy.FORCE_SPARSE):
        """
        Configures the format to be used for the ground truth of training examples.

        :param output_format:   The format to be used
        :return:                The builder itself
        """
        if output_format:
            self.args.append('--output-format')
            self.args.append(output_format)

        return self

    def prediction_format(self, prediction_format: Optional[str] = SparsePolicy.FORCE_SPARSE):
        """
        Configures the format to be used for predictions.

        :param prediction_format:   The format to be used
        :return:                    The builder itself
        """
        if prediction_format:
            self.args.append('--prediction-format')
            self.args.append(prediction_format)

        return self

    def instance_sampling(self, instance_sampling: Optional[str]):
        """
        Configures the rule learner to sample from the available training examples.

        :param instance_sampling:   The name of the sampling method that should be used
        :return:                    The builder itself
        """
        if instance_sampling:
            self.args.append('--instance-sampling')
            self.args.append(instance_sampling)

        return self

    def feature_sampling(self, feature_sampling: Optional[str] = SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available features.

        :param feature_sampling:    The name of the sampling method that should be used
        :return:                    The builder itself
        """
        if feature_sampling:
            self.args.append('--feature-sampling')
            self.args.append(feature_sampling)

        return self

    def output_sampling(self, output_sampling: Optional[str] = SAMPLING_WITHOUT_REPLACEMENT):
        """
        Configures the rule learner to sample from the available outputs.

        :param output_sampling: The name of the sampling method that should be used
        :return:                The builder itself
        """
        if output_sampling:
            self.args.append('--output-sampling')
            self.args.append(output_sampling)

        return self

    def rule_pruning(self, rule_pruning: Optional[str] = RulePruningParameter.RULE_PRUNING_IREP):
        """
        Configures the rule learner to use a specific method for pruning individual rules.

        :param rule_pruning:    The name of the pruning method that should be used
        :return:                The builder itself
        """
        if rule_pruning:
            self.args.append('--rule-pruning')
            self.args.append(rule_pruning)

        return self

    def rule_induction(self, rule_induction: Optional[str] = RuleInductionParameter.RULE_INDUCTION_TOP_DOWN_GREEDY):
        """
        Configures the rule learner to use a specific algorithm for the induction of individual rules.

        :param rule_induction:  The name of the algorithm that should be used
        :return:                The builder itself
        """
        if rule_induction:
            self.args.append('--rule-induction')
            self.args.append(rule_induction)

        return self

    def post_optimization(self,
                          post_optimization: Optional[str] = PostOptimizationParameter.POST_OPTIMIZATION_SEQUENTIAL):
        """
        Configures the post-optimization method to be used by the algorithm.

        :param post_optimization:   The name of the method that should be used for post-optimization
        :return:                    The builder itself
        """
        if post_optimization:
            self.args.append('--post-optimization')
            self.args.append(post_optimization)

        return self

    def holdout(self, holdout: Optional[str] = PartitionSamplingParameter.PARTITION_SAMPLING_RANDOM):
        """
        Configures the algorithm to use a holdout set.

        :param holdout: The name of the sampling method that should be used to create the holdout set
        :return:        The builder itself
        """
        if holdout:
            self.args.append('--holdout')
            self.args.append(holdout)

        return self

    def feature_binning(self, feature_binning: Optional[str] = BINNING_EQUAL_WIDTH):
        """
        Configures the algorithm to use a specific method for feature binning.

        :param feature_binning: The name of the method that should be used for feature binning
        :return:                The builder itself
        """
        if feature_binning:
            self.args.append('--feature-binning')
            self.args.append(feature_binning)

        return self
