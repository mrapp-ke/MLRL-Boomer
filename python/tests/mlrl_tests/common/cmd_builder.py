"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path
from typing import Any, List, Optional

from .datasets import Dataset

from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.state import ExperimentMode

from mlrl.util.options import Options


class CmdBuilder:
    """
    A builder that allows to configure a command for running an experiment.
    """

    RESOURCE_DIR = Path('python', 'tests', 'res')

    CONFIG_DIR = RESOURCE_DIR / 'config'

    EXPECTED_OUTPUT_DIR = RESOURCE_DIR / 'out'

    INPUT_DIR = RESOURCE_DIR / 'in'

    RERUN_DIR = Path('rerun')

    def __init__(self,
                 expected_output_dir: Path,
                 input_dir: Path,
                 batch_config: Path,
                 runnable_module_name: str,
                 runnable_class_name: Optional[str] = None,
                 dataset: str = Dataset.EMOTIONS):
        """
        :param expected_output_dir:     The path to the directory that contains the file with the expected output
        :param input_dir:               The path to the directory from which input data should be read
        :param runnable_module_name:    The fully qualified name of the runnable to be invoked by the program
                                        'mlrl-testbed'
        :param batch_config:            The path to the config file that should be used in batch mode
        :param runnable_class_name:     The class name of the runnable to be invoked by the program 'mlrl-testbed'
        :param dataset:                 The name of the dataset
        """
        self.expected_output_dir = expected_output_dir
        self.input_dir = input_dir
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
        self.control_args: List[str] = []
        self.algorithmic_args: List[str] = []
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

        rerun = self.mode in {ExperimentMode.RUN, ExperimentMode.READ}
        base_dir = self.base_dir / self.RERUN_DIR if rerun else self.base_dir

        args.extend(('--log-level', 'debug'))
        args.extend(('--base-dir', str(base_dir)))

        if self.mode == ExperimentMode.BATCH:
            args.extend(('--config', str(self.batch_config)))

            if self.runner:
                args.extend(('--runner', self.runner))

                if self.runner == 'slurm':
                    args.extend(('--slurm-config', str(self.CONFIG_DIR / 'slurm_config.yml'), '--print-slurm-scripts',
                                 'true', '--save-slurm-scripts', 'true', '--slurm-save-dir', str(base_dir)))
        else:
            if rerun:
                args.extend(('--input-dir', str(self.base_dir)))
            else:
                args.extend(('--data-dir', str(self.RESOURCE_DIR / 'data')))
                args.extend(('--dataset', self.dataset))
                args.extend(('--result-dir', str(self.result_dir)))

                if self.model_load_dir:
                    args.append('--load-models')
                    args.append(str(True).lower())
                    args.append('--model-load-dir')
                    args.append(str(self.model_load_dir))

                if self.model_save_dir:
                    args.append('--model-save-dir')
                    args.append(str(self.model_save_dir))

                if self.parameter_save_dir:
                    args.append('--parameter-save-dir')
                    args.append(str(self.parameter_save_dir))

        if self.problem_type and not rerun:
            args.extend(('--problem-type', self.problem_type))

        args.extend(self.control_args)

        if self.mode not in {ExperimentMode.RUN, ExperimentMode.READ}:
            args.extend(self.algorithmic_args)

        return args

    def add_control_argument(self, name: str, value: Optional[Any] = None):
        """
        Adds a control argument to the command.

        :param name:    The name of the argument
        :param value:   The value of the argument
        :return:        The builder itself
        """
        self.control_args.append(name)

        if value is not None:
            self.control_args.append(str(value))
        return self

    def add_algorithmic_argument(self, name: str, value: Optional[Any] = None):
        """
        Adds an algorithmic argument to the command.

        :param name:    The name of the argument
        :param value:   The value of the argument
        :return:        The builder itself
        """
        self.algorithmic_args.append(name)

        if value is not None:
            self.algorithmic_args.append(str(value))

        return self

    def set_mode(self, mode: Optional[str], *extra_args: str):
        """
        Configures the mode of operation to be used.

        :param mode:        The mode of operation to be used
        :param extra_args:  Additional arguments to be added
        :return:            The builder itself
        """
        self.mode = mode
        self.control_args.extend(extra_args)
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
        self.add_control_argument('--save-models', str(True).lower())
        return self

    def load_parameters(self):
        """
        Configures the rule learner to load parameter settings from a directory, if available.

        :return: The builder itself
        """
        self.add_control_argument('--load-parameters', str(True).lower())
        self.add_control_argument('--parameter-load-dir', str(self.input_dir))
        return self

    def save_parameters(self):
        """
        Configures the rule learner to save parameter settings to a directory.

        :return: The builder itself
        """
        self.parameter_save_dir = self.result_dir
        self.add_control_argument('--save-parameters', str(True).lower())
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
        num_folds = 0
        current_fold = None

        if data_split:
            if data_split == DatasetSplitterArguments.VALUE_CROSS_VALIDATION:
                num_folds = options.get_int(DatasetSplitterArguments.OPTION_NUM_FOLDS, 10)
                first_fold = options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, 0)

                if first_fold > 0 and first_fold == options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, 0):
                    current_fold = first_fold

            self.add_control_argument('--data-split', data_split + (str(options) if options else ''))

        self.num_folds = num_folds
        self.current_fold = current_fold
        return self

    def predict_for_training_data(self, predict_for_training_data: bool = True):
        """
        Configures whether predictions should be obtained for the training data or not.

        :param predict_for_training_data:   True, if predictions should be obtained for the training data, False
                                            otherwise
        :return:                            The builder itself
        """
        self.add_control_argument('--predict-for-training-data', str(predict_for_training_data).lower())
        return self

    def print_all(self, print_all: bool = True):
        """
        Configures whether all experimental results should be printed on the console or not.

        :param print_all:   True, if all experimental results should be printed, False otherwise
        :return:            The builder itself
        """
        self.add_control_argument('--print-all', str(print_all).lower())
        return self

    def save_all(self, save_all: bool = True):
        """
        Configures whether all experimental results should be written to output files or not.

        :param save_all:    True, if the all experimental results should be written to output files, False otherwise
        :return:            The builder itself
        """
        self.parameter_save_dir = self.result_dir
        self.model_save_dir = self.model_dir
        self.add_control_argument('--save-all', str(save_all).lower())
        return self

    def print_meta_data(self, print_meta_data: bool = True):
        """
        Configures whether the meta-data of the experiment should be printed on the console or not.

        :param print_meta_data: True, if the meta-data should be printed, False otherwise
        :return:                The builder itself
        """
        self.add_control_argument('--print-meta-data', str(print_meta_data).lower())
        return self

    def save_meta_data(self, save_meta_data: bool = True):
        """
        Configures whether the meta-data of the experiment should be written to output files or not.

        :param save_meta_data:  True, if the meta-data should be written to output files, False otherwise
        :return:                The builder itself
        """
        self.add_control_argument('--save-meta-data', str(save_meta_data).lower())
        return self

    def print_evaluation(self, print_evaluation: bool = True):
        """
        Configures whether the evaluation results should be printed on the console or not.

        :param print_evaluation:    True, if the evaluation results should be printed, False otherwise
        :return:                    The builder self
        """
        self.add_control_argument('--print-evaluation', str(print_evaluation).lower())
        return self

    def save_evaluation(self, save_evaluation: bool = True):
        """
        Configures whether the evaluation results should be written to output files or not.

        :param save_evaluation: True, if the evaluation results should be written to output files, False otherwise
        :return:                The builder itself
        """
        self.add_control_argument('--save-evaluation', str(save_evaluation).lower())
        return self

    def print_parameters(self, print_parameters: bool = True):
        """
        Configures whether the parameters should be printed on the console or not.

        :param print_parameters:    True, if the parameters should be printed, False otherwise
        :return:                    The builder itself
        """
        self.add_control_argument('--print-parameters', str(print_parameters).lower())
        return self

    def print_predictions(self, print_predictions: bool = True):
        """
        Configures whether the predictions should be printed on the console or not.

        :param print_predictions:   True, if the predictions should be printed, False otherwise
        :return:                    The builder itself
        """
        self.add_control_argument('--print-predictions', str(print_predictions).lower())
        return self

    def save_predictions(self, save_predictions: bool = True):
        """
        Configures whether the predictions should be written to output files or not.

        :param save_predictions:    True, if the predictions should be written to output files, False otherwise
        :return:                    The builder itself
        """
        self.add_control_argument('--save-predictions', str(save_predictions).lower())
        return self

    def print_ground_truth(self, print_ground_truth: bool = True):
        """
        Configures whether the ground truth should be printed on the console or not.

        :param print_ground_truth:  True, if the ground truth should be printed, False otherwise
        :return:                    The builder itself
        """
        self.add_control_argument('--print-ground-truth', str(print_ground_truth).lower())
        return self

    def save_ground_truth(self, save_ground_truth: bool = True):
        """
        Configures whether the ground truth should be written to output files or not.

        :param save_ground_truth:   True, if the ground truth should be written to output files, False otherwise
        :return:                    The builder itself
        """
        self.add_control_argument('--save-ground-truth', str(save_ground_truth).lower())
        return self

    def print_prediction_characteristics(self, print_prediction_characteristics: bool = True):
        """
        Configures whether the characteristics of predictions should be printed on the console or not.

        :param print_prediction_characteristics:    True, if the characteristics of predictions should be printed, False
                                                    otherwise
        :return:                                    The builder itself
        """
        self.add_control_argument('--print-prediction-characteristics', str(print_prediction_characteristics).lower())
        return self

    def save_prediction_characteristics(self, save_prediction_characteristics: bool = True):
        """
        Configures whether the characteristics of predictions should be written to output files or not.

        :param save_prediction_characteristics: True, if the characteristics of predictions should be written to
                                                output files, False otherwise
        :return:                                The builder itself
        """
        self.add_control_argument('--save-prediction-characteristics', str(save_prediction_characteristics).lower())
        return self

    def print_data_characteristics(self, print_data_characteristics: bool = True):
        """
        Configures whether the characteristics of datasets should be printed on the console or not.

        :param print_data_characteristics:  True, if the characteristics of datasets should be printed, False otherwise
        :return:                            The builder itself
        """
        self.add_control_argument('--print-data-characteristics', str(print_data_characteristics).lower())
        return self

    def save_data_characteristics(self, save_data_characteristics: bool = True):
        """
        Configures whether the characteristics of datasets should be written to output files or not.

        :param save_data_characteristics:   True, if the characteristics of datasets should be written to output
                                            files, False otherwise
        :return:                            The builder itself
        """
        self.add_control_argument('--save-data-characteristics', str(save_data_characteristics).lower())
        return self
