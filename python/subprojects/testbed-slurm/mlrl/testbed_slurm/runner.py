"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run experiments via the Slurm Workload Manager.
"""
import logging as log
import sys

from argparse import Namespace
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, cast, override

import yamale

from tabulate import tabulate

from mlrl.testbed_slurm.arguments import SlurmArguments
from mlrl.testbed_slurm.sbatch import Sbatch

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.input.dataset.splitters.arguments import DatasetSplitterArguments
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.modes.mode_batch import Batch, BatchMode
from mlrl.testbed.util.io import open_readable_file, open_writable_file

from mlrl.util.options import Options


@dataclass
class JobArray:
    """
    Defines a Slurm job array.

    Attributes:
        command:            The command to be executed
        command_modifier:   A function that modifies a given command such that it can be used in the job array
        file_name_modifier: A function that modifies a given file name such that it can be used in the job array
        task_ids:           The task ids to be used in the job array
    """
    command: Command
    command_modifier: Callable[[Command], Command]
    file_name_modifier: Callable[[str], str]
    task_ids: Set[int] = field(default_factory=set)

    @property
    def modified_command(self) -> Command:
        """
        The command to be executed, modified such that it can be used in the job array.
        """
        command = self.command
        return self.command_modifier(command) if len(self.task_ids) > 1 else command

    @property
    def formatted_task_ids(self) -> Optional[str]:
        """
        The task ids to be used in the job array formatted such that they can be used as the value or the sbatch
        argument "--array".
        """
        task_ids = list(self.task_ids)

        if len(task_ids) > 1:
            task_ids.sort()
            task_id_strings: List[str] = []
            current_range = (0, 0)

            for task_id in task_ids:
                start = current_range[0]
                end = current_range[1]

                if end < 1:
                    current_range = (task_id, task_id)
                elif end == task_id - 1:
                    current_range = (start, task_id)
                else:
                    task_id_strings.append(self.__format_range(current_range))
                    current_range = (task_id, task_id)

            task_id_strings.append(self.__format_range(current_range))
            return ','.join(task_id_strings)

        return None

    @staticmethod
    def __format_range(start_and_end: Tuple[int, int]) -> str:
        start = start_and_end[0]
        end = start_and_end[1]
        return str(start) + '-' + str(end) if start < end else str(start)


class SlurmRunner(BatchMode.Runner):
    """
    A `BatchMode.Runner` that allows to run experiments via the Slurm Workload manager.
    """

    class ConfigFile:
        """
        A YAML configuration file that configures Slurm jobs to be run.
        """

        def __init__(self, file_path: str, schema_file_path: str):
            """
            :param file_path:           The path to the configuration file
            :param schema_file_path:    The path to a YAML schema file
            """
            schema = yamale.make_schema(schema_file_path)
            data = yamale.make_data(file_path)
            yamale.validate(schema, data)
            self.file_path = file_path
            self.yaml_dict = data[0][0]

        @property
        def sbatch_arguments(self) -> List[str]:
            """
            The sbatch arguments (starting with #SBATCH) to be passed to a Slurm job.
            """
            return [
                '#SBATCH ' + str(argument) for argument in self.yaml_dict.get('sbatch-arguments', [])
                if len(argument) > 0 and not argument.isspace()
            ]

        @property
        def before_script(self) -> List[str]:
            """
            The shell commands to be executed before the experiment is started.
            """
            return [
                line.strip() for line in self.yaml_dict.get('before-script', '').split('\n')
                if len(line) > 0 and not line.isspace()
            ]

        @property
        def after_script(self) -> List[str]:
            """
            The shell commands to be executed after the experiment has finished.
            """
            return [
                line.strip() for line in self.yaml_dict.get('after-script', '').split('\n')
                if len(line) > 0 and not line.isspace()
            ]

        @override
        def __str__(self) -> str:
            return self.file_path

    @staticmethod
    def __is_command_available() -> bool:
        sbatch = Sbatch()
        version = sbatch.version().run()

        if not version.ok:
            log.error('Command "%s" not found: %s', sbatch.command, version.output)
            return False

        return True

    @staticmethod
    def __read_config_file(args: Namespace) -> Optional[ConfigFile]:
        config_file_path = SlurmArguments.SLURM_CONFIG_FILE.get_value(args)

        if config_file_path:
            schema_file_path = Path(__file__).parent / 'slurm_config.schema.yml'
            return SlurmRunner.ConfigFile(file_path=config_file_path, schema_file_path=str(schema_file_path))

        return None

    @staticmethod
    def __create_sbatch_script(command: Command, config_file: Optional[ConfigFile],
                               job_array: Optional[JobArray]) -> str:
        base_dir = Path(command.argument_dict[OutputArguments.BASE_DIR.name])
        result_dir = base_dir / command.argument_dict[ResultDirectoryArguments.RESULT_DIR.name]
        job_array_task_ids = job_array.formatted_task_ids if job_array else None
        std_file_name = SlurmRunner.__get_std_file_name(command)

        if job_array:
            std_file_name = job_array.file_name_modifier(std_file_name)

        content = '#!/bin/sh\n\n'
        content += '#SBATCH --output=' + str(result_dir / f'{std_file_name}.out') + '\n'
        content += '#SBATCH --error=' + str(result_dir / f'{std_file_name}.err') + '\n'

        if job_array_task_ids:
            content += '#SBATCH --array=' + job_array_task_ids + '\n'

        if config_file:
            for argument in config_file.sbatch_arguments:
                content += argument + '\n'

            content += '\n'

            for line in config_file.before_script:
                content += line + '\n'

        content += str(command) \
            .replace('{', '\'{') \
            .replace('}', '}\'') + '\n'

        if config_file and config_file.after_script:
            for line in config_file.after_script:
                content += line + '\n'

        return content

    @staticmethod
    def __get_std_file_name(command: Command) -> str:
        file_name = 'std'

        try:
            command_args = command.apply_to_namespace(Namespace())
            value, options = DatasetSplitterArguments.DATASET_SPLITTER.get_value_and_options(command_args)

            if value == DatasetSplitterArguments.VALUE_CROSS_VALIDATION:
                first_fold = options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, 0)
                last_fold = options.get_int(DatasetSplitterArguments.OPTION_LAST_FOLD, 0)
                file_name += '_fold'

                if first_fold > 0:
                    file_name += '-' + str(first_fold)

                if last_fold > first_fold:
                    file_name += '-' + str(last_fold)
        except ValueError:
            pass

        return file_name

    @staticmethod
    def __write_sbatch_file(args: Namespace, command: Command, config_file: Optional[ConfigFile],
                            job_array: Optional[JobArray], file_name: str) -> Path:
        path = Path(SlurmArguments.SLURM_SAVE_DIR.get_value(args)) / file_name

        with open_writable_file(path) as sbatch_file:
            sbatch_file.write(SlurmRunner.__create_sbatch_script(command, config_file=config_file, job_array=job_array))

        return path

    @staticmethod
    def __read_sbatch_file(path: Path) -> str:
        with open_readable_file(path) as sbatch_file:
            return sbatch_file.read()

    @staticmethod
    def __submit_command(sbatch_file: Path) -> int:
        result = Sbatch().script(sbatch_file).run()

        if result.ok:
            job_name = sbatch_file.stem
            job_id = result.output.split(' ')[-1]
            log.info('Successfully submitted job:\n\n%s',
                     tabulate([['JOBID', job_id], ['NAME', job_name]], tablefmt='plain'))
            return 0

        log.error('Submission to Slurm failed:\n%s', result.output)
        return result.exit_code

    @staticmethod
    def __assign_to_job_arrays(batch: Batch) -> List[JobArray | Command]:
        job_arrays: Dict[Tuple[str, ...], JobArray] = {}
        result: List[JobArray | Command] = []

        for command in batch:
            command_args = command.apply_to_namespace(Namespace())
            value, options = DatasetSplitterArguments.DATASET_SPLITTER.get_value_and_options(command_args)
            job_array: Optional[JobArray] = None

            if value == DatasetSplitterArguments.VALUE_CROSS_VALIDATION:
                first_fold = options.get_int(DatasetSplitterArguments.OPTION_FIRST_FOLD, 0)
                last_fold = options.get_int(DatasetSplitterArguments.OPTION_LAST_FOLD, 0)

                if 0 < first_fold <= last_fold:
                    stripped_down_options = Options({
                        key: value
                        for key, value in options.dictionary.items() if key not in
                        {DatasetSplitterArguments.OPTION_FIRST_FOLD, DatasetSplitterArguments.OPTION_LAST_FOLD}
                    })
                    modified_command = Command.with_argument(command, DatasetSplitterArguments.DATASET_SPLITTER.name,
                                                             value + str(stripped_down_options))
                    key = tuple(string for string in modified_command)

                    if key in job_arrays:
                        existing_job_array = job_arrays[key]
                        existing_job_array.task_ids.update(range(first_fold, last_fold + 1))
                        continue

                    modified_options = Options({
                        key:
                            '$SLURM_ARRAY_TASK_ID' if key in {
                                DatasetSplitterArguments.OPTION_FIRST_FOLD, DatasetSplitterArguments.OPTION_LAST_FOLD
                            } else value
                        for key, value in options.dictionary.items()
                    })
                    job_array = JobArray(command=command,
                                         task_ids=set(range(first_fold, last_fold + 1)),
                                         command_modifier=partial(SlurmRunner.__modify_command,
                                                                  DatasetSplitterArguments.DATASET_SPLITTER.name, value,
                                                                  modified_options),
                                         file_name_modifier=lambda file_name: f'{file_name}_fold-%a')
                    job_arrays[key] = job_array

            result.append(job_array if job_array else command)

        return result

    @staticmethod
    def __modify_command(argument_name: str, argument_value, options: Options, command: Command) -> Command:
        return Command.with_argument(command, argument_name, argument_value + str(options))

    def __init__(self):
        super().__init__(name='slurm')

    @override
    def run_batch(self, args: Namespace, batch: Batch, _: Recipe):
        """
        See :func:`mlrl.testbed.modes.mode_batch.BatchMode.Runner.run_batch`
        """
        save_file = SlurmArguments.SAVE_SLURM_SCRIPTS.get_value(args)
        print_file = SlurmArguments.PRINT_SLURM_SCRIPTS.get_value(args)
        submit_command = not save_file and not print_file and self.__is_command_available()
        num_experiments = len(batch)
        log.info('Submitting %s %s to Slurm...', num_experiments,
                 'experiments' if num_experiments > 1 else 'experiment')
        command_or_job_arrays = self.__assign_to_job_arrays(batch)
        num_jobs = len(command_or_job_arrays)

        for i, command_or_job_array in enumerate(command_or_job_arrays):
            job_array = command_or_job_array if isinstance(command_or_job_array, JobArray) else None
            command = job_array.modified_command if job_array else cast(Command, command_or_job_array)
            log.info('\nSubmitting Slurm job (%s / %s): "%s"', i + 1, num_jobs, str(command))
            slurm_config_file = SlurmRunner.__read_config_file(args)
            sbatch_file = SlurmRunner.__write_sbatch_file(args,
                                                          command,
                                                          config_file=slurm_config_file,
                                                          job_array=job_array,
                                                          file_name=f'sbatch_{i + 1}.sh')

            if save_file:
                log.info('Slurm script saved to file "%s"', sbatch_file)

            if print_file:
                log.info('Content of Slurm script is:\n\n%s', self.__read_sbatch_file(sbatch_file))

            exit_code = self.__submit_command(sbatch_file) if submit_command else 0

            if not save_file:
                sbatch_file.unlink()

            if exit_code != 0:
                sys.exit(exit_code)
