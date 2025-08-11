"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run experiments via the Slurm Workload Manager.
"""
import logging as log
import sys

from argparse import Namespace
from pathlib import Path
from typing import List, Optional, override

import yamale

from tabulate import tabulate

from mlrl.testbed_slurm.arguments import SlurmArguments
from mlrl.testbed_slurm.sbatch import Sbatch

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.modes.mode_batch import Batch, BatchMode
from mlrl.testbed.util.io import open_readable_file, open_writable_file


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
    def __create_sbatch_script(command: Command, config_file: Optional[ConfigFile]) -> str:
        base_dir = Path(command.argument_dict[OutputArguments.BASE_DIR.name])
        result_dir = base_dir / command.argument_dict[ResultDirectoryArguments.RESULT_DIR.name]
        content = '#!/bin/sh\n\n'
        content += '#SBATCH --output=' + str(result_dir / 'std.out') + '\n'
        content += '#SBATCH --error=' + str(result_dir / 'std.err') + '\n'

        if config_file:
            for argument in config_file.sbatch_arguments:
                content += argument + '\n'

            content += '\n'

            for line in config_file.before_script:
                content += line + '\n'

        content += str(command) + '\n'

        if config_file and config_file.after_script:
            for line in config_file.after_script:
                content += line + '\n'

        return content

    @staticmethod
    def __write_sbatch_file(args: Namespace, command: Command, config_file: Optional[ConfigFile],
                            file_name: str) -> Path:
        path = Path(SlurmArguments.SLURM_SAVE_DIR.get_value(args)) / file_name

        with open_writable_file(path) as sbatch_file:
            sbatch_file.write(SlurmRunner.__create_sbatch_script(command, config_file))

        return path

    @staticmethod
    def __read_sbatch_file(path: Path) -> str:
        with open_readable_file(path) as sbatch_file:
            return sbatch_file.read()

    def __submit_command(self, sbatch_file: Path) -> int:
        result = Sbatch().script(sbatch_file).run()

        if result.ok:
            job_name = sbatch_file.stem
            job_id = result.output.split(' ')[-1]
            log.info('Successfully submitted job:\n\n%s',
                     tabulate([['JOBID', job_id], ['NAME', job_name]], tablefmt='plain'))
            return 0

        log.error('Submission to Slurm failed:\n%s', result.output)
        return result.exit_code

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

        for i, command in enumerate(batch):
            log.info('\nSubmitting experiment (%s / %s): "%s"', i + 1, num_experiments, str(command))
            slurm_config_file = SlurmRunner.__read_config_file(args)
            sbatch_file = SlurmRunner.__write_sbatch_file(args,
                                                          command,
                                                          slurm_config_file,
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
