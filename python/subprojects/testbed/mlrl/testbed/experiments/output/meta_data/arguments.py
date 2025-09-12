"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to write meta-data to one or several sinks.
"""
from mlrl.util.cli import AUTO, BoolArgument, SetArgument
from mlrl.util.options import BooleanOption


class MetaDataArguments:
    """
    Defines command line arguments for configuring the functionality to write meta-data to one or several sinks.
    """

    PRINT_META_DATA = BoolArgument(
        '--print-meta-data',
        default=False,
        description='Whether meta-data should be printed on the console or not.',
    )

    SAVE_META_DATA = SetArgument(
        '--save-meta-data',
        default=AUTO,
        values={AUTO, BooleanOption.TRUE, BooleanOption.FALSE},
        description='Whether meta-data should be saved to output files or not. If set to "' + AUTO + '", meta-data is '
        + 'saved whenever other output files are written as well.',
    )
