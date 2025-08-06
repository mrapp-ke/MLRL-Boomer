"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to write meta-data to one or several sinks.
"""
from mlrl.util.cli import AUTO, SetArgument
from mlrl.util.options import BooleanOption


class MetaDataArguments:
    """
    Defines command line arguments for configuring the functionality to write meta-data to one or several sinks.
    """

    SAVE_META_DATA = SetArgument(
        '--save-meta-data',
        default=AUTO,
        values={AUTO, BooleanOption.TRUE, BooleanOption.FALSE},
        description='Whether meta-data should be saved to output files or not. If set to "' + AUTO + '", meta-data is '
        + 'saved whenever other output files are written as well.',
    )
