"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for reading YAML files.
"""
from pathlib import Path
from typing import Any, Dict

import yamale
import yaml

from mlrl.testbed.util.io import open_readable_file

from mlrl.util.validation import ValidationError


def read_yaml(yaml_file_path: Path) -> Dict[Any, Any]:
    """
    Reads and returns the content of a YAML file as a dictionary.

    :param yaml_file_path:  The path to a YAML file
    :return:                The content of the YAML file as a dictionary
    """
    with open_readable_file(yaml_file_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def read_and_validate_yaml(yaml_file_path: Path, schema_file_path: Path) -> Dict[Any, Any]:
    """
    Reads and returns the content of a YAML file as a dictionary. The content is validated against a given schema file.
    If it is malformed, a `ValidationError` is raised.

    :param yaml_file_path:      The path to a YAML file
    :param schema_file_path:    The path to a schema file
    :return:                    The content of the YAML file as a dictionary
    """
    schema = yamale.make_schema(schema_file_path)
    data = yamale.make_data(yaml_file_path)

    try:
        yamale.validate(schema, data)
    except yamale.YamaleError as error:
        with open_readable_file(schema_file_path) as schema_file:
            raise ValidationError(
                f'YAML file "{yaml_file_path}" is malformed: {error}\nThe expected schema is:\n{schema_file.read()}'
            ) from error

    return data[0][0]
