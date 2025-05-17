"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for representing datasets.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class AttributeType(Enum):
    """
    All supported types of attributes.
    """
    NUMERICAL = auto()
    ORDINAL = auto()
    NOMINAL = auto()


@dataclass
class Attribute:
    """
    An attribute, e.g., a feature, a ground truth label, or a regression score, that is contained by a data set.

    Attributes:
        name:           The name of the attribute
        attribute_type: The type of the attribute
        nominal_values: A list that contains the possible values in case of a nominal feature
    """
    name: str
    attribute_type: AttributeType
    nominal_values: Optional[List[str]] = None
