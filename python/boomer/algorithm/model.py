#!/usr/bin/python

"""
@author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides classes for representing the model learned by a classifier or ranker.
"""
from typing import List

from boomer.algorithm._model import Rule

# Type alias for a theory, which is a list containing several rules
Theory = List[Rule]
