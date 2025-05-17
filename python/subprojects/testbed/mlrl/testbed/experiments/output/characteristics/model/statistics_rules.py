"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that store statistics of rule models.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional

from mlrl.testbed.util.math import divide_or_zero


@dataclass
class BodyStatistics:
    """
    Stores statistics of a rule's body.

    Attributes:
        num_numerical_leq:  The number of numerical conditions using the <= operator
        num_numerical_gr:   The number of numerical conditions using the > operator
        num_ordinal_leq:    The number of ordinal conditions using the <= operator
        num_ordinal_gr:     The number of ordinal conditions using the > operator
        num_nominal_eq:     The number of nominal conditions using the == operator
        num_nominal_neq:    The number of nominal conditions using the != operator
    """
    num_numerical_leq: int = 0
    num_numerical_gr: int = 0
    num_ordinal_leq: int = 0
    num_ordinal_gr: int = 0
    num_nominal_eq: int = 0
    num_nominal_neq: int = 0

    @property
    def num_numerical(self) -> int:
        """
        The number of numerical conditions, regardless of their operator.
        """
        return self.num_numerical_leq + self.num_numerical_gr

    @property
    def num_ordinal(self) -> int:
        """
        The number of ordinal conditions, regardless of their operator.
        """
        return self.num_ordinal_leq + self.num_ordinal_gr

    @property
    def num_nominal(self) -> int:
        """
        The number of nominal conditions, regardless of their operator.
        """
        return self.num_nominal_eq + self.num_nominal_neq

    @property
    def num_conditions(self) -> int:
        """
        The total number of conditions.
        """
        return self.num_numerical + self.num_ordinal + self.num_nominal

    @property
    def fraction_numerical_leq(self) -> float:
        """
        The fraction of numerical conditions using the <= operator.
        """
        return divide_or_zero(self.num_numerical_leq, self.num_conditions)

    @property
    def fraction_numerical_gr(self) -> float:
        """
        The fraction of numerical conditions using the > operator.
        """
        return divide_or_zero(self.num_numerical_gr, self.num_conditions)

    @property
    def fraction_ordinal_leq(self) -> float:
        """
        The fraction of ordinal conditions using the <= operator.
        """
        return divide_or_zero(self.num_ordinal_leq, self.num_conditions)

    @property
    def fraction_ordinal_gr(self) -> float:
        """
        The fraction of ordinal conditions using the > operator.
        """
        return divide_or_zero(self.num_ordinal_gr, self.num_conditions)

    @property
    def fraction_nominal_eq(self) -> float:
        """
        The fraction of nominal conditions using the == operator.
        """
        return divide_or_zero(self.num_nominal_eq, self.num_conditions)

    @property
    def fraction_nominal_neq(self) -> float:
        """
        The fraction of nominal conditions using the != operator.
        """
        return divide_or_zero(self.num_nominal_neq, self.num_conditions)

    def __add__(self, other: 'BodyStatistics') -> 'BodyStatistics':
        return BodyStatistics(num_numerical_leq=self.num_numerical_leq + other.num_numerical_leq,
                              num_numerical_gr=self.num_numerical_gr + other.num_numerical_gr,
                              num_ordinal_leq=self.num_ordinal_leq + other.num_ordinal_leq,
                              num_ordinal_gr=self.num_ordinal_gr + other.num_ordinal_gr,
                              num_nominal_eq=self.num_nominal_eq + other.num_nominal_eq,
                              num_nominal_neq=self.num_nominal_neq + other.num_nominal_neq)


@dataclass
class HeadStatistics:
    """
    Stores statistics of a rule's head.

    Attributes:
        num_positive_predictions:   The number of positive predictions
        num_negative_predictions:   The number of negative predictions
    """
    num_positive_predictions: int = 0
    num_negative_predictions: int = 0

    @property
    def num_predictions(self) -> int:
        """
        The total number of predictions.
        """
        return self.num_positive_predictions + self.num_negative_predictions

    @property
    def fraction_positive_predictions(self) -> float:
        """
        The fraction of positive predictions.
        """
        return divide_or_zero(self.num_positive_predictions, self.num_predictions)

    @property
    def fraction_negative_predictions(self) -> float:
        """
        The fraction of negative predictions.
        """
        return 1 - self.fraction_positive_predictions

    def __add__(self, other: 'HeadStatistics') -> 'HeadStatistics':
        return HeadStatistics(num_positive_predictions=self.num_positive_predictions + other.num_positive_predictions,
                              num_negative_predictions=self.num_negative_predictions + other.num_negative_predictions)


@dataclass
class RuleStatistics:
    """
    Stores statistics of a rule.

    Attributes:
        body_statistics:    The statistics of the rule's body
        head_statistics:    The statistics of the rule's head
    """
    body_statistics: BodyStatistics = field(default_factory=BodyStatistics)
    head_statistics: HeadStatistics = field(default_factory=HeadStatistics)

    def __add__(self, other: 'RuleStatistics') -> 'RuleStatistics':
        return RuleStatistics(body_statistics=self.body_statistics + other.body_statistics,
                              head_statistics=self.head_statistics + other.head_statistics)


@dataclass
class RuleModelStatistics:
    """
    Stores statistics of a rule model.

    Attributes:
        default_rule_statistics:    The statistics of the default rule, if any
        rule_statistics:            A list that stores the statistics all other rules
    """
    default_rule_statistics: Optional[RuleStatistics] = None
    rule_statistics: List[RuleStatistics] = field(default_factory=list)

    @property
    def has_default_rule(self) -> bool:
        """
        True, if the model has a default rule, False otherwise.
        """
        return self.default_rule_statistics is not None

    @property
    def num_rules(self) -> int:
        """
        The number of rules.
        """
        return len(self.rule_statistics)

    @cached_property
    def min_conditions(self) -> int:
        """
        The minimum number of conditions per rule.
        """
        rule_statistics = self.rule_statistics

        if rule_statistics:
            return min(map(lambda statistics: statistics.body_statistics.num_conditions, rule_statistics))
        return 0

    @cached_property
    def max_conditions(self) -> int:
        """
        The maximum number of conditions per rule.
        """
        rule_statistics = self.rule_statistics

        if rule_statistics:
            return max(map(lambda statistics: statistics.body_statistics.num_conditions, rule_statistics))
        return 0

    @cached_property
    def min_predictions(self) -> int:
        """
        The minimum number of predictions per rule.
        """
        rule_statistics = self.rule_statistics

        if rule_statistics:
            return min(map(lambda statistics: statistics.head_statistics.num_predictions, rule_statistics))
        return 0

    @cached_property
    def max_predictions(self) -> int:
        """
        The maximum number of predictions per rule.
        """
        rule_statistics = self.rule_statistics

        if rule_statistics:
            return max(map(lambda statistics: statistics.head_statistics.num_predictions, rule_statistics))
        return 0
