"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Tests for pickling and unpickling fitted SeCo learners.
"""

from typing import Any

import pytest
from mlrl.common.cython.rule_model import RuleList
from mlrl.seco.learners import SeCoClassifier

from ..common.pickling import assert_identical_after_pickling, create_classification_dataset


@pytest.mark.seco
@pytest.mark.classification
class TestSeCoClassifierPickling:
    """
    Tests for pickling and unpickling fitted instances of the class `SeCoClassifier`.
    """

    @pytest.mark.parametrize(
        'parameters',
        [
            pytest.param({}, id='default'),
            pytest.param({'head_type': 'partial'}, id='partial-heads'),
            pytest.param({'feature_binning': 'equal-width'}, id='feature-binning'),
            pytest.param({'prediction_format': 'sparse'}, id='sparse-predictions'),
        ],
    )
    def test_pickle(self, parameters: dict[str, Any]):
        x, y, ordinal_feature_indices, nominal_feature_indices = create_classification_dataset()
        learner = SeCoClassifier(max_rules=10, random_state=1, **parameters)
        learner.fit(
            x, y, ordinal_feature_indices=ordinal_feature_indices, nominal_feature_indices=nominal_feature_indices
        )
        assert isinstance(learner.model_, RuleList)
        assert_identical_after_pickling(learner, x)
