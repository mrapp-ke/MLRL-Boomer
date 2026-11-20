"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Tests for pickling and unpickling fitted BOOMER learners.
"""

import pickle
from typing import Any

import pytest
from mlrl.boosting.learners import BoomerClassifier, BoomerRegressor
from mlrl.common.cython.output_space_info import LabelVectorSet, NoOutputSpaceInfo
from mlrl.common.cython.probability_calibration import (
    IsotonicJointProbabilityCalibrationModel,
    IsotonicMarginalProbabilityCalibrationModel,
)
from mlrl.common.cython.rule_model import RuleList
from scipy.sparse import csr_matrix

from ..common.pickling import assert_identical_after_pickling, create_classification_dataset, create_regression_dataset


@pytest.mark.boosting
@pytest.mark.classification
class TestBoomerClassifierPickling:
    """
    Tests for pickling and unpickling fitted instances of the class `BoomerClassifier`.
    """

    @pytest.mark.parametrize(
        'parameters',
        [
            pytest.param({}, id='default'),
            pytest.param({'head_type': 'single'}, id='single-output-heads'),
            pytest.param({'head_type': 'partial-fixed'}, id='partial-heads'),
            pytest.param({'head_type': 'complete', 'loss': 'logistic-non-decomposable'}, id='complete-heads'),
            pytest.param({'feature_binning': 'equal-width'}, id='feature-binning'),
            pytest.param({'default_rule': 'false'}, id='no-default-rule'),
            pytest.param({'statistic_type': '32-bit'}, id='32-bit-statistics'),
            pytest.param({'binary_predictor': 'example-wise'}, id='binary-predictor-example-wise'),
            pytest.param(
                {
                    'binary_predictor': 'gfm',
                    'probability_predictor': 'marginalized',
                    'marginal_probability_calibration': 'isotonic',
                    'joint_probability_calibration': 'isotonic',
                },
                id='gfm-isotonic-calibration',
            ),
            pytest.param(
                {
                    'probability_predictor': 'output-wise',
                    'marginal_probability_calibration': 'isotonic',
                },
                id='output-wise-isotonic-calibration',
            ),
            pytest.param({'prediction_format': 'sparse'}, id='sparse-predictions'),
        ],
    )
    def test_pickle(self, parameters: dict[str, Any]):
        x, y, ordinal_feature_indices, nominal_feature_indices = create_classification_dataset()
        learner = BoomerClassifier(max_rules=15, random_state=1, **parameters)
        learner.fit(
            x, y, ordinal_feature_indices=ordinal_feature_indices, nominal_feature_indices=nominal_feature_indices
        )
        assert isinstance(learner.model_, RuleList)
        assert_identical_after_pickling(learner, x)

    def test_pickle_sparse_features(self):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=15, random_state=1, feature_format='sparse')
        learner.fit(csr_matrix(x), y)
        assert_identical_after_pickling(learner, csr_matrix(x))

    def test_pickle_native_objects(self):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(
            max_rules=15,
            random_state=1,
            binary_predictor='gfm',
            probability_predictor='marginalized',
            marginal_probability_calibration='isotonic',
            joint_probability_calibration='isotonic',
        )
        learner.fit(x, y)
        assert isinstance(learner.output_space_info_, LabelVectorSet)
        assert isinstance(learner.marginal_probability_calibration_model_, IsotonicMarginalProbabilityCalibrationModel)
        assert isinstance(learner.joint_probability_calibration_model_, IsotonicJointProbabilityCalibrationModel)

        for native_object in (
            learner.model_,
            learner.output_space_info_,
            learner.marginal_probability_calibration_model_,
            learner.joint_probability_calibration_model_,
        ):
            reduced = native_object.__reduce__()
            assert reduced[0] is type(native_object)
            assert isinstance(reduced[2], bytes), 'The state of a native object must be a bytes object'
            loaded = pickle.loads(pickle.dumps(native_object))
            assert type(loaded) is type(native_object)

    def test_no_output_space_info(self):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=5, random_state=1)
        learner.fit(x, y)
        assert isinstance(learner.output_space_info_, NoOutputSpaceInfo)
        assert isinstance(pickle.loads(pickle.dumps(learner.output_space_info_)), NoOutputSpaceInfo)

    @pytest.mark.parametrize(
        'cls, state',
        [
            pytest.param(RuleList, (4, ([], False, 0)), id='rule-list'),
            pytest.param(LabelVectorSet, (3, []), id='label-vector-set'),
            pytest.param(IsotonicMarginalProbabilityCalibrationModel, (0, []), id='isotonic-marginal'),
            pytest.param(IsotonicJointProbabilityCalibrationModel, (0, []), id='isotonic-joint'),
        ],
    )
    def test_unpickling_old_state_fails(self, cls, state):
        native_object = cls.__new__(cls)

        with pytest.raises(ValueError, match='pickled by an older version of the library'):
            native_object.__setstate__(state)

    def test_unpickling_corrupted_state_fails(self):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=5, random_state=1)
        learner.fit(x, y)
        state = learner.model_.__reduce__()[2]
        rule_list = RuleList.__new__(RuleList)

        with pytest.raises(RuntimeError):
            rule_list.__setstate__(state[: len(state) // 2])

        with pytest.raises(RuntimeError):
            rule_list.__setstate__(b'')

    def test_unpickling_wrong_type_fails(self):
        x, y, _, _ = create_classification_dataset()
        learner = BoomerClassifier(max_rules=5, random_state=1, marginal_probability_calibration='isotonic')
        learner.fit(x, y)
        state = learner.model_.__reduce__()[2]

        # The state of a rule model does not contain a model for the calibration of probabilities
        with pytest.raises(RuntimeError):
            IsotonicMarginalProbabilityCalibrationModel.__new__(
                IsotonicMarginalProbabilityCalibrationModel
            ).__setstate__(state)


@pytest.mark.boosting
@pytest.mark.regression
class TestBoomerRegressorPickling:
    """
    Tests for pickling and unpickling fitted instances of the class `BoomerRegressor`.
    """

    @pytest.mark.parametrize(
        'parameters',
        [
            pytest.param({}, id='default'),
            pytest.param({'head_type': 'single'}, id='single-output-heads'),
            pytest.param({'head_type': 'partial-fixed'}, id='partial-heads'),
            pytest.param({'statistic_type': '32-bit'}, id='32-bit-statistics'),
        ],
    )
    def test_pickle(self, parameters: dict[str, Any]):
        x, y = create_regression_dataset()
        learner = BoomerRegressor(max_rules=15, random_state=1, **parameters)
        learner.fit(x, y)
        assert_identical_after_pickling(learner, x)
