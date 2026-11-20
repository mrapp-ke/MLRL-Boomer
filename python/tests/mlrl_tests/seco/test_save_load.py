"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Tests for saving SeCo learners to HDF5 files and loading them from such files.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlrl.common import load_model
from mlrl.common.cython.model_file import ModelFileReader
from mlrl.seco.learners import SeCoClassifier

from ..common.pickling import assert_identical_learners, create_classification_dataset


@pytest.mark.seco
@pytest.mark.classification
class TestSeCoClassifierSaveLoad:
    """
    Tests for saving fitted instances of the class `SeCoClassifier` and loading them.
    """

    @pytest.mark.parametrize(
        'parameters',
        [
            pytest.param({}, id='default'),
            pytest.param({'head_type': 'partial'}, id='partial-heads'),
            pytest.param({'heuristic': 'precision', 'max_rules': 7}, id='non-default-hyperparameters'),
            pytest.param({'prediction_format': 'sparse'}, id='sparse-predictions'),
        ],
    )
    def test_save_load(self, parameters: dict[str, Any], tmp_path: Path):
        x, y, ordinal_feature_indices, nominal_feature_indices = create_classification_dataset()
        learner = SeCoClassifier(max_rules=10, random_state=1)
        learner.set_params(**parameters)
        learner.fit(
            x, y, ordinal_feature_indices=ordinal_feature_indices, nominal_feature_indices=nominal_feature_indices
        )
        path = tmp_path / 'model.h5'
        learner.save_model(path)
        loaded = load_model(path)
        assert type(loaded) is SeCoClassifier
        assert loaded.get_params() == learner.get_params()
        assert_identical_learners(learner, loaded, x)
        assert ModelFileReader(path.read_bytes()).get_attribute('learner', 'package') == 'mlrl-seco'

    def test_constant_labels(self, tmp_path: Path):
        x, y, _, _ = create_classification_dataset()
        learner = SeCoClassifier(max_rules=10, random_state=1).fit(x, np.zeros_like(y))
        path = tmp_path / 'model.h5'
        learner.save_model(path)
        loaded = SeCoClassifier.load(path)
        np.testing.assert_array_equal(loaded.predict(x), learner.predict(x))
